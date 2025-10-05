"""
MoEFsDiC U-Net Model for Super-Resolution
[SR 专用] 超分辨率专用模型，支持 LR 图像条件输入
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append('..')

from ..modules.conv_blocks import MoE_ConvBlock, Dilated_Fusion_Block
from ..modules.freq_module import Freq_Global_Module
from ..modules.experts import DepthwiseSeparableConv2d


class TimestepEmbedding(nn.Module):
    """时间步嵌入模块"""
    def __init__(self, dim, max_period=10000):
        super(TimestepEmbedding, self).__init__()
        self.dim = dim
        self.max_period = max_period
        
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)
        
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period)) * 
            torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.float() * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        
        emb = self.mlp(embedding)
        return emb


class ConditionEmbedding(nn.Module):
    """条件嵌入模块（用于类别等条件）"""
    def __init__(self, num_classes, dim):
        super(ConditionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, dim)
    
    def forward(self, c):
        return self.embedding(c)


class DownBlock(nn.Module):
    """
    [SR 增强] 编码器下采样块，支持 LR 特征融合
    """
    def __init__(self, in_channels, out_channels, t_dim, c_dim, 
                 num_experts, k_active, use_dsconv, moe_enabled, lr_channels=0):
        super(DownBlock, self).__init__()
        self.moe_enabled = moe_enabled
        
        # 选择使用 MoE 块或扩张融合块
        if moe_enabled:
            self.conv_block = MoE_ConvBlock(
                in_channels, out_channels, t_dim, c_dim, 
                num_experts, k_active, use_dsconv, lr_channels=lr_channels
            )
        else:
            self.conv_block = Dilated_Fusion_Block(
                in_channels, out_channels, use_dsconv
            )
        
        # 下采样
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, t_emb, c_emb, lr_feat=None):
        """
        Args:
            x: 输入 [B, in_channels, H, W]
            t_emb: 时间嵌入 [B, t_dim]
            c_emb: 条件嵌入 [B, c_dim]
            lr_feat: LR 特征 [B, lr_channels, H, W]
        Returns:
            x_down: 下采样后的特征
            x_skip: 跳跃连接特征
            logits: Router logits
        """
        x_skip, logits = self.conv_block(x, t_emb, c_emb, lr_feat=lr_feat)
        x_down = self.downsample(x_skip)
        return x_down, x_skip, logits


class UpBlock(nn.Module):
    """
    [SR 增强] 解码器上采样块，支持 LR 特征融合
    """
    def __init__(self, in_channels, skip_channels, out_channels, t_dim, c_dim,
                 num_experts, k_active, use_dsconv, moe_enabled, lr_channels=0):
        super(UpBlock, self).__init__()
        self.moe_enabled = moe_enabled
        
        # 上采样
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        # 合并跳跃连接后的卷积块
        # [SR 增强] 跳跃连接中已经融合了 LR 特征，所以这里不需要额外通道
        merged_channels = in_channels + skip_channels
        if moe_enabled:
            self.conv_block = MoE_ConvBlock(
                merged_channels, out_channels, t_dim, c_dim,
                num_experts, k_active, use_dsconv, lr_channels=lr_channels
            )
        else:
            self.conv_block = Dilated_Fusion_Block(
                merged_channels, out_channels, use_dsconv
            )
    
    def forward(self, x, x_skip, t_emb, c_emb, lr_feat=None):
        """
        Args:
            x: 来自下层的特征
            x_skip: 跳跃连接特征（已融合 LR）
            t_emb: 时间嵌入
            c_emb: 条件嵌入
            lr_feat: 当前尺度的 LR 特征
        Returns:
            out: 输出特征
            logits: Router logits
        """
        # 上采样
        x_up = self.upsample(x)
        
        # 合并跳跃连接
        x_merged = torch.cat([x_up, x_skip], dim=1)
        
        # 卷积处理（传递 LR 特征）
        out, logits = self.conv_block(x_merged, t_emb, c_emb, lr_feat=lr_feat)
        return out, logits


class MoEFsDiC_SR_UNet(nn.Module):
    """
    MoEFsDiC U-Net for Super-Resolution
    [SR 专用] 集成 LR 图像条件的超分辨率扩散模型
    """
    def __init__(self, config):
        super(MoEFsDiC_SR_UNet, self).__init__()
        
        # 解析配置
        self.in_channels = config['model']['in_channels']
        self.model_channels = config['model']['model_channels']
        self.channel_mults = config['model']['channel_mults']
        self.t_dim = config['model']['t_dim']
        self.c_dim = config['model']['c_dim']
        
        self.moe_enabled = config['moe']['enabled']
        self.num_experts = config['moe']['num_experts']
        self.k_active = config['moe']['k_active']
        
        self.freq_enabled = config['freq']['enabled']
        self.use_dsconv = config['train']['use_dsconv_global']
        
        # [SR 配置]
        self.scale_factor = config['sr']['scale_factor']
        self.lr_channels = config['sr']['lr_channels']
        
        # Logits 收集列表
        self.moe_logits_list = []
        
        # 时间步嵌入
        self.time_embedding = TimestepEmbedding(self.t_dim)
        
        # 条件嵌入（假设 10 个类别，可根据需要调整）
        self.condition_embedding = ConditionEmbedding(num_classes=10, dim=self.c_dim)
        
        # [SR 增强] LR 图像编码器
        # 将 LR 图像编码为潜在特征
        self.lr_encoder = nn.Sequential(
            nn.Conv2d(self.in_channels, self.lr_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.lr_channels, self.lr_channels, kernel_size=3, padding=1)
        )
        
        # [SR 增强] LR 特征的多尺度下采样路径
        # 为每个 U-Net 层级创建对应的 LR 下采样
        self.lr_downsamplers = nn.ModuleList()
        for i in range(len(self.channel_mults)):
            self.lr_downsamplers.append(
                nn.Conv2d(self.lr_channels, self.lr_channels, kernel_size=3, stride=2, padding=1)
            )
        
        # 输入投影
        self.input_proj = nn.Conv2d(self.in_channels, self.model_channels, kernel_size=3, padding=1)
        
        # 编码器
        self.encoder = nn.ModuleList()
        in_ch = self.model_channels
        self.encoder_channels = [in_ch]
        
        for mult in self.channel_mults[:-1]:  # 除了最后一层
            out_ch = self.model_channels * mult
            self.encoder.append(DownBlock(
                in_ch, out_ch, self.t_dim, self.c_dim,
                self.num_experts, self.k_active, self.use_dsconv, self.moe_enabled,
                lr_channels=self.lr_channels  # [SR] 传递 LR 通道数
            ))
            in_ch = out_ch
            self.encoder_channels.append(in_ch)
        
        # 瓶颈层
        bottleneck_ch = self.model_channels * self.channel_mults[-1]
        self.bottleneck_proj = nn.Conv2d(in_ch, bottleneck_ch, kernel_size=3, stride=2, padding=1)
        
        if self.freq_enabled:
            # 使用频率域全局模块
            self.bottleneck = Freq_Global_Module(bottleneck_ch, self.t_dim)
        else:
            # 使用标准卷积块
            if self.moe_enabled:
                self.bottleneck = MoE_ConvBlock(
                    bottleneck_ch, bottleneck_ch, self.t_dim, self.c_dim,
                    self.num_experts, self.k_active, self.use_dsconv,
                    lr_channels=self.lr_channels  # [SR]
                )
            else:
                self.bottleneck = Dilated_Fusion_Block(bottleneck_ch, bottleneck_ch, self.use_dsconv)
        
        # 解码器
        self.decoder = nn.ModuleList()
        in_ch = bottleneck_ch
        
        for i in range(len(self.channel_mults) - 1, 0, -1):
            skip_ch = self.encoder_channels[i]
            out_ch = self.model_channels * self.channel_mults[i-1]
            self.decoder.append(UpBlock(
                in_ch, skip_ch, out_ch, self.t_dim, self.c_dim,
                self.num_experts, self.k_active, self.use_dsconv, self.moe_enabled,
                lr_channels=self.lr_channels  # [SR]
            ))
            in_ch = out_ch
        
        # 最后一个上采样块（回到原始通道数）
        self.decoder.append(UpBlock(
            in_ch, self.encoder_channels[0], self.model_channels,
            self.t_dim, self.c_dim, self.num_experts, self.k_active, 
            self.use_dsconv, self.moe_enabled,
            lr_channels=self.lr_channels  # [SR]
        ))
        
        # [SR 增强] 最终上采样层（空间分辨率提升）
        # 使用 Sub-pixel Convolution (Pixel Shuffle) 实现高效上采样
        self.final_upsample = nn.Sequential(
            nn.Conv2d(self.model_channels, self.model_channels * (self.scale_factor ** 2), 
                     kernel_size=3, padding=1),
            nn.PixelShuffle(self.scale_factor),  # 空间上采样
            nn.GELU()
        )
        
        # 输出投影（预测噪声）
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, self.model_channels),
            nn.GELU(),
            nn.Conv2d(self.model_channels, self.in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t, c, lr_image):
        """
        前向传播
        
        Args:
            x: 输入噪声图像/特征 [B, in_channels, H, W]
            t: 时间步 [B]
            c: 条件标签 [B]
            lr_image: LR 图像 [B, in_channels, H_lr, W_lr] (H_lr = H / scale_factor)
        
        Returns:
            out: 预测的噪声 [B, in_channels, H, W]
            moe_logits_list: 所有 MoE 块的 logits 列表
        """
        # 清空 Logits 列表（每次前向传播开始时）
        self.moe_logits_list = []
        
        # 嵌入
        t_emb = self.time_embedding(t)  # [B, t_dim]
        c_emb = self.condition_embedding(c)  # [B, c_dim]
        
        # [SR 核心] 编码 LR 图像
        lr_feat = self.lr_encoder(lr_image)  # [B, lr_channels, H_lr, W_lr]
        
        # [SR 核心] 生成多尺度 LR 特征
        # 将 LR 特征上采样到与输入 x 相同的尺寸
        lr_feat_base = F.interpolate(lr_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        # 为每个层级生成对应尺度的 LR 特征
        lr_feats_pyramid = [lr_feat_base]  # 第一层：原始尺寸
        current_lr = lr_feat_base
        for downsampler in self.lr_downsamplers[:-1]:  # 生成下采样的 LR 特征
            current_lr = downsampler(current_lr)
            lr_feats_pyramid.append(current_lr)
        
        # 输入投影
        x = self.input_proj(x)  # [B, model_channels, H, W]
        
        # 编码器：收集跳跃连接
        skip_connections = []
        for idx, down_block in enumerate(self.encoder):
            # 使用对应尺度的 LR 特征
            lr_at_scale = lr_feats_pyramid[idx]
            x, x_skip, logits = down_block(x, t_emb, c_emb, lr_feat=lr_at_scale)
            skip_connections.append(x_skip)
            if logits is not None:
                self.moe_logits_list.append(logits)
        
        # 瓶颈层
        x = self.bottleneck_proj(x)
        if self.freq_enabled:
            x = self.bottleneck(x, t_emb)
        else:
            # 瓶颈层的 LR 特征
            lr_bottleneck = lr_feats_pyramid[len(self.encoder)] if len(lr_feats_pyramid) > len(self.encoder) else lr_feats_pyramid[-1]
            lr_bottleneck = F.adaptive_avg_pool2d(lr_bottleneck, x.shape[2:])
            x, logits = self.bottleneck(x, t_emb, c_emb, lr_feat=lr_bottleneck)
            if logits is not None:
                self.moe_logits_list.append(logits)
        
        # 解码器：使用跳跃连接
        for i, up_block in enumerate(self.decoder):
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0:
                x_skip = skip_connections[skip_idx]
            else:
                # 最后一层没有对应的 skip connection
                x_skip = torch.zeros(x.size(0), self.encoder_channels[0], 
                                    x.size(2) * 2, x.size(3) * 2, 
                                    device=x.device, dtype=x.dtype)
            
            # 使用对应尺度的 LR 特征
            lr_idx = len(self.encoder) - 1 - i
            lr_idx = max(0, min(lr_idx, len(lr_feats_pyramid) - 1))
            lr_at_scale = lr_feats_pyramid[lr_idx]
            
            # 确保 LR 特征尺寸匹配
            if lr_at_scale.shape[2:] != x_skip.shape[2:]:
                lr_at_scale = F.interpolate(lr_at_scale, size=x_skip.shape[2:], 
                                           mode='bilinear', align_corners=False)
            
            x, logits = up_block(x, x_skip, t_emb, c_emb, lr_feat=lr_at_scale)
            if logits is not None:
                self.moe_logits_list.append(logits)
        
        # [SR 关键] 最终上采样：将特征上采样到 HR 尺寸
        x = self.final_upsample(x)  # [B, model_channels, H_hr, W_hr]
        
        # 输出投影：预测噪声
        out = self.output_proj(x)  # [B, in_channels, H_hr, W_hr]
        
        return out, self.moe_logits_list

