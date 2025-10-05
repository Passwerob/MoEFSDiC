"""
MoEFsDiC U-Net Model
主模型：集成 MoE、频率域增强的 U-Net 骨干网络
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
    """
    时间步嵌入模块
    将标量时间步转换为高维嵌入向量
    """
    def __init__(self, dim, max_period=10000):
        super(TimestepEmbedding, self).__init__()
        self.dim = dim
        self.max_period = max_period
        
        # MLP 用于进一步处理正弦嵌入
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        """
        Args:
            t: 时间步 [B] 或 [B, 1]
        Returns:
            emb: 时间嵌入 [B, dim]
        """
        if len(t.shape) == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        # 正弦位置编码
        half_dim = self.dim // 2
        freqs = torch.exp(
            -torch.log(torch.tensor(self.max_period)) * 
            torch.arange(0, half_dim, dtype=torch.float32, device=t.device) / half_dim
        )
        args = t.float() * freqs.unsqueeze(0)  # [B, half_dim]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
        
        # MLP 处理
        emb = self.mlp(embedding)
        return emb


class ConditionEmbedding(nn.Module):
    """
    条件嵌入模块
    将条件信息（如类别标签）转换为嵌入向量
    """
    def __init__(self, num_classes, dim):
        super(ConditionEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_classes, dim)
    
    def forward(self, c):
        """
        Args:
            c: 条件标签 [B]
        Returns:
            emb: 条件嵌入 [B, dim]
        """
        return self.embedding(c)


class DownBlock(nn.Module):
    """
    编码器下采样块
    """
    def __init__(self, in_channels, out_channels, t_dim, c_dim, 
                 num_experts, k_active, use_dsconv, moe_enabled):
        super(DownBlock, self).__init__()
        self.moe_enabled = moe_enabled
        
        # 选择使用 MoE 块或扩张融合块
        if moe_enabled:
            self.conv_block = MoE_ConvBlock(
                in_channels, out_channels, t_dim, c_dim, 
                num_experts, k_active, use_dsconv
            )
        else:
            self.conv_block = Dilated_Fusion_Block(
                in_channels, out_channels, use_dsconv
            )
        
        # 下采样
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x, t_emb, c_emb):
        """
        Args:
            x: 输入 [B, in_channels, H, W]
            t_emb: 时间嵌入 [B, t_dim]
            c_emb: 条件嵌入 [B, c_dim]
        Returns:
            x_down: 下采样后的特征 [B, out_channels, H/2, W/2]
            x_skip: 跳跃连接特征 [B, out_channels, H, W]
            logits: Router logits (如果 MoE 启用)
        """
        x_skip, logits = self.conv_block(x, t_emb, c_emb)
        x_down = self.downsample(x_skip)
        return x_down, x_skip, logits


class UpBlock(nn.Module):
    """
    解码器上采样块
    """
    def __init__(self, in_channels, skip_channels, out_channels, t_dim, c_dim,
                 num_experts, k_active, use_dsconv, moe_enabled):
        super(UpBlock, self).__init__()
        self.moe_enabled = moe_enabled
        
        # 上采样
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1)
        
        # 合并跳跃连接后的卷积块
        merged_channels = in_channels + skip_channels
        if moe_enabled:
            self.conv_block = MoE_ConvBlock(
                merged_channels, out_channels, t_dim, c_dim,
                num_experts, k_active, use_dsconv
            )
        else:
            self.conv_block = Dilated_Fusion_Block(
                merged_channels, out_channels, use_dsconv
            )
    
    def forward(self, x, x_skip, t_emb, c_emb):
        """
        Args:
            x: 来自下层的特征 [B, in_channels, H, W]
            x_skip: 跳跃连接特征 [B, skip_channels, 2H, 2W]
            t_emb: 时间嵌入 [B, t_dim]
            c_emb: 条件嵌入 [B, c_dim]
        Returns:
            out: 输出特征 [B, out_channels, 2H, 2W]
            logits: Router logits (如果 MoE 启用)
        """
        # 上采样
        x_up = self.upsample(x)
        
        # 合并跳跃连接
        x_merged = torch.cat([x_up, x_skip], dim=1)
        
        # 卷积处理
        out, logits = self.conv_block(x_merged, t_emb, c_emb)
        return out, logits


class MoEFsDiC_UNet(nn.Module):
    """
    MoEFsDiC U-Net 主模型
    混合专家系统 + 频率域增强的扩散模型骨干网络
    """
    def __init__(self, config):
        super(MoEFsDiC_UNet, self).__init__()
        
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
        
        # Logits 收集列表
        self.moe_logits_list = []
        
        # 时间步嵌入
        self.time_embedding = TimestepEmbedding(self.t_dim)
        
        # 条件嵌入（假设 10 个类别，可根据需要调整）
        self.condition_embedding = ConditionEmbedding(num_classes=10, dim=self.c_dim)
        
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
                self.num_experts, self.k_active, self.use_dsconv, self.moe_enabled
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
                    self.num_experts, self.k_active, self.use_dsconv
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
                self.num_experts, self.k_active, self.use_dsconv, self.moe_enabled
            ))
            in_ch = out_ch
        
        # 最后一个上采样块（回到原始通道数）
        self.decoder.append(UpBlock(
            in_ch, self.encoder_channels[0], self.model_channels,
            self.t_dim, self.c_dim, self.num_experts, self.k_active, 
            self.use_dsconv, self.moe_enabled
        ))
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, self.model_channels),
            nn.GELU(),
            nn.Conv2d(self.model_channels, self.in_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, t, c):
        """
        前向传播
        
        Args:
            x: 输入图像 [B, in_channels, H, W]
            t: 时间步 [B]
            c: 条件标签 [B]
        
        Returns:
            out: 预测的噪声 [B, in_channels, H, W]
            moe_logits_list: 所有 MoE 块的 logits 列表
        """
        # 清空 Logits 列表（每次前向传播开始时）
        self.moe_logits_list = []
        
        # 嵌入
        t_emb = self.time_embedding(t)  # [B, t_dim]
        c_emb = self.condition_embedding(c)  # [B, c_dim]
        
        # 输入投影
        x = self.input_proj(x)  # [B, model_channels, H, W]
        
        # 编码器：收集跳跃连接
        skip_connections = []
        for down_block in self.encoder:
            x, x_skip, logits = down_block(x, t_emb, c_emb)
            skip_connections.append(x_skip)
            if logits is not None:
                self.moe_logits_list.append(logits)
        
        # 瓶颈层
        x = self.bottleneck_proj(x)
        if self.freq_enabled:
            x = self.bottleneck(x, t_emb)
        else:
            x, logits = self.bottleneck(x, t_emb, c_emb)
            if logits is not None:
                self.moe_logits_list.append(logits)
        
        # 解码器：使用跳跃连接
        for i, up_block in enumerate(self.decoder):
            skip_idx = len(skip_connections) - 1 - i
            if skip_idx >= 0:
                x_skip = skip_connections[skip_idx]
            else:
                # 最后一层没有对应的 skip connection，使用输入投影后的特征
                # 需要上采样到匹配尺寸
                x_skip = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
                x_skip = torch.zeros_like(x_skip)  # 占位
            
            x, logits = up_block(x, x_skip, t_emb, c_emb)
            if logits is not None:
                self.moe_logits_list.append(logits)
        
        # 输出投影
        out = self.output_proj(x)
        
        return out, self.moe_logits_list

