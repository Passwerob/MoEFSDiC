"""
Convolutional Blocks for MoE-FSNL-DiC
核心卷积块：MoE_ConvBlock 和 Dilated_Fusion_Block
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import DepthwiseSeparableConv2d, Expert, Router


class MoE_ConvBlock(nn.Module):
    """
    混合专家卷积块 (MoE ConvBlock)
    U-Net 中主要的特征处理块，集成了稀疏 MoE 机制
    [SR 增强] 支持 LR 特征融合，根据 LR 内容动态选择专家
    """
    def __init__(self, in_channels, out_channels, t_dim, c_dim, 
                 num_experts=8, k_active=2, use_dsconv=True, lr_channels=0):
        super(MoE_ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_experts = num_experts
        self.k_active = k_active
        self.lr_channels = lr_channels
        
        # [SR 增强] 如果有 LR 特征，投影层需要处理拼接后的通道
        proj_in_channels = in_channels + lr_channels if lr_channels > 0 else in_channels
        
        # 投影层：将输入通道映射到专家所需的通道数
        if use_dsconv:
            self.proj = DepthwiseSeparableConv2d(proj_in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.proj = nn.Conv2d(proj_in_channels, out_channels, kernel_size=3, padding=1)
        
        # 路由器（接受 LR 特征）
        self.router = Router(out_channels, t_dim, c_dim, num_experts, lr_channels=lr_channels)
        
        # 专家池
        self.experts = nn.ModuleList([
            Expert(out_channels, use_dsconv=use_dsconv) 
            for _ in range(num_experts)
        ])
    
    def forward(self, x, t_emb, c_emb, lr_feat=None):
        """
        MoE 块前向传播
        
        Args:
            x: 输入特征 [B, in_channels, H, W]
            t_emb: 时间嵌入 [B, t_dim]
            c_emb: 条件嵌入 [B, c_dim]
            lr_feat: LR 图像特征 [B, lr_channels, H, W] (可选，用于 SR)
        
        Returns:
            out: 输出特征 [B, out_channels, H, W]
            router_logits: 路由 Logits [B, num_experts]
        """
        B = x.size(0)
        
        # [SR 增强] 1. 如果提供了 LR 特征，先与输入特征融合
        if lr_feat is not None and self.lr_channels > 0:
            # 拼接融合：保留 LR 图像的结构信息
            x = torch.cat([x, lr_feat], dim=1)  # [B, in_channels + lr_channels, H, W]
        
        # 2. 投影到专家通道空间
        x_proj = self.proj(x)  # [B, out_channels, H, W]
        
        # 3. 路由：获取专家分数（传递 LR 特征到路由器）
        router_logits = self.router(x_proj, t_emb, c_emb, lr_feat=lr_feat)  # [B, num_experts]
        
        # 4. 稀疏激活：Top-K 选择
        topk_weights, topk_indices = torch.topk(router_logits, self.k_active, dim=1)  # [B, K]
        
        # 5. 归一化 Top-K 权重
        topk_weights_norm = F.softmax(topk_weights, dim=1)  # [B, K]
        
        # 6. 混合专家输出
        # 初始化输出为零
        expert_output = torch.zeros_like(x_proj)  # [B, out_channels, H, W]
        
        # 遍历每个 Top-K 位置
        for k_idx in range(self.k_active):
            # 对于批次中的每个样本，获取其第 k_idx 个激活的专家索引
            for b_idx in range(B):
                expert_idx = topk_indices[b_idx, k_idx].item()
                weight = topk_weights_norm[b_idx, k_idx]
                
                # 获取当前样本的特征
                x_sample = x_proj[b_idx:b_idx+1]  # [1, out_channels, H, W]
                
                # 专家处理
                expert_out = self.experts[expert_idx](x_sample)  # [1, out_channels, H, W]
                
                # 加权累加
                expert_output[b_idx:b_idx+1] += weight * expert_out
        
        return expert_output, router_logits


class Dilated_Fusion_Block(nn.Module):
    """
    扩张融合块 (Dilated Fusion Block)
    使用不同扩张率的卷积并行处理，用于消融实验（MoE 禁用时的替代方案）
    """
    def __init__(self, in_channels, out_channels, use_dsconv=True, dilations=[1, 2, 4]):
        super(Dilated_Fusion_Block, self).__init__()
        self.dilations = dilations
        self.num_branches = len(dilations)
        
        # 投影层
        if use_dsconv:
            self.proj = DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        else:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # 并行的扩张卷积分支
        self.branches = nn.ModuleList()
        for dilation in dilations:
            padding = dilation  # 保持相同的输出尺寸
            if use_dsconv:
                branch = DepthwiseSeparableConv2d(
                    out_channels, out_channels, 
                    kernel_size=3, padding=padding, dilation=dilation
                )
            else:
                branch = nn.Conv2d(
                    out_channels, out_channels, 
                    kernel_size=3, padding=padding, dilation=dilation
                )
            self.branches.append(branch)
        
        # 融合层：1x1 卷积
        self.fusion = nn.Conv2d(out_channels * self.num_branches, out_channels, kernel_size=1)
        
        # 归一化和激活
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
    
    def forward(self, x, t_emb=None, c_emb=None, lr_feat=None):
        """
        扩张融合块前向传播
        
        Args:
            x: 输入特征 [B, in_channels, H, W]
            t_emb: 时间嵌入 (可选，保持接口一致)
            c_emb: 条件嵌入 (可选，保持接口一致)
            lr_feat: LR 图像特征 (可选，保持接口一致)
        
        Returns:
            out: 输出特征 [B, out_channels, H, W]
            None: 占位，保持接口一致（无 router_logits）
        """
        # [SR 增强] 如果提供了 LR 特征，融合（虽然该块不使用路由）
        # 为了保持接口一致，这里也支持 lr_feat 但不做特殊处理
        
        # 投影
        x_proj = self.proj(x)
        identity = x_proj
        
        # 并行扩张卷积
        branch_outputs = []
        for branch in self.branches:
            branch_out = branch(x_proj)
            branch_outputs.append(branch_out)
        
        # 拼接所有分支
        concat_out = torch.cat(branch_outputs, dim=1)  # [B, out_channels * num_branches, H, W]
        
        # 融合
        fused = self.fusion(concat_out)  # [B, out_channels, H, W]
        
        # 残差连接
        out = fused + identity
        out = self.norm(out)
        out = self.activation(out)
        
        return out, None  # 返回 None 作为 logits 占位

