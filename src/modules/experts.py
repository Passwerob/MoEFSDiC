"""
Experts and Router modules for MoE-FSNL-DiC
实现深度可分离卷积、专家模块和路由器
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv2d(nn.Module):
    """
    深度可分离卷积：Depthwise Conv + Pointwise Conv
    高效的卷积操作，减少参数量和计算量
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False):
        super(DepthwiseSeparableConv2d, self).__init__()
        # Depthwise Convolution: 每个通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            dilation=dilation,
            groups=in_channels,  # 关键：groups=in_channels 实现 depthwise
            bias=bias
        )
        # Pointwise Convolution: 1x1 卷积混合通道信息
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=bias
        )
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class Expert(nn.Module):
    """
    专家模块：使用 DS-Conv 的残差块
    每个专家是一个独立的特征处理单元
    """
    def __init__(self, channels, use_dsconv=True):
        super(Expert, self).__init__()
        self.use_dsconv = use_dsconv
        
        if use_dsconv:
            # 使用深度可分离卷积
            self.conv1 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = DepthwiseSeparableConv2d(channels, channels, kernel_size=3, padding=1)
        else:
            # 标准卷积（用于消融实验）
            self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        self.norm1 = nn.BatchNorm2d(channels)
        self.norm2 = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()
    
    def forward(self, x):
        """
        残差块前向传播
        """
        identity = x
        
        # 第一层卷积
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.activation(out)
        
        # 第二层卷积
        out = self.conv2(out)
        out = self.norm2(out)
        
        # 残差连接
        out = out + identity
        out = self.activation(out)
        
        return out


class Router(nn.Module):
    """
    路由器模块：计算专家分数（Logits）
    融合空间特征、时间嵌入、条件嵌入和 LR 特征来决定专家权重
    [SR 增强] 添加 LR 特征感知，根据 LR 图像内容动态路由
    """
    def __init__(self, feature_channels, t_dim, c_dim, num_experts, lr_channels=0):
        super(Router, self).__init__()
        self.feature_channels = feature_channels
        self.t_dim = t_dim
        self.c_dim = c_dim
        self.num_experts = num_experts
        self.lr_channels = lr_channels
        
        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 计算拼接后的总维度（包含 LR 特征）
        concat_dim = feature_channels + t_dim + c_dim + lr_channels
        
        # 轻量级 MLP 路由网络
        self.mlp = nn.Sequential(
            nn.Linear(concat_dim, concat_dim // 2),
            nn.GELU(),
            nn.Linear(concat_dim // 2, num_experts)
        )
    
    def forward(self, x, t_emb, c_emb, lr_feat=None):
        """
        计算专家路由 Logits
        
        Args:
            x: 特征图 [B, C, H, W]
            t_emb: 时间嵌入 [B, t_dim]
            c_emb: 条件嵌入 [B, c_dim]
            lr_feat: LR 图像特征 [B, lr_channels, H, W] (可选，用于 SR)
        
        Returns:
            logits: 专家分数 [B, num_experts]
        """
        # 全局平均池化获取空间特征表示
        B = x.size(0)
        spatial_feat = self.gap(x).view(B, -1)  # [B, feature_channels]
        
        # 构建拼接特征列表
        feat_list = [spatial_feat, t_emb, c_emb]
        
        # [SR 增强] 如果提供了 LR 特征，融合 LR 图像的结构信息
        if lr_feat is not None and self.lr_channels > 0:
            lr_spatial = self.gap(lr_feat).view(B, -1)  # [B, lr_channels]
            feat_list.append(lr_spatial)
        
        # 在通道维度拼接所有输入
        concat_feat = torch.cat(feat_list, dim=1)  # [B, concat_dim]
        
        # 通过 MLP 计算专家 Logits
        logits = self.mlp(concat_feat)  # [B, num_experts]
        
        return logits

