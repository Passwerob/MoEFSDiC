"""
Frequency Global Module for MoE-FSNL-DiC
频率域全局感知模块，在 U-Net 瓶颈层使用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Freq_Global_Module(nn.Module):
    """
    频率域全局模块 (F-Module)
    在频率域进行特征调制，增强全局感知能力
    """
    def __init__(self, channels, t_dim):
        super(Freq_Global_Module, self).__init__()
        self.channels = channels
        self.t_dim = t_dim
        
        # 全局平均池化，用于提取上下文信息
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # 轻量级 MLP 调制网络
        # 输入：全局特征 + 时间嵌入
        # 输出：频率域调制权重
        self.modulation_mlp = nn.Sequential(
            nn.Linear(channels + t_dim, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
            nn.Sigmoid()  # 调制权重范围 [0, 1]
        )
    
    def forward(self, x, t_emb):
        """
        频率域增强前向传播
        
        Args:
            x: 输入特征图 [B, C, H, W]
            t_emb: 时间嵌入 [B, t_dim]
        
        Returns:
            out: 频率域增强后的特征 [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x  # 保存用于残差连接
        
        # 1. 提取全局上下文
        global_feat = self.gap(x).view(B, C)  # [B, C]
        
        # 2. 融合全局特征和时间嵌入
        concat_feat = torch.cat([global_feat, t_emb], dim=1)  # [B, C + t_dim]
        
        # 3. 计算调制权重
        modulation_weights = self.modulation_mlp(concat_feat)  # [B, C]
        modulation_weights = modulation_weights.view(B, C, 1, 1)  # [B, C, 1, 1]
        
        # 4. 转换到频率域 (FFT2)
        # torch.fft.fft2 返回复数类型
        freq_x = torch.fft.fft2(x, dim=(-2, -1))  # [B, C, H, W] complex
        
        # 5. 在频率域进行调制
        # 将调制权重广播到频率域的每个位置
        freq_modulated = freq_x * modulation_weights  # [B, C, H, W] complex
        
        # 6. 逆 FFT 转回空间域
        spatial_x = torch.fft.ifft2(freq_modulated, dim=(-2, -1))  # [B, C, H, W] complex
        
        # 7. 只取实部（理论上虚部应该很小）
        spatial_x = spatial_x.real  # [B, C, H, W] real
        
        # 8. 残差连接
        out = spatial_x + identity
        
        return out

