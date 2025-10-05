"""
MoEFsDiC-SR 超分辨率训练脚本
[SR 专用] 训练基于扩散模型的图像超分辨率系统
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
from tqdm import tqdm
import argparse

from src.models.moefsndic_sr_unet import MoEFsDiC_SR_UNet
from src.utils.loss import MoELoss


class DiffusionProcess:
    """
    扩散过程（用于 SR）
    """
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # 线性 beta 调度
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # 用于采样的系数
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
    
    def add_noise(self, x0, t, noise=None):
        """
        前向扩散过程：添加噪声
        
        Args:
            x0: 原始 HR 图像 [B, C, H, W]
            t: 时间步 [B]
            noise: 噪声（可选）[B, C, H, W]
        
        Returns:
            xt: 加噪后的图像 [B, C, H, W]
            noise: 使用的噪声 [B, C, H, W]
        """
        if noise is None:
            noise = torch.randn_like(x0)
        
        # 获取对应时间步的系数
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # q(x_t | x_0)
        xt = sqrt_alpha_cumprod_t * x0 + sqrt_one_minus_alpha_cumprod_t * noise
        
        return xt, noise


class SRDataset(Dataset):
    """
    超分辨率数据集
    生成 LR-HR 图像对
    """
    def __init__(self, hr_dir, hr_size=256, lr_size=64, scale_factor=4):
        self.hr_dir = hr_dir
        self.hr_size = hr_size
        self.lr_size = lr_size
        self.scale_factor = scale_factor
        
        # 获取所有图像文件
        self.image_files = [f for f in os.listdir(hr_dir) 
                           if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        # HR 变换
        self.hr_transform = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # LR 变换（下采样）
        self.lr_transform = transforms.Compose([
            transforms.Resize((lr_size, lr_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 加载 HR 图像
        img_path = os.path.join(self.hr_dir, self.image_files[idx])
        hr_image = Image.open(img_path).convert('RGB')
        
        # 生成 HR 和 LR 图像
        hr = self.hr_transform(hr_image)
        lr = self.lr_transform(hr_image)
        
        # 类别标签（简单示例，可以根据实际情况修改）
        label = 0
        
        return hr, lr, label


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_dataloader(config):
    """
    创建数据加载器
    """
    # 这里使用占位数据集，实际使用时需要准备真实数据
    # 例如：DIV2K, Flickr2K, ImageNet 等数据集
    
    # 如果没有数据，创建合成数据
    class SyntheticSRDataset(Dataset):
        def __init__(self, num_samples=1000, hr_size=256, lr_size=64):
            self.num_samples = num_samples
            self.hr_size = hr_size
            self.lr_size = lr_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # 生成随机图像
            hr = torch.randn(3, self.hr_size, self.hr_size)
            lr = torch.randn(3, self.lr_size, self.lr_size)
            label = torch.randint(0, 10, (1,)).item()
            return hr, lr, label
    
    dataset = SyntheticSRDataset(
        num_samples=1000,
        hr_size=config['train']['hr_size'],
        lr_size=config['train']['lr_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['train']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    return dataloader


def train_epoch(model, dataloader, optimizer, criterion, diffusion, device, epoch, config):
    """
    训练一个 epoch
    """
    model.train()
    total_loss = 0.0
    total_dm_loss = 0.0
    total_load_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    
    for step, (hr_images, lr_images, labels) in enumerate(pbar):
        hr_images = hr_images.to(device)
        lr_images = lr_images.to(device)
        labels = labels.to(device)
        
        batch_size = hr_images.size(0)
        
        # 随机采样时间步
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        
        # 前向扩散：对 HR 图像添加噪声
        hr_noisy, noise = diffusion.add_noise(hr_images, t)
        
        # [SR 核心] 模型预测噪声，以 LR 图像为条件
        noise_pred, moe_logits_list = model(hr_noisy, t, labels, lr_images)
        
        # 计算损失
        loss, dm_loss, load_loss = criterion(noise_pred, noise, moe_logits_list)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        total_dm_loss += dm_loss
        total_load_loss += load_loss
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dm': f"{dm_loss:.4f}",
            'load': f"{load_loss:.4f}"
        })
        
        # 定期打印日志
        if (step + 1) % config['train']['log_interval'] == 0:
            avg_loss = total_loss / (step + 1)
            avg_dm_loss = total_dm_loss / (step + 1)
            avg_load_loss = total_load_loss / (step + 1)
            print(f"\n[Epoch {epoch+1}, Step {step+1}] "
                  f"Loss: {avg_loss:.4f}, DM: {avg_dm_loss:.4f}, Load: {avg_load_loss:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_dm_loss = total_dm_loss / len(dataloader)
    avg_load_loss = total_load_loss / len(dataloader)
    
    return avg_loss, avg_dm_loss, avg_load_loss


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='Train MoEFsDiC-SR U-Net')
    parser.add_argument('--config', type=str, default='configs/sr_default.yaml',
                        help='Path to SR config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print("=== MoEFsDiC-SR Training ===")
    print(f"Config: {args.config}")
    print(f"Scale Factor: {config['sr']['scale_factor']}x")
    print(f"MoE Enabled: {config['moe']['enabled']}")
    print(f"Freq Enabled: {config['freq']['enabled']}")
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建模型
    model = MoEFsDiC_SR_UNet(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 损失函数
    criterion = MoELoss(lambda_load=config['train']['lambda_load'])
    
    # 优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['train']['learning_rate'],
        weight_decay=config['train']['weight_decay']
    )
    
    # 扩散过程
    diffusion = DiffusionProcess(
        num_timesteps=config['diffusion']['num_timesteps'],
        beta_start=config['diffusion']['beta_start'],
        beta_end=config['diffusion']['beta_end'],
        device=device
    )
    
    # 数据加载器
    dataloader = get_dataloader(config)
    print(f"Dataset size: {len(dataloader.dataset)}")
    
    # 创建检查点目录
    os.makedirs(config['train']['checkpoint_dir'], exist_ok=True)
    
    # 恢复训练（如果指定）
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resumed from epoch {start_epoch}")
    
    # 训练循环
    print("\n=== Starting Training ===")
    for epoch in range(start_epoch, config['train']['num_epochs']):
        avg_loss, avg_dm_loss, avg_load_loss = train_epoch(
            model, dataloader, optimizer, criterion, diffusion, device, epoch, config
        )
        
        print(f"\n[Epoch {epoch+1}/{config['train']['num_epochs']}] "
              f"Avg Loss: {avg_loss:.4f}, Avg DM: {avg_dm_loss:.4f}, "
              f"Avg Load: {avg_load_loss:.4f}\n")
        
        # 保存检查点
        if (epoch + 1) % (config['train']['save_interval'] // len(dataloader) + 1) == 0:
            checkpoint_path = os.path.join(
                config['train']['checkpoint_dir'],
                f"sr_checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()

