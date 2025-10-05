"""
MoEFsDiC 训练脚本
训练混合专家系统的扩散模型
"""
import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse

from src.models.moefsndic_unet import MoEFsDiC_UNet
from src.utils.loss import MoELoss


class DiffusionProcess:
    """
    简化的扩散过程
    实现前向加噪和损失计算
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
            x0: 原始图像 [B, C, H, W]
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


def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def get_dataloader(config):
    """
    创建数据加载器
    使用 CIFAR-10 作为示例数据集
    """
    transform = transforms.Compose([
        transforms.Resize(config['train']['image_size']),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
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
    
    for step, (images, labels) in enumerate(pbar):
        images = images.to(device)
        labels = labels.to(device)
        
        batch_size = images.size(0)
        
        # 随机采样时间步
        t = torch.randint(0, diffusion.num_timesteps, (batch_size,), device=device)
        
        # 前向扩散：添加噪声
        xt, noise = diffusion.add_noise(images, t)
        
        # 模型预测噪声
        noise_pred, moe_logits_list = model(xt, t, labels)
        
        # 计算损失
        loss, dm_loss, load_loss = criterion(noise_pred, noise, moe_logits_list)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # 累计损失
        total_loss += loss.item()
        total_dm_loss += dm_loss
        total_load_loss += load_loss
        
        # 更新进度条
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dm_loss': f"{dm_loss:.4f}",
            'load_loss': f"{load_loss:.4f}"
        })
        
        # 定期打印日志
        if (step + 1) % config['train']['log_interval'] == 0:
            avg_loss = total_loss / (step + 1)
            avg_dm_loss = total_dm_loss / (step + 1)
            avg_load_loss = total_load_loss / (step + 1)
            print(f"\n[Epoch {epoch+1}, Step {step+1}] "
                  f"Loss: {avg_loss:.4f}, DM Loss: {avg_dm_loss:.4f}, Load Loss: {avg_load_loss:.4f}")
    
    avg_loss = total_loss / len(dataloader)
    avg_dm_loss = total_dm_loss / len(dataloader)
    avg_load_loss = total_load_loss / len(dataloader)
    
    return avg_loss, avg_dm_loss, avg_load_loss


def main():
    # 命令行参数
    parser = argparse.ArgumentParser(description='Train MoEFsDiC U-Net')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    print("=== MoEFsDiC Training ===")
    print(f"Config: {args.config}")
    print(f"MoE Enabled: {config['moe']['enabled']}")
    print(f"Freq Enabled: {config['freq']['enabled']}")
    print(f"DS-Conv: {config['train']['use_dsconv_global']}")
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # 创建模型
    model = MoEFsDiC_UNet(config).to(device)
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
              f"Avg Loss: {avg_loss:.4f}, Avg DM Loss: {avg_dm_loss:.4f}, "
              f"Avg Load Loss: {avg_load_loss:.4f}\n")
        
        # 保存检查点
        if (epoch + 1) % (config['train']['save_interval'] // len(dataloader) + 1) == 0:
            checkpoint_path = os.path.join(
                config['train']['checkpoint_dir'],
                f"checkpoint_epoch_{epoch+1}.pth"
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")
    
    print("\n=== Training Complete ===")


if __name__ == '__main__':
    main()

