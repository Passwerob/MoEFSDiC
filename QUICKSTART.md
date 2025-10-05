# MoEFsDiC 快速入门指南

## 环境设置

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装：
```bash
pip install torch>=2.0.0 torchvision>=0.15.0 pyyaml tqdm numpy
```

### 2. 验证安装

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
```

## 快速测试

### 1. 测试模型结构（需要 PyTorch）

```bash
python test_model.py
```

这将测试：
- 模型实例化
- 前向传播
- 不同消融配置

### 2. 开始训练

```bash
# 使用默认配置训练
python train.py --config configs/default.yaml --device cuda

# 使用 CPU 训练（调试用）
python train.py --config configs/default.yaml --device cpu

# 从检查点恢复训练
python train.py --config configs/default.yaml --resume checkpoints/checkpoint_epoch_10.pth
```

## 消融实验指南

### 实验 1: 完整模型（基线）

`configs/default.yaml`:
```yaml
moe:
  enabled: true
  num_experts: 8
  k_active: 2
freq:
  enabled: true
train:
  use_dsconv_global: true
```

### 实验 2: 无 MoE（使用 Dilated Fusion）

创建 `configs/no_moe.yaml`:
```yaml
moe:
  enabled: false  # 关键改动
  num_experts: 8
  k_active: 2
freq:
  enabled: true
train:
  use_dsconv_global: true
```

训练：
```bash
python train.py --config configs/no_moe.yaml
```

### 实验 3: 无频率域模块

创建 `configs/no_freq.yaml`:
```yaml
moe:
  enabled: true
  num_experts: 8
  k_active: 2
freq:
  enabled: false  # 关键改动
train:
  use_dsconv_global: true
```

### 实验 4: 无 DS-Conv（标准卷积）

创建 `configs/no_dsconv.yaml`:
```yaml
moe:
  enabled: true
  num_experts: 8
  k_active: 2
freq:
  enabled: true
train:
  use_dsconv_global: false  # 关键改动
```

### 实验 5: 调整专家数量和 Top-K

创建 `configs/moe_variants.yaml`:
```yaml
moe:
  enabled: true
  num_experts: 16  # 增加专家数量
  k_active: 4      # 增加激活数量
freq:
  enabled: true
train:
  use_dsconv_global: true
```

## 训练监控

### 查看训练日志

训练过程中会输出：
- `loss`: 总损失（扩散损失 + 负载均衡损失）
- `dm_loss`: 扩散模型 MSE 损失
- `load_loss`: MoE 负载均衡损失

示例输出：
```
Epoch 1/100: 100%|████████| 3125/3125 [10:23<00:00, 5.01it/s, loss=0.1234, dm_loss=0.1200, load_loss=0.0034]
[Epoch 1/100] Avg Loss: 0.1234, Avg DM Loss: 0.1200, Avg Load Loss: 0.0034
```

### 检查点

模型会定期保存在 `checkpoints/` 目录：
- `checkpoint_epoch_1.pth`
- `checkpoint_epoch_2.pth`
- ...

## 代码使用示例

### 基本使用

```python
import torch
import yaml
from src.models.moefsndic_unet import MoEFsDiC_UNet

# 加载配置
with open('configs/default.yaml', 'r') as f:
    config = yaml.safe_load(f)

# 创建模型
model = MoEFsDiC_UNet(config)
model.eval()

# 准备输入
batch_size = 4
x = torch.randn(batch_size, 3, 64, 64)  # [B, C, H, W]
t = torch.randint(0, 1000, (batch_size,))  # 时间步
c = torch.randint(0, 10, (batch_size,))    # 类别条件

# 前向传播
with torch.no_grad():
    noise_pred, moe_logits = model(x, t, c)

print(f"Predicted noise shape: {noise_pred.shape}")
print(f"Number of MoE blocks: {len(moe_logits)}")
```

### 自定义训练循环

```python
from src.utils.loss import MoELoss

# 创建模型和损失
model = MoEFsDiC_UNet(config).cuda()
criterion = MoELoss(lambda_load=0.01)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# 训练步骤
for images, labels in dataloader:
    images, labels = images.cuda(), labels.cuda()
    
    # 采样时间步
    t = torch.randint(0, 1000, (images.size(0),)).cuda()
    
    # 添加噪声
    noise = torch.randn_like(images)
    noisy_images = add_noise(images, t, noise)
    
    # 预测噪声
    noise_pred, moe_logits = model(noisy_images, t, labels)
    
    # 计算损失
    loss, dm_loss, load_loss = criterion(noise_pred, noise, moe_logits)
    
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 性能优化建议

### 1. 批次大小
- GPU 内存充足：batch_size = 32 或更大
- GPU 内存有限：batch_size = 8-16
- 调试阶段：batch_size = 2-4

### 2. 模型大小
- 小模型：`model_channels = 64`, `channel_mults = [1, 2, 4]`
- 中模型：`model_channels = 128`, `channel_mults = [1, 2, 4, 8]`（默认）
- 大模型：`model_channels = 256`, `channel_mults = [1, 2, 4, 8, 16]`

### 3. MoE 配置
- 少量专家（快速）：`num_experts = 4`, `k_active = 1`
- 中等专家（平衡）：`num_experts = 8`, `k_active = 2`（默认）
- 大量专家（容量大）：`num_experts = 16`, `k_active = 4`

### 4. 训练技巧
- 使用混合精度训练：可以使用 `torch.cuda.amp`
- 梯度累积：如果显存不足，可以累积多个 batch 再更新
- 学习率调度：使用 CosineAnnealingLR 或 ReduceLROnPlateau

## 常见问题

### Q: 显存不足（CUDA out of memory）
A: 
- 减少 `batch_size`
- 减少 `model_channels`
- 减少 `image_size`
- 启用梯度检查点（gradient checkpointing）

### Q: 训练很慢
A:
- 启用 `use_dsconv_global = true` 使用深度可分离卷积
- 减少 `num_experts`
- 使用更小的 `image_size`

### Q: 负载均衡损失很大
A:
- 增加 `lambda_load` 权重
- 减少 `num_experts` 或增加 `k_active`
- 检查路由器是否工作正常

### Q: MoE 不起作用
A:
- 确保 `moe.enabled = true`
- 检查是否收集到 `moe_logits`（应该 > 0）
- 验证路由器输出是否合理

## 下一步

1. 阅读 `README.md` 了解架构细节
2. 查看源代码注释理解实现
3. 运行消融实验比较不同配置
4. 根据数据集调整超参数
5. 实现自定义的专家或模块

## 引用

如果使用本项目，请引用：
```bibtex
@software{moefsndic2025,
  title={MoEFsDiC: Mixture-of-Experts Frequency-Separable Dilated Conv Diffusion},
  year={2025}
}
```

