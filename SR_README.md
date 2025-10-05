# MoEFsDiC-SR: 超分辨率扩散模型

基于 MoEFsDiC 架构的高效图像超分辨率系统，利用混合专家系统和频率域增强实现快速、高质量的高清细节重建。

## 核心特点

### 🎯 SR 专用设计

1. **LR 图像条件输入**
   - LR 图像作为强制性条件输入到每个 U-Net 块
   - 保留 LR 图像的结构信息，引导 HR 重建

2. **多尺度 LR 特征融合**
   - 在编码器和解码器的每一层融合对应尺度的 LR 特征
   - 跳跃连接中包含 LR 信息

3. **内容感知的 MoE 路由**
   - Router 根据 LR 图像内容（边缘、平滑区域等）动态选择专家
   - 不同区域使用不同的重建策略

4. **高效上采样**
   - 使用 Sub-pixel Convolution (Pixel Shuffle)
   - 最终上采样层实现空间分辨率提升

## 架构变更

### 相比通用 MoEFsDiC 的修改

#### 1. Router 增强
```python
# 原版
router_logits = Router(x, t_emb, c_emb)

# SR 版本
router_logits = Router(x, t_emb, c_emb, lr_feat)
```

Router 现在感知：
- 时间步（扩散进度）
- LR 图像结构（边缘、纹理）
- 条件信息

#### 2. MoE_ConvBlock 增强
```python
# 原版
out = MoE_ConvBlock(x, t_emb, c_emb)

# SR 版本
out = MoE_ConvBlock(x, t_emb, c_emb, lr_feat)
```

特征融合方式：
```python
# 拼接融合
x_fused = torch.cat([x, lr_feat], dim=1)
x_proj = proj_layer(x_fused)
```

#### 3. U-Net 架构增强

**LR 特征金字塔**：
```
LR Image [B, 3, 64, 64]
    ↓ LR Encoder
LR Feat [B, 64, 64, 64]
    ↓ 多尺度下采样
├─ Level 0: [B, 64, 64, 64]   → 编码器层 0
├─ Level 1: [B, 64, 32, 32]   → 编码器层 1  
├─ Level 2: [B, 64, 16, 16]   → 编码器层 2
└─ Level 3: [B, 64, 8, 8]     → 瓶颈层
```

**跳跃连接融合**：
```
编码器输出 + LR 特征 → 跳跃连接 → 解码器
```

**最终上采样**：
```
Feature [B, C, 64, 64]
    ↓ Conv + PixelShuffle(4x)
HR Output [B, 3, 256, 256]
```

## 使用方法

### 1. 训练 SR 模型

```bash
# 使用默认 SR 配置
python train_sr.py --config configs/sr_default.yaml --device cuda

# 使用自定义配置
python train_sr.py --config configs/sr_custom.yaml
```

### 2. 准备数据

数据集结构：
```
data/
├── train/
│   ├── hr/           # 高分辨率图像
│   └── lr/           # 低分辨率图像（可选，会自动生成）
└── val/
    ├── hr/
    └── lr/
```

推荐数据集：
- **DIV2K**: 800 张 2K 分辨率图像
- **Flickr2K**: 2650 张高质量图像
- **Set5, Set14, BSD100**: 标准测试集

### 3. 配置说明

#### SR 专用配置 (`configs/sr_default.yaml`)

```yaml
sr:
  scale_factor: 4        # 超分倍数 (2x/4x/8x)
  lr_channels: 64        # LR 特征通道数

train:
  hr_size: 256          # HR 图像尺寸
  lr_size: 64           # LR 图像尺寸 (hr_size / scale_factor)
  batch_size: 8         # 批次大小
  learning_rate: 2e-4   # 学习率
```

### 4. 推理/采样

```python
import torch
from src.models.moefsndic_sr_unet import MoEFsDiC_SR_UNet

# 加载模型
model = MoEFsDiC_SR_UNet(config).cuda()
model.load_state_dict(torch.load('checkpoint.pth'))
model.eval()

# 准备输入
lr_image = load_lr_image()  # [1, 3, 64, 64]
t = torch.tensor([0])       # 采样从 t=0 开始
c = torch.tensor([0])       # 类别条件

# 生成 HR 图像（简化示例）
with torch.no_grad():
    # 从噪声开始
    hr_noise = torch.randn(1, 3, 256, 256).cuda()
    
    # 迭代去噪
    for t in reversed(range(num_timesteps)):
        noise_pred, _ = model(hr_noise, torch.tensor([t]), c, lr_image)
        hr_noise = denoise_step(hr_noise, noise_pred, t)
    
    hr_image = hr_noise
```

## 消融实验

### 实验配置

| 配置 | MoE | Freq | LR Fusion | 说明 |
|------|-----|------|-----------|------|
| SR Full | ✓ | ✓ | ✓ | 完整 SR 模型 |
| SR No-MoE | ✗ | ✓ | ✓ | 无 MoE，使用 Dilated |
| SR No-Freq | ✓ | ✗ | ✓ | 无频率域增强 |
| SR Baseline | ✗ | ✗ | ✓ | 最小 SR 基线 |

### 运行实验

```bash
# 完整模型
python train_sr.py --config configs/sr_default.yaml

# 无 MoE
python train_sr.py --config configs/sr_no_moe.yaml

# 无频率域
python train_sr.py --config configs/sr_no_freq.yaml
```

## 训练技巧

### 1. 数据增强

```yaml
augmentation:
  random_flip: true      # 水平/垂直翻转
  random_crop: true      # 随机裁剪
  color_jitter: false    # 颜色抖动（慎用）
```

### 2. 学习率调度

```python
# Cosine Annealing
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=num_epochs
)

# Warm Restart
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=10, T_mult=2
)
```

### 3. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

with autocast():
    noise_pred, moe_logits = model(hr_noisy, t, labels, lr_images)
    loss, dm_loss, load_loss = criterion(noise_pred, noise, moe_logits)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4. 渐进式训练

```python
# 阶段 1: 低分辨率快速预训练
config['train']['hr_size'] = 128
config['train']['lr_size'] = 32

# 阶段 2: 目标分辨率精细训练
config['train']['hr_size'] = 256
config['train']['lr_size'] = 64
```

## 性能优化

### 显存优化

1. **减少批次大小**
   ```yaml
   batch_size: 4  # 从 8 降到 4
   ```

2. **梯度累积**
   ```python
   accumulation_steps = 4
   for i, batch in enumerate(dataloader):
       loss = compute_loss(batch)
       loss = loss / accumulation_steps
       loss.backward()
       
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```

3. **梯度检查点**
   ```python
   from torch.utils.checkpoint import checkpoint
   
   # 在模型中使用
   out = checkpoint(self.conv_block, x, t_emb, c_emb, lr_feat)
   ```

### 速度优化

1. **使用 DS-Conv**
   ```yaml
   use_dsconv_global: true  # 参数量减少 ~9 倍
   ```

2. **减少专家数量**
   ```yaml
   num_experts: 4   # 从 8 降到 4
   k_active: 1      # 从 2 降到 1
   ```

3. **数据加载优化**
   ```python
   DataLoader(dataset, num_workers=8, pin_memory=True, prefetch_factor=2)
   ```

## 评估指标

### 图像质量指标

```python
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

# PSNR
psnr_value = psnr(hr_gt, hr_pred)

# SSIM
ssim_value = ssim(hr_gt, hr_pred, multichannel=True)

# LPIPS (感知损失)
import lpips
loss_fn = lpips.LPIPS(net='alex')
lpips_value = loss_fn(hr_gt, hr_pred)
```

### MoE 负载分析

```python
# 分析专家激活分布
expert_activation_counts = torch.zeros(num_experts)
for logits in moe_logits_list:
    topk_indices = torch.topk(logits, k=2, dim=1)[1]
    for idx in topk_indices.flatten():
        expert_activation_counts[idx] += 1

print(f"Expert utilization: {expert_activation_counts / expert_activation_counts.sum()}")
```

## 常见问题

### Q: SR 模型显存消耗大怎么办？
A: 
- 减少 `batch_size` (推荐 4-8)
- 减少 `hr_size` 和 `lr_size`
- 使用梯度累积
- 启用混合精度训练

### Q: 生成的图像有伪影？
A: 
- 增加训练轮数
- 调整 `lambda_load`（可能过大）
- 检查数据质量
- 使用更平滑的 beta schedule

### Q: MoE 路由不均衡？
A: 
- 增加 `lambda_load` (0.01 → 0.05)
- 减少 `num_experts` 或增加 `k_active`
- 检查 LR 特征是否正确传递

### Q: 如何可视化 LR 特征影响？
A: 
```python
# 禁用 LR 特征
noise_pred_no_lr, _ = model(hr_noisy, t, c, None)

# 启用 LR 特征
noise_pred_with_lr, _ = model(hr_noisy, t, c, lr_image)

# 可视化差异
diff = torch.abs(noise_pred_with_lr - noise_pred_no_lr)
```

## 扩展方向

### 1. 多尺度 SR
支持 2x, 4x, 8x 同时训练：
```python
scale = random.choice([2, 4, 8])
lr_size = hr_size // scale
```

### 2. 真实世界退化
添加模糊、噪声等真实退化：
```python
from degradations import blur_kernel, add_noise
lr_degraded = add_noise(blur_kernel(hr_image))
```

### 3. 参考图像 SR
使用额外的参考图像：
```python
model(hr_noisy, t, c, lr_image, ref_image)
```

### 4. 视频 SR
扩展到时序一致性：
```python
# 添加时序注意力
temporal_attn = TemporalAttention(channels)
```

## 引用

如果使用本项目的 SR 功能，请引用：

```bibtex
@software{moefsndic_sr_2025,
  title={MoEFsDiC-SR: Super-Resolution with Mixture-of-Experts Diffusion},
  year={2025}
}
```

## 相关资源

- **原始 MoEFsDiC**: `README.md`
- **架构详解**: `ARCHITECTURE.md`
- **快速入门**: `QUICKSTART.md`
- **SR 配置**: `configs/sr_default.yaml`
- **SR 训练**: `train_sr.py`

---

**项目状态**: ✅ SR 功能已完成并可用

**最后更新**: 2025-10-05

