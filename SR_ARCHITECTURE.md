# MoEFsDiC-SR 架构详解

本文档详细说明 MoEFsDiC-SR 超分辨率模型的架构设计和关键修改。

## 从通用扩散到 SR 的转变

### 核心差异对比

| 方面 | 通用扩散模型 | SR 扩散模型 |
|------|-------------|------------|
| **输入** | 噪声图像 x_t | 噪声 HR 图像 x_t + LR 图像 |
| **条件** | 时间步 t, 类别 c | 时间步 t, 类别 c, **LR 图像** |
| **输出** | 预测噪声 ε | 预测噪声 ε (HR 空间) |
| **目标** | 生成任意图像 | 从 LR 重建 HR |
| **上采样** | 无 | 最终 scale_factor× 上采样 |

## SR 增强的关键组件

### 1. LR 图像编码器

**作用**: 将 LR 图像编码为潜在特征

```python
class LREncoder(nn.Module):
    def __init__(self, in_channels, lr_channels):
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, lr_channels, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(lr_channels, lr_channels, 3, padding=1)
        )
    
    def forward(self, lr_image):
        # lr_image: [B, 3, 64, 64]
        # 输出: [B, lr_channels, 64, 64]
        return self.encoder(lr_image)
```

**设计考虑**:
- 轻量级设计（2 层卷积）
- 保留空间分辨率
- 提取结构和纹理信息

### 2. LR 特征金字塔

**作用**: 为 U-Net 的每个层级生成对应尺度的 LR 特征

```python
# 构建 LR 金字塔
lr_feat_base = lr_encoder(lr_image)                    # [B, 64, 64, 64]
lr_feat_base = F.interpolate(lr_feat_base, size=(256, 256))  # 上采样到 HR 尺寸

lr_pyramid = [lr_feat_base]  # Level 0: 256×256

# 生成下采样层级
for downsampler in lr_downsamplers:
    lr_feat = downsampler(lr_feat)
    lr_pyramid.append(lr_feat)
    # Level 1: 128×128
    # Level 2: 64×64
    # Level 3: 32×32
```

**金字塔结构**:
```
LR Image [64×64]
    ↓ Encoder
LR Feat [64×64]
    ↓ Interpolate to HR size
LR Pyramid:
    Level 0: [64, 256, 256]  ← 编码器层 0
    Level 1: [64, 128, 128]  ← 编码器层 1
    Level 2: [64, 64, 64]    ← 编码器层 2
    Level 3: [64, 32, 32]    ← 瓶颈层
```

### 3. 增强的 Router

**原版 Router**:
```python
def forward(self, x, t_emb, c_emb):
    spatial_feat = GAP(x)
    concat = [spatial_feat, t_emb, c_emb]
    logits = MLP(concat(concat))
    return logits
```

**SR Router**:
```python
def forward(self, x, t_emb, c_emb, lr_feat):
    spatial_feat = GAP(x)
    lr_spatial = GAP(lr_feat)  # 提取 LR 全局信息
    
    concat = [spatial_feat, t_emb, c_emb, lr_spatial]
    logits = MLP(concat(concat))
    return logits
```

**为什么添加 LR 特征到 Router**:

1. **内容感知路由**:
   ```
   LR 边缘区域 → 激活边缘增强专家
   LR 平滑区域 → 激活纹理合成专家
   LR 复杂纹理 → 激活细节重建专家
   ```

2. **动态专家选择**:
   - 不同图像内容需要不同的重建策略
   - Router 根据 LR 内容特征决定专家组合
   - 提高重建质量和效率

### 4. 增强的 MoE_ConvBlock

**特征融合流程**:

```python
def forward(self, x, t_emb, c_emb, lr_feat):
    # 1. 融合 LR 特征
    if lr_feat is not None:
        x_fused = torch.cat([x, lr_feat], dim=1)
        # x: [B, C, H, W]
        # lr_feat: [B, lr_C, H, W]
        # x_fused: [B, C + lr_C, H, W]
    else:
        x_fused = x
    
    # 2. 投影到统一通道空间
    x_proj = self.proj(x_fused)  # [B, out_C, H, W]
    
    # 3. Router 计算（包含 LR 感知）
    router_logits = self.router(x_proj, t_emb, c_emb, lr_feat)
    
    # 4. Top-K 专家选择
    topk_weights, topk_indices = torch.topk(router_logits, k=2)
    
    # 5. 专家混合
    output = mix_experts(x_proj, experts, topk_weights, topk_indices)
    
    return output, router_logits
```

**融合方式对比**:

| 方式 | 优点 | 缺点 | 何时使用 |
|------|------|------|----------|
| **Concatenation** | 保留完整信息 | 增加通道数 | 推荐（本实现） |
| Addition | 不增加参数 | 信息可能冲突 | 通道数受限时 |
| Attention Fusion | 自适应权重 | 计算开销大 | 需要精细控制时 |

### 5. 跳跃连接中的 LR 融合

**编码器端**:
```python
# 编码器输出已经包含 LR 信息
for idx, down_block in enumerate(encoder):
    lr_feat = lr_pyramid[idx]
    x, x_skip, logits = down_block(x, t_emb, c_emb, lr_feat)
    skip_connections.append(x_skip)  # x_skip 已融合 LR
```

**解码器端**:
```python
# 跳跃连接传递包含 LR 的特征
for idx, up_block in enumerate(decoder):
    x_skip = skip_connections[...]  # 来自编码器，已含 LR
    lr_feat = lr_pyramid[...]       # 当前尺度的 LR
    
    x, logits = up_block(x, x_skip, t_emb, c_emb, lr_feat)
```

**优势**:
- 编码器和解码器都感知 LR 结构
- 跨层传递 LR 信息
- 保持空间一致性

### 6. 最终上采样层

**Sub-pixel Convolution (Pixel Shuffle)**:

```python
class FinalUpsample(nn.Module):
    def __init__(self, in_channels, scale_factor):
        self.conv = nn.Conv2d(
            in_channels, 
            in_channels * (scale_factor ** 2),
            kernel_size=3, padding=1
        )
        self.pixel_shuffle = nn.PixelShuffle(scale_factor)
        self.activation = nn.GELU()
    
    def forward(self, x):
        # x: [B, C, 64, 64]
        x = self.conv(x)              # [B, C×16, 64, 64] (for 4x)
        x = self.pixel_shuffle(x)     # [B, C, 256, 256]
        x = self.activation(x)
        return x
```

**Pixel Shuffle 原理**:
```
输入: [B, C×r², H, W]
输出: [B, C, H×r, W×r]

例如 4× 上采样:
[B, 128, 64, 64] → Conv → [B, 2048, 64, 64]
                 → Shuffle → [B, 128, 256, 256]
```

**为什么用 Pixel Shuffle**:
- **高效**: 单次卷积 + 重排，无需学习插值
- **无伪影**: 避免转置卷积的棋盘效应
- **参数少**: 比多层转置卷积更轻量

**替代方案对比**:

| 方法 | 参数量 | 质量 | 速度 | 伪影 |
|------|--------|------|------|------|
| Pixel Shuffle | 低 | 高 | 快 | 少 |
| 转置卷积 | 中 | 中 | 中 | 多（棋盘） |
| 双线性+卷积 | 低 | 中 | 快 | 中 |
| 多层上采样 | 高 | 高 | 慢 | 少 |

## 完整前向传播流程

### 输入准备

```python
# 输入
hr_noisy = x_t              # [B, 3, 256, 256] 噪声 HR
lr_image = x_lr             # [B, 3, 64, 64] LR 图像
t = timestep                # [B] 时间步
c = condition               # [B] 条件

# 嵌入
t_emb = time_embedding(t)   # [B, 512]
c_emb = cond_embedding(c)   # [B, 512]
```

### LR 特征处理

```python
# 1. 编码 LR
lr_feat = lr_encoder(lr_image)  # [B, 64, 64, 64]

# 2. 上采样到 HR 尺寸
lr_feat = F.interpolate(lr_feat, size=(256, 256))  # [B, 64, 256, 256]

# 3. 构建金字塔
lr_pyramid = [lr_feat]
for downsampler in lr_downsamplers:
    lr_feat = downsampler(lr_feat)
    lr_pyramid.append(lr_feat)
# lr_pyramid: [256, 128, 64, 32, 16]
```

### U-Net 编码器

```python
x = input_proj(hr_noisy)  # [B, 128, 256, 256]

skip_connections = []
for idx, down_block in enumerate(encoder):
    lr_at_scale = lr_pyramid[idx]
    x, x_skip, logits = down_block(x, t_emb, c_emb, lr_at_scale)
    skip_connections.append(x_skip)
    moe_logits_list.append(logits)

# 编码器输出: [B, 1024, 32, 32]
# 跳跃连接: [(B,128,256), (B,256,128), (B,512,64), (B,1024,32)]
```

### 瓶颈层

```python
x = bottleneck_proj(x)  # [B, 1024, 16, 16]

if freq_enabled:
    x = freq_module(x, t_emb)
else:
    lr_bottleneck = lr_pyramid[-1]
    x, logits = bottleneck_conv(x, t_emb, c_emb, lr_bottleneck)
```

### U-Net 解码器

```python
for idx, up_block in enumerate(decoder):
    x_skip = skip_connections[-(idx+1)]
    lr_at_scale = lr_pyramid[...] # 对应尺度
    
    x, logits = up_block(x, x_skip, t_emb, c_emb, lr_at_scale)
    moe_logits_list.append(logits)

# 解码器输出: [B, 128, 256, 256]
```

### 最终输出

```python
# 上采样（如果需要）
if scale_factor > 1:
    x = final_upsample(x)  # [B, 128, 256, 256] (已是目标尺寸)

# 输出投影
noise_pred = output_proj(x)  # [B, 3, 256, 256]

return noise_pred, moe_logits_list
```

## 训练过程

### 数据准备

```python
# 1. 加载 HR 图像
hr_image = load_image()  # [B, 3, 256, 256]

# 2. 生成 LR 图像（双三次下采样）
lr_image = F.interpolate(
    hr_image, 
    size=(64, 64), 
    mode='bicubic', 
    antialias=True
)

# 3. 添加噪声到 HR
t = sample_timestep()
hr_noisy, noise = add_noise(hr_image, t)
```

### 训练步骤

```python
# 前向传播
noise_pred, moe_logits = model(hr_noisy, t, c, lr_image)

# 计算损失
L_DM = MSE(noise_pred, noise)
L_Load = load_balance_loss(moe_logits)
L_Total = L_DM + λ_load × L_Load

# 反向传播
loss.backward()
optimizer.step()
```

### 损失函数详解

**扩散损失** (L_DM):
```python
L_DM = ||ε_θ(x_t, t, c, x_lr) - ε||²
```
- 预测噪声与真实噪声的 MSE
- 在 HR 图像空间计算
- 以 LR 图像为条件

**负载均衡损失** (L_Load):
```python
for logits in moe_logits_list:
    probs = Softmax(logits)          # [B, N_experts]
    variance = Var(probs, dim=1)     # [B]
    L_Load += variance.mean()

L_Load = L_Load / len(moe_logits_list)
```
- 鼓励专家均匀激活
- 最小化专家选择的方差
- 防止专家崩溃（某些专家从不使用）

## 推理/采样

### DDPM 采样（标准）

```python
def sample_ddpm(model, lr_image, num_steps=1000):
    # 从纯噪声开始
    x_T = torch.randn(B, 3, 256, 256)
    
    x_t = x_T
    for t in reversed(range(num_steps)):
        # 预测噪声
        t_tensor = torch.tensor([t] * B)
        c = torch.zeros(B, dtype=torch.long)
        
        noise_pred, _ = model(x_t, t_tensor, c, lr_image)
        
        # 去噪一步
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t-1] if t > 0 else 1.0
        
        # 重参数化
        pred_x0 = (x_t - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
        
        # 添加噪声（除了最后一步）
        if t > 0:
            noise = torch.randn_like(x_t)
            x_t = sqrt(alpha_t_prev) * pred_x0 + sqrt(1 - alpha_t_prev) * noise
        else:
            x_t = pred_x0
    
    return x_t  # 生成的 HR 图像
```

### DDIM 采样（快速）

```python
def sample_ddim(model, lr_image, num_steps=50, eta=0.0):
    # DDIM 允许更少的步数
    timesteps = np.linspace(0, 999, num_steps, dtype=int)
    
    x_t = torch.randn(B, 3, 256, 256)
    
    for i in reversed(range(len(timesteps))):
        t = timesteps[i]
        t_prev = timesteps[i-1] if i > 0 else 0
        
        # 预测噪声
        noise_pred, _ = model(x_t, torch.tensor([t]), c, lr_image)
        
        # DDIM 更新
        alpha_t = alphas_cumprod[t]
        alpha_t_prev = alphas_cumprod[t_prev]
        
        pred_x0 = (x_t - sqrt(1 - alpha_t) * noise_pred) / sqrt(alpha_t)
        
        # 方向向量
        dir_xt = sqrt(1 - alpha_t_prev - eta²) * noise_pred
        
        # 更新
        x_t = sqrt(alpha_t_prev) * pred_x0 + dir_xt
        
        if eta > 0 and i > 0:
            noise = torch.randn_like(x_t)
            x_t += eta * sqrt(1 - alpha_t_prev) * noise
    
    return x_t
```

## 性能分析

### 计算复杂度

**LR 特征处理**:
- LR Encoder: O(C_lr × H_lr × W_lr)
- LR 金字塔生成: O(C_lr × Σ(H_i × W_i))
- 总开销: < 5% (相对主网络)

**MoE 额外开销**:
- Router: O(C × N_experts) (轻量级 MLP)
- 专家混合: O(K/N × Expert_FLOPs)
- 稀疏激活节省: ~(N-K)/N

**总体复杂度**:
```
SR U-Net ≈ Base U-Net + LR_Encoder + K/N × N_experts × Expert
```

### 参数量估计

假设配置：
- `model_channels = 128`
- `lr_channels = 64`
- `num_experts = 8`
- `k_active = 2`

| 组件 | 参数量 |
|------|--------|
| LR Encoder | ~0.1M |
| LR Downsamplers | ~0.3M |
| Base U-Net | ~30M |
| MoE 专家池 | ~15M |
| Router 网络 | ~0.5M |
| Final Upsample | ~2M |
| **总计** | **~48M** |

### 显存占用

| 批次大小 | 分辨率 | 显存 (FP32) | 显存 (FP16) |
|----------|--------|-------------|-------------|
| 1 | 256×256 | ~4 GB | ~2 GB |
| 4 | 256×256 | ~12 GB | ~6 GB |
| 8 | 256×256 | ~22 GB | ~11 GB |
| 16 | 256×256 | OOM | ~20 GB |

## 设计权衡

### 1. LR 通道数选择

| lr_channels | 优点 | 缺点 | 推荐场景 |
|-------------|------|------|----------|
| 32 | 轻量、快速 | 信息有限 | 2x SR, 快速推理 |
| 64 | 平衡 | - | 4x SR, 通用场景 |
| 128 | 丰富信息 | 显存消耗大 | 8x SR, 复杂纹理 |

### 2. MoE 配置

**专家数量**:
- 4 专家: 轻量，适合简单场景
- 8 专家: 平衡，推荐
- 16 专家: 高容量，复杂场景

**激活数量**:
- k=1: 最稀疏，最快
- k=2: 推荐（性能平衡）
- k=4: 更丰富的混合

### 3. 上采样策略

**一次性上采样** (本实现):
```
[B, C, 64, 64] → PixelShuffle(4×) → [B, C, 256, 256]
```
优点: 简单、高效
缺点: 大倍数上采样压力大

**渐进式上采样**:
```
[B, C, 64, 64] → PixelShuffle(2×) → [B, C, 128, 128]
               → PixelShuffle(2×) → [B, C, 256, 256]
```
优点: 更平滑的重建
缺点: 更多层，显存消耗大

## 总结

MoEFsDiC-SR 的核心创新：

1. **LR 条件注入**: 在每个 U-Net 块融合 LR 特征
2. **内容感知路由**: Router 根据 LR 内容动态选择专家
3. **多尺度融合**: LR 金字塔保证跨尺度的信息流动
4. **高效上采样**: Pixel Shuffle 实现高质量空间提升

这些设计使得模型能够：
- 保留 LR 图像的结构信息
- 根据不同内容使用不同的重建策略
- 高效地进行高清细节重建

---

**文档版本**: 1.0  
**最后更新**: 2025-10-05

