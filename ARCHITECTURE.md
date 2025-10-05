# MoEFsDiC 架构详解

本文档详细说明 MoEFsDiC 项目的架构设计和实现细节。

## 整体架构

MoEFsDiC 是一个基于 U-Net 的扩散模型骨干网络，集成了三个核心创新：

1. **混合专家系统 (MoE)**: 稀疏激活的专家网络
2. **频率域增强 (FSNL)**: 在频率域进行全局调制
3. **高效卷积 (DiC)**: 深度可分离卷积减少参数

```
输入图像 (x_t) + 时间步 (t) + 条件 (c)
    ↓
[编码器] → 多层 MoE_ConvBlock + 下采样
    ↓
[瓶颈] → Freq_Global_Module (频率域增强)
    ↓
[解码器] → 多层 MoE_ConvBlock + 上采样 + 跳跃连接
    ↓
输出 (预测噪声) + MoE Logits
```

## 核心组件详解

### 1. DepthwiseSeparableConv2d

**目的**: 减少卷积参数量和计算量

**结构**:
```
输入 [B, C_in, H, W]
    ↓
Depthwise Conv (groups=C_in) → [B, C_in, H, W]
    ↓
Pointwise Conv (1x1) → [B, C_out, H, W]
```

**优势**:
- 标准卷积参数量: `C_in × C_out × K × K`
- DS-Conv 参数量: `C_in × K × K + C_in × C_out`
- 对于 K=3, C_in=C_out=256，参数减少约 9 倍

**代码关键点**:
```python
self.depthwise = nn.Conv2d(in_channels, in_channels, 
                           kernel_size=3, groups=in_channels)
self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
```

### 2. Expert 模块

**目的**: MoE 系统中的专家单元

**结构**:
```
输入 x
    ↓
DS-Conv + BN + GELU
    ↓
DS-Conv + BN
    ↓
残差连接: out = conv(x) + x
    ↓
GELU
```

**特点**:
- 每个专家是独立的残差块
- 使用 GELU 激活函数（比 ReLU 更平滑）
- BatchNorm 用于稳定训练

### 3. Router 模块

**目的**: 计算专家选择概率

**输入融合**:
```
特征图 x [B, C, H, W]
    ↓ GAP
空间特征 [B, C]
    ↓
拼接: [spatial_feat, t_emb, c_emb]
    ↓
MLP
    ↓
专家 Logits [B, N_experts]
```

**设计考虑**:
- **空间特征**: 通过 GAP 获取全局上下文
- **时间嵌入**: 反映当前扩散步骤
- **条件嵌入**: 融合类别等条件信息
- **轻量级 MLP**: 避免路由开销过大

**数学表示**:
```
f_spatial = GAP(x) ∈ R^C
f_concat = Concat(f_spatial, t_emb, c_emb) ∈ R^(C+T+D)
logits = MLP(f_concat) ∈ R^N
```

### 4. MoE_ConvBlock

**目的**: 实现稀疏混合专家机制

**流程**:
```
1. 投影: x → x_proj (通道对齐)
2. 路由: Router(x_proj, t, c) → logits [B, N]
3. Top-K 选择: TopK(logits, K) → weights, indices
4. 归一化: Softmax(weights) → norm_weights
5. 混合专家:
   output = Σ(norm_weights[i] × Expert[i](x_proj))
6. 返回: (output, logits)
```

**稀疏激活**:
- 只激活 Top-K 个专家（K << N）
- 例如：8 个专家中只激活 2 个
- 大幅减少计算量，同时保持模型容量

**负载均衡**:
- 收集所有 logits 用于计算负载损失
- 鼓励专家均匀分配任务
- 避免某些专家过载，其他专家闲置

**代码关键点**:
```python
# Top-K 选择
topk_weights, topk_indices = torch.topk(router_logits, k=2, dim=1)
topk_weights_norm = F.softmax(topk_weights, dim=1)

# 专家混合
for k_idx in range(K):
    expert_idx = topk_indices[b, k_idx]
    weight = topk_weights_norm[b, k_idx]
    expert_output += weight * experts[expert_idx](x)
```

### 5. Freq_Global_Module

**目的**: 在频率域进行全局特征调制

**流程**:
```
输入 x [B, C, H, W]
    ↓
1. 提取全局上下文: GAP(x) → [B, C]
2. 融合时间信息: Concat(global, t_emb) → [B, C+T]
3. 计算调制权重: MLP → [B, C]
4. FFT 到频率域: FFT2(x) → X_freq (complex)
5. 频率域调制: X_freq * weights
6. IFFT 回空间域: IFFT2(X_modulated) → x_spatial
7. 残差连接: x_spatial + x
```

**为什么使用频率域**:
- **全局感知**: FFT 天然具有全局感受野
- **频率调制**: 可以选择性增强/抑制不同频率成分
- **高效**: FFT 计算复杂度 O(N log N)

**数学表示**:
```
w = MLP(Concat(GAP(x), t_emb)) ∈ R^C
X_freq = FFT2(x) ∈ C^(B×C×H×W)
X_mod = X_freq ⊙ w
x_out = Real(IFFT2(X_mod)) + x
```

**适用场景**:
- U-Net 瓶颈层（特征维度最小）
- 需要全局信息融合的位置
- 扩散模型中低分辨率特征处理

### 6. Dilated_Fusion_Block

**目的**: MoE 禁用时的替代方案

**结构**:
```
输入 x
    ↓
投影 → x_proj
    ↓
并行分支:
  - DS-Conv (dilation=1)
  - DS-Conv (dilation=2)
  - DS-Conv (dilation=4)
    ↓
拼接所有分支 → Concat
    ↓
1x1 融合 → fused
    ↓
残差连接: fused + x_proj
```

**扩张卷积的作用**:
- 不同扩张率捕获不同尺度的上下文
- 不增加参数的情况下扩大感受野
- 多尺度特征融合

**参数量对比**:
- MoE 版本: N_experts × Expert_params + Router_params
- Dilated 版本: N_branches × Conv_params + Fusion_params

### 7. MoEFsDiC_UNet

**完整架构**:

```
输入 [B, 3, 64, 64]
    ↓
时间嵌入: TimestepEmbedding(t) → [B, 512]
条件嵌入: ConditionEmbedding(c) → [B, 512]
    ↓
输入投影: Conv3x3 → [B, 128, 64, 64]
    ↓
┌─────────────────────────────────────────┐
│ 编码器 (Encoder)                        │
│                                         │
│ Layer 1: [B, 128, 64, 64]              │
│   → MoE_ConvBlock + Downsample         │
│   → [B, 256, 32, 32] (skip_1)          │
│                                         │
│ Layer 2: [B, 256, 32, 32]              │
│   → MoE_ConvBlock + Downsample         │
│   → [B, 512, 16, 16] (skip_2)          │
│                                         │
│ Layer 3: [B, 512, 16, 16]              │
│   → MoE_ConvBlock + Downsample         │
│   → [B, 1024, 8, 8] (skip_3)           │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 瓶颈 (Bottleneck)                       │
│                                         │
│ [B, 1024, 8, 8] → Conv (stride=2)      │
│   → [B, 1024, 4, 4]                    │
│   → Freq_Global_Module                 │
│   → [B, 1024, 4, 4]                    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│ 解码器 (Decoder)                        │
│                                         │
│ Layer 1: [B, 1024, 4, 4]               │
│   → Upsample + Concat(skip_3)          │
│   → [B, 1536, 8, 8]                    │
│   → MoE_ConvBlock                      │
│   → [B, 512, 8, 8]                     │
│                                         │
│ Layer 2: [B, 512, 8, 8]                │
│   → Upsample + Concat(skip_2)          │
│   → [B, 768, 16, 16]                   │
│   → MoE_ConvBlock                      │
│   → [B, 256, 16, 16]                   │
│                                         │
│ Layer 3: [B, 256, 16, 16]              │
│   → Upsample + Concat(skip_1)          │
│   → [B, 384, 32, 32]                   │
│   → MoE_ConvBlock                      │
│   → [B, 128, 32, 32]                   │
│                                         │
│ Layer 4: [B, 128, 32, 32]              │
│   → Upsample                           │
│   → MoE_ConvBlock                      │
│   → [B, 128, 64, 64]                   │
└─────────────────────────────────────────┘
    ↓
输出投影: GroupNorm + GELU + Conv3x3
    ↓
输出 [B, 3, 64, 64] + MoE Logits
```

**关键设计**:

1. **跳跃连接**: 编码器特征直接传递到解码器
2. **U-Net 对称性**: 编码器和解码器层数对称
3. **通道倍增**: `[1, 2, 4, 8]` 倍递增
4. **Logits 收集**: 每个 MoE 块的 logits 都被收集

**Forward 方法关键逻辑**:
```python
def forward(self, x, t, c):
    # 清空 logits 列表
    self.moe_logits_list = []
    
    # 嵌入
    t_emb = self.time_embedding(t)
    c_emb = self.condition_embedding(c)
    
    # 编码器
    skip_connections = []
    for down_block in self.encoder:
        x, x_skip, logits = down_block(x, t_emb, c_emb)
        skip_connections.append(x_skip)
        if logits is not None:
            self.moe_logits_list.append(logits)
    
    # 瓶颈
    x = self.bottleneck(x, t_emb)
    
    # 解码器
    for up_block in self.decoder:
        x_skip = skip_connections.pop()
        x, logits = up_block(x, x_skip, t_emb, c_emb)
        if logits is not None:
            self.moe_logits_list.append(logits)
    
    return self.output_proj(x), self.moe_logits_list
```

### 8. MoELoss

**损失函数组成**:

```
L_total = L_DM + λ_load × L_Load
```

**扩散损失 (L_DM)**:
```python
L_DM = MSE(noise_pred, target_noise)
```
- 标准的扩散模型损失
- 预测噪声与真实噪声的均方误差

**负载均衡损失 (L_Load)**:
```python
for logits in all_moe_logits:
    probs = Softmax(logits)  # [B, N_experts]
    variance = Var(probs, dim=1)  # [B]
    L_Load += variance.mean()

L_Load = L_Load / len(all_moe_logits)
```

**为什么使用方差**:
- 方差大 → 专家分布不均 → 某些专家过载
- 方差小 → 专家分布均匀 → 负载平衡
- 最小化方差 = 鼓励均匀分布

**超参数 λ_load**:
- 太小 (< 0.001): 负载可能不均衡
- 适中 (0.01): 平衡性能和负载
- 太大 (> 0.1): 可能牺牲性能强制均衡

## 训练流程

### 前向扩散过程

```python
# 1. 采样时间步
t ~ Uniform(0, T)

# 2. 采样噪声
ε ~ N(0, I)

# 3. 加噪
x_t = √(α̅_t) × x_0 + √(1 - α̅_t) × ε
```

### 模型训练

```python
# 4. 预测噪声
ε_pred = Model(x_t, t, c)

# 5. 计算损失
L = MSE(ε_pred, ε) + λ × L_Load(moe_logits)

# 6. 反向传播
L.backward()
optimizer.step()
```

## 消融实验矩阵

| 实验 | MoE | Freq | DS-Conv | 说明 |
|------|-----|------|---------|------|
| Full | ✓ | ✓ | ✓ | 完整模型 |
| No-MoE | ✗ | ✓ | ✓ | 使用 Dilated Fusion |
| No-Freq | ✓ | ✗ | ✓ | 瓶颈层用 MoE 块 |
| No-DS | ✓ | ✓ | ✗ | 标准卷积 |
| Minimal | ✗ | ✗ | ✗ | 最小基线 |

## 参数量分析

假设：
- `C = 128` (model_channels)
- `N = 8` (num_experts)
- `K = 2` (k_active)
- `T = 512` (t_dim)

**MoE_ConvBlock 参数**:
```
投影层: C_in × C_out (DS-Conv)
Router: (C + T + C) × (C + T + C)/2 × N
专家池: N × Expert_params
```

**完整模型参数**:
```
编码器: ~3 个 MoE 块
瓶颈: 1 个 Freq 模块
解码器: ~4 个 MoE 块
总计: 约 50-100M 参数（取决于配置）
```

## 计算复杂度

**标准卷积**: O(C² × H × W × K²)
**DS-Conv**: O(C × H × W × K² + C² × H × W)
**MoE**: O(K/N × N × Expert_FLOPs)
**FFT**: O(C × H × W × log(H × W))

## 总结

MoEFsDiC 通过以下创新实现高效扩散模型：

1. **MoE**: 增加模型容量，稀疏激活控制计算
2. **频率域**: 全局感知，高效特征调制
3. **DS-Conv**: 减少参数，保持性能

这些组件可以独立启用/禁用，便于进行消融实验和性能分析。

