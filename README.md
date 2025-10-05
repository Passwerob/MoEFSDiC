# MoEFsDiC: Mixture-of-Experts Frequency-Separable Dilated Conv Diffusion

基于 DiC 的纯 Conv3x3 哲学，通过集成 MoE 结构和频率域（FSNL）增强，构建高效、可扩展的扩散模型 U-Net 骨干网络。

## 项目特点

- **混合专家系统 (MoE)**: 使用稀疏激活的专家网络，提高模型容量和效率
- **频率域增强 (FSNL)**: 在 U-Net 瓶颈层使用频率域全局感知模块
- **高效卷积**: 采用深度可分离卷积 (DS-Conv) 减少参数量和计算量
- **灵活的消融实验**: 通过配置文件轻松控制各个组件的开关

## 项目结构

```
MoEFsDiC/
├── src/
│   ├── modules/
│   │   ├── experts.py          # 专家和路由模块
│   │   ├── freq_module.py      # 频率域模块
│   │   └── conv_blocks.py      # 核心卷积块
│   ├── models/
│   │   └── moefsndic_unet.py   # 主模型
│   └── utils/
│       └── loss.py             # 损失函数
├── configs/
│   └── default.yaml            # 默认配置
└── train.py                    # 训练脚本
```

## 核心组件

### 1. DepthwiseSeparableConv2d
高效的深度可分离卷积，包含 Depthwise Conv + Pointwise Conv。

### 2. Expert 模块
每个专家是一个残差块，使用 DS-Conv 进行特征处理。

### 3. Router 模块
负责计算专家分数，融合空间特征、时间嵌入和条件嵌入。

### 4. Freq_Global_Module
在频率域进行特征调制，增强全局感知能力。

### 5. MoE_ConvBlock
混合专家卷积块，实现 Top-K 稀疏激活。

### 6. MoEFsDiC_UNet
完整的 U-Net 架构，集成所有组件。

## 安装依赖

```bash
pip install torch torchvision pyyaml tqdm
```

## 使用方法

### 训练模型

```bash
python train.py --config configs/default.yaml --device cuda
```

### 消融实验

通过修改 `configs/default.yaml` 进行不同的消融实验：

#### 1. 禁用 MoE（使用 Dilated Fusion Block）
```yaml
moe:
  enabled: false
```

#### 2. 禁用频率域模块
```yaml
freq:
  enabled: false
```

#### 3. 禁用深度可分离卷积
```yaml
train:
  use_dsconv_global: false
```

#### 4. 调整专家数量和激活数
```yaml
moe:
  num_experts: 16  # 专家数量
  k_active: 4      # Top-K 激活数
```

## 配置说明

### 模型配置
- `in_channels`: 输入通道数（默认 3 for RGB）
- `model_channels`: 基础通道数
- `channel_mults`: 每层的通道倍数
- `t_dim`: 时间嵌入维度
- `c_dim`: 条件嵌入维度

### MoE 配置
- `enabled`: MoE 开关
- `num_experts`: 专家数量
- `k_active`: Top-K 激活专家数

### 频率域配置
- `enabled`: 频率域模块开关

### 训练配置
- `lambda_load`: 负载均衡损失权重
- `learning_rate`: 学习率
- `batch_size`: 批次大小
- `use_dsconv_global`: 是否全局使用深度可分离卷积

## 损失函数

模型使用组合损失：

```
L_Total = L_DM + λ_load × L_Load
```

- **L_DM**: 扩散模型 MSE 损失
- **L_Load**: 负载均衡损失（专家概率分布方差）
- **λ_load**: 负载均衡损失权重

## 架构细节

### U-Net 结构
- **编码器**: 多层下采样，使用 MoE_ConvBlock
- **瓶颈**: 使用 Freq_Global_Module 进行频率域增强
- **解码器**: 多层上采样，结合跳跃连接

### MoE 机制
1. Router 计算专家分数
2. Top-K 稀疏激活
3. 加权混合专家输出
4. 负载均衡正则化

### 频率域增强
1. FFT 转换到频率域
2. MLP 生成调制权重
3. 频率域调制
4. IFFT 转回空间域
5. 残差连接

## 实验建议

### 基线实验
1. **Full Model**: 所有组件启用
2. **No MoE**: 禁用 MoE，使用 Dilated Fusion
3. **No Freq**: 禁用频率域模块
4. **No DS-Conv**: 使用标准卷积

### 超参数搜索
- 专家数量: [4, 8, 16, 32]
- Top-K: [1, 2, 4, 8]
- λ_load: [0.001, 0.01, 0.1]

## License

MIT License

## 引用

如果您使用了本项目，请引用：

```bibtex
@software{moefsndic2025,
  title={MoEFsDiC: Mixture-of-Experts Frequency-Separable Dilated Conv Diffusion},
  year={2025}
}
```
