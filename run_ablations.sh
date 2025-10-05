#!/bin/bash
# 批量运行消融实验脚本

echo "=== MoEFsDiC Ablation Study ==="
echo ""

# 设置设备
DEVICE=${1:-cuda}
EPOCHS=${2:-100}

echo "Device: $DEVICE"
echo "Epochs: $EPOCHS"
echo ""

# 实验配置列表
configs=(
    "configs/default.yaml"
    "configs/no_moe.yaml"
    "configs/no_freq.yaml"
    "configs/no_dsconv.yaml"
    "configs/minimal.yaml"
    "configs/large_moe.yaml"
)

# 运行每个实验
for config in "${configs[@]}"; do
    echo "----------------------------------------"
    echo "Running experiment: $config"
    echo "----------------------------------------"
    
    python train.py \
        --config "$config" \
        --device "$DEVICE" \
        2>&1 | tee "logs/$(basename $config .yaml).log"
    
    echo ""
    echo "Completed: $config"
    echo ""
done

echo "=== All experiments completed ==="

