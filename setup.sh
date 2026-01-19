#!/bin/bash

# 创建并激活 conda 环境
echo "创建 conda 环境..."
conda env create -f environment.yml

# 激活环境
echo "激活环境..."
eval "$(conda shell.bash hook)"
conda activate hev_rl

# 创建必要的目录
mkdir -p plots
mkdir -p logs
mkdir -p models
mkdir -p checkpoints

echo "环境设置完成!"
echo "使用 'conda activate hev_rl' 激活环境"
