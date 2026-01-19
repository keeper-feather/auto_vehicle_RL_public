# 混合动力汽车能量管理策略 - 强化学习项目

## 项目概述

使用 **PPO (Proximal Policy Optimization)** 强化学习算法优化氢燃料电池+氢内燃机混合动力汽车的能量管理策略。

---

## 快速开始

### 1. 创建环境
```bash
conda env create -f environment.yml
conda activate hev_rl
```

### 2. 测试环境
```bash
python test_env.py
```

### 3. 开始训练
```bash
python train_ppo.py
```

### 4. 监控训练
```bash
tensorboard --logdir ./logs
```

### 5. 查看数据
```bash
python view_data.py --list
python view_data.py --compare
```

---

## 项目结构详解

```
auto_vehicle_RL/
│
├──  核心代码文件
│   ├── hev_env.py              #  Gymnasium环境定义
│   ├── train_ppo.py            #  PPO训练脚本
│   ├── cycle_loader.py         #  工况数据加载模块
│   ├── config.py               #  配置参数集中管理
│   ├── utils.py                # 工具函数（绘图、评估）
│   └── data_manager.py         # 数据保存管理
│
├──  测试与查看
│   ├── test_env.py             # 环境测试脚本
│   ├── test_cycles.py          # 工况切换测试脚本
│   ├── test_data_manager.py    # 数据管理测试
│   ├── view_data.py            # 查看保存的训练数据
│   └── monitor_training.py     # 实时监控训练进度
│
├──  环境配置
│   ├── environment.yml         # Conda环境配置
│   ├── requirements.txt        # Pip依赖备份
│   └── setup.sh                # 自动化安装脚本
│
│
├──  工况数据
│   └── state_data/             # 驾驶工况数据目录
│       ├── CWTVC.xlsx          # 中国轻型汽车测试工况
│       ├── NEDC.xlsx           # 新欧洲驾驶工况
│       └── WLTC.xlsx           # 全球轻型汽车测试工况
│
├──  输出目录（自动生成）
│   ├── data/                   # 训练数据存储
│   │   └── YYYYMMDD_HHMMSS/   # 训练会话时间戳
│   │       └── agent_N/       # 第N个符合条件的agent
│   │           ├── summary.json
│   │           ├── episode_data.npz
│   │           └── *.png      # 4张分析图表
│   │
│   ├── models/                 # 训练好的模型
│   │   └── ppo_hev_final.zip
│   │
│   ├── checkpoints/            # 训练检查点
│   │   └── ppo_hev_100000_steps.zip
│   │
│   ├── logs/                   # Tensorboard日志
│   │   └── PPO_1/
│   │
│   └── plots/                  # 临时图表
│       └── *.png
│
└──  其他
    ├── .gitignore              # Git忽略文件
    └── training_log.txt        # 训练日志
```

---

##  关键参数调整

### 0️⃣ 工况配置  新功能
**位置**: [`train_ppo.py`](train_ppo.py:25)

支持多种标准驾驶工况的灵活切换，只需修改一个变量：

```python
# ═══════════════════════════════════════════════════════════════
#  工况配置 - 修改这里切换工况
# ═══════════════════════════════════════════════════════════════
# 可选工况: 'cwtvc', 'nedc', 'wltp'
TARGET_CYCLE = 'cwtvc'  #  修改这个变量切换工况
# ═══════════════════════════════════════════════════════════════
```

**支持的工况**:
| 工况代码 | 全称 | 地区 | 特点 |
|---------|------|------|------|
| `cwtvc` | CWTVC | 中国 | 中国轻型汽车测试工况 (30分钟, 0-87.8 km/h) |
| `nedc` | NEDC | 欧洲 | 新欧洲驾驶工况 (20分钟, 0-120.0 km/h) |
| `wltp` | WLTP | 全球 | 全球轻型汽车测试工况 (30分钟, 0-131.3 km/h) |

**使用方法**:
```bash
# 1. 测试工况加载
python test_cycles.py

# 2. 修改 train_ppo.py 中的 TARGET_CYCLE
# 3. 开始训练
python train_ppo.py
```

**工况数据格式**:
- 文件位置: [`state_data/`](state_data/)
- 文件格式: Excel (.xlsx)
- 数据列: 第1列 = 时间(秒), 第2列 = 速度(km/h)
- 自动处理: 单位转换、加速度计算、错误回退

---

### 1️⃣ 训练超参数
**位置**: [`train_ppo.py`](train_ppo.py:246-266)

```python
# PPO算法参数
learning_rate = 3e-4      # 学习率
n_steps = 2048           # 每次更新采样步数
batch_size = 64          # 批次大小
n_epochs = 10            # 训练轮数
gamma = 0.99             # 折扣因子

# 训练规模
total_timesteps = 500000  # 总训练步数
n_envs = 8               # 并行环境数量
```

**调整建议**:
- **快速验证**: `total_timesteps = 50000`, `n_envs = 4`
- **高精度训练**: `total_timesteps = 1000000`, `n_envs = 16`
- **加速训练**: 增加 `n_envs`（需要更多内存）

---

### 2️⃣ 环境参数
**位置**: [`hev_env.py`](hev_env.py:168-173)

```python
# 奖励权重  重要！
self.reward_h2_weight = 1.0         # 氢耗权重
self.reward_soc_weight = 500.0      # SOC维持权重 - 提高以强制使用氢源
self.reward_violation_weight = 1000 # 越界惩罚权重

# SOC目标
self.soc_target = 0.5       # 目标SOC值
self.soc_init = 0.5         # 初始SOC
```
---

### 3️⃣ 车辆物理参数
**位置**: [`hev_env.py`](hev_env.py:58-85)

```python
# 整车参数
self.m_eq = 7020           # 等效质量 (kg)
self.r_eff = 0.3764        # 轮胎有效半径 (m)
self.i_0 = 16.04           # 主减速比

# 阻力系数
self.F_const = 447.37      # 常值阻力 (N)
self.F_lin_coef = 1.1805   # 一次速度阻力系数
self.F_quad_coef = 0.1968  # 二次速度阻力系数
```

---

### 4️⃣ 电池参数
**位置**: [`hev_env.py`](hev_env.py:51-58)

```python
# 容量
self.bat_capacity = 10     # 电池容量 (Ah) - 减小以强制使用氢源
self.n_cell = 102          # 串联电芯数

# SOC限制
self.soc_init = 0.5        # 初始SOC (50%)
self.soc_min = 0.2         # SOC下限 (20%)
self.soc_max = 0.8         # SOC上限 (80%)
```

---

### 5️⃣ 动力系统参数
**位置**: [`hev_env.py`](hev_env.py:144-164)

```python
# ICE（氢内燃机）
self.ice_max_power = 60000  # 最大功率 (W) = 60 kW

# FC（燃料电池）
self.fc_max_power = 60000   # 最大功率 (W) = 60 kW

# DC/DC效率
self.fc_dcdc_eff = 0.85     # 燃料电池DC/DC效率
self.ice_dc_dc_eff = 0.95   # ICE发电机DC/DC效率
```

---

### 6️⃣ 数据保存阈值
**位置**: [`train_ppo.py`](train_ppo.py:337)

```python
# 只保存平均奖励 > -500 的数据
data_manager = DataManager(
    base_dir="data",
    reward_threshold=-500  #  修改这里调整保存阈值
)
```
##  输出目录说明

### `/data/` - 训练数据存储
**内容**: 自动保存符合标准的训练数据

**结构**:
```
data/
├── 20260118_235750/           # 训练会话时间戳
│   ├── agent_1/               # 第1个符合条件的agent
│   │   ├── summary.json       # 性能摘要（JSON格式）
│   │   ├── episode_data.npz   # 详细数据（NPZ压缩格式）
│   │   ├── agent_results_soc.png       # SOC轨迹图
│   │   ├── agent_results_power.png     # 功率分配图
│   │   ├── agent_results_reward.png    # 奖励曲线图
│   │   └── agent_results_metrics.png   # 性能指标图
│   └── agent_2/               # 第2个agent
└── ...
```
**查看数据**:
```bash
python view_data.py --list           # 列出所有会话
python view_data.py --compare         # 对比top 5
python view_data.py --compare 10      # 对比top 10
```

---

### `/models/` - 训练模型
**内容**: 训练完成后的最终模型

**文件**:
- `ppo_hev_final.zip` - 完整的PPO模型（包含网络参数）

**使用模型**:
```python
from stable_baselines3 import PPO
model = PPO.load("models/ppo_hev_final")
```

---

### `/checkpoints/` - 训练检查点
**内容**: 训练过程中的中间模型

**文件**:
- `ppo_hev_100000_steps.zip` - 第100K步的模型
- `ppo_hev_200000_steps.zip` - 第200K步的模型
- ...

**用途**:
- 恢复中断的训练
- 对比不同阶段的模型性能
- 选择最佳检查点

---

### `/logs/` - Tensorboard日志
**内容**: 训练过程的实时数据

**查看**:
```bash
tensorboard --logdir ./logs
# 然后打开 http://localhost:6006
```

**可查看**:
- 奖励曲线
- 损失函数
- 策略熵
- Episode长度
- 学习率变化

---

### `/plots/` - 临时图表
**内容**: 每次训练生成的临时图表

**文件**:
- `cycle_visualization.png` - 工况数据可视化
- `validation_results*.png` - 验证结果图表

**注意**: 这些是未筛选的图表，符合条件的数据会保存到 `data/` 目录

---

##  常用操作

### 查看训练进度
```bash
# 方法1: 查看Tensorboard
tensorboard --logdir ./logs

# 方法2: 实时监控脚本
python monitor_training.py

# 方法3: 查看训练日志
tail -f training_log.txt
```

### 对比不同训练
```bash
# 列出所有训练会话
python view_data.py --list

# 对比top 5 agents
python view_data.py --compare

# 对比top 10 agents
python view_data.py --compare 10

# 查看特定agent详情
python view_data.py --agent data/20260118_235750/agent_1
```

### 重新训练
```bash
# 方法1: 直接训练（会覆盖之前的模型）
python train_ppo.py

# 方法2: 修改参数后训练
# 1. 编辑 train_ppo.py 或 hev_env.py
# 2. 运行训练
python train_ppo.py
```

### 清理数据
```bash
# 清理测试数据
rm -rf data/test/

# 清理旧模型
rm -rf models/* checkpoints/* logs/*

# 清理临时图表
rm -rf plots/*.png
```
##  工况切换功能详解

### 快速切换
```python
# 在 train_ppo.py 第 25 行修改
TARGET_CYCLE = 'cwtvc'  # 可选: 'cwtvc', 'nedc', 'wltp'
```

### 编程方式使用
```python
from hev_env import ContinuousHEVEnv
from cycle_loader import load_cycle_data

# 方式1: 使用工况名称
env = ContinuousHEVEnv(cycle_name='nedc')

# 方式2: 直接传入工况数据
cycle_data = load_cycle_data('wltp')
env = ContinuousHEVEnv(cycle_data=cycle_data)
```

### API 参考
```python
from cycle_loader import load_cycle_data, print_cycle_info, get_available_cycles

# 加载工况数据
data = load_cycle_data('cwtvc')
# 返回: numpy array shape=(N, 2), 列0=速度(m/s), 列1=加速度(m/s²)

# 打印工况信息
print_cycle_info('nedc')

# 获取可用工况列表
cycles = get_available_cycles()
print(f"可用工况: {cycles}")
```

##  训练示例

### 基础训练（推荐新手）
```python
# train_ppo.py
total_timesteps = 200000  # 20万步，约1-2分钟
n_envs = 4                # 4个并行环境
```

### 标准训练（默认）
```python
total_timesteps = 500000  # 50万步，约3-5分钟
n_envs = 8                # 8个并行环境
```

### 高精度训练
```python
total_timesteps = 1000000 # 100万步，约8-10分钟
n_envs = 16               # 16个并行环境
learning_rate = 1e-4      # 更小的学习率
```

---



## 作者

Auto Vehicle RL ：张涵 车31 hzhang23@mails.tsinghua.edu.cn