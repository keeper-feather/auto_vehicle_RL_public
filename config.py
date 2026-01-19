"""
配置文件
包含所有训练和环境参数
"""

import numpy as np


class VehicleConfig:
    """车辆参数配置"""

    # 整车参数
    M_EQ = 7020  # 等效质量 (kg)
    R_EFF = 0.3764  # 轮胎有效半径 (m)
    I_0 = 16.04  # 主减速比
    WHEEL_CIRCUMFERENCE = 2.365  # 轮胎周长 (m)

    # 阻力系数
    F_CONST = 447.37  # 常值阻力 (N)
    F_LIN_COEF = 1.1805  # 一次速度阻力系数
    F_QUAD_COEF = 0.1968  # 二次速度阻力系数
    F_BRAKE_MAX = 500  # 最大制动力 (N)


class BatteryConfig:
    """电池参数配置"""

    # 容量参数
    CAPACITY_AH = 25  # 电池容量 (Ah)
    N_CELL = 102  # 串联电芯数
    SOC_INIT = 0.975  # 初始 SOC
    SOC_TARGET = 0.6  # 目标 SOC
    SOC_MIN = 0.1  # SOC 下限
    SOC_MAX = 0.9  # SOC 上限

    # DC/DC 效率
    FC_DCDC_EFF = 0.85  # 燃料电池 DC/DC 效率
    ICE_DCDC_EFF = 0.95  # ICE 发电机 DC/DC 效率

    # 电池查表数据 (25°C)
    SOC_POINTS = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    VOC_CELLS = np.array([3.46, 3.58, 3.64, 3.68, 3.73, 3.78, 3.87, 3.98, 4.08])  # V/cell
    R_DIS = np.array([1.89e-3, 1.85e-3, 1.83e-3, 1.81e-3, 1.81e-3, 1.82e-3,
                      1.83e-3, 1.86e-3, 1.86e-3])  # Ohm (放电)
    R_CHG = np.array([1.56e-3, 1.50e-3, 1.50e-3, 1.46e-3, 1.47e-3, 1.47e-3,
                      1.48e-3, 1.53e-3, 1.54e-3])  # Ohm (充电)


class MotorConfig:
    """电机参数配置"""

    # 最大转矩表
    VOLT_POINTS = np.array([278, 336, 402])  # V
    SPEED_POINTS = np.array([700, 1400, 2100, 2800, 3100, 3400, 3800, 4100, 4800,
                             5500, 6200, 6900, 7600, 8300, 9000, 9700, 10400,
                             11000, 11500, 12000])  # rpm

    # 简化的效率表
    TORQUE_POINTS = np.array([14, 70, 140, 211, 281, 365])

    # 电机效率系数
    DRIVE_EFF = 0.97
    REGENERATIVE_EFF = 0.89


class PowertrainConfig:
    """动力系统配置"""

    # ICE 参数
    ICE_MAX_POWER = 60000  # W (60 kW)
    ICE_POWER_POINTS = np.array([4.79, 10.69, 21.12, 30.15, 40.71, 51.21, 60.82, 69.98, 78.29])
    ICE_H2_CONSUMPTION = np.array([0.11, 0.20, 0.38, 0.54, 0.72, 0.91, 1.09, 1.28, 1.46])

    # FC 参数
    FC_MAX_POWER = 60000  # W (60 kW)
    FC_POWER_POINTS = np.array([0, 12.04, 22.27, 38.72, 58.66, 77.49, 99.86])
    FC_H2_CONSUMPTION = np.array([0, 0.15, 0.29, 0.54, 0.84, 1.14, 1.51])


class TrainingConfig:
    """训练配置"""

    # 环境参数
    DT = 1.0  # 时间步长 (秒)
    MAX_SPEED = 90.0 / 3.6  # 最大速度 (m/s)
    MAX_ACC = 2.0  # 最大加速度 (m/s²)
    MAX_POWER = 120000  # 最大功率需求 (W)

    # PPO 超参数
    LEARNING_RATE = 3e-4
    N_STEPS = 2048
    BATCH_SIZE = 64
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    TARGET_KL = 0.01

    # 训练参数
    TOTAL_TIMESTEPS = 500000
    N_ENVS = 8  # 并行环境数量
    CHECKPOINT_FREQ = 10000  # 保存检查点频率

    # 奖励权重
    REWARD_H2_WEIGHT = 1.0
    REWARD_SOC_WEIGHT = 150.0
    REWARD_VIOLATION_WEIGHT = 1000.0


class PathConfig:
    """路径配置"""

    # 数据路径
    CYCLE_DATA_PATH = "./data/cycle_data.csv"

    # 输出路径
    LOG_DIR = "./logs"
    MODEL_DIR = "./models"
    CHECKPOINT_DIR = "./checkpoints"
    PLOT_DIR = "./plots"

    # 文件名
    MODEL_NAME = "ppo_hev_final"
    CYCLE_PLOT_NAME = "cycle_visualization.png"
    VALIDATION_PLOT_NAME = "validation_results.png"
