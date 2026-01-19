"""
工具函数
包含数据加载、可视化、评估等辅助功能
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import os


def load_cycle_data(filepath):
    """
    从 CSV 文件加载工况数据

    参数:
        filepath: CSV 文件路径

    返回:
        cycle_data: numpy 数组, shape=(N, 2), 列0=速度(m/s), 列1=加速度(m/s²)
    """
    try:
        data = np.loadtxt(filepath, delimiter=',')
        print(f"从 {filepath} 加载工况数据: {len(data)} 个时间步")
        return data
    except Exception as e:
        print(f"加载工况数据失败: {e}")
        return None


def plot_comparison(results_dict, save_path="comparison.png"):
    """
    对比多个策略的性能

    参数:
        results_dict: 字典,键为策略名称,值为指标字典
        save_path: 保存路径
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    strategies = list(results_dict.keys())

    # 1. SOC 曲线对比
    ax = axes[0, 0]
    for strategy in strategies:
        metrics = results_dict[strategy]
        ax.plot(metrics['soc_history'], label=strategy, linewidth=2)
    ax.axhline(y=0.6, color='red', linestyle='--', label='Target SOC', alpha=0.5)
    ax.set_ylabel('SOC', fontsize=12)
    ax.set_title('SOC Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. 累计氢耗对比
    ax = axes[0, 1]
    for strategy in strategies:
        metrics = results_dict[strategy]
        # 计算累计氢耗 (简化)
        ax.plot(np.arange(len(metrics['soc_history'])),
                np.linspace(0, metrics['total_h2_consumption_g'], len(metrics['soc_history'])),
                label=strategy, linewidth=2)
    ax.set_ylabel('Cumulative H2 (g)', fontsize=12)
    ax.set_title('Hydrogen Consumption Comparison', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. 功率分配对比 (选择第一个策略)
    ax = axes[1, 0]
    strategy = strategies[0]
    metrics = results_dict[strategy]
    ax.plot(metrics['power_ice_history'] / 1000, label='ICE Power', linewidth=1.5, alpha=0.8)
    ax.plot(metrics['power_fc_history'] / 1000, label='FC Power', linewidth=1.5, alpha=0.8)
    ax.plot(metrics['power_batt_history'] / 1000, label='Battery Power', linewidth=1.5, alpha=0.8)
    ax.set_ylabel('Power (kW)', fontsize=12)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_title(f'Power Distribution - {strategy}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 性能指标对比
    ax = axes[1, 1]
    ax.axis('off')

    metrics_names = ['Total H2 (g)', 'Final SOC', 'Avg Reward']
    metrics_data = []

    for strategy in strategies:
        metrics = results_dict[strategy]
        metrics_data.append([
            f"{metrics['total_h2_consumption_g']:.2f}",
            f"{metrics['final_soc']:.4f}",
            f"{metrics['average_reward']:.2f}"
        ])

    # 创建表格
    table = ax.table(
        cellText=[['Strategy'] + metrics_names] +
                 [[strategies[i]] + metrics_data[i] for i in range(len(strategies))],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)

    ax.set_title('Performance Metrics Comparison', fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"对比图已保存到: {save_path}")
    plt.close()


def plot_soc_trajectory(metrics, save_path="soc_trajectory.png"):
    """
    绘制SOC轨迹图 (独立图表)

    参数:
        metrics: 环境返回的指标字典
        save_path: 保存路径
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制SOC曲线
    ax.plot(metrics['soc_history'], linewidth=2.5, color='#2E86AB', label='SOC')

    # 目标线
    ax.axhline(y=0.6, color='#E94F37', linestyle='--', linewidth=2, label='Target SOC (0.6)')

    # 填充区域
    soc_array = np.array(metrics['soc_history'])
    ax.fill_between(range(len(soc_array)), 0.6, soc_array,
                    where=soc_array >= 0.6, color='#27AE60', alpha=0.2, label='SOC ≥ Target')
    ax.fill_between(range(len(soc_array)), 0.6, soc_array,
                    where=soc_array < 0.6, color='#E74C3C', alpha=0.2, label='SOC < Target')

    ax.set_xlabel('Time Step (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('SOC', fontsize=13, fontweight='bold')
    ax.set_title('Battery State of Charge Trajectory', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"SOC轨迹图已保存到: {save_path}")


def plot_power_allocation(metrics, save_path="power_allocation.png"):
    """
    绘制功率分配图 (独立图表 - 折线图)

    参数:
        metrics: 环境返回的指标字典
        save_path: 保存路径
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 6))

    # 绘制功率曲线 (折线图)
    ax.plot(metrics['power_ice_history'] / 1000, label='ICE Power',
            linewidth=1.8, color='#E74C3C', alpha=0.85)
    ax.plot(metrics['power_fc_history'] / 1000, label='FC Power',
            linewidth=1.8, color='#3498DB', alpha=0.85)
    ax.plot(metrics['power_batt_history'] / 1000, label='Battery Power',
            linewidth=1.8, color='#27AE60', alpha=0.85)

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)

    ax.set_xlabel('Time Step (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Power (kW)', fontsize=13, fontweight='bold')
    ax.set_title('Power Allocation Over Time', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=3)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"功率分配图已保存到: {save_path}")


def plot_reward_curve(metrics, save_path="reward_curve.png"):
    """
    绘制奖励曲线图 (独立图表)

    参数:
        metrics: 环境返回的指标字典
        save_path: 保存路径
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 6))

    rewards = np.array(metrics['reward_history'])
    window = 50

    if len(rewards) >= window:
        # 滑动平均
        rewards_ma = np.convolve(rewards, np.ones(window) / window, mode='valid')
        x_ma = np.arange(window - 1, len(rewards))

        # 原始数据 (半透明)
        ax.plot(rewards, linewidth=0.5, color='#95A5A6', alpha=0.4, label='Raw Reward')
        # 滑动平均 (主曲线)
        ax.plot(x_ma, rewards_ma, linewidth=2.5, color='#8E44AD', label=f'Moving Average (window={window})')
    else:
        ax.plot(rewards, linewidth=1.5, color='#8E44AD', label='Reward')

    # 添加零线
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_xlabel('Time Step (s)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Reward', fontsize=13, fontweight='bold')
    ax.set_title('Reward Trajectory', fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"奖励曲线图已保存到: {save_path}")


def plot_performance_metrics(metrics, save_path="performance_metrics.png"):
    """
    绘制性能指标统计图 (独立图表)

    参数:
        metrics: 环境返回的指标字典
        save_path: 保存路径
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. 累计氢耗
    ax1 = axes[0, 0]
    h2_cumsum = np.linspace(0, metrics['total_h2_consumption_g'], len(metrics['soc_history']))
    ax1.plot(h2_cumsum, linewidth=2.5, color='#E67E22')
    ax1.fill_between(range(len(h2_cumsum)), 0, h2_cumsum, alpha=0.3)
    ax1.set_xlabel('Time Step (s)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative H2 Consumption (g)', fontsize=12, fontweight='bold')
    ax1.set_title('Hydrogen Consumption Over Time', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # 2. SOC分布直方图
    ax2 = axes[0, 1]
    ax2.hist(metrics['soc_history'], bins=50, color='#3498DB', alpha=0.7, edgecolor='black')
    ax2.axvline(x=0.6, color='#E74C3C', linestyle='--', linewidth=2, label='Target SOC')
    ax2.set_xlabel('SOC', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax2.set_title('SOC Distribution', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. 功率统计箱线图
    ax3 = axes[1, 0]
    power_data = [
        metrics['power_ice_history'] / 1000,
        metrics['power_fc_history'] / 1000,
        metrics['power_batt_history'] / 1000
    ]
    bp = ax3.boxplot(power_data, labels=['ICE', 'FC', 'Battery'],
                     patch_artist=True, widths=0.6)
    colors = ['#E74C3C', '#3498DB', '#27AE60']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_ylabel('Power (kW)', fontsize=12, fontweight='bold')
    ax3.set_title('Power Distribution Statistics', fontsize=13, fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')

    # 4. 关键指标文本
    ax4 = axes[1, 1]
    ax4.axis('off')

    ice_mean = np.mean(metrics['power_ice_history']) / 1000
    ice_max = np.max(metrics['power_ice_history']) / 1000
    fc_mean = np.mean(metrics['power_fc_history']) / 1000
    fc_max = np.max(metrics['power_fc_history']) / 1000
    batt_mean = np.mean(metrics['power_batt_history']) / 1000
    batt_max = np.max(metrics['power_batt_history']) / 1000

    stats_text = f"""
    ╔════════════════════════════════════════╗
    ║     PERFORMANCE METRICS SUMMARY       ║
    ╠════════════════════════════════════════╣

    Total H2 Consumption: {metrics['total_h2_consumption_g']:.2f} g
    Final SOC: {metrics['final_soc']:.4f}
    Average Reward: {metrics['average_reward']:.2f}

    ────────────────────────────────────────

    ICE Power:
      • Mean: {ice_mean:.2f} kW
      • Max: {ice_max:.2f} kW

    FC Power:
      • Mean: {fc_mean:.2f} kW
      • Max: {fc_max:.2f} kW

    Battery Power:
      • Mean: {batt_mean:.2f} kW
      • Max: {batt_max:.2f} kW

    ╚════════════════════════════════════════╝
    """

    ax4.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center', color='#2C3E50')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"性能指标统计图已保存到: {save_path}")


def plot_detailed_analysis(metrics, save_path="detailed_analysis.png"):
    """
    绘制所有详细分析图表 (4张独立图表)

    参数:
        metrics: 环境返回的指标字典
        save_path: 保存路径 (会自动添加后缀)
    """
    base_path = save_path.replace('.png', '')

    # 生成4张独立图表
    plot_soc_trajectory(metrics, f"{base_path}_soc.png")
    plot_power_allocation(metrics, f"{base_path}_power.png")
    plot_reward_curve(metrics, f"{base_path}_reward.png")
    plot_performance_metrics(metrics, f"{base_path}_metrics.png")

    print(f"\n 所有分析图表已生成完成!")
    print(f"   1. SOC轨迹: {base_path}_soc.png")
    print(f"   2. 功率分配: {base_path}_power.png")
    print(f"   3. 奖励曲线: {base_path}_reward.png")
    print(f"   4. 性能指标: {base_path}_metrics.png")


def evaluate_model(model, env, num_episodes=5):
    """
    评估训练好的模型

    参数:
        model: 训练好的模型
        env: 环境
        num_episodes: 评估回合数

    返回:
        results: 结果列表,每个元素是一个回合的指标
    """
    results = []

    for episode in range(num_episodes):
        print(f"\n评估回合 {episode + 1}/{num_episodes}")

        obs, info = env.reset()
        done = False
        step_count = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        metrics = env.get_metrics()
        results.append(metrics)

        print(f"  氢耗: {metrics['total_h2_consumption_g']:.2f} g")
        print(f"  最终 SOC: {metrics['final_soc']:.4f}")
        print(f"  平均奖励: {metrics['average_reward']:.2f}")

    # 计算平均指标
    avg_h2 = np.mean([r['total_h2_consumption_g'] for r in results])
    avg_soc = np.mean([r['final_soc'] for r in results])
    avg_reward = np.mean([r['average_reward'] for r in results])

    print(f"\n平均结果 ({num_episodes} 回合):")
    print(f"  平均氢耗: {avg_h2:.2f} g")
    print(f"  平均最终 SOC: {avg_soc:.4f}")
    print(f"  平均奖励: {avg_reward:.2f}")

    return results


def create_baseline_policy(env):
    """
    创建基线策略用于对比

    参数:
        env: 环境

    返回:
        policy: 基线策略函数
    """
    def baseline_policy(obs):
        """
        简单的基线策略:
        - SOC 低时增加 ICE 和 FC 输出
        - SOC 高时减少输出
        - 优先使用 FC (效率更高)
        """
        soc = obs[2]
        p_demand_norm = obs[3]

        # 根据 SOC 调整
        if soc < 0.5:
            # SOC 低,增加输出
            ice_ratio = 0.6 * p_demand_norm
            fc_ratio = 0.8 * p_demand_norm
        elif soc > 0.7:
            # SOC 高,减少输出
            ice_ratio = 0.2 * p_demand_norm
            fc_ratio = 0.3 * p_demand_norm
        else:
            # SOC 正常
            ice_ratio = 0.4 * p_demand_norm
            fc_ratio = 0.5 * p_demand_norm

        return np.array([ice_ratio, fc_ratio])

    return baseline_policy


def run_baseline(env, policy):
    """
    运行基线策略

    参数:
        env: 环境
        policy: 策略函数

    返回:
        metrics: 性能指标
    """
    obs, info = env.reset()
    done = False

    while not done:
        action = policy(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    metrics = env.get_metrics()
    return metrics


def save_training_log(log_data, filepath="training_log.txt"):
    """
    保存训练日志

    参数:
        log_data: 日志数据字典
        filepath: 保存路径
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("训练日志\n")
        f.write("="*60 + "\n\n")

        for key, value in log_data.items():
            f.write(f"{key}: {value}\n")

    print(f"训练日志已保存到: {filepath}")
