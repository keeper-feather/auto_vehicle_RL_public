"""
PPO 训练脚本
使用 Stable-Baselines3 训练混合动力汽车能量管理策略
"""

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.monitor import Monitor
import os
from datetime import datetime
from pathlib import Path
import torch

from hev_env import ContinuousHEVEnv
from data_manager import DataManager
from utils import plot_soc_trajectory, plot_power_allocation, plot_reward_curve, plot_performance_metrics

# ═══════════════════════════════════════════════════════════════
# 工况配置 - 修改这里切换工况
# ═══════════════════════════════════════════════════════════════
# 可选工况: 'cwtvc', 'nedc', 'wltp'
TARGET_CYCLE = 'cwtvc'  # 修改这个变量切换工况
# ═══════════════════════════════════════════════════════════════


def linear_schedule(initial_value: float):
    """
    线性学习率调度函数

    参数:
        initial_value: 初始学习率

    返回:
        schedule: 根据训练进度返回当前学习率的函数
    """
    def func(progress_remaining: float) -> float:
        """
        progress_remaining: 从 1.0 递减到 0.0
        """
        return initial_value * progress_remaining

    return func


class RewardCallback(BaseCallback):
    """自定义回调函数,记录训练过程中的奖励"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        # 每一步都会调用
        return True


class TrainingPlotCallback(BaseCallback):
    """训练过程中的可视化回调"""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.plot_freq = 1000  # 每1000步绘制一次

    def _on_step(self) -> bool:
        if self.n_calls % self.plot_freq == 0 and self.n_calls > 0:
            # 打印当前训练信息
            if len(self.model.ep_info_buffer) > 0:
                latest_ep = self.model.ep_info_buffer[-1]
                print(f"\n{'='*60}")
                print(f"Step: {self.num_timesteps}")
                print(f"Episode Reward: {latest_ep.get('r', 'N/A'):.2f}")
                print(f"Episode Length: {latest_ep.get('l', 'N/A'):.0f}")
                print(f"{'='*60}\n")
        return True


def validate_environment(env, num_steps=100):
    """验证环境是否正常工作"""
    print("\n" + "="*60)
    print("验证环境...")
    print("="*60)

    obs, info = env.reset()
    print(f"观测空间形状: {obs.shape}")
    print(f"初始观测: {obs}")

    total_reward = 0
    for i in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            print(f"Step {i}: Reward={reward:.2f}, SOC={info['soc']:.4f}, "
                  f"H2={info['h2_consumption']:.2f}g")

        if terminated or truncated:
            break

    print(f"\n验证完成! 平均奖励: {total_reward/num_steps:.2f}")
    print("="*60 + "\n")


def plot_training_results(metrics, save_path="validation_results.png"):
    """
    绘制训练结果 - 生成4张独立图表

    参数:
        metrics: 环境返回的指标字典
        save_path: 保存路径 (会自动添加后缀)
    """
    from utils import plot_soc_trajectory, plot_power_allocation, plot_reward_curve, plot_performance_metrics

    base_path = save_path.replace('.png', '')

    print("\n生成验证结果图表...")

    # 生成4张独立图表
    plot_soc_trajectory(metrics, f"{base_path}_soc.png")
    plot_power_allocation(metrics, f"{base_path}_power.png")
    plot_reward_curve(metrics, f"{base_path}_reward.png")
    plot_performance_metrics(metrics, f"{base_path}_metrics.png")

    print(f"\n验证结果图表已生成完成!")
    print(f"   1. SOC轨迹: {base_path}_soc.png")
    print(f"   2. 功率分配: {base_path}_power.png")
    print(f"   3. 奖励曲线: {base_path}_reward.png")
    print(f"   4. 性能指标: {base_path}_metrics.png")


def train():
    """主训练函数"""
    print("\n" + "="*60)
    print("混合动力汽车能量管理策略训练")
    print("算法: PPO (Proximal Policy Optimization)")
    print("="*60 + "\n")

    # 检查 GPU 可用性
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print()

    # === 1. 创建环境 ===
    print(f"创建训练环境 (工况: {TARGET_CYCLE.upper()})...")

    # 单环境用于验证
    env = ContinuousHEVEnv(cycle_name=TARGET_CYCLE)

    # 向量化环境用于训练 (并行环境)
    n_envs = 8  # 并行环境数量
    vec_env = make_vec_env(
        lambda: ContinuousHEVEnv(cycle_name=TARGET_CYCLE),
        n_envs=n_envs
    )

    # 使用 VecNormalize 归一化观测和奖励
    # 这将自动处理不同量级的观测值 (速度 vs 功率 vs SOC)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,        # 归一化观测
        norm_reward=True,     # 归一化奖励
        clip_obs=10.0,        # 裁剪观测值
        clip_reward=10.0,     # 裁剪奖励值
        gamma=0.99,           # 折扣因子，用于奖励归一化
    )

    print(" 环境归一化已启用 (VecNormalize)")

    # === 3. 验证环境 ===
    validate_environment(env, num_steps=50)

    # === 4. 创建回调函数 ===
    print("创建回调函数...")

    # 保存检查点
    checkpoint_dir = "./checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="ppo_hev"
    )

    # 训练可视化回调
    plot_callback = TrainingPlotCallback(verbose=1)

    # === 5. 创建 PPO 模型 ===
    print("创建 PPO 模型...")
    print("超参数:")
    print("  - learning_rate: 3e-4 (线性递减)")
    print("  - n_steps: 2048")
    print("  - batch_size: 64")
    print("  - n_epochs: 10")
    print("  - gamma: 0.99")
    print("  - ent_coef: 0.01 (增加探索)")
    print()

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=linear_schedule(3e-4),  # 使用线性学习率调度
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,         # 增加熵系数，促进探索
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,         # 不使用状态依赖探索
        sde_sample_freq=-1,
        target_kl=0.01,
        tensorboard_log="./logs/",
        verbose=1,
        device=device
    )

    # === 6. 训练模型 ===
    print("\n" + "="*60)
    print("开始训练...")
    print("="*60 + "\n")

    total_timesteps = 500000  # 总训练步数
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, plot_callback]
    )

    print("\n" + "="*60)
    print("训练完成!")
    print("="*60 + "\n")

    # === 7. 保存模型 ===
    model_dir = "./models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = f"{model_dir}/ppo_hev_final"
    model.save(model_path)
    print(f"模型已保存到: {model_path}")

    # 保存 VecNormalize 统计信息
    vec_env.save(f"{model_path}_vecnormalize.pkl")
    print(f"归一化统计已保存到: {model_path}_vecnormalize.pkl")

    # === 8. 验证训练好的模型 ===
    print("\n" + "="*60)
    print("验证训练好的模型...")
    print("="*60 + "\n")

    # 初始化数据管理器
    data_manager = DataManager(base_dir="data", reward_threshold=-500)

    # 使用训练好的模型进行验证
    obs, info = env.reset()
    done = False
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        if step_count % 200 == 0:
            print(f"Validation Step {step_count}: SOC={info['soc']:.4f}, "
                  f"H2={info['h2_consumption']:.2f}g")

    # 获取指标
    metrics = env.get_metrics()

    print("\n验证结果:")
    print(f"  总氢耗: {metrics['total_h2_consumption_g']:.2f} g")
    print(f"  最终 SOC: {metrics['final_soc']:.4f}")
    print(f"  平均奖励: {metrics['average_reward']:.2f}")
    print(f"  验证步数: {step_count}")
    print()

    # === 9. 检查是否符合保存条件 ===
    if data_manager.should_save(metrics):
        print(" 验证结果符合保存标准 (平均奖励 > -100)!")

        # 获取agent目录
        agent_dir, agent_id = data_manager.get_next_agent_dir()

        # 生成图表并保存到agent目录
        base_path = agent_dir / "agent_results"
        plot_files = [
            f"{base_path}_soc.png",
            f"{base_path}_power.png",
            f"{base_path}_reward.png",
            f"{base_path}_metrics.png"
        ]

        # 绘制图表
        plot_training_results(metrics, str(base_path) + ".png")

        # 保存数据和图表
        episode_info = {
            'total_timesteps': total_timesteps,
            'validation_steps': step_count,
            'model_path': model_path
        }
        data_manager.save_episode_data(metrics, agent_dir, agent_id, episode_info)
        data_manager.copy_plots_to_agent_dir(agent_dir, plot_files)

        print(f"\nAgent {agent_id} 数据已保存到: {agent_dir}")
    else:
        print(f"验证结果未达到保存标准 (平均奖励 {metrics['average_reward']:.2f} <= -500)")
        print("   数据未保存")

        # 仍然保存到常规plots目录
        os.makedirs("plots", exist_ok=True)
        plot_training_results(metrics, "plots/validation_results.png")

    # === 10. 打印会话摘要 ===
    data_manager.print_session_summary()

    print("="*60)
    print("训练和验证流程全部完成!")
    print("="*60)

    return model, env


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    torch.manual_seed(42)

    # 创建必要的目录
    os.makedirs("plots", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # 开始训练
    model, env = train()

    print("\n提示: 使用 'tensorboard --logdir ./logs' 查看训练曲线")
