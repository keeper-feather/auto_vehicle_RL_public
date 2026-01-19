"""
环境测试脚本
快速验证环境是否正常工作
"""

import numpy as np
from hev_env import ContinuousHEVEnv
from utils import plot_detailed_analysis


def test_environment():
    """测试环境功能"""
    print("="*60)
    print("环境测试")
    print("="*60 + "\n")

    # 创建环境
    print("1. 创建环境...")
    env = ContinuousHEVEnv()
    print(f"   观测空间: {env.observation_space}")
    print(f"   动作空间: {env.action_space}")
    print(f"   工况长度: {env.cycle_length} 步\n")

    # 测试重置
    print("2. 测试重置...")
    obs, info = env.reset()
    print(f"   初始观测: {obs}")
    print(f"   初始 SOC: {env.soc:.4f}\n")

    # 测试随机动作
    print("3. 测试随机动作 (100 步)...")
    total_reward = 0
    for i in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if i % 20 == 0:
            print(f"   Step {i}: Reward={reward:.2f}, SOC={info['soc']:.4f}, "
                  f"P_ice={info['p_ice']/1000:.1f}kW, P_fc={info['p_fc']/1000:.1f}kW")

        if terminated or truncated:
            break

    print(f"\n   100 步平均奖励: {total_reward/100:.2f}\n")

    # 测试完整回合
    print("4. 测试完整回合...")
    obs, info = env.reset()
    done = False
    step_count = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        if step_count % 300 == 0:
            print(f"   Step {step_count}: SOC={info['soc']:.4f}, "
                  f"H2={info['h2_consumption']:.2f}g")

    metrics = env.get_metrics()

    print(f"\n   回合完成:")
    print(f"   总步数: {step_count}")
    print(f"   总氢耗: {metrics['total_h2_consumption_g']:.2f} g")
    print(f"   最终 SOC: {metrics['final_soc']:.4f}")
    print(f"   平均奖励: {metrics['average_reward']:.2f}\n")

    # 生成分析图
    print("5. 生成分析图...")
    plot_detailed_analysis(metrics, "test_results.png")

    print("="*60)
    print("环境测试完成!")
    print("="*60)

    return env, metrics


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)

    # 运行测试
    env, metrics = test_environment()
