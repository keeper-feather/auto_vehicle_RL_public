"""
测试数据管理功能
"""

import numpy as np
from data_manager import DataManager


def test_data_manager():
    """测试数据管理器"""
    print("="*60)
    print("测试数据管理器")
    print("="*60 + "\n")

    # 创建数据管理器
    dm = DataManager(base_dir="data/test", reward_threshold=-100)

    # 测试数据1: 符合条件
    print("测试1: 符合条件的数据 (奖励 > -100)")
    metrics1 = {
        'average_reward': -85.5,
        'total_h2_consumption_g': 1200.0,
        'final_soc': 0.65,
        'soc_history': np.random.rand(1800) * 0.3 + 0.6,
        'power_ice_history': np.random.rand(1800) * 30000,
        'power_fc_history': np.random.rand(1800) * 30000,
        'power_batt_history': np.random.rand(1800) * 40000 - 20000,
        'reward_history': np.random.randn(1800) * 10 - 85
    }

    agent_dir1, agent_id1 = dm.get_next_agent_dir()
    print(f"Agent目录: {agent_dir1}")
    print(f"Agent ID: {agent_id1}")
    print(f"符合条件: {dm.should_save(metrics1)}")

    if dm.should_save(metrics1):
        dm.save_episode_data(metrics1, agent_dir1, agent_id1)

    print()

    # 测试数据2: 不符合条件
    print("测试2: 不符合条件的数据 (奖励 < -100)")
    metrics2 = {
        'average_reward': -150.0,
        'total_h2_consumption_g': 2000.0,
        'final_soc': 0.4,
        'soc_history': np.random.rand(1800) * 0.3 + 0.3,
        'power_ice_history': np.random.rand(1800) * 30000,
        'power_fc_history': np.random.rand(1800) * 30000,
        'power_batt_history': np.random.rand(1800) * 40000 - 20000,
        'reward_history': np.random.randn(1800) * 10 - 150
    }

    print(f"符合条件: {dm.should_save(metrics2)}")
    print("数据未保存")

    print()

    # 测试数据3: 再次符合条件
    print("测试3: 再次符合条件的数据 (奖励 > -100)")
    metrics3 = {
        'average_reward': -75.0,
        'total_h2_consumption_g': 1100.0,
        'final_soc': 0.62,
        'soc_history': np.random.rand(1800) * 0.3 + 0.55,
        'power_ice_history': np.random.rand(1800) * 30000,
        'power_fc_history': np.random.rand(1800) * 30000,
        'power_batt_history': np.random.rand(1800) * 40000 - 20000,
        'reward_history': np.random.randn(1800) * 10 - 75
    }

    agent_dir3, agent_id3 = dm.get_next_agent_dir()
    print(f"Agent目录: {agent_dir3}")
    print(f"Agent ID: {agent_id3}")
    print(f"符合条件: {dm.should_save(metrics3)}")

    if dm.should_save(metrics3):
        dm.save_episode_data(metrics3, agent_dir3, agent_id3)

    print()

    # 打印会话摘要
    dm.print_session_summary()

    print("="*60)
    print("测试完成!")
    print("="*60)


if __name__ == "__main__":
    test_data_manager()
