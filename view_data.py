"""
查看已保存的训练数据
"""

import json
import numpy as np
from pathlib import Path
import argparse


def list_sessions(data_dir="data"):
    """列出所有训练会话"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"数据目录不存在: {data_dir}")
        return

    sessions = sorted(data_path.iterdir(), key=lambda x: x.name)

    if not sessions:
        print(f"没有找到训练会话")
        return

    print(f"\n找到 {len(sessions)} 个训练会话:\n")

    for session in sessions:
        print(f" {session.name}")

        # 统计agent数量
        agents = list(session.glob("agent_*"))
        print(f"   保存的Agent: {len(agents)}")

        # 列出每个agent的摘要
        for agent_dir in sorted(agents):
            summary_file = agent_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                print(f"   - {agent_dir.name}: "
                      f"奖励={summary['average_reward']:.2f}, "
                      f"氢耗={summary['total_h2_consumption_g']:.2f}g, "
                      f"SOC={summary['final_soc']:.4f}")
        print()


def view_agent(agent_path):
    """查看特定agent的详细信息"""
    agent_path = Path(agent_path)

    if not agent_path.exists():
        print(f"Agent目录不存在: {agent_path}")
        return

    print(f"\n{'='*60}")
    print(f"Agent详细信息: {agent_path.name}")
    print(f"{'='*60}\n")

    # 读取摘要
    summary_file = agent_path / "summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            summary = json.load(f)

        print(" 性能摘要:")
        print(f"   Agent ID: {summary['agent_id']}")
        print(f"   时间戳: {summary['timestamp']}")
        print(f"   平均奖励: {summary['average_reward']:.2f}")
        print(f"   总氢耗: {summary['total_h2_consumption_g']:.2f} g")
        print(f"   最终SOC: {summary['final_soc']:.4f}")
        print(f"   Episode长度: {summary['episode_length']} 步")

        if 'total_timesteps' in summary:
            print(f"   训练步数: {summary['total_timesteps']}")

    # 列出文件
    print(f"\n 包含文件:")
    for file in sorted(agent_path.iterdir()):
        size = file.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / 1024 / 1024:.2f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.2f} KB"
        else:
            size_str = f"{size} B"

        print(f"   - {file.name} ({size_str})")


def load_agent_data(agent_path):
    """加载agent的详细数据"""
    agent_path = Path(agent_path)
    data_file = agent_path / "episode_data.npz"

    if not data_file.exists():
        print(f"数据文件不存在: {data_file}")
        return None

    data = np.load(data_file)
    return {
        'soc_history': data['soc_history'],
        'power_ice_history': data['power_ice_history'],
        'power_fc_history': data['power_fc_history'],
        'power_batt_history': data['power_batt_history'],
        'reward_history': data['reward_history']
    }


def compare_agents(data_dir="data", top_n=5):
    """比较表现最好的agents"""
    data_path = Path(data_dir)

    if not data_path.exists():
        print(f"数据目录不存在: {data_dir}")
        return

    # 收集所有agents
    all_agents = []

    for session in data_path.iterdir():
        for agent_dir in session.glob("agent_*"):
            summary_file = agent_dir / "summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary = json.load(f)
                all_agents.append({
                    'path': agent_dir,
                    'session': session.name,
                    'agent_id': summary['agent_id'],
                    'reward': summary['average_reward'],
                    'h2': summary['total_h2_consumption_g'],
                    'soc': summary['final_soc']
                })

    if not all_agents:
        print("没有找到任何agent数据")
        return

    # 按奖励排序
    all_agents.sort(key=lambda x: x['reward'], reverse=True)

    # 显示top N
    print(f"\n 表现最好的 {min(top_n, len(all_agents))} 个Agents:\n")
    print(f"{'排名':<4} {'Agent':<20} {'平均奖励':<12} {'氢耗(g)':<10} {'最终SOC':<10}")
    print("-" * 60)

    for i, agent in enumerate(all_agents[:top_n], 1):
        print(f"{i:<4} {agent['session']}/{agent['agent_id']:<13} "
              f"{agent['reward']:<12.2f} {agent['h2']:<10.2f} {agent['soc']:<10.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看训练数据")
    parser.add_argument('--data-dir', default='data', help='数据目录')
    parser.add_argument('--list', action='store_true', help='列出所有会话')
    parser.add_argument('--agent', type=str, help='查看特定agent')
    parser.add_argument('--compare', type=int, nargs='?', const=5, help='对比top N agents')

    args = parser.parse_args()

    if args.list:
        list_sessions(args.data_dir)
    elif args.agent:
        view_agent(args.agent)
    elif args.compare:
        compare_agents(args.data_dir, args.compare)
    else:
        # 默认列出所有会话
        list_sessions(args.data_dir)
        compare_agents(args.data_dir, 5)
