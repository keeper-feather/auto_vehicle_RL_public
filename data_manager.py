"""
数据管理模块
负责保存符合标准的训练数据
"""

import os
import json
import shutil
from datetime import datetime
from pathlib import Path


class DataManager:
    """训练数据管理器"""

    def __init__(self, base_dir="data", reward_threshold=-100):
        """
        初始化数据管理器

        参数:
            base_dir: 基础数据目录
            reward_threshold: 奖励阈值，高于此值才保存数据
        """
        self.base_dir = Path(base_dir)
        self.reward_threshold = reward_threshold

        # 创建基础目录
        self.base_dir.mkdir(parents=True, exist_ok=True)

        # 创建当前训练会话目录（以时间命名）
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / timestamp
        self.session_dir.mkdir(exist_ok=True)

        # 计数器：用于agent编号
        self.agent_counter = 0

        print(f"数据管理器初始化完成:")
        print(f"  - 基础目录: {self.base_dir}")
        print(f"  - 会话目录: {self.session_dir}")
        print(f"  - 奖励阈值: {self.reward_threshold}")

    def get_next_agent_dir(self):
        """
        获取下一个agent目录编号

        返回:
            agent_dir: agent目录路径
            agent_id: agent编号
        """
        self.agent_counter += 1
        agent_id = self.agent_counter
        agent_dir = self.session_dir / f"agent_{agent_id}"
        agent_dir.mkdir(exist_ok=True)

        return agent_dir, agent_id

    def should_save(self, metrics):
        """
        判断是否应该保存数据

        参数:
            metrics: 性能指标字典

        返回:
            bool: 是否符合保存条件
        """
        avg_reward = metrics.get('average_reward', float('-inf'))

        # 判断平均奖励是否高于阈值
        if avg_reward > self.reward_threshold:
            return True

        return False

    def save_episode_data(self, metrics, agent_dir, agent_id, episode_info=None):
        """
        保存episode数据

        参数:
            metrics: 性能指标字典
            agent_dir: agent目录路径
            agent_id: agent编号
            episode_info: 额外的episode信息（如训练步数、模型参数等）
        """
        # 保存指标为JSON
        summary = {
            'agent_id': agent_id,
            'timestamp': datetime.now().isoformat(),
            'average_reward': float(metrics.get('average_reward', 0)),
            'total_h2_consumption_g': float(metrics['total_h2_consumption_g']),
            'final_soc': float(metrics['final_soc']),
            'episode_length': len(metrics['soc_history'])
        }

        # 添加额外信息
        if episode_info:
            summary.update(episode_info)

        # 保存JSON摘要
        summary_path = agent_dir / 'summary.json'
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 保存详细数据为NPZ（压缩格式）
        data_path = agent_dir / 'episode_data.npz'
        import numpy as np
        np.savez_compressed(
            data_path,
            soc_history=metrics['soc_history'],
            power_ice_history=metrics['power_ice_history'],
            power_fc_history=metrics['power_fc_history'],
            power_batt_history=metrics['power_batt_history'],
            reward_history=metrics['reward_history']
        )

        print(f" 数据已保存到: {agent_dir}")
        print(f"   - 平均奖励: {summary['average_reward']:.2f}")
        print(f"   - 氢耗: {summary['total_h2_consumption_g']:.2f} g")
        print(f"   - 最终SOC: {summary['final_soc']:.4f}")

        return agent_dir

    def copy_plots_to_agent_dir(self, agent_dir, plot_files):
        """
        将图表复制到agent目录

        参数:
            agent_dir: agent目录路径
            plot_files: 图表文件路径列表
        """
        for plot_file in plot_files:
            if os.path.exists(plot_file):
                dest = agent_dir / os.path.basename(plot_file)
                # 避免复制自身
                if str(plot_file) != str(dest):
                    shutil.copy2(plot_file, dest)
                    print(f"   - 图表已复制: {dest.name}")
                else:
                    print(f"   - 图表已存在: {dest.name}")

    def get_session_stats(self):
        """
        获取当前会话的统计信息

        返回:
            stats: 统计信息字典
        """
        # 统计保存的agent数量
        agent_dirs = list(self.session_dir.glob("agent_*"))
        saved_count = len(agent_dirs)

        stats = {
            'session_dir': str(self.session_dir),
            'total_agents': self.agent_counter,
            'saved_agents': saved_count,
            'threshold': self.reward_threshold
        }

        return stats

    def print_session_summary(self):
        """打印会话摘要"""
        stats = self.get_session_stats()

        print("\n" + "="*60)
        print("数据保存摘要")
        print("="*60)
        print(f"会话目录: {stats['session_dir']}")
        print(f"总评估次数: {stats['total_agents']}")
        print(f"符合条件保存: {stats['saved_agents']}")
        print(f"奖励阈值: >{stats['threshold']}")
        print("="*60 + "\n")
