"""
实时监控训练进度
"""

import time
import subprocess
import re

def monitor_training():
    """监控训练进度"""
    print("="*60)
    print("实时监控训练进度")
    print("="*60 + "\n")

    output_file = "/tmp/claude/-home-keeper-auto-vehicle-RL/tasks/ba264ca.output"

    try:
        while True:
            # 读取最新输出
            result = subprocess.run(
                ["tail", "-100", output_file],
                capture_output=True,
                text=True
            )

            lines = result.stdout.split('\n')

            # 提取关键信息
            for line in lines:
                if 'Step:' in line and 'Episode Reward:' in lines[lines.index(line) + 1]:
                    step_line = line
                    reward_line = lines[lines.index(line) + 1]

                    step = re.search(r'Step:\s+(\d+)', step_line)
                    reward = re.search(r'Episode Reward:\s+([\d.-]+)', reward_line)

                    if step and reward:
                        print(f"\r当前步数: {step.group(1)} | "
                              f"最近奖励: {float(reward.group(1)):.2f} | "
                              f"进度: {int(step.group(1)) / 500000 * 100:.1f}%", end='')
                        break

            # 等待5秒
            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\n监控已停止")

if __name__ == "__main__":
    monitor_training()
