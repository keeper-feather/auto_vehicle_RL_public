"""
工况数据加载模块
支持多种标准工况（CWTVC, NEDC, WLTC）的灵活加载
"""

import numpy as np
from pathlib import Path
import pandas as pd


def load_cycle_data(cycle_name='cwtvc', data_dir='state_data'):
    """
    加载工况数据，支持多种标准工况

    参数:
        cycle_name: 工况名称 ('cwtvc', 'nedc', 'wltc')
        data_dir: 工况数据目录

    返回:
        cycle_data: numpy数组, shape=(N, 2)
                   列0: 速度 (m/s)
                   列1: 加速度 (m/s²)

    支持的工况:
        - cwtvc: 中国轻型汽车测试工况
        - nedc: 新欧洲驾驶工况
        - wltp/wltc: 全球轻型汽车测试工况
    """
    # 标准化工况名称（不区分大小写）
    cycle_name = cycle_name.lower()

    # 工况文件映射
    cycle_files = {
        'cwtvc': 'CWTVC.xlsx',
        'nedc': 'NEDC.xlsx',
        'wltp': 'WLTC.xlsx',
        'wltc': 'WLTC.xlsx'  # wltp 和 wltc 都指向同一个文件
    }

    # 检查工况名称是否支持
    if cycle_name not in cycle_files:
        print(f"  不支持的工况名称: {cycle_name}")
        print(f"   支持的工况: {list(cycle_files.keys())}")
        print(f"   回退到合成工况数据")
        return generate_synthetic_cwtvc()

    # 构造文件路径
    filename = cycle_files[cycle_name]
    file_path = Path(data_dir) / filename

    # 尝试加载文件
    try:
        if file_path.exists():
            print(f" 加载工况文件: {file_path}")

            # 检查文件类型
            if file_path.suffix.lower() in ['.xlsx', '.xls']:
                # Excel 文件，使用 pandas 读取
                try:
                    df = pd.read_excel(file_path)
                    print(f"   使用 pandas 读取 Excel 文件")
                    print(f"   数据形状: {df.shape}")
                    print(f"   列名: {df.columns.tolist()}")

                    # 处理数据：假设第一列是时间(s)，第二列是速度(km/h)
                    if len(df.columns) >= 2:
                        # 第一列通常是时间，第二列是速度(km/h)
                        time_col = df.iloc[:, 0].values
                        speed_kmh = df.iloc[:, 1].values

                        # 转换速度从 km/h 到 m/s
                        speed = speed_kmh / 3.6

                        # 计算加速度 (m/s²)
                        acc = np.gradient(speed, 1.0)

                        cycle_data = np.column_stack([speed, acc])
                        print(f"   速度范围: {speed_kmh.min():.1f} - {speed_kmh.max():.1f} km/h")
                    else:
                        # 只有一列数据
                        speed_kmh = df.iloc[:, 0].values
                        speed = speed_kmh / 3.6
                        acc = np.gradient(speed, 1.0)
                        cycle_data = np.column_stack([speed, acc])

                except Exception as e:
                    print(f"     Excel 读取失败: {e}")
                    raise

            else:
                # CSV 文件，尝试多种编码方式
                encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'latin1']
                cycle_data = None

                for encoding in encodings:
                    try:
                        cycle_data = np.loadtxt(file_path, delimiter=',', encoding=encoding)
                        print(f"   使用编码: {encoding}")
                        break
                    except UnicodeDecodeError:
                        continue
                    except Exception:
                        continue

                if cycle_data is None:
                    raise ValueError("无法解码文件，尝试了所有编码方式")

            # 验证数据格式
            if cycle_data.ndim == 1:
                # 如果是一维数组，假设只有速度数据
                speed = cycle_data
                acc = np.gradient(speed, 1.0)
                cycle_data = np.column_stack([speed, acc])
            elif cycle_data.shape[1] < 2:
                # 如果列数不足，补充加速度
                speed = cycle_data[:, 0]
                acc = np.gradient(speed, 1.0)
                cycle_data = np.column_stack([speed, acc])

            print(f"   工况长度: {len(cycle_data)} 步")
            print(f"   速度范围: {np.min(cycle_data[:, 0])*3.6:.1f} - {np.max(cycle_data[:, 0])*3.6:.1f} km/h")
            print(f"   加速度范围: {np.min(cycle_data[:, 1]):.2f} - {np.max(cycle_data[:, 1]):.2f} m/s²")

            return cycle_data
        else:
            raise FileNotFoundError(f"文件不存在: {file_path}")

    except Exception as e:
        print(f"  加载工况文件失败: {e}")
        print(f"   文件路径: {file_path}")
        print(f"   回退到合成工况数据")
        return generate_synthetic_cwtvc()


def generate_synthetic_cwtvc():
    """
    生成合成的 CWTVC 工况数据（正弦波模拟）
    作为文件加载失败时的回退方案

    返回:
        cycle_data: numpy数组, shape=(N, 2)
    """
    print(" 生成合成 CWTVC 工况数据（正弦波模拟）...")

    dt = 1.0
    t_max = 1800  # 30分钟
    t = np.arange(0, t_max, dt)

    # 基础速度 40 km/h + 变化
    base_speed = 40 / 3.6  # m/s

    # 叠加多个频率的正弦波
    speed = (base_speed +
             15 / 3.6 * np.sin(2 * np.pi * t / 300) +  # 5分钟周期
             8 / 3.6 * np.sin(2 * np.pi * t / 120) +   # 2分钟周期
             5 / 3.6 * np.sin(2 * np.pi * t / 60))      # 1分钟周期

    # 添加一些随机噪声
    np.random.seed(42)
    speed += np.random.normal(0, 0.5, len(t))
    speed = np.maximum(speed, 0)  # 确保速度非负

    # 计算加速度
    acc = np.gradient(speed, dt)
    acc = np.clip(acc, -2.0, 2.0)  # 限制加速度范围

    # 组合成工况数据
    cycle_data = np.column_stack([speed, acc])

    print(f"   合成工况长度: {len(cycle_data)} 步")
    print(f"   速度范围: {np.min(speed)*3.6:.1f} - {np.max(speed)*3.6:.1f} km/h")
    print(f"   加速度范围: {np.min(acc):.2f} - {np.max(acc):.2f} m/s²")

    return cycle_data


def get_available_cycles(data_dir='state_data'):
    """
    获取可用的工况列表

    参数:
        data_dir: 工况数据目录

    返回:
        list: 可用的工况名称列表
    """
    available = []
    data_path = Path(data_dir)

    if data_path.exists():
        # 检查哪些工况文件存在
        cycle_files = {
            'cwtvc': 'CWTVC.xlsx',
            'nedc': 'NEDC.xlsx',
            'wltp': 'WLTC.xlsx'
        }

        for name, filename in cycle_files.items():
            if (data_path / filename).exists():
                available.append(name)

    return available


def print_cycle_info(cycle_name, data_dir='state_data'):
    """
    打印工况信息

    参数:
        cycle_name: 工况名称
        data_dir: 工况数据目录
    """
    cycle_data = load_cycle_data(cycle_name, data_dir)

    print(f"\n{'='*60}")
    print(f"工况信息: {cycle_name.upper()}")
    print(f"{'='*60}")
    print(f"数据点数: {len(cycle_data)}")
    print(f"时长: {len(cycle_data)} 秒 ({len(cycle_data)/60:.1f} 分钟)")
    print(f"速度范围: {np.min(cycle_data[:, 0])*3.6:.1f} - {np.max(cycle_data[:, 0])*3.6:.1f} km/h")
    print(f"平均速度: {np.mean(cycle_data[:, 0])*3.6:.1f} km/h")
    print(f"加速度范围: {np.min(cycle_data[:, 1]):.2f} - {np.max(cycle_data[:, 1]):.2f} m/s²")
    print(f"{'='*60}\n")
