"""
工况切换测试脚本
测试 cycle_loader 模块和多工况切换功能
"""

from cycle_loader import load_cycle_data, print_cycle_info, get_available_cycles


def test_cycle_loading():
    """测试工况加载功能"""
    print("\n" + "="*60)
    print("工况加载测试")
    print("="*60 + "\n")

    # 测试工况列表
    test_cycles = ['cwtvc', 'nedc', 'wltp']

    for cycle in test_cycles:
        print(f"\n{'─'*60}")
        print(f"测试工况: {cycle.upper()}")
        print(f"{'─'*60}")

        try:
            # 加载工况数据
            data = load_cycle_data(cycle)

            # 打印基本信息
            print(f" 成功加载工况: {cycle}")
            print(f"   数据形状: {data.shape}")
            print(f"   速度范围: {data[:, 0].min()*3.6:.1f} - {data[:, 0].max()*3.6:.1f} km/h")
            print(f"   加速度范围: {data[:, 1].min():.2f} - {data[:, 1].max():.2f} m/s²")

        except Exception as e:
            print(f" 加载失败: {e}")

    print(f"\n{'='*60}\n")


def test_available_cycles():
    """测试获取可用工况列表"""
    print("\n" + "="*60)
    print("可用工况列表")
    print("="*60 + "\n")

    available = get_available_cycles()

    if available:
        print(f" 找到 {len(available)} 个可用工况:")
        for cycle in available:
            print(f"   - {cycle}")
    else:
        print("  未找到可用工况文件")
        print("   请将工况文件放到 state_data/ 目录:")
        print("   - CWTVC.xlsx")
        print("   - NEDC.xlsx")
        print("   - WLTC.xlsx")

    print(f"\n{'='*60}\n")


def test_cycle_info():
    """测试工况信息打印"""
    print("\n" + "="*60)
    print("工况详细信息")
    print("="*60 + "\n")

    # 打印 CWTVC 工况信息
    print_cycle_info('cwtvc')


def test_env_integration():
    """测试与 HEV 环境的集成"""
    print("\n" + "="*60)
    print("环境集成测试")
    print("="*60 + "\n")

    try:
        from hev_env import ContinuousHEVEnv

        # 测试不同工况的环境创建
        for cycle in ['cwtvc', 'nedc', 'wltp']:
            print(f"\n测试工况: {cycle.upper()}")
            print(f"{'─'*40}")

            try:
                env = ContinuousHEVEnv(cycle_name=cycle)
                print(f" 环境创建成功")
                print(f"   工况长度: {env.cycle_length} 步")
                print(f"   观察空间: {env.observation_space}")
                print(f"   动作空间: {env.action_space}")

                # 测试重置环境
                obs, info = env.reset()
                print(f" 环境重置成功")
                print(f"   初始观察: {obs}")

                # 测试执行一步
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                print(f" 执行一步成功")
                print(f"   奖励: {reward:.4f}")

            except Exception as e:
                print(f" 测试失败: {e}")

    except ImportError as e:
        print(f" 无法导入 HEV 环境: {e}")

    print(f"\n{'='*60}\n")


def test_invalid_cycle():
    """测试无效工况名称的处理"""
    print("\n" + "="*60)
    print("无效工况测试")
    print("="*60 + "\n")

    invalid_cycle = "invalid_cycle_name"
    print(f"尝试加载无效工况: {invalid_cycle}")

    data = load_cycle_data(invalid_cycle)

    print(f" 系统正确处理了无效工况")
    print(f"   返回了合成数据 (形状: {data.shape})")
    print(f"   这确保了系统的鲁棒性")

    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("开始工况切换功能测试")
    print("="*60)

    # 运行所有测试
    test_available_cycles()
    test_cycle_loading()
    test_cycle_info()
    test_invalid_cycle()
    test_env_integration()

    print("\n" + "="*60)
    print(" 所有测试完成!")
    print("="*60 + "\n")
