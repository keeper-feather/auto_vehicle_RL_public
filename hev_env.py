"""
混合动力汽车 (HEV) 能量管理环境
使用 Gymnasium 和后向准静态仿真
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.interpolate import RegularGridInterpolator, interp1d
from cycle_loader import load_cycle_data


class ContinuousHEVEnv(gym.Env):
    """
    连续动作空间的混合动力汽车环境

    动力系统架构: 燃料电池(FC) + 氢内燃机(ICE) + 动力电池
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, cycle_name='cwtvc', cycle_data=None, render_mode=None):
        """
        初始化环境

        参数:
            cycle_name: 工况名称 ('cwtvc', 'nedc', 'wltp')，默认为 'cwtvc'
            cycle_data: 直接传入工况数据（可选，优先级高于cycle_name）
            render_mode: 渲染模式
        """
        super().__init__()

        # === 仿真参数 ===
        self.dt = 1.0  # 时间步长 (秒)
        self.max_speed = 90.0 / 3.6  # 最大速度 (m/s), 90 km/h
        self.max_acc = 2.0  # 最大加速度 (m/s²)
        self.max_power = 120000  # 最大功率需求 (W)

        # === 车辆参数 (来自 vehicle.md) ===
        self.m_eq = 7020  # 等效质量 (kg)
        self.r_eff = 0.3764  # 轮胎有效半径 (m)
        self.i_0 = 16.04  # 主减速比
        self.wheel_circumference = 2.365  # 轮胎周长 (m)

        # 阻力系数
        self.F_const = 447.37  # 常值阻力 (N)
        self.F_lin_coef = 1.1805  # 一次速度阻力系数
        self.F_quad_coef = 0.1968  # 二次速度阻力系数
        self.F_brake_max = 500  # 最大制动力 (N)

        # === 电池参数 (来自 power battery.md) ===
        self.bat_capacity = 15  # 电池容量 (Ah) - 适中的容量
        self.bat_capacity_coulomb = self.bat_capacity * 3600  # 转换为库仑
        self.n_cell = 102  # 串联电芯数
        self.soc_init = 0.6  # 初始 SOC (60%) - 适中的初始值
        self.soc_target = 0.6  # 目标 SOC
        self.soc_min = 0.3  # SOC 下限 (30%) - 防止过度放电
        self.soc_max = 0.9  # SOC 上限 (90%) - 允许一定范围

        # DC/DC 效率
        self.fc_dcdc_eff = 0.85  # 燃料电池 DC/DC 效率
        self.ice_dc_dc_eff = 0.95  # ICE 发电机 DC/DC 效率

        # 电池查表数据 (25°C)
        self.soc_points = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        self.voc_cells = np.array([3.46, 3.58, 3.64, 3.68, 3.73, 3.78, 3.87, 3.98, 4.08])  # V/cell
        self.r_dis = np.array([1.89e-3, 1.85e-3, 1.83e-3, 1.81e-3, 1.81e-3, 1.82e-3, 1.83e-3, 1.86e-3, 1.86e-3])  # Ohm
        self.r_chg = np.array([1.56e-3, 1.50e-3, 1.50e-3, 1.46e-3, 1.47e-3, 1.47e-3, 1.48e-3, 1.53e-3, 1.54e-3])  # Ohm

        # 创建插值函数
        self.voc_interp = interp1d(self.soc_points, self.voc_cells, kind='linear', bounds_error=False, fill_value="extrapolate")
        self.r_dis_interp = interp1d(self.soc_points, self.r_dis, kind='linear', bounds_error=False, fill_value="extrapolate")
        self.r_chg_interp = interp1d(self.soc_points, self.r_chg, kind='linear', bounds_error=False, fill_value="extrapolate")

        # === 电机参数 (来自 motor.md) ===
        # 最大转矩表
        self.mot_volt_points = np.array([278, 336, 402])  # V
        self.mot_speed_points = np.array([700, 1400, 2100, 2800, 3100, 3400, 3800, 4100, 4800,
                                          5500, 6200, 6900, 7600, 8300, 9000, 9700, 10400,
                                          11000, 11500, 12000])  # rpm

        # 简化的最大转矩表 (在不同电压下相同)
        self.mot_max_torque = np.array([
            [365.4, 364.0, 363.1, 362.5, 349.6, 323.9, 292.0, 268.0, 229.6,
             201.9, 178.2, 160.8, 146.6, 135.1, 124.2, 114.6, 108.3, 102.5, 100.3, 93.8],
            [365.4, 364.0, 363.1, 362.5, 349.6, 323.9, 292.0, 268.0, 229.6,
             201.9, 178.2, 160.8, 146.6, 135.1, 124.2, 114.6, 108.3, 102.5, 100.3, 93.8],
            [365.4, 364.0, 363.1, 362.5, 349.6, 323.9, 292.0, 268.0, 229.6,
             201.9, 178.2, 160.8, 146.6, 135.1, 124.2, 114.6, 108.3, 102.5, 100.3, 93.8]
        ])

        # 创建最大转矩插值器
        self.max_torque_interp = RegularGridInterpolator(
            (self.mot_volt_points, self.mot_speed_points),
            self.mot_max_torque,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # 电机效率表 (简化版, 使用较少的速度点以匹配效率矩阵)
        self.mot_torque_points = np.array([14, 70, 140, 211, 281, 365])
        # 使用简化的速度点 (6个点)
        self.mot_speed_points_eff = np.array([700, 2800, 5500, 8300, 10400, 12000])
        # 创建简化的效率矩阵 (6x6)
        self.mot_efficiency = np.array([
            [0.70, 0.75, 0.78, 0.80, 0.79, 0.75],
            [0.75, 0.85, 0.88, 0.90, 0.89, 0.85],
            [0.78, 0.88, 0.92, 0.94, 0.92, 0.88],
            [0.80, 0.90, 0.94, 0.95, 0.94, 0.90],
            [0.79, 0.89, 0.92, 0.94, 0.92, 0.88],
            [0.75, 0.85, 0.88, 0.90, 0.89, 0.85]
        ])

        # 创建效率插值器
        self.eff_interp = RegularGridInterpolator(
            (self.mot_torque_points, self.mot_speed_points_eff),
            self.mot_efficiency,
            method='linear',
            bounds_error=False,
            fill_value=None
        )

        # 电机驱动/回馈系数
        self.mot_drive_eff = 0.97
        self.mot_regenerative_eff = 0.89

        # === 氢耗参数 (来自 agent skill.md) ===
        # ICE 氢耗 MAP (kW -> g/s)
        self.ice_power_points = np.array([4.79, 10.69, 21.12, 30.15, 40.71, 51.21, 60.82, 69.98, 78.29])
        self.ice_h2_consumption = np.array([0.11, 0.20, 0.38, 0.54, 0.72, 0.91, 1.09, 1.28, 1.46])
        self.ice_h2_interp = interp1d(self.ice_power_points, self.ice_h2_consumption,
                                      kind='linear', bounds_error=False, fill_value=(0, 1.46))

        # FC 氢耗 MAP (kW -> g/s)
        self.fc_power_points = np.array([0, 12.04, 22.27, 38.72, 58.66, 77.49, 99.86])
        self.fc_h2_consumption = np.array([0, 0.15, 0.29, 0.54, 0.84, 1.14, 1.51])
        self.fc_h2_interp = interp1d(self.fc_power_points, self.fc_h2_consumption,
                                     kind='linear', bounds_error=False, fill_value=(0, 1.51))

        # ICE 和 FC 最大功率
        self.ice_max_power = 60000  # W
        self.fc_max_power = 60000  # W

        # === 奖励权重系数 ===
        self.reward_h2_weight = 0.3           # 氢耗权重 - 降低权重，让氢耗的负面影响更小
        self.reward_soc_weight = 200.0        # SOC维持权重 - 适中的权重
        self.reward_violation_weight = 1000   # 违规惩罚权重
        self.reward_smoothness_weight = 5.0   # 动作平滑权重 (新增)
        self.reward_incomplete_weight = 100.0 # 未完成全程的惩罚权重 (新增)

        # === 工况数据 ===
        if cycle_data is None:
            # 使用 cycle_name 加载工况数据
            print(f"\n{'='*60}")
            print(f"加载工况: {cycle_name.upper()}")
            print(f"{'='*60}")
            cycle_data = load_cycle_data(cycle_name)
            self.cycle_speed = cycle_data[:, 0]  # m/s
            self.cycle_acc = cycle_data[:, 1]  # m/s²
            self.cycle_name = cycle_name
        else:
            self.cycle_speed = cycle_data[:, 0]  # m/s
            self.cycle_acc = cycle_data[:, 1]  # m/s²
            self.cycle_name = 'custom'

        self.cycle_length = len(self.cycle_speed)

        # === 动作空间 ===
        # 连续动作空间: [ICE功率比例, FC功率比例]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0]),
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )

        # === 观察空间 ===
        # [速度(归一化), 加速度(归一化), SOC, 功率需求(归一化)]
        self.observation_space = spaces.Box(
            low=np.array([0.0, -1.0, 0.0, 0.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )

        # === 状态变量 ===
        self.current_step = 0
        self.soc = self.soc_init
        self.bat_voltage = 400  # 初始电压 (V)
        self.bat_current = 0  # 初始电流 (A)

        # === 奖励权重 ===
        self.reward_h2_weight = 1.0  # 氢耗权重
        self.reward_soc_weight = 150.0  # SOC维持权重
        self.reward_violation_weight = 1000.0  # 越界惩罚权重

        # === 记录变量 ===
        self.total_h2_consumption = 0.0
        self.soc_history = []
        self.power_ice_history = []
        self.power_fc_history = []
        self.power_batt_history = []
        self.reward_history = []

        # === 渲染模式 ===
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        """重置环境"""
        super().reset(seed=seed)

        # 重置状态变量
        self.current_step = 0
        self.soc = self.soc_init
        self.bat_voltage = 400
        self.bat_current = 0

        # 重置记录变量
        self.total_h2_consumption = 0.0
        self.soc_history = [self.soc]
        self.power_ice_history = []
        self.power_fc_history = []
        self.power_batt_history = []
        self.reward_history = []

        # 重置上一步动作 (用于动作平滑奖励)
        self.last_action = np.array([0.5, 0.5])

        # 获取初始观测
        observation = self._get_observation()
        info = {}

        return observation, info

    def step(self, action):
        """执行一步仿真"""
        # === 1. 获取工况数据 ===
        if self.current_step >= self.cycle_length:
            # 如果超过工况长度，重置
            return self.reset()

        v_target = self.cycle_speed[self.current_step]
        a_target = self.cycle_acc[self.current_step]

        # === 2. 轮端需求反推 (Backward: Wheel -> Motor) ===
        # 计算总阻力
        v_abs = abs(v_target)
        f_loss = (self.F_const + self.F_lin_coef * v_abs + self.F_quad_coef * v_abs**2)

        # 计算需要的净力
        f_net = self.m_eq * a_target + f_loss

        # 转换为轮端转矩
        t_wheel = f_net * self.r_eff

        # 转换为电机转矩
        t_mot_req = t_wheel / self.i_0

        # 转换为电机转速
        n_mot = (v_target / self.r_eff) * self.i_0 * (60 / (2 * np.pi))

        # === 3. 电机损耗计算 ===
        # 限制转矩在最大范围内
        t_mot_max = self._get_max_torque(self.bat_voltage, n_mot)
        t_mot = np.clip(t_mot_req, -t_mot_max, t_mot_max)

        # 计算机械功率
        omega_mot = n_mot / 9.55  # rpm -> rad/s
        p_mech = t_mot * omega_mot  # W

        # 查表获取效率
        eff_mot = self._get_motor_efficiency(abs(t_mot), n_mot)

        # 计算电功率
        if p_mech >= 0:
            # 驱动模式
            p_elec = (p_mech / eff_mot) * self.mot_drive_eff
        else:
            # 回馈模式
            p_elec = (p_mech * eff_mot) * self.mot_regenerative_eff

        # === 4. 策略分配 (Agent Action) ===
        # Action[0]: ICE功率比例, Action[1]: FC功率比例
        p_ice_cmd = action[0] * self.ice_max_power
        p_fc_cmd = action[1] * self.fc_max_power

        # 限制在最大功率范围内
        p_ice_cmd = np.clip(p_ice_cmd, 0, self.ice_max_power)
        p_fc_cmd = np.clip(p_fc_cmd, 0, self.fc_max_power)

        # === 强制最小H2使用策略 ===
        # 当功率需求为正（驱动模式）时，强制使用最小H2功率
        if p_elec > 0:
            # 计算最小H2功率：功率需求的30%，但不超过最大H2功率的50%
            min_h2_power = min(p_elec * 0.3, (self.ice_max_power + self.fc_max_power) * 0.5)

            # 当前H2总功率
            current_h2_power = p_ice_cmd + p_fc_cmd

            # 如果当前H2功率不足，强制提升到最小值
            if current_h2_power < min_h2_power:
                # 按比例分配额外的H2功率到ICE和FC
                deficit = min_h2_power - current_h2_power

                # 优先使用FC（效率更高），然后使用ICE
                if p_fc_cmd < self.fc_max_power:
                    fc_add = min(deficit, self.fc_max_power - p_fc_cmd)
                    p_fc_cmd += fc_add
                    deficit -= fc_add

                if deficit > 0 and p_ice_cmd < self.ice_max_power:
                    p_ice_cmd += deficit

        # === 5. 电池物理响应 (R-int 模型) ===
        # 功率平衡: P_batt = P_demand + P_aux - P_ice_net - P_fc_net
        # 这里 P_aux 忽略, P_demand = P_elec
        p_ice_net = p_ice_cmd * self.ice_dc_dc_eff
        p_fc_net = p_fc_cmd * self.fc_dcdc_eff

        p_batt = p_elec - p_ice_net - p_fc_net

        # 求解电池电流: I²R - V_oc*I + P_batt = 0
        # 根据当前 SOC 查表获得 V_oc 和 R_int
        voc_cell = self.voc_interp(self.soc)
        voc_pack = voc_cell * self.n_cell

        # 根据充放电选择内阻
        if p_batt >= 0:
            # 放电
            r_int = self.r_dis_interp(self.soc) * self.n_cell
        else:
            # 充电
            r_int = self.r_chg_interp(self.soc) * self.n_cell

        # 求解二次方程
        delta = voc_pack**2 - 4 * r_int * p_batt
        if delta >= 0:
            # 两个解,选择合理的那个
            i1 = (voc_pack + np.sqrt(delta)) / (2 * r_int)
            i2 = (voc_pack - np.sqrt(delta)) / (2 * r_int)

            # 选择电流较小的解 (更符合物理实际)
            self.bat_current = min(abs(i1), abs(i2))
            if p_batt < 0:
                self.bat_current = -self.bat_current  # 充电电流为负

            # 计算端电压
            self.bat_voltage = voc_pack - self.bat_current * r_int

            # SOC 更新
            soc_delta = (self.bat_current * self.dt) / self.bat_capacity_coulomb
            self.soc = self.soc - soc_delta

            violation_flag = False
        else:
            # 功率需求过大,无解 - 使用物理极限电流
            violation_flag = True
            # 强制设电流为物理极限 I = V_oc / (2R)
            self.bat_current = voc_pack / (2 * r_int)
            if p_batt < 0:
                self.bat_current = -self.bat_current

            # 计算端电压
            self.bat_voltage = voc_pack - self.bat_current * r_int

            # SOC 更新 (使用极限电流)
            soc_delta = (self.bat_current * self.dt) / self.bat_capacity_coulomb
            self.soc = self.soc - soc_delta

        # 限制 SOC 范围 (硬约束，防止数值越界)
        self.soc = np.clip(self.soc, 0.0, 1.0)

        # SOC 违规检测
        soc_violation = False
        soc_terminate_violation = False
        if self.soc < self.soc_min:
            soc_violation = True
            # SOC 严重过低，终止 episode
            if self.soc < 0.05:  # 5% 以下强制终止
                soc_terminate_violation = True
        elif self.soc > self.soc_max:
            soc_violation = True
            # SOC 过高，终止 episode
            soc_terminate_violation = True

        # === 6. 奖励计算 ===
        # 计算氢耗
        h2_ice = self.ice_h2_interp(p_ice_cmd / 1000) * self.dt  # g
        h2_fc = self.fc_h2_interp(p_fc_cmd / 1000) * self.dt  # g
        h2_total = h2_ice + h2_fc

        self.total_h2_consumption += h2_total

        # 计算基础奖励
        reward = -(
            self.reward_h2_weight * h2_total +
            self.reward_soc_weight * (self.soc - self.soc_target)**2
        )

        # 添加动作平滑奖励 (惩罚高频振荡)
        action_smoothness_penalty = np.sum((action - self.last_action) ** 2)
        reward -= self.reward_smoothness_weight * action_smoothness_penalty

        # 更新上一步动作
        self.last_action = action.copy()

        # 添加违规惩罚
        if violation_flag:
            reward -= self.reward_violation_weight
        if soc_violation:
            reward -= self.reward_violation_weight / 2
        if soc_terminate_violation:
            # SOC 严重违规，大额失败惩罚
            reward -= 500

        # === 7. 记录历史 ===
        self.soc_history.append(self.soc)
        self.power_ice_history.append(p_ice_cmd)
        self.power_fc_history.append(p_fc_cmd)
        self.power_batt_history.append(p_batt)
        self.reward_history.append(reward)

        # === 8. 更新步数 ===
        self.current_step += 1

        # === 9. 判断是否终止 ===
        terminated = False
        truncated = False

        if self.current_step >= self.cycle_length:
            # 完成全程，给予奖励
            reward += 1000  # 完成全程的大额奖励
            terminated = True
        elif soc_terminate_violation:
            # SOC 严重违规，提前终止
            # 计算未完成的比例
            incomplete_ratio = (self.cycle_length - self.current_step) / self.cycle_length
            # 对未完成的部分施加惩罚
            reward -= self.reward_incomplete_weight * incomplete_ratio * 1000
            terminated = True

        # === 10. 获取观测 ===
        observation = self._get_observation(v_target, a_target, p_batt)

        # === 11. 构造 info ===
        info = {
            "soc": self.soc,
            "h2_consumption": self.total_h2_consumption,
            "p_ice": p_ice_cmd,
            "p_fc": p_fc_cmd,
            "p_batt": p_batt,
            "voltage": self.bat_voltage,
            "current": self.bat_current,
            "violation": violation_flag
        }

        # === 12. 渲染 ===
        if self.render_mode == "human":
            self._render()

        return observation, reward, terminated, truncated, info

    def _get_observation(self, v=None, a=None, p_batt=None):
        """获取归一化的观测值"""
        if v is None:
            v = self.cycle_speed[min(self.current_step, self.cycle_length - 1)]
        if a is None:
            a = self.cycle_acc[min(self.current_step, self.cycle_length - 1)]
        if p_batt is None:
            p_batt = 0

        v_norm = np.clip(v / self.max_speed, 0, 1)
        a_norm = np.clip(a / self.max_acc, -1, 1)
        soc_norm = np.clip(self.soc, 0, 1)
        p_batt_norm = np.clip(abs(p_batt) / self.max_power, 0, 1)

        return np.array([v_norm, a_norm, soc_norm, p_batt_norm], dtype=np.float32)

    def _get_max_torque(self, voltage, speed_rpm):
        """查表获取最大转矩"""
        try:
            torque = self.max_torque_interp(np.array([voltage, speed_rpm]))[0]
            return max(0, torque)
        except:
            return 100  # 默认值

    def _get_motor_efficiency(self, torque, speed_rpm):
        """查表获取电机效率"""
        try:
            # 限制在查表范围内
            torque_clipped = np.clip(torque, 14, 365)
            speed_clipped = np.clip(speed_rpm, 700, 12000)
            eff = self.eff_interp(np.array([torque_clipped, speed_clipped]))[0]
            return max(0.7, min(0.95, eff))
        except:
            return 0.85  # 默认效率

    def _render(self):
        """渲染当前状态"""
        if self.current_step % 100 == 0:
            print(f"Step: {self.current_step}/{self.cycle_length}, "
                  f"SOC: {self.soc:.4f}, "
                  f"H2: {self.total_h2_consumption:.2f}g, "
                  f"V: {self.bat_voltage:.1f}V, "
                  f"I: {self.bat_current:.1f}A")

    def get_metrics(self):
        """获取性能指标"""
        return {
            "total_h2_consumption_g": self.total_h2_consumption,
            "final_soc": self.soc,
            "soc_history": np.array(self.soc_history),
            "power_ice_history": np.array(self.power_ice_history),
            "power_fc_history": np.array(self.power_fc_history),
            "power_batt_history": np.array(self.power_batt_history),
            "reward_history": np.array(self.reward_history),
            "average_reward": np.mean(self.reward_history) if self.reward_history else 0
        }
