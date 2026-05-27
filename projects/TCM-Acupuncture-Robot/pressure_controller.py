"""
TCM-Acupuncture-Robot E01
机械臂推拿力度-穴位-形变闭环控制
 pressure_controller.py - PID闭环压力控制器
"""

import time
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class PIDConfig:
    """PID控制器配置"""
    kp: float = 0.8       # 比例系数
    ki: float = 0.1       # 积分系数
    kd: float = 0.3       # 微分系数
    output_min: float = -50.0  # 输出下限 (N/s)
    output_max: float = 50.0   # 输出上限 (N/s)
    integral_limit: float = 20.0  # 积分限幅


@dataclass
class ControlState:
    """控制器状态"""
    timestamp: float
    target_pressure: float      # 目标压力 N
    current_pressure: float      # 当前压力 N
    error: float                 # 误差
    output: float                # 控制输出
    p_term: float                # P项
    i_term: float                # I项
    d_term: float                # D项


class PressureController:
    """PID闭环压力控制器
    
    实现机械臂推拿压力闭环控制
    输入: 目标压力(N)
    输出: 速度指令(N/s)
    """

    def __init__(self, config: Optional[PIDConfig] = None):
        self.config = config or PIDConfig()
        self._reset()
        
    def _reset(self):
        """重置控制器状态"""
        self._prev_error = 0.0
        self._integral = 0.0
        self._prev_time = None
        self._history: list[ControlState] = []
        
    def compute(self, target: float, current: float, 
                timestamp: Optional[float] = None) -> float:
        """计算PID控制输出
        
        Args:
            target: 目标压力 (N)
            current: 当前压力 (N)
            timestamp: 时间戳 (可选)
            
        Returns:
            控制输出 (N/s 速度指令)
        """
        if timestamp is None:
            timestamp = time.time()
            
        # 计算误差
        error = target - current
        
        # 计算时间步长
        dt = 0.01  # 默认10ms
        if self._prev_time is not None:
            dt = timestamp - self._prev_time
            dt = max(dt, 0.001)  # 防止除零
        
        # PID三项
        p_term = self.config.kp * error
        self._integral += error * dt
        self._integral = max(-self.config.integral_limit, 
                             min(self.config.integral_limit, self._integral))
        i_term = self.config.ki * self._integral
        
        d_term = 0.0
        if dt > 0:
            d_term = self.config.kd * (error - self._prev_error) / dt
        
        # 总输出
        output = p_term + i_term + d_term
        output = max(self.config.output_min, 
                    min(self.config.output_max, output))
        
        # 保存状态
        state = ControlState(
            timestamp=timestamp,
            target_pressure=target,
            current_pressure=current,
            error=error,
            output=output,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term
        )
        self._history.append(state)
        
        # 更新previous值
        self._prev_error = error
        self._prev_time = timestamp
        
        return output
    
    def compute_and_step(self, target: float, current: float,
                         delta_time: float = 0.01) -> Tuple[float, ControlState]:
        """一步计算并返回详细状态
        
        Args:
            target: 目标压力 (N)
            current: 当前压力 (N)
            delta_time: 控制周期 (s)
            
        Returns:
            (输出, 控制状态)
        """
        timestamp = time.time()
        output = self.compute(target, current, timestamp)
        
        # 获取最新状态
        state = self._history[-1] if self._history else None
        
        return output, state
    
    def set_tunings(self, kp: float, ki: float, kd: float):
        """在线调整PID参数"""
        self.config.kp = kp
        self.config.ki = ki
        self.config.kd = kd
        
    def reset(self):
        """重置控制器"""
        self._reset()
        
    def get_history(self, last_n: Optional[int] = None) -> list[ControlState]:
        """获取控制历史"""
        if last_n is None:
            return self._history.copy()
        return self._history[-last_n:]
    
    def get_steady_state_error(self) -> float:
        """计算稳态误差 (最后10个点的平均误差)"""
        if len(self._history) < 10:
            return float('inf')
        recent = self._history[-10:]
        return sum(s.error for s in recent) / len(recent)


class AdaptivePressureController(PressureController):
    """自适应PID压力控制器
    
    根据压力误差自动调整PID参数
    适用于不同穴位的差异化控制
    """

    def __init__(self, config: Optional[PIDConfig] = None):
        super().__init__(config)
        self._adaptation_enabled = True
        self._error_threshold = 2.0  # 误差阈值 N
        
    def compute_adaptive(self, target: float, current: float,
                        depth: float, tissue_stiffness: float,
                        timestamp: Optional[float] = None) -> float:
        """自适应PID计算
        
        Args:
            target: 目标压力 (N)
            current: 当前压力 (N)
            depth: 当前深度 (mm)
            tissue_stiffness: 组织刚度 (N/mm)
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = time.time()
            
        error = target - current
        
        # 根据组织刚度自适应调整KP
        # 硬组织(肌肉): 需要较小的P
        # 软组织(脂肪): 需要较大的P
        if tissue_stiffness > 10.0:  # 硬组织
            adaptive_kp = self.config.kp * 0.8
        elif tissue_stiffness < 3.0:  # 软组织
            adaptive_kp = self.config.kp * 1.2
        else:
            adaptive_kp = self.config.kp
            
        # 根据深度微调KI (防止过深时积分饱和)
        depth_factor = 1.0 - 0.01 * max(0, depth - 20)
        adaptive_ki = self.config.ki * depth_factor
        
        # 计算控制输出
        dt = 0.01
        if self._prev_time is not None:
            dt = timestamp - self._prev_time
            dt = max(dt, 0.001)
            
        p_term = adaptive_kp * error
        self._integral += error * dt
        self._integral = max(-self.config.integral_limit,
                            min(self.config.integral_limit, self._integral))
        i_term = adaptive_ki * self._integral
        
        d_term = 0.0
        if dt > 0:
            d_term = self.config.kd * (error - self._prev_error) / dt
            
        output = p_term + i_term + d_term
        output = max(self.config.output_min,
                    min(self.config.output_max, output))
        
        # 记录状态
        state = ControlState(
            timestamp=timestamp,
            target_pressure=target,
            current_pressure=current,
            error=error,
            output=output,
            p_term=p_term,
            i_term=i_term,
            d_term=d_term
        )
        self._history.append(state)
        
        self._prev_error = error
        self._prev_time = timestamp
        
        return output


def tune_pid(controller: PressureController, 
             target: float,
             simulation_steps: int = 500,
             noise_std: float = 0.5) -> dict:
    """PID参数调优 (Ziegler-Nichols方法模拟)
    
    Returns:
        调优结果和建议参数
    """
    # 简化的Ziegler-Nichols调优
    # 记录极限增益测试
    test_kp = controller.config.kp
    
    # 记录响应数据
    pressures = []
    outputs = []
    current = 0.0
    
    for i in range(simulation_steps):
        output = controller.compute(target, current, timestamp=i*0.01)
        
        # 模拟被控对象 (一阶惯性环节)
        # P(s) = K / (Ts + 1)
        K = 1.0  # 过程增益
        T = 0.1  # 时间常数
        dt = 0.01
        current += (K * output - current) / T * dt
        
        # 添加测量噪声
        import random
        current += random.gauss(0, noise_std)
        current = max(0, current)
        
        pressures.append(current)
        outputs.append(output)
    
    # 计算性能指标
    overshoot = max(pressures) - target if max(pressures) > target else 0
    rise_time = next((i * 0.01 for i, p in enumerate(pressures) 
                      if p >= target * 0.9), 0)
    settling_error = controller.get_steady_state_error()
    
    return {
        'overshoot': overshoot,
        'rise_time': rise_time,
        'steady_state_error': settling_error,
        'final_kp': controller.config.kp,
        'final_ki': controller.config.ki,
        'final_kd': controller.config.kd
    }


def main():
    """测试PID控制器"""
    print("=" * 60)
    print("PID闭环压力控制器测试")
    print("=" * 60)
    
    # 创建控制器
    pid = PressureController(PIDConfig(kp=0.8, ki=0.1, kd=0.3))
    
    target_pressure = 12.0  # 12N
    current_pressure = 0.0
    
    print(f"\n目标压力: {target_pressure}N")
    print(f"{'Step':<6} {'Time':<8} {'Current':<10} {'Error':<10} {'Output':<10} {'P':<8} {'I':<8} {'D':<8}")
    print("-" * 80)
    
    # 模拟控制200步
    for step in range(200):
        timestamp = step * 0.01
        output = pid.compute(target_pressure, current_pressure, timestamp)
        
        # 模拟被控对象响应
        # 模拟机械臂+组织响应
        import random
        response = output * 0.15 + random.gauss(0, 0.1)
        current_pressure += response
        current_pressure = max(0, min(20, current_pressure))
        
        # 每20步打印一次
        if step % 20 == 0:
            state = pid.get_history()[-1]
            print(f"{step:<6} {timestamp:<8.2f} {current_pressure:<10.2f} "
                  f"{state.error:<10.2f} {output:<10.2f} "
                  f"{state.p_term:<8.2f} {state.i_term:<8.2f} {state.d_term:<8.2f}")
    
    # 测试自适应控制器
    print("\n" + "=" * 60)
    print("自适应PID控制器测试")
    print("=" * 60)
    
    adaptive = AdaptivePressureController()
    
    # 模拟不同组织刚度
    test_cases = [
        (12.0, 5.0, 2.0),    # 软组织
        (12.0, 10.0, 6.0),   # 中等
        (12.0, 15.0, 12.0),  # 硬组织
    ]
    
    for target, depth, stiffness in test_cases:
        adaptive.reset()
        current = 0.0
        print(f"\n目标:{target}N | 深度:{depth}mm | 刚度:{stiffness}N/mm")
        
        for step in range(100):
            output = adaptive.compute_adaptive(target, current, depth, stiffness)
            current += output * 0.1
            current = max(0, min(20, current))
        
        final_error = adaptive.get_steady_state_error()
        print(f"  最终误差: {final_error:.3f}N")


if __name__ == "__main__":
    main()
