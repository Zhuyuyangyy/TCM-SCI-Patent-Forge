"""
fire_sensor_sim.py - 炮制温度传感器模拟器
模拟真实炮制过程中的温度曲线数据
"""

import numpy as np
from typing import List, Tuple, Optional
import time


class FireSensorSimulator:
    """炮制火候传感器模拟器"""
    
    def __init__(self, noise_level: float = 2.0):
        """
        初始化模拟器
        
        Args:
            noise_level: 噪声水平（标准差），模拟传感器误差
        """
        self.noise_level = noise_level
        self.time_series = []
        self.temp_series = []
        self.humidity_series = []
        self.start_time = None
        
    def generate_standard_curve(
        self,
        target_temp: float,
        duration_min: float,
        sampling_rate: float = 1.0,
        phases: Optional[List[dict]] = None
    ) -> Tuple[List[float], List[float]]:
        """
        生成标准温度曲线
        
        Args:
            target_temp: 目标温度(℃)
            duration_min: 持续时间(分钟)
            sampling_rate: 采样频率(Hz)
            phases: 自定义阶段 [(start_time_min, end_time_min, temp), ...]
        
        Returns:
            (时间序列秒, 温度序列)
        """
        if phases is None:
            # 默认单段升温到目标温度并保持
            phases = [
                (0, duration_min * 0.1, 30),           # 预热阶段
                (duration_min * 0.1, duration_min * 0.2, target_temp * 0.5),  # 升温
                (duration_min * 0.2, duration_min * 0.8, target_temp),        # 保温
                (duration_min * 0.8, duration_min, target_temp * 0.8),       # 降温
            ]
        
        total_seconds = int(duration_min * 60)
        times = []
        temps = []
        
        for t in range(0, total_seconds, int(1/sampling_rate)):
            t_min = t / 60.0
            temp = self._interpolate_temp(t_min, phases)
            # 添加传感器噪声
            temp += np.random.normal(0, self.noise_level)
            times.append(t)
            temps.append(max(20, temp))  # 确保温度不低于环境温度
        
        self.time_series = times
        self.temp_series = temps
        self.start_time = time.time()
        
        return times, temps
    
    def _interpolate_temp(self, t_min: float, phases: List[dict]) -> float:
        """分段线性插值计算温度"""
        for i, (start, end, temp) in enumerate(phases):
            if start <= t_min < end:
                # 在该阶段内，进行线性插值
                if i < len(phases) - 1 and end < phases[i+1][0]:
                    # 过渡到下一阶段
                    next_temp = phases[i+1][2]
                    segment_duration = phases[i+1][0] - end
                    if segment_duration > 0 and t_min >= end:
                        progress = (t_min - end) / segment_duration
                        return temp + (next_temp - temp) * min(1, progress * 2)
                return temp
        return phases[-1][2] if phases else 25
    
    def generate_realistic_curve(
        self,
        target_temp: float,
        duration_min: float,
        fire_level: str = "medium",
        instability: float = 0.05
    ) -> Tuple[List[float], List[float]]:
        """
        生成更真实的炮制温度曲线（考虑火力波动）
        
        Args:
            target_temp: 目标温度
            duration_min: 持续时间
            fire_level: 火力等级 ("low", "medium", "high")
            instability: 不稳定系数
        
        Returns:
            (时间序列, 温度序列)
        """
        fire_intensity = {"low": 0.8, "medium": 1.0, "high": 1.2}.get(fire_level, 1.0)
        
        total_seconds = int(duration_min * 60)
        times = []
        temps = []
        
        # 温度曲线参数
        heat_rate = 5 * fire_intensity  # 升温速率 ℃/min
        overshoot = 0.05 * fire_intensity  # 过冲量
        
        for t in range(total_seconds):
            t_min = t / 60.0
            
            # 预热阶段 (0-10%)
            if t_min < duration_min * 0.1:
                temp = 25 + (target_temp * 0.3) * (t_min / (duration_min * 0.1))
            # 快速升温阶段 (10-25%)
            elif t_min < duration_min * 0.25:
                prev_temp = target_temp * 0.3
                target_in_phase = target_temp * 0.95
                phase_progress = (t_min - duration_min * 0.1) / (duration_min * 0.15)
                temp = prev_temp + (target_in_phase - prev_temp) * min(1, phase_progress * 1.5)
            # 保温阶段 (25-75%) - 带有周期性波动
            elif t_min < duration_min * 0.75:
                base = target_temp
                # 添加周期性波动 (模拟火力不均匀)
                wave1 = np.sin(t_min * 0.5) * target_temp * 0.02
                wave2 = np.sin(t_min * 1.7) * target_temp * 0.01
                # 添加随机噪声
                noise = np.random.normal(0, target_temp * instability)
                temp = base + wave1 + wave2 + noise
            # 缓慢降温阶段 (75-90%)
            elif t_min < duration_min * 0.9:
                phase_progress = (t_min - duration_min * 0.75) / (duration_min * 0.15)
                temp = target_temp * (1 - phase_progress * 0.15)
                temp += np.random.normal(0, 2)
            # 快速降温阶段 (90-100%)
            else:
                phase_progress = (t_min - duration_min * 0.9) / (duration_min * 0.1)
                temp = target_temp * 0.85 * (1 - min(1, phase_progress * 1.2))
                temp += np.random.normal(0, 3)
            
            times.append(t)
            temps.append(max(20, temp))
        
        self.time_series = times
        self.temp_series = temps
        self.start_time = time.time()
        
        return times, temps
    
    def get_realtime_reading(self) -> float:
        """获取实时温度读数（模拟传感器）"""
        if not self.temp_series or not self.time_series:
            return 25.0
        
        elapsed = time.time() - self.start_time if self.start_time else 0
        # 找到对应时间点的温度
        for i, t in enumerate(self.time_series):
            if t >= elapsed:
                return self.temp_series[i]
        return self.temp_series[-1] if self.temp_series else 25.0
    
    def get_curve_statistics(self) -> dict:
        """获取温度曲线统计信息"""
        if not self.temp_series:
            return {}
        
        temps = np.array(self.temp_series)
        return {
            "mean_temp": float(np.mean(temps)),
            "std_temp": float(np.std(temps)),
            "min_temp": float(np.min(temps)),
            "max_temp": float(np.max(temps)),
            "duration_sec": len(self.temp_series),
            "stability": 1.0 - min(1.0, np.std(temps) / max(1, np.mean(temps)))
        }
    
    def simulate_sensor_failure(self, failure_duration_sec: int = 5) -> List[float]:
        """模拟传感器故障期间的读数（返回异常值）"""
        if not self.temp_series:
            return []
        
        failure_start = np.random.randint(len(self.temp_series) // 4, len(self.temp_series) // 2)
        failure_values = []
        
        for i in range(failure_duration_sec):
            idx = failure_start + i
            if idx < len(self.temp_series):
                # 模拟故障读数（跳跃或卡死）
                if np.random.random() > 0.5:
                    self.temp_series[idx] = np.random.uniform(0, 50)  # 跳跃到异常低值
                else:
                    self.temp_series[idx] = self.temp_series[idx-1]  # 卡死
                failure_values.append(self.temp_series[idx])
        
        return failure_values
    
    def export_curve_data(self, filepath: str):
        """导出温度曲线数据到文件"""
        if not self.time_series or not self.temp_series:
            print("No data to export")
            return
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("time_sec,temp_c\n")
            for t, temp in zip(self.time_series, self.temp_series):
                f.write(f"{t},{temp:.2f}\n")
        
        print(f"Curve data exported to {filepath}")


def generate_fuzhou_processing_curve(
    method: str = "制附子",
    duration_min: float = 240
) -> Tuple[List[float], List[float]]:
    """生成附子炮制的标准温度曲线"""
    sim = FireSensorSimulator(noise_level=1.5)
    
    # 附子炮制特有的多阶段曲线
    phases = [
        (0, 10, 25),                           # 常温预热
        (10, 30, 80),                          # 快速升温
        (30, 60, 110),                         # 继续升温
        (60, duration_min - 30, 120),          # 保温阶段（关键）
        (duration_min - 30, duration_min - 10, 115),  # 缓慢降温
        (duration_min - 10, duration_min, 80), # 出锅前降温
    ]
    
    return sim.generate_standard_curve(
        target_temp=120,
        duration_min=duration_min,
        phases=phases
    )


def generate_custom_herb_curve(
    herb_name: str,
    processing_method: str,
    duration_min: float,
    fire_level: str = "medium"
) -> Tuple[List[float], List[float]]:
    """为指定药材和方法生成温度曲线"""
    from herb_database import get_fire_parameters
    
    params = get_fire_parameters(herb_name, processing_method)
    if not params:
        # 使用默认值
        target_temp = 150
    else:
        target_temp = params.get("temp_target", 150)
    
    sim = FireSensorSimulator(noise_level=2.0)
    return sim.generate_realistic_curve(
        target_temp=target_temp,
        duration_min=duration_min,
        fire_level=fire_level,
        instability=0.03
    )


if __name__ == "__main__":
    print("=== 附子炮制温度曲线模拟 ===")
    times, temps = generate_fuzhou_processing_curve("制附子", duration_min=240)
    print(f"生成数据点: {len(temps)}")
    print(f"温度范围: {min(temps):.1f} - {max(temps):.1f} ℃")
    print(f"平均温度: {sum(temps)/len(temps):.1f} ℃")
    
    # 模拟实时读取
    sim = FireSensorSimulator()
    times2, temps2 = sim.generate_realistic_curve(120, 60, fire_level="medium")
    print(f"\n实时模拟曲线: {len(temps2)} 点")
    stats = sim.get_curve_statistics()
    print(f"曲线稳定性: {stats['stability']:.2%}")
