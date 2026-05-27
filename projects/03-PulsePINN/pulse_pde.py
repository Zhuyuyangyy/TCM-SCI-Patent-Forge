"""
Pulse Wave PDE Model
动脉脉波传播的偏微分方程模型

基于一维血管流体力学:
∂u/∂t + c ∂u/∂x = 0 (对流方程)
+ 生物力学弹性项

参数:
- c: 波速 (与血管弹性相关)
- α, β: 非线性项系数
- R: 远端阻力
"""
import numpy as np

class PulsePDE:
    """
    脉波传播PDE模型
    ∂²u/∂t² = c² ∂²u/∂x² - δ ∂u/∂t + f(x,t)
    """
    def __init__(self, c=10.0, delta=0.5, L=1.0, nx=100, nt=200):
        """
        c: 波速 (m/s), 通常 5-15 m/s
        delta: 阻尼系数
        L: 血管长度 (m)
        nx: 空间网格数
        nt: 时间步数
        """
        self.c = c
        self.delta = delta
        self.L = L
        self.nx = nx
        self.nt = nt
        self.dx = L / nx
        self.dt = None  # 由CFL条件决定
        
    def solve(self, u0=None, T=1.0):
        """
        求解PDE
        u0: 初始条件 (nx+1,)
        T: 总时间
        返回: u(x,t) 数组 (nt+1, nx+1)
        """
        self.dt = self.dx * 0.5 / self.c  # CFL条件
        nt = int(T / self.dt)
        
        u = np.zeros((nt+1, self.nx+1))
        if u0 is not None:
            u[0] = u0
        else:
            # 高斯脉搏初始条件
            x = np.linspace(0, self.L, self.nx+1)
            u[0] = np.exp(-((x - 0.3) ** 2) / 0.01)
        
        # 有限差分求解
        r = (self.c * self.dt / self.dx) ** 2
        D = self.delta * self.dt
        
        for n in range(nt):
            for i in range(1, self.nx):
                u[n+1, i] = (2*u[n, i] - u[n-1, i] 
                             + r*(u[n, i+1] - 2*u[n, i] + u[n, i-1])
                             - D*(u[n, i] - u[n-1, i]))
            # 边界条件
            u[n+1, 0] = 0
            u[n+1, -1] = u[n+1, -2]
        
        return u
    
    def get_pulse_at_position(self, u, x_pos, time_points=None):
        """获取特定位置的脉波信号"""
        i = int(x_pos / self.dx)
        if time_points is None:
            return u[:, i]
        return u[time_points, i]
    
    def classify_pulse_type(self, pulse_signal):
        """
        简单脉象分类（基于峰值特征）
        返回: 脉象名称
        """
        peaks = self._find_peaks(pulse_signal)
        if len(peaks) == 0:
            return '平脉'
        # 简化分类
        height = np.max(pulse_signal) - np.min(pulse_signal)
        width = peaks[0] if len(peaks) > 0 else len(pulse_signal)//2
        
        if height > 0.8:
            return '实脉'
        elif height < 0.3:
            return '虚脉'
        else:
            return '平脉'
    
    def _find_peaks(self, signal):
        peaks = []
        for i in range(1, len(signal)-1):
            if signal[i] > signal[i-1] and signal[i] > signal[i+1]:
                peaks.append(i)
        return peaks
