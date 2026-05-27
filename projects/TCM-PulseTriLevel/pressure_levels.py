#!/usr/bin/env python3
"""P03 Pressure Levels - Fu/Zhong/Chen Three-Level TCM Pulse Modeling"""
import numpy as np


class PressureLevel:
    FU = "浮"   # Superficial - light touch, 0-20mmHg
    ZHONG = "中"  # Middle - moderate pressure, 20-80mmHg
    CHEN = "沉"  # Deep - heavy pressure, 80-200mmHg


class ThreeLevelPulseModel:
    """Three-level pulse waveform model based on TCM theory.

    TCM pulse diagnosis: Fu (浮/floating), Zhong (中/middle), Chen (沉/deep).
    Each level reflects different organ and zang-fu states.
    """

    def __init__(self, vessel_params):
        self.E = vessel_params['E']        # Elastic modulus (Pa)
        self.C = vessel_params['C']        # Compliance (m³/Pa)
        self.R = vessel_params['R']        # Peripheral resistance
        self.zeta = vessel_params.get('zeta', 0.1)  # Damping ratio

        # Pressure level definitions
        self.levels = {
            PressureLevel.FU: {'p_range': (0, 20), 'target': 10, 'description': '浮脉 - 主表证'},
            PressureLevel.ZHONG: {'p_range': (20, 80), 'target': 50, 'description': '中脉 - 主里证'},
            PressureLevel.CHEN: {'p_range': (80, 200), 'target': 140, 'description': '沉脉 - 主寒证'},
        }

    def get_level_params(self, level_name):
        return self.levels.get(level_name, self.levels[PressureLevel.ZHONG])

    def calculate_waveform(self, t, level_name):
        """Generate pulse waveform at specified pressure level.

        Args:
            t: time array (seconds)
            level_name: '浮', '中', or '沉'

        Returns:
            pressure waveform array (mmHg)
        """
        params = self.get_level_params(level_name)
        target_p = params['target']

        # Base heart rate ~75 BPM = 1.25 Hz
        hr = 1.25
        cardiac_period = 1.0 / hr

        # Multi-harmonic representation of pulse waveform
        waveform = np.zeros_like(t, dtype=float)

        for i, ti in enumerate(t):
            phase = (ti % cardiac_period) / cardiac_period
            omega = 2 * np.pi * hr

            # Fundamental + harmonics (1st to 4th)
            fundamental = np.sin(omega * ti)
            h2 = 0.4 * np.sin(2 * omega * ti + 0.3)
            h3 = 0.15 * np.sin(3 * omega * ti + 0.7)
            h4 = 0.08 * np.sin(4 * omega * ti + 1.1)

            raw = (fundamental + h2 + h3 + h4) / (1 + 0.4 + 0.15 + 0.08)
            waveform[i] = target_p * (0.5 + 0.5 * raw)

        return waveform

    def calc_cfl(self, dt, dx=None):
        """Calculate CFL stability condition for numerical methods.

        Returns:
            cfl: CFL number (< 1 for stability)
            dt_max: maximum stable time step
        """
        if dx is None:
            # Use vessel length scale
            dx = 0.01  # 1cm spatial step
        pwave = self.calculate_waveform(np.array([0, 0.001]), PressureLevel.ZHONG)
        c = np.sqrt(self.E * self.C)  # wave speed estimate
        cfl = c * dt / dx
        return cfl, dt / c * dx if c > 0 else float('inf')

    def get_level_characteristics(self):
        """Return physiological interpretation of each level."""
        return {
            PressureLevel.FU: {
                'description': '浮脉',
                'TCM_meaning': '主表证、外感风寒/风热、气虚下陷',
                'cardiovascular': '浅层动脉、外周血管扩张',
                'expected_pulse': '轻取即得、重按稍减',
            },
            PressureLevel.ZHONG: {
                'description': '中脉',
                'TCM_meaning': '主里证、脏腑病证、气血失调',
                'cardiovascular': '中层血管、正常循环状态',
                'expected_pulse': '不轻不重、应手有力',
            },
            PressureLevel.CHEN: {
                'description': '沉脉',
                'TCM_meaning': '主寒证、里实证、阳气内郁',
                'cardiovascular': '深层血管、血管收缩、外周阻力升高',
                'expected_pulse': '轻取不应、重按始得',
            },
        }


def demo():
    """Demonstrate three-level pulse modeling."""
    vessel_params = {'E': 1e5, 'C': 1e-7, 'R': 1e4, 'zeta': 0.1}
    model = ThreeLevelPulseModel(vessel_params)

    t = np.linspace(0, 2, 500)

    print("=== 浮中沉三压力层级脉波建模 ===")
    for level in [PressureLevel.FU, PressureLevel.ZHONG, PressureLevel.CHEN]:
        p = model.calculate_waveform(t, level)
        chars = model.get_level_characteristics()[level]
        print(f"\n[{level}] {chars['description']}")
        print(f"  范围: {model.levels[level]['p_range']} mmHg")
        print(f"  TCM: {chars['TCM_meaning']}")
        print(f"  心血管: {chars['cardiovascular']}")
        print(f"  脉象: {chars['expected_pulse']}")
        print(f"  信号范围: [{p.min():.1f}, {p.max():.1f}] mmHg")

    cfl, dt_max = model.calc_cfl(0.001)
    print(f"\nCFL条件验证: CFL={cfl:.4f} {'✓稳定' if cfl < 1 else '✗不稳定'} (dt_max={dt_max:.6f}s)")


if __name__ == '__main__':
    demo()
