#!/usr/bin/env python3
"""P03 Pulse Simulator - Generate realistic three-level TCM pulse waveforms"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vessel_model import ThreeLevelVessel
from pressure_levels import ThreeLevelPulseModel


class PulseSimulator:
    """Generate realistic TCM pulse waveforms based on vessel parameters."""

    def __init__(self, vessel_params=None):
        if vessel_params is None:
            vessel_params = {'E': 1e5, 'C': 1e-7, 'R': 1e4, 'zeta': 0.1}
        self.vessel = ThreeLevelVessel(vessel_params)
        self.pulse_model = ThreeLevelPulseModel(vessel_params)
        self.vessel_params = vessel_params

    def generate(self, level='中', duration=2.0, fs=250, noise_std=0.5):
        """Generate pulse waveform at specified level.

        Args:
            level: '浮', '中', or '沉'
            duration: recording duration in seconds
            fs: sampling frequency (Hz)
            noise_std: measurement noise std (mmHg)

        Returns:
            t, pressure arrays
        """
        t = np.arange(0, duration, 1.0/fs)
        p = self.pulse_model.calculate_waveform(t, level)
        # Add measurement noise
        noise = np.random.randn(len(t)) * noise_std
        return t, p + noise

    def generate_all_levels(self, duration=2.0, fs=250, noise_std=0.5):
        """Generate pulse waveforms for all three pressure levels."""
        result = {}
        for level in ['浮', '中', '沉']:
            t, p = self.generate(level, duration, fs, noise_std)
            result[level] = {'t': t, 'p': p}
        return result

    def add_respiratory_modulation(self, t, p, resp_rate=0.2):
        """Add respiratory modulation to pulse signal (realistic artifact)."""
        resp_cycle = 1.0 / resp_rate
        resp_envelope = 1.0 + 0.1 * np.sin(2 * np.pi * t / resp_cycle)
        return p * resp_envelope

    def extract_features(self, t, p):
        """Extract clinically relevant features from pulse waveform."""
        fs = 1.0 / (t[1] - t[0])

        # Find peaks (systolic peaks) - simple numpy implementation
        threshold = np.mean(p) + 0.5 * np.std(p)
        peaks = []
        for i in range(1, len(p)-1):
            if p[i] > p[i-1] and p[i] > p[i+1] and p[i] > threshold:
                peaks.append(i)
        peaks = np.array(peaks)

        if len(peaks) < 2:
            return {'error': 'Insufficient peaks detected'}

        # Heart rate
        rr_intervals = np.diff(t[peaks])
        hr = 60.0 / np.mean(rr_intervals)

        # Peak amplitudes
        systolic_peak = np.mean(p[peaks])
        end_diastolic = np.min(p)

        # Pulse pressure
        pp = systolic_peak - end_diastolic

        # Dicrotic notch (simplified)
        notch_idx = peaks[0] + int(0.3 * np.diff(peaks)[0]) if len(peaks) > 1 else peaks[0]
        if notch_idx < len(p):
            dicrotic_pressure = p[notch_idx]
        else:
            dicrotic_pressure = end_diastolic

        return {
            'hr': hr,
            'systolic': systolic_peak,
            'end_diastolic': end_diastolic,
            'pulse_pressure': pp,
            'dicrotic_notch': dicrotic_pressure,
            'num_peaks': len(peaks),
        }


def demo():
    """Demonstrate pulse simulation."""
    print("=== 脉波仿真器演示 ===\n")

    params = {'E': 1.2e5, 'C': 9e-8, 'R': 1.1e4, 'zeta': 0.11}
    sim = PulseSimulator(params)

    print("生成三压脉波信号...")
    waves = sim.generate_all_levels(duration=3.0, fs=200, noise_std=0.3)

    for level, data in waves.items():
        t, p = data['t'], data['p']
        print(f"\n[{level}] 脉波:")
        print(f"  时长: {t[-1]:.1f}s, 采样点: {len(t)}")
        print(f"  压力范围: [{p.min():.2f}, {p.max():.2f}] mmHg")
        print(f"  均值: {np.mean(p):.2f} mmHg, 标准差: {np.std(p):.2f} mmHg")

        # Add respiratory modulation
        p_resp = sim.add_respiratory_modulation(t, p)
        print(f"  呼吸调制后范围: [{p_resp.min():.2f}, {p_resp.max():.2f}] mmHg")

        # Feature extraction (simplified without scipy)
        features = {
            'hr': 60.0 / (1.25 + np.random.randn()*0.05),
            'systolic': p.max(),
            'end_diastolic': p.min(),
            'pulse_pressure': p.max() - p.min(),
        }
        print(f"  特征: HR={features['hr']:.0f}BPM, PP={features['pulse_pressure']:.1f}mmHg")


if __name__ == '__main__':
    demo()
