"""
B-013: Acupuncture Robot Adaptive PID v2
针灸机器人自适应PID力控 — 根据组织刚度/深度自动调参

Innovation: Tissue-aware adaptive PID + safety envelope + acupoint-specific profiles
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
# 1. Acupoint Profile Database
# ═══════════════════════════════════════════════════════════
@dataclass
class AcupointProfile:
    """Per-acupoint force control parameters."""
    name: str
    target_force: float      # N - target insertion force
    max_force: float         # N - safety limit
    max_depth: float         # mm - maximum safe depth
    tissue_stiffness: float  # N/mm - expected tissue stiffness
    kp: float = 0.8
    ki: float = 0.1
    kd: float = 0.3


# Pre-defined acupoint profiles
ACUPOINT_PROFILES = {
    '合谷': AcupointProfile('合谷', 2.0, 5.0, 15.0, 3.0, 0.8, 0.1, 0.3),
    '足三里': AcupointProfile('足三里', 3.0, 8.0, 25.0, 4.0, 0.7, 0.15, 0.35),
    '三阴交': AcupointProfile('三阴交', 2.5, 6.0, 20.0, 3.5, 0.75, 0.12, 0.3),
    '太冲': AcupointProfile('太冲', 1.5, 4.0, 12.0, 2.5, 0.85, 0.08, 0.25),
    '内关': AcupointProfile('内关', 2.0, 5.0, 15.0, 3.0, 0.8, 0.1, 0.3),
    '百会': AcupointProfile('百会', 1.0, 3.0, 8.0, 2.0, 0.9, 0.05, 0.2),
    '气海': AcupointProfile('气海', 2.5, 6.0, 20.0, 3.5, 0.75, 0.12, 0.3),
    '关元': AcupointProfile('关元', 2.5, 6.0, 20.0, 3.5, 0.75, 0.12, 0.3),
}


# ═══════════════════════════════════════════════════════════
# 2. PID Controller with Anti-windup
# ═══════════════════════════════════════════════════════════
@dataclass
class PIDState:
    timestamp: float = 0.0
    error: float = 0.0
    integral: float = 0.0
    derivative: float = 0.0
    output: float = 0.0


class AdaptivePID:
    """
    PID controller with:
    - Anti-windup (integral clamping)
    - Derivative filtering (low-pass)
    - Output saturation
    - Online parameter adaptation
    """
    def __init__(self, kp=0.8, ki=0.1, kd=0.3,
                 output_min=-50.0, output_max=50.0,
                 integral_limit=20.0, derivative_filter=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_min = output_min
        self.output_max = output_max
        self.integral_limit = integral_limit
        self.derivative_filter = derivative_filter

        self._prev_error = 0.0
        self._integral = 0.0
        self._filtered_d = 0.0
        self._prev_time = None
        self.history: List[PIDState] = []

    def compute(self, target: float, current: float, timestamp: float = None) -> Tuple[float, PIDState]:
        if timestamp is None:
            import time; timestamp = time.time()

        error = target - current
        dt = 0.01 if self._prev_time is None else max(timestamp - self._prev_time, 0.001)

        # P
        p_term = self.kp * error

        # I with anti-windup
        self._integral += error * dt
        self._integral = max(-self.integral_limit, min(self.integral_limit, self._integral))
        i_term = self.ki * self._integral

        # D with low-pass filter
        raw_d = (error - self._prev_error) / dt if dt > 0 else 0
        self._filtered_d = self.derivative_filter * raw_d + (1 - self.derivative_filter) * self._filtered_d
        d_term = self.kd * self._filtered_d

        # Output with saturation
        output = max(self.output_min, min(self.output_max, p_term + i_term + d_term))

        state = PIDState(timestamp=timestamp, error=error, integral=self._integral,
                        derivative=self._filtered_d, output=output)
        self.history.append(state)

        self._prev_error = error
        self._prev_time = timestamp

        return output, state

    def reset(self):
        self._prev_error = 0.0
        self._integral = 0.0
        self._filtered_d = 0.0
        self._prev_time = None
        self.history.clear()

    def set_params(self, kp, ki, kd):
        self.kp, self.ki, self.kd = kp, ki, kd

    def steady_state_error(self, n=10) -> float:
        if len(self.history) < n:
            return float('inf')
        return abs(sum(s.error for s in self.history[-n:])) / n


# ═══════════════════════════════════════════════════════════
# 3. Tissue-Aware Adaptive Controller
# ═══════════════════════════════════════════════════════════
class TissueAdaptiveController:
    """
    Adaptive PID that adjusts parameters based on:
    - Tissue stiffness (硬组织→小KP, 软组织→大KP)
    - Insertion depth (深→减KI防积分饱和)
    - Acupoint-specific profiles
    """
    def __init__(self):
        self.pid = AdaptivePID()
        self.current_acupoint: Optional[AcupointProfile] = None
        self.safety_envelope = True

    def set_acupoint(self, acupoint_name: str):
        profile = ACUPOINT_PROFILES.get(acupoint_name)
        if profile:
            self.current_acupoint = profile
            self.pid.set_params(profile.kp, profile.ki, profile.kd)
        self.pid.reset()

    def adapt_params(self, tissue_stiffness: float, depth: float):
        """Adapt PID parameters based on tissue properties."""
        base = self.current_acupoint
        kp = base.kp if base else 0.8
        ki = base.ki if base else 0.1
        kd = base.kd if base else 0.3

        # Stiffness adaptation
        if tissue_stiffness > 10.0:     # 硬组织 (肌肉)
            kp *= 0.8
        elif tissue_stiffness < 3.0:    # 软组织 (脂肪)
            kp *= 1.2

        # Depth adaptation (reduce KI at depth to prevent windup)
        depth_factor = max(0.5, 1.0 - 0.01 * max(0, depth - 15))
        ki *= depth_factor

        self.pid.set_params(kp, ki, kd)

    def compute(self, target_force: float, current_force: float,
                depth: float, tissue_stiffness: float,
                timestamp: float = None) -> Tuple[float, PIDState]:
        """Compute control output with adaptation and safety checks."""
        self.adapt_params(tissue_stiffness, depth)

        output, state = self.pid.compute(target_force, current_force, timestamp)

        # Safety envelope
        if self.safety_envelope and self.current_acupoint:
            max_f = self.current_acupoint.max_force
            max_d = self.current_acupoint.max_depth

            if current_force > max_f:
                output = min(output, 0)  # stop pushing
                state.output = output

            if depth > max_d:
                output = min(output, 0)  # stop deeper
                state.output = output

        return output, state


# ═══════════════════════════════════════════════════════════
# 4. Simulation Environment
# ═══════════════════════════════════════════════════════════
class TissueSimulator:
    """Simulate needle-tissue interaction for testing."""
    def __init__(self, stiffness=3.0, damping=0.5, noise_std=0.1):
        self.stiffness = stiffness
        self.damping = damping
        self.noise = noise_std
        self.depth = 0.0
        self.force = 0.0

    def step(self, velocity_cmd: float, dt=0.01) -> Tuple[float, float]:
        """Simulate one timestep. Returns (force, depth)."""
        # Tissue resistance model
        resistance = self.stiffness * self.depth + self.damping * velocity_cmd
        self.force = max(0, velocity_cmd * 0.15 - resistance * 0.01 + np.random.randn() * self.noise)
        self.depth = max(0, self.depth + velocity_cmd * dt * 0.5)
        return self.force, self.depth

    def reset(self):
        self.depth = 0.0
        self.force = 0.0


# ═══════════════════════════════════════════════════════════
# 5. Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("B-013: Acupuncture Robot Adaptive PID v2")
    print("=" * 60)

    controller = TissueAdaptiveController()
    sim = TissueSimulator(stiffness=3.0, damping=0.5, noise_std=0.2)

    # Test with different acupoints
    for acupoint in ['合谷', '足三里', '百会']:
        print(f"\n[{acupoint}]")
        profile = ACUPOINT_PROFILES[acupoint]
        controller.set_acupoint(acupoint)
        sim.reset()

        target = profile.target_force
        print(f"  Target: {target}N, Max: {profile.max_force}N, Depth limit: {profile.max_depth}mm")

        for step in range(200):
            output, state = controller.compute(
                target, sim.force, sim.depth, profile.tissue_stiffness,
                timestamp=step * 0.01
            )
            sim.step(output)

        sse = controller.pid.steady_state_error()
        print(f"  Final force: {sim.force:.3f}N, Final depth: {sim.depth:.2f}mm, SSE: {sse:.4f}")

    # Parameter adaptation demo
    print("\n[Adaptation Demo]")
    controller.set_acupoint('足三里')
    controller.adapt_params(tissue_stiffness=15.0, depth=5.0)   # 硬组织
    print(f"  硬组织: KP={controller.pid.kp:.3f}, KI={controller.pid.ki:.3f}, KD={controller.pid.kd:.3f}")
    controller.adapt_params(tissue_stiffness=2.0, depth=5.0)    # 软组织
    print(f"  软组织: KP={controller.pid.kp:.3f}, KI={controller.pid.ki:.3f}, KD={controller.pid.kd:.3f}")
    controller.adapt_params(tissue_stiffness=5.0, depth=25.0)   # 深层
    print(f"  深层:   KP={controller.pid.kp:.3f}, KI={controller.pid.ki:.3f}, KD={controller.pid.kd:.3f}")

    print("\nDone!")
