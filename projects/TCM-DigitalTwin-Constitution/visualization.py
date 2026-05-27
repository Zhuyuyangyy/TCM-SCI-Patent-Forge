#!/usr/bin/env python3
"""D01 Visualization - Text-based Radar/Trajectory/Risk Charts"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ConstitutionRadarChart:
    """Text-based radar chart for constitution distribution."""

    CONSTITUTION_LABELS = ['气虚', '阳虚', '阴虚', '痰湿', '湿热', '血瘀', '气郁', '特禀', '平和']

    @staticmethod
    def plot(state_vector, labels=None):
        if labels is None:
            labels = ConstitutionRadarChart.CONSTITUTION_LABELS
        print("\n  [体质雷达图]")
        print("  " + "-" * 52)
        for label, val in zip(labels, state_vector):
            bar_len = int(val * 30)
            bar = "█" * bar_len
            print(f"  {label:6s} | {bar:<30s} {val:.3f}")
        print("  " + "-" * 52)


class ConstitutionTrajectoryChart:
    """Text-based trajectory chart for constitution evolution."""

    @staticmethod
    def plot(trajectory, title="体质演化轨迹"):
        print(f"\n  [{title}]")
        print("  " + "-" * 52)
        print(f"  {'时间点':^8s} | {'主导体质':^10s} | {'阴阳指数':^10s} | {'气血指数':^10s}")
        print("  " + "-" * 52)
        for i, (dominant, yin_yang, qi_blood) in enumerate(trajectory):
            print(f"  T{i:<6d} | {dominant:^10s} | {yin_yang:^10.3f} | {qi_blood:^10.3f}")
        print("  " + "-" * 52)


class RiskChart:
    """Text-based risk comparison chart."""

    RISK_LABELS = ['代谢综合征', '心血管', '免疫下降', '肿瘤风险', '亚健康']

    @staticmethod
    def plot(risk_levels, labels=None):
        if labels is None:
            labels = RiskChart.RISK_LABELS
        print("\n  [风险评估]")
        print("  " + "-" * 52)
        for label, val in zip(labels, risk_levels):
            bar_len = int(val * 30)
            bar = "█" * bar_len
            risk_tag = "低" if val < 0.3 else ("中" if val < 0.6 else "高")
            print(f"  {label:8s} | {bar:<30s} {val:.3f} [{risk_tag}]")
        print("  " + "-" * 52)


class DashboardGenerator:
    """Generate comprehensive text-based dashboard report."""

    @staticmethod
    def generate(state, trajectory, intervention_comparison, risk_levels):
        print("\n" + "=" * 60)
        print("  中医体质数字孪生 - 综合报告")
        print("=" * 60)

        # Radar chart
        ConstitutionRadarChart.plot(state.constitution_vector)

        # Trajectory
        ConstitutionTrajectoryChart.plot(trajectory)

        # Risk chart
        RiskChart.plot(risk_levels)

        # Intervention comparison
        print("\n  [干预效果对比]")
        print("  " + "-" * 52)
        for name, effect in intervention_comparison.items():
            bar_len = int(effect * 30)
            bar = "█" * bar_len
            print(f"  {name:10s} | {bar:<30s} {effect:.3f}")
        print("  " + "-" * 52)

        print("\n" + "=" * 60)
