"""
fire_judge_model.py - Fire Control Intelligent Judgment Model
Hybrid judgment system combining rule-based and neural network approaches
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class FireJudgmentResult:
    """Fire judgment result"""
    is_qualified: bool
    fire_level: str
    score: float
    confidence: float
    reasons: List[str]
    suggestions: List[str]
    temp_deviation: float
    duration_deviation: float
    phase_analysis: Dict[str, dict]


class RuleBasedJudge:
    """Rule-based fire judgment"""

    def __init__(self, tolerance_temp: float = 10.0, tolerance_time: float = 0.1):
        self.tolerance_temp = tolerance_temp
        self.tolerance_time = tolerance_time

    def judge(
        self,
        actual_curve: List[float],
        time_points: List[float],
        target_temp: float,
        target_duration: float,
        phases: Optional[List[dict]] = None
    ) -> dict:
        if not actual_curve or not time_points:
            return {"qualified": False, "score": 0, "reasons": ["No temperature data"]}

        temps = np.array(actual_curve)
        times = np.array(time_points) / 60.0

        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        max_temp = np.max(temps)
        min_temp = np.min(temps)

        temp_deviation = abs(mean_temp - target_temp)

        duration_in_range = np.sum((temps >= target_temp - self.tolerance_temp) &
                                   (temps <= target_temp + self.tolerance_temp))
        time_ratio = duration_in_range / len(temps)

        overheat_ratio = np.sum(temps > target_temp + self.tolerance_temp * 2) / len(temps)
        underheat_ratio = np.sum(temps < target_temp - self.tolerance_temp * 2) / len(temps)

        score = 100
        score -= min(30, temp_deviation * 0.5)
        score -= min(20, std_temp * 2)

        if time_ratio < 0.8:
            score -= (0.8 - time_ratio) * 50

        if overheat_ratio > 0.1:
            score -= overheat_ratio * 100

        if underheat_ratio > 0.2:
            score -= underheat_ratio * 50

        score = max(0, score)

        if score >= 85 and overheat_ratio < 0.05:
            level = "qualified"
        elif score >= 70:
            level = "acceptable"
        elif overheat_ratio > 0.15:
            level = "overheat"
        elif underheat_ratio > 0.3:
            level = "insufficient"
        else:
            level = "failed"

        reasons = []
        if temp_deviation > self.tolerance_temp:
            reasons.append(f"Temperature deviation too large: {temp_deviation:.1f} C")
        if std_temp > 10:
            reasons.append(f"Temperature fluctuation too large: std {std_temp:.1f} C")
        if overheat_ratio > 0.1:
            reasons.append(f"Overheat detected: {overheat_ratio*100:.1f}% time over temp")
        if underheat_ratio > 0.2:
            reasons.append(f"Insufficient holding time: only {time_ratio*100:.1f}% time qualified")

        return {
            "qualified": level in ["qualified", "acceptable"],
            "score": round(score, 2),
            "level": level,
            "mean_temp": round(mean_temp, 2),
            "std_temp": round(std_temp, 2),
            "temp_deviation": round(temp_deviation, 2),
            "time_ratio": round(time_ratio, 4),
            "overheat_ratio": round(overheat_ratio, 4),
            "underheat_ratio": round(underheat_ratio, 4),
            "reasons": reasons,
            "max_temp": round(max_temp, 2),
            "min_temp": round(min_temp, 2)
        }


class NeuralFireJudge:
    """Neural network based fire judgment (simplified MLP)"""

    def __init__(self):
        self.input_dim = 10
        self.hidden_dim = 32
        self.output_dim = 5

        np.random.seed(42)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim) * 0.1
        self.b1 = np.zeros(self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim) * 0.1
        self.b2 = np.zeros(self.output_dim)

        self.feature_mean = np.array([100, 50, 20, 5, 0.5, 0.3, 0.8, 120, 80, 150])
        self.feature_std = np.array([50, 30, 15, 5, 0.3, 0.2, 0.2, 30, 20, 50])

    def extract_features(self, curve: List[float], times: List[float]) -> np.ndarray:
        temps = np.array(curve)

        mean_temp = np.mean(temps)
        std_temp = np.std(temps)
        max_temp = np.max(temps)
        min_temp = np.min(temps)

        temp_diff = np.diff(temps)
        mean_diff = np.mean(np.abs(temp_diff))
        max_diff = np.max(np.abs(temp_diff))

        cv = std_temp / mean_temp if mean_temp > 0 else 0

        target_range = mean_temp * 0.2
        in_range = np.sum((temps > mean_temp - target_range) &
                         (temps < mean_temp + target_range)) / len(temps)

        high_temp_ratio = np.sum(temps > mean_temp * 0.9) / len(temps)

        features = np.array([
            mean_temp, std_temp, max_temp - min_temp, mean_diff, max_diff,
            cv, in_range, high_temp_ratio, min_temp, max_temp
        ])

        return features

    def _relu(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def predict(self, curve: List[float], times: List[float]) -> dict:
        features = self.extract_features(curve, times)
        features_norm = (features - self.feature_mean) / (self.feature_std + 1e-8)

        hidden = self._relu(np.dot(features_norm, self.W1) + self.b1)
        output = np.dot(hidden, self.W2) + self.b2
        probs = self._softmax(output)

        levels = ["failed", "poor", "acceptable", "good", "excellent"]
        pred_idx = np.argmax(probs)

        return {
            "level": levels[pred_idx],
            "probabilities": probs.tolist(),
            "confidence": float(probs[pred_idx]),
            "features": features.tolist()
        }


class HybridFireJudge:
    """Hybrid fire judgment system (rule + neural network)"""

    def __init__(self):
        self.rule_judge = RuleBasedJudge()
        self.nn_judge = NeuralFireJudge()
        self.weights = {"rule": 0.6, "neural": 0.4}

    def judge(
        self,
        herb_name: str,
        processing_method: str,
        actual_curve: List[float],
        time_points: List[float],
        target_temp: float,
        target_duration: float
    ) -> FireJudgmentResult:
        rule_result = self.rule_judge.judge(
            actual_curve, time_points, target_temp, target_duration
        )

        nn_result = self.nn_judge.predict(actual_curve, time_points)

        rule_score = rule_result["score"]
        nn_score = nn_result["probabilities"][3] * 100 + nn_result["probabilities"][4] * 100

        final_score = self.weights["rule"] * rule_score + self.weights["neural"] * nn_score

        if final_score >= 85:
            fire_level = "qualified"
        elif final_score >= 70:
            fire_level = "acceptable"
        elif nn_result["level"] == "excellent" and rule_score >= 60:
            fire_level = "qualified"
        elif rule_result.get("overheat_ratio", 0) > 0.15:
            fire_level = "overheat"
        elif rule_result.get("underheat_ratio", 0) > 0.3:
            fire_level = "insufficient"
        else:
            fire_level = "failed"

        suggestions = []
        if rule_result.get("temp_deviation", 0) > 10:
            suggestions.append("Recommend adjusting fire power to approach target temperature")
        if rule_result.get("std_temp", 0) > 10:
            suggestions.append("Temperature fluctuation is large, recommend keeping fire stable")
        if rule_result.get("overheat_ratio", 0) > 0.1:
            suggestions.append("Overheat detected, should reduce fire power or shorten holding time")
        if rule_result.get("underheat_ratio", 0) > 0.2:
            suggestions.append("Insufficient holding time, recommend extending processing time or increasing fire power")
        if final_score >= 85:
            suggestions.append("Fire control is good, processing quality is qualified")

        phase_analysis = self._analyze_phases(actual_curve, time_points, target_temp)

        return FireJudgmentResult(
            is_qualified=final_score >= 70,
            fire_level=fire_level,
            score=round(final_score, 2),
            confidence=nn_result["confidence"],
            reasons=rule_result.get("reasons", []),
            suggestions=suggestions,
            temp_deviation=rule_result.get("temp_deviation", 0),
            duration_deviation=0,
            phase_analysis=phase_analysis
        )

    def _analyze_phases(
        self,
        curve: List[float],
        times: List[float],
        target_temp: float
    ) -> Dict[str, dict]:
        temps = np.array(curve)
        times_min = np.array(times) / 60.0
        duration = times_min[-1] if len(times_min) > 0 else 1

        phases = {
            "Preheating": {"score": 100, "analysis": ""},
            "Heating": {"score": 100, "analysis": ""},
            "Holding": {"score": 100, "analysis": ""},
            "Cooling": {"score": 100, "analysis": ""}
        }

        n = len(temps)
        if n < 4:
            return phases

        phase_ranges = [
            (0, int(n * 0.1), "Preheating"),
            (int(n * 0.1), int(n * 0.25), "Heating"),
            (int(n * 0.25), int(n * 0.75), "Holding"),
            (int(n * 0.75), n, "Cooling")
        ]

        for start, end, name in phase_ranges:
            if start >= n:
                continue
            end = min(end, n)
            phase_temps = temps[start:end]

            phase_mean = np.mean(phase_temps)
            phase_std = np.std(phase_temps)

            if name == "Holding":
                deviation = abs(phase_mean - target_temp)
                stability_score = max(0, 100 - deviation * 2 - phase_std * 3)
            else:
                stability_score = max(0, 100 - phase_std * 5)

            phases[name] = {
                "score": round(stability_score, 1),
                "mean_temp": round(phase_mean, 1),
                "std_temp": round(phase_std, 2),
                "analysis": self._get_phase_analysis(name, stability_score, phase_mean, target_temp)
            }

        return phases

    def _get_phase_analysis(self, phase_name: str, score: float, mean_temp: float, target: float) -> str:
        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 60:
            status = "Average"
        else:
            status = "Needs improvement"

        deviation = mean_temp - target
        dev_str = f"+{deviation:.1f}C" if deviation > 0 else f"{deviation:.1f}C" if deviation < 0 else ""

        return f"{status} {dev_str}"


def judge_fire(
    herb_name: str,
    processing_method: str,
    actual_curve: List[float],
    time_points: List[float],
    target_temp: float,
    target_duration: float
) -> FireJudgmentResult:
    """Quick judgment function"""
    judge = HybridFireJudge()
    return judge.judge(
        herb_name, processing_method,
        actual_curve, time_points,
        target_temp, target_duration
    )


if __name__ == "__main__":
    print("=== Fire Judgment System Test ===")

    sim_times = list(range(0, 3600, 1))
    sim_temps = [120 + np.random.randn() * 3 for _ in sim_times]

    result = judge_fire(
        herb_name="Aconite",
        processing_method="Processed Aconite",
        actual_curve=sim_temps,
        time_points=sim_times,
        target_temp=120,
        target_duration=240
    )

    print(f"Judgment: {result.fire_level}")
    print(f"Score: {result.score}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Qualified: {result.is_qualified}")
    print(f"Reasons: {result.reasons}")
    print(f"Suggestions: {result.suggestions}")
