"""
八段锦动作质量评分模块
基于偏差幅度/稳定性/节奏感三个维度评分(100分制)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from movement_db import MOVEMENT_TEMPLATES


class ScoreEvaluator:
    """动作质量评分器"""
    
    def __init__(self, movement_id: int):
        self.movement_id = movement_id
        self.template_info = MOVEMENT_TEMPLATES.get(movement_id, {})
        self.difficulty = self.template_info.get("difficulty", 2)
        
        # 评分权重
        self.weights = {
            "deviation": 0.50,
            "stability": 0.30,
            "rhythm": 0.20,
        }
        
        # 历史记录
        self.frame_scores = []
        self.max_history = 100
        
    def evaluate_deviation(self, deviations: Dict[str, float]) -> float:
        if not deviations:
            return 100.0
        
        joint_scores = []
        for joint, deviation in deviations.items():
            abs_dev = abs(deviation)
            tolerance = self._get_tolerance(joint)
            
            if abs_dev <= tolerance:
                score = 100.0
            else:
                excess = abs_dev - tolerance
                score = max(0, 100 - (excess / tolerance) * 100)
            
            joint_scores.append(score)
        
        return np.mean(joint_scores)
    
    def _get_tolerance(self, joint: str) -> float:
        base_tolerances = {
            "l_shoulder_angle": 15,
            "r_shoulder_angle": 15,
            "l_elbow_angle": 10,
            "r_elbow_angle": 10,
            "spine_angle": 10,
            "l_arm_raise": 0.15,
            "r_arm_raise": 0.15,
            "l_hip_angle": 15,
            "r_hip_angle": 15,
            "l_knee_angle": 10,
            "r_knee_angle": 10,
        }
        
        base = base_tolerances.get(joint, 10)
        return base * (1.1 - self.difficulty * 0.05)
    
    def evaluate_stability(self, features_history: List[Dict[str, float]]) -> float:
        if len(features_history) < 3:
            return 50.0
        
        stability_scores = []
        joints = features_history[0].keys()
        
        for joint in joints:
            values = [f[joint] for f in features_history if joint in f]
            if len(values) < 3:
                continue
            
            mean_val = np.mean(values)
            std_val = np.std(values)
            
            if abs(mean_val) > 1e-6:
                cv = abs(std_val / mean_val)
            else:
                cv = std_val
            
            if cv < 0.02:
                score = 100.0
            elif cv > 0.2:
                score = 0.0
            else:
                score = 100 * (1 - (cv - 0.02) / 0.18)
            
            stability_scores.append(score)
        
        return np.mean(stability_scores) if stability_scores else 50.0
    
    def evaluate_rhythm(self, timestamps: List[float], 
                       expected_duration: Optional[float] = None) -> float:
        if len(timestamps) < 2:
            return 50.0
        
        intervals = np.diff(timestamps)
        interval_std = np.std(intervals)
        interval_mean = np.mean(intervals)
        
        if interval_mean > 0:
            cv = interval_std / interval_mean
        else:
            cv = 1.0
        
        if cv < 0.1:
            rhythm_score = 100.0
        elif cv > 0.5:
            rhythm_score = 0.0
        else:
            rhythm_score = 100 * (1 - (cv - 0.1) / 0.4)
        
        if expected_duration is not None:
            total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
            if total_duration > 0:
                duration_deviation = abs(total_duration - expected_duration) / expected_duration
                if duration_deviation < 0.1:
                    duration_score = 100.0
                elif duration_deviation > 0.5:
                    duration_score = 0.0
                else:
                    duration_score = 100 * (1 - (duration_deviation - 0.1) / 0.4)
                rhythm_score = (rhythm_score + duration_score) / 2
        
        return rhythm_score
    
    def evaluate_comprehensive(self, deviations: Dict[str, float],
                              features_history: List[Dict[str, float]],
                              timestamps: Optional[List[float]] = None) -> Dict:
        deviation_score = self.evaluate_deviation(deviations)
        stability_score = self.evaluate_stability(features_history)
        expected_duration = self._get_expected_duration()
        
        if timestamps:
            rhythm_score = self.evaluate_rhythm(timestamps, expected_duration)
        else:
            rhythm_score = 50.0
        
        total_score = (
            deviation_score * self.weights["deviation"] +
            stability_score * self.weights["stability"] +
            rhythm_score * self.weights["rhythm"]
        )
        
        self.frame_scores.append({
            "deviation": deviation_score,
            "stability": stability_score,
            "rhythm": rhythm_score,
            "total": total_score,
        })
        
        if len(self.frame_scores) > self.max_history:
            self.frame_scores.pop(0)
        
        return {
            "movement_id": self.movement_id,
            "movement_name": self.template_info.get("name", ""),
            "difficulty": self.difficulty,
            "deviation_score": deviation_score,
            "stability_score": stability_score,
            "rhythm_score": rhythm_score,
            "total_score": total_score,
            "grade": self._score_to_grade(total_score),
            "suggestion": self._get_suggestion(deviation_score, stability_score, rhythm_score),
        }
    
    def _get_expected_duration(self) -> float:
        base_durations = {1: 4.0, 2: 6.0, 3: 8.0, 4: 10.0, 5: 12.0}
        return base_durations.get(self.difficulty, 6.0)
    
    def _score_to_grade(self, score: float) -> str:
        if score >= 95: return "S"
        elif score >= 85: return "A"
        elif score >= 75: return "B"
        elif score >= 65: return "C"
        elif score >= 55: return "D"
        else: return "E"
    
    def _get_suggestion(self, dev_score: float, stab_score: float, 
                       rhythm_score: float) -> str:
        suggestions = []
        
        if dev_score < 70:
            suggestions.append("注意保持动作标准")
        if stab_score < 70:
            suggestions.append("保持身体稳定性")
        if rhythm_score < 70:
            suggestions.append("调整动作节奏")
        
        if not suggestions:
            return "动作完成良好，继续保持"
        
        return "，".join(suggestions)
    
    def get_summary(self) -> Dict:
        if not self.frame_scores:
            return {}
        
        total_scores = [f["total"] for f in self.frame_scores]
        
        return {
            "count": len(self.frame_scores),
            "avg_score": np.mean(total_scores),
            "max_score": np.max(total_scores),
            "min_score": np.min(total_scores),
            "latest_score": total_scores[-1],
        }


def quick_score(deviations: Dict[str, float], 
               movement_id: int) -> Tuple[float, str]:
    evaluator = ScoreEvaluator(movement_id)
    result = evaluator.evaluate_comprehensive(deviations, [])
    return result["total_score"], result["grade"]


def generate_score_report(eval_result: Dict) -> str:
    lines = []
    lines.append("=" * 40)
    lines.append("    八段锦动作质量评分报告")
    lines.append("=" * 40)
    lines.append("动作名称: " + str(eval_result['movement_name']))
    lines.append("难度等级: " + str(eval_result['difficulty']))
    lines.append("")
    lines.append("【分项得分】")
    lines.append("  偏差幅度: {:.1f}/100".format(eval_result['deviation_score']))
    lines.append("  稳定性:   {:.1f}/100".format(eval_result['stability_score']))
    lines.append("  节奏感:   {:.1f}/100".format(eval_result['rhythm_score']))
    lines.append("")
    lines.append("【综合评分】")
    lines.append("  总分: {:.1f}/100".format(eval_result['total_score']))
    lines.append("  等级: " + str(eval_result['grade']))
    lines.append("")
    lines.append("【建议】: " + str(eval_result['suggestion']))
    lines.append("=" * 40)
    return "\n".join(lines)


if __name__ == "__main__":
    print("=== 动作质量评分测试 ===")
    print()
    
    evaluator = ScoreEvaluator(movement_id=0)
    
    features_history = []
    for i in range(20):
        features = {
            "l_shoulder_angle": 160 + np.random.randn() * 3,
            "r_shoulder_angle": 162 + np.random.randn() * 3,
            "l_elbow_angle": 177 + np.random.randn() * 2,
            "r_elbow_angle": 178 + np.random.randn() * 2,
            "l_arm_raise": 0.95 + np.random.randn() * 0.05,
            "r_arm_raise": 0.93 + np.random.randn() * 0.05,
            "spine_angle": 2 + np.random.randn() * 2,
        }
        features_history.append(features)
    
    current_deviations = {
        "l_shoulder_angle": -5,
        "r_shoulder_angle": -3,
        "l_elbow_angle": -3,
        "r_elbow_angle": -2,
        "l_arm_raise": -0.05,
        "r_arm_raise": -0.07,
        "spine_angle": 2,
    }
    
    result = evaluator.evaluate_comprehensive(
        deviations=current_deviations,
        features_history=features_history,
    )
    
    print(generate_score_report(result))
    print()
    
    print("=== 较差姿态测试 ===")
    poor_features = [{"l_shoulder_angle": 120 + np.random.randn() * 20} for _ in range(10)]
    poor_deviations = {"l_shoulder_angle": 40}
    result2 = evaluator.evaluate_comprehensive(poor_deviations, poor_features)
    print("较差姿态得分: {:.1f}, 等级: {}".format(result2['total_score'], result2['grade']))
