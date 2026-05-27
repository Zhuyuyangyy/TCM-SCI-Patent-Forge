"""
八段锦康复处方推荐模块
根据动作评分+体质推荐运动强度/时长/频率
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# 中医体质分类
CONSTITUTION_TYPES = {
    "气虚质": {
        "description": "元气不足，容易疲劳",
        "suitable_movements": [0, 2, 7],  # 托天、单举、收势
        "intensity_modifier": 0.7,
        "contraindications": ["摇头摆尾"],
    },
    "阳虚质": {
        "description": "阳气不足，畏寒怕冷",
        "suitable_movements": [0, 5, 7],
        "intensity_modifier": 0.7,
        "contraindications": [],
    },
    "阴虚质": {
        "description": "阴液不足，口干咽燥",
        "suitable_movements": [2, 3, 6],
        "intensity_modifier": 0.8,
        "contraindications": ["攒拳怒目"],
    },
    "痰湿质": {
        "description": "痰湿内盛，体形肥胖",
        "suitable_movements": [1, 4, 6],
        "intensity_modifier": 1.0,
        "contraindications": [],
    },
    "湿热质": {
        "description": "湿热内蕴，面油长痘",
        "suitable_movements": [3, 4, 6],
        "intensity_modifier": 1.0,
        "contraindications": [],
    },
    "血瘀质": {
        "description": "血行不畅，肤色晦暗",
        "suitable_movements": [1, 4, 5],
        "intensity_modifier": 0.9,
        "contraindications": [],
    },
    "气郁质": {
        "description": "气机郁结，情绪低落",
        "suitable_movements": [0, 1, 5],
        "intensity_modifier": 0.85,
        "contraindications": [],
    },
    "特禀质": {
        "description": "先天特异，易过敏",
        "suitable_movements": [0, 2, 7],
        "intensity_modifier": 0.75,
        "contraindications": [],
    },
    "平和质": {
        "description": "阴阳平衡，健康体质",
        "suitable_movements": [0, 1, 2, 3, 4, 5, 6, 7],
        "intensity_modifier": 1.0,
        "contraindications": [],
    },
}


# 运动处方基础参数
BASE_PRESCRIPTION = {
    "frequency": {
        "min_times_per_week": 3,
        "max_times_per_week": 7,
        "recommended_times_per_week": 5,
    },
    "duration": {
        "min_minutes": 15,
        "max_minutes": 60,
        "per_movement_seconds": 30,  # 每个动作建议时长
    },
    "intensity": {
        "low": {"hr_percent": 50, "borg_scale": 6},
        "medium": {"hr_percent": 60, "borg_scale": 8},
        "high": {"hr_percent": 70, "borg_scale": 10},
    },
}


class RehabPrescriber:
    """康复处方推荐器"""
    
    def __init__(self, constitution: str = "平和质"):
        self.constitution = constitution
        self.constitution_info = CONSTITUTION_TYPES.get(
            constitution, CONSTITUTION_TYPES["平和质"]
        )
        
    def recommend_prescription(self, 
                                movement_scores: Dict[int, float],
                                age: int = 60,
                                has_injury: bool = False,
                                injury_type: Optional[str] = None) -> Dict:
        """
        生成康复处方
        movement_scores: {动作ID: 评分}
        """
        # 计算平均得分
        if movement_scores:
            avg_score = np.mean(list(movement_scores.values()))
            min_score = min(movement_scores.values())
        else:
            avg_score = 70
            min_score = 70
        
        # 确定运动强度
        intensity_level = self._determine_intensity(avg_score, age)
        
        # 计算运动参数
        base_duration = self._calculate_duration(avg_score, min_score)
        frequency = self._calculate_frequency(avg_score)
        repetitions = self._calculate_repetitions(avg_score, intensity_level)
        
        # 推荐动作序列
        recommended_movements = self._recommend_movements(movement_scores)
        
        # 禁忌检查
        contraindications = self._check_contraindications()
        
        # 生成处方
        prescription = {
            "patient_info": {
                "constitution": self.constitution,
                "constitution_desc": self.constitution_info["description"],
                "age": age,
                "has_injury": has_injury,
                "injury_type": injury_type,
            },
            "exercise_prescription": {
                "frequency": {
                    "times_per_week": frequency,
                    "rest_days": 7 - frequency,
                },
                "duration": {
                    "total_minutes": base_duration,
                    "per_movement_seconds": 30,
                    "sets": max(1, frequency // 3),
                },
                "intensity": {
                    "level": intensity_level,
                    "target_hr_percent": BASE_PRESCRIPTION["intensity"][intensity_level]["hr_percent"],
                    "borg_scale": BASE_PRESCRIPTION["intensity"][intensity_level]["borg_scale"],
                },
                "repetitions_per_movement": repetitions,
            },
            "recommended_movements": recommended_movements,
            "contraindications": contraindications,
            "warnings": self._generate_warnings(has_injury, injury_type),
            "tips": self._generate_tips(avg_score, intensity_level),
        }
        
        return prescription
    
    def _determine_intensity(self, avg_score: float, age: int) -> str:
        """确定运动强度"""
        # 基础强度根据评分确定
        if avg_score >= 85:
            base_intensity = "high"
        elif avg_score >= 70:
            base_intensity = "medium"
        else:
            base_intensity = "low"
        
        # 根据体质调整
        modifier = self.constitution_info["intensity_modifier"]
        
        if modifier < 0.75:
            # 虚弱体质，降低强度
            if base_intensity == "high":
                return "medium"
            elif base_intensity == "medium":
                return "low"
        
        # 根据年龄调整 (假设60岁以上需要降低)
        if age > 65 and base_intensity == "high":
            return "medium"
        
        return base_intensity
    
    def _calculate_duration(self, avg_score: float, min_score: float) -> int:
        """计算运动时长(分钟)"""
        base = BASE_PRESCRIPTION["duration"]["min_minutes"]
        
        # 评分高可增加时长
        if avg_score >= 85:
            duration = base + 20
        elif avg_score >= 75:
            duration = base + 15
        elif avg_score >= 65:
            duration = base + 10
        else:
            duration = base
        
        # 体质调整
        modifier = self.constitution_info["intensity_modifier"]
        duration = int(duration * modifier)
        
        return max(BASE_PRESCRIPTION["duration"]["min_minutes"], 
                   min(BASE_PRESCRIPTION["duration"]["max_minutes"], duration))
    
    def _calculate_frequency(self, avg_score: float) -> int:
        """计算运动频率(每周次数)"""
        base = BASE_PRESCRIPTION["frequency"]["recommended_times_per_week"]
        
        if avg_score >= 80:
            return min(BASE_PRESCRIPTION["frequency"]["max_times_per_week"], base + 1)
        elif avg_score >= 65:
            return base
        else:
            return max(BASE_PRESCRIPTION["frequency"]["min_times_per_week"], base - 1)
    
    def _calculate_repetitions(self, avg_score: float, intensity: str) -> int:
        """计算每个动作的重复次数"""
        if intensity == "low":
            base_rep = 4
        elif intensity == "medium":
            base_rep = 6
        else:
            base_rep = 8
        
        # 评分高可增加次数
        if avg_score >= 85:
            return base_rep + 2
        elif avg_score >= 70:
            return base_rep + 1
        else:
            return max(2, base_rep - 1)
    
    def _recommend_movements(self, movement_scores: Dict[int, float]) -> List[Dict]:
        """推荐动作序列"""
        suitable = self.constitution_info["suitable_movements"]
        
        recommendations = []
        for mid in suitable:
            if mid in movement_scores:
                score = movement_scores[mid]
            else:
                score = 70  # 默认分数
            
            # 分数高且适合该体质的动作优先推荐
            priority = score * self.constitution_info["intensity_modifier"]
            
            recommendations.append({
                "movement_id": mid,
                "name": self._get_movement_name(mid),
                "score": score,
                "priority": priority,
                "suggestion": self._get_movement_suggestion(mid, score),
            })
        
        # 按优先级排序
        recommendations.sort(key=lambda x: x["priority"], reverse=True)
        
        return recommendations[:4]  # 最多推荐4个动作
    
    def _get_movement_name(self, mid: int) -> str:
        names = [
            "双手托天理三焦",
            "左右开弓似射雕",
            "调理脾胃须单举",
            "五劳七伤往后瞧",
            "摇头摆尾去心火",
            "攒拳怒目增气力",
            "背后七颠百病消",
            "收势",
        ]
        return names[mid] if 0 <= mid < len(names) else "未知"
    
    def _get_movement_suggestion(self, mid: int, score: float) -> str:
        """获取动作具体建议"""
        if score >= 85:
            return "保持当前水平，注意细节"
        elif score >= 70:
            return "继续练习，巩固动作"
        elif score >= 55:
            return "加强练习，注意规范"
        else:
            return "从基础开始，循序渐进"
    
    def _check_contraindications(self) -> List[str]:
        """检查禁忌动作"""
        return self.constitution_info.get("contraindications", [])
    
    def _generate_warnings(self, has_injury: bool, injury_type: Optional[str]) -> List[str]:
        """生成注意事项"""
        warnings = []
        
        if has_injury:
            if injury_type:
                warnings.append(f"注意保护{injury_type}，如有不适立即停止")
            else:
                warnings.append("如有不适立即停止运动")
        
        if self.constitution in ["气虚质", "阳虚质"]:
            warnings.append("运动后注意保暖，避免受凉")
        elif self.constitution in ["阴虚质"]:
            warnings.append("避免在高温环境下练习，及时补充水分")
        
        return warnings
    
    def _generate_tips(self, avg_score: float, intensity: str) -> List[str]:
        """生成练习提示"""
        tips = []
        
        if avg_score < 70:
            tips.append("建议从单式动作开始，逐步过渡到完整套路")
        
        if intensity == "low":
            tips.append("以舒展柔和为主，不必追求动作幅度")
        
        tips.append("练习前做好热身，练习后适当放松")
        tips.append("保持呼吸自然，与动作协调配合")
        
        return tips


def generate_prescription_report(prescription: Dict) -> str:
    """生成康复处方报告"""
    lines = []
    lines.append("=" * 50)
    lines.append("       八段锦康复处方")
    lines.append("=" * 50)
    lines.append("")
    
    # 患者信息
    lines.append("【患者信息】")
    lines.append(f"  体质类型: {prescription['patient_info']['constitution']}")
    lines.append(f"  体质描述: {prescription['patient_info']['constitution_desc']}")
    lines.append(f"  年龄: {prescription['patient_info']['age']}岁")
    if prescription['patient_info']['has_injury']:
        lines.append(f"  伤情: {prescription['patient_info']['injury_type']}")
    lines.append("")
    
    # 运动处方
    lines.append("【运动处方】")
    rx = prescription['exercise_prescription']
    lines.append(f"  频率: 每周{rx['frequency']['times_per_week']}次")
    lines.append(f"  时长: 每次{rx['duration']['total_minutes']}分钟")
    lines.append(f"  强度: {rx['intensity']['level']} (靶心率{rx['intensity']['target_hr_percent']}%)")
    lines.append(f"  重复: 每个动作{rx['repetitions_per_movement']}次")
    lines.append("")
    
    # 推荐动作
    lines.append("【推荐动作】")
    for i, mv in enumerate(prescription['recommended_movements'], 1):
        lines.append(f"  {i}. {mv['name']} (得分:{mv['score']:.0f})")
        lines.append(f"     建议: {mv['suggestion']}")
    lines.append("")
    
    # 禁忌
    if prescription['contraindications']:
        lines.append("【禁忌动作】")
        for c in prescription['contraindications']:
            lines.append(f"  - {c}")
        lines.append("")
    
    # 警告
    if prescription['warnings']:
        lines.append("【注意事项】")
        for w in prescription['warnings']:
            lines.append(f"  ! {w}")
        lines.append("")
    
    # 提示
    lines.append("【练习提示】")
    for tip in prescription['tips']:
        lines.append(f"  • {tip}")
    
    lines.append("")
    lines.append("=" * 50)
    
    return "\n".join(lines)


def quick_prescribe(constitution: str, avg_score: float) -> Dict:
    """快速处方接口"""
    prescriber = RehabPrescriber(constitution)
    return prescriber.recommend_prescription({0: avg_score})


if __name__ == "__main__":
    print("=== 康复处方推荐测试 ===")
    print()
    
    # 测试各体质
    constitutions = ["气虚质", "痰湿质", "平和质"]
    
    for const in constitutions:
        prescriber = RehabPrescriber(constitution=const)
        scores = {
            0: 78,  # 双手托天
            1: 65,  # 左右开弓
            2: 82,  # 调理脾胃
            3: 70,  # 五劳七伤
        }
        
        prescription = prescriber.recommend_prescription(
            movement_scores=scores,
            age=62,
            has_injury=False,
        )
        
        report = generate_prescription_report(prescription)
        print(report)
        print()
