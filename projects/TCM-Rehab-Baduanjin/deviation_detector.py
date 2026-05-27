"""
八段锦动作纠偏检测模块
实时对比当前姿态与标准模板，计算偏差角度，输出纠偏指令
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from movement_db import get_template, MOVEMENT_TEMPLATES, is_within_range


# 纠偏指令映射
CORRECTION_COMMANDS = {
    "l_shoulder_angle": {
        "low": "左肩下沉",
        "high": "左肩抬起",
    },
    "r_shoulder_angle": {
        "low": "右肩下沉",
        "high": "右肩抬起",
    },
    "l_elbow_angle": {
        "low": "左肘伸直",
        "high": "左肘微屈",
    },
    "r_elbow_angle": {
        "low": "右肘伸直",
        "high": "右肘微屈",
    },
    "spine_angle": {
        "low": "脊柱向左调整",
        "high": "脊柱向右调整",
    },
    "l_arm_raise": {
        "low": "左臂抬高",
        "high": "左臂降低",
    },
    "r_arm_raise": {
        "low": "右臂抬高",
        "high": "右臂降低",
    },
    "l_hip_angle": {
        "low": "左髋前送",
        "high": "左髋后收",
    },
    "r_hip_angle": {
        "low": "右髋前送",
        "high": "右髋后收",
    },
    "l_knee_angle": {
        "low": "左膝伸直",
        "high": "左膝微屈",
    },
    "r_knee_angle": {
        "low": "右膝伸直",
        "high": "右膝微屈",
    },
}


class DeviationDetector:
    """动作偏差检测器"""
    
    def __init__(self, movement_id: int, phase: str = None):
        self.movement_id = movement_id
        self.phase = phase
        self.template = get_template(movement_id, phase) if phase else None
        self.history = []  # 存储历史偏差用于稳定性分析
        self.max_history = 30  # 保留最近30帧
        
    def update_phase(self, phase: str):
        """更新目标阶段"""
        self.phase = phase
        self.template = get_template(self.movement_id, phase)
        self.history = []
    
    def compute_deviation(self, current_features: Dict[str, float]) -> Dict[str, float]:
        """
        计算当前姿态与模板的偏差
        返回: {关节名: 偏差角度值}
        """
        if self.template is None:
            return {}
        
        deviations = {}
        for joint, target_range in self.template.items():
            if joint in current_features:
                value = current_features[joint]
                center = (target_range[0] + target_range[1]) / 2
                deviation = value - center
                deviations[joint] = deviation
        
        return deviations
    
    def get_correction_commands(self, deviations: Dict[str, float], 
                                threshold: float = 10.0) -> List[Tuple[str, float, str]]:
        """
        根据偏差生成纠偏指令
        threshold: 偏差阈值(度或米)，小于此值不提示
        返回: [(关节名, 偏差值, 纠偏指令), ...]
        """
        commands = []
        
        for joint, deviation in deviations.items():
            if abs(deviation) < threshold:
                continue
            
            if joint in CORRECTION_COMMANDS:
                if deviation < 0:
                    cmd = CORRECTION_COMMANDS[joint]["low"]
                else:
                    cmd = CORRECTION_COMMANDS[joint]["high"]
                
                commands.append((joint, deviation, cmd))
        
        # 按偏差幅度排序，优先提示偏差大的
        commands.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return commands
    
    def analyze_deviation_pattern(self, deviations: Dict[str, float]) -> str:
        """
        分析偏差模式，返回总体评价
        """
        if not deviations:
            return "姿态标准"
        
        total_abs_deviation = sum(abs(d) for d in deviations.values())
        max_deviation = max(abs(d) for d in deviations.values())
        
        if max_deviation < 5:
            return "姿态基本标准"
        elif max_deviation < 15:
            return "存在轻微偏差"
        elif max_deviation < 25:
            return "存在明显偏差"
        else:
            return "存在较大偏差，请调整"
    
    def detect_stability(self, features_history: List[Dict[str, float]]) -> Tuple[float, bool]:
        """
        检测姿态稳定性
        返回: (稳定性得分0-100, 是否稳定)
        """
        if len(features_history) < 5:
            return 50.0, False
        
        # 计算各特征的变化幅度
        stability_scores = []
        joints = features_history[0].keys()
        
        for joint in joints:
            values = [f[joint] for f in features_history]
            std_dev = np.std(values)
            # 标准差越小越稳定
            score = max(0, 100 - std_dev * 10)
            stability_scores.append(score)
        
        avg_score = np.mean(stability_scores)
        is_stable = avg_score > 70 and np.std([np.mean(list(f.values())) for f in features_history]) < 10
        
        return avg_score, is_stable
    
    def process_frame(self, current_features: Dict[str, float]) -> Dict:
        """
        处理单帧数据，返回完整分析结果
        """
        # 存储历史
        self.history.append(current_features.copy())
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # 计算偏差
        deviations = self.compute_deviation(current_features)
        
        # 获取纠偏指令
        commands = self.get_correction_commands(deviations)
        
        # 分析偏差模式
        pattern = self.analyze_deviation_pattern(deviations)
        
        # 检测稳定性
        stability_score, is_stable = self.detect_stability(self.history)
        
        # 计算总体偏差幅度
        if deviations:
            max_deviation = max(abs(d) for d in deviations.values())
            avg_deviation = np.mean([abs(d) for d in deviations.values()])
        else:
            max_deviation = 0.0
            avg_deviation = 0.0
        
        return {
            "movement_id": self.movement_id,
            "phase": self.phase,
            "deviations": deviations,
            "max_deviation": max_deviation,
            "avg_deviation": avg_deviation,
            "correction_commands": commands,
            "pattern": pattern,
            "stability_score": stability_score,
            "is_stable": is_stable,
        }


def generate_correction_report(detector_result: Dict) -> str:
    """生成可读的纠偏报告"""
    lines = []
    lines.append(f"=== 动作纠偏报告 ===")
    lines.append(f"动作: {MOVEMENT_TEMPLATES[detector_result['movement_id']]['name']}")
    lines.append(f"阶段: {detector_result['phase']}")
    lines.append(f"总体评价: {detector_result['pattern']}")
    lines.append(f"")
    lines.append(f"偏差分析:")
    lines.append(f"  最大偏差: {detector_result['max_deviation']:.1f}°")
    lines.append(f"  平均偏差: {detector_result['avg_deviation']:.1f}°")
    lines.append(f"  稳定性: {detector_result['stability_score']:.1f}%")
    lines.append(f"")
    
    if detector_result['correction_commands']:
        lines.append(f"纠偏指令 (共{len(detector_result['correction_commands'])}条):")
        for joint, dev, cmd in detector_result['correction_commands']:
            lines.append(f"  [{joint}] 偏差{dev:+.1f}° -> {cmd}")
    else:
        lines.append("姿态标准，无需纠偏")
    
    return "\n".join(lines)


def quick_detect(current_features: Dict[str, float], 
                 movement_id: int, 
                 phase: str) -> Tuple[bool, List[str]]:
    """
    快速检测接口
    返回: (是否需要纠偏, 纠偏指令列表)
    """
    detector = DeviationDetector(movement_id, phase)
    result = detector.process_frame(current_features)
    
    commands = [cmd for _, _, cmd in result['correction_commands']]
    needs_correction = len(commands) > 0 or result['max_deviation'] > 10
    
    return needs_correction, commands


if __name__ == "__main__":
    print("=== 动作纠偏检测测试 ===")
    print()
    
    # 模拟偏差姿态
    test_features = {
        "l_shoulder_angle": 140,   # 标准165，偏低
        "r_shoulder_angle": 160,   # 标准165，正常
        "l_elbow_angle": 172,      # 标准180，偏低
        "r_elbow_angle": 178,      # 标准180，正常
        "l_arm_raise": 0.7,       # 标准1.0，偏低
        "r_arm_raise": 0.9,       # 标准1.0，正常
        "spine_angle": 3.0,        # 标准0，轻微偏
    }
    
    detector = DeviationDetector(movement_id=0, phase="托天(保持)")
    result = detector.process_frame(test_features)
    
    report = generate_correction_report(result)
    print(report)
    print()
    
    # 测试稳定性检测
    print("=== 稳定性测试 ===")
    stable_features = [{"l_shoulder_angle": 165 + np.random.randn()*2} for _ in range(10)]
    stability, is_stable = detector.detect_stability(stable_features)
    print(f"稳定姿态: 稳定性={stability:.1f}%, 是否稳定={is_stable}")
    
    unstable_features = [{"l_shoulder_angle": 165 + np.random.randn()*15} for _ in range(10)]
    stability, is_stable = detector.detect_stability(unstable_features)
    print(f"不稳定姿态: 稳定性={stability:.1f}%, 是否稳定={is_stable}")
