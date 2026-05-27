"""
八段锦标准动作模板库
每个动作的关节点角度标准值及可接受范围
"""

import numpy as np
from typing import Dict, List, Tuple

# 标准姿态模板 (每个动作的多个阶段)
MOVEMENT_TEMPLATES = {
    0: {  # 双手托天理三焦
        "name": "双手托天理三焦",
        "phases": {
            "预备": {
                "l_shoulder_angle": (70, 90),
                "r_shoulder_angle": (70, 90),
                "l_elbow_angle": (150, 180),
                "r_elbow_angle": (150, 180),
                "l_arm_raise": (-0.1, 0.2),
                "r_arm_raise": (-0.1, 0.2),
                "spine_angle": (-5, 5),
            },
            "上举": {
                "l_shoulder_angle": (120, 160),
                "r_shoulder_angle": (120, 160),
                "l_elbow_angle": (170, 180),
                "r_elbow_angle": (170, 180),
                "l_arm_raise": (0.4, 0.8),
                "r_arm_raise": (0.4, 0.8),
                "spine_angle": (-5, 5),
            },
            "托天(保持)": {
                "l_shoulder_angle": (150, 180),
                "r_shoulder_angle": (150, 180),
                "l_elbow_angle": (175, 185),
                "r_elbow_angle": (175, 185),
                "l_arm_raise": (0.8, 1.2),
                "r_arm_raise": (0.8, 1.2),
                "spine_angle": (-3, 3),
            },
            "回落": {
                "l_shoulder_angle": (120, 160),
                "r_shoulder_angle": (120, 160),
                "l_elbow_angle": (170, 180),
                "r_elbow_angle": (170, 180),
                "l_arm_raise": (0.4, 0.8),
                "r_arm_raise": (0.4, 0.8),
                "spine_angle": (-5, 5),
            },
        },
        "key_joints": ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "spine"],
        "difficulty": 2,  # 难度系数 1-5
    },
    
    1: {  # 左右开弓似射雕
        "name": "左右开弓似射雕",
        "phases": {
            "预备": {
                "l_shoulder_angle": (80, 100),
                "r_shoulder_angle": (80, 100),
                "l_elbow_angle": (160, 180),
                "r_elbow_angle": (160, 180),
                "spine_angle": (-5, 5),
            },
            "搭箭": {
                "l_shoulder_angle": (90, 110),
                "r_shoulder_angle": (90, 110),
                "l_elbow_angle": (160, 180),
                "r_elbow_angle": (160, 180),
                "spine_angle": (-5, 5),
            },
            "开弓(左拉)": {
                "l_shoulder_angle": (140, 170),
                "r_shoulder_angle": (80, 100),
                "l_elbow_angle": (20, 40),
                "r_elbow_angle": (160, 180),
                "spine_angle": (-10, 10),
            },
            "开弓(右拉)": {
                "l_shoulder_angle": (80, 100),
                "r_shoulder_angle": (140, 170),
                "l_elbow_angle": (160, 180),
                "r_elbow_angle": (20, 40),
                "spine_angle": (-10, 10),
            },
        },
        "key_joints": ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "spine"],
        "difficulty": 3,
    },
    
    2: {  # 调理脾胃须单举
        "name": "调理脾胃须单举",
        "phases": {
            "预备": {
                "l_shoulder_angle": (70, 90),
                "r_shoulder_angle": (70, 90),
                "l_elbow_angle": (150, 170),
                "r_elbow_angle": (150, 170),
                "l_arm_raise": (-0.1, 0.2),
                "r_arm_raise": (-0.1, 0.2),
                "spine_angle": (-5, 5),
            },
            "左托天": {
                "l_shoulder_angle": (150, 180),
                "r_shoulder_angle": (70, 90),
                "l_elbow_angle": (175, 185),
                "r_elbow_angle": (150, 170),
                "l_arm_raise": (0.8, 1.1),
                "r_arm_raise": (-0.1, 0.2),
                "spine_angle": (-5, 5),
            },
            "右托天": {
                "l_shoulder_angle": (70, 90),
                "r_shoulder_angle": (150, 180),
                "l_elbow_angle": (150, 170),
                "r_elbow_angle": (175, 185),
                "l_arm_raise": (-0.1, 0.2),
                "r_arm_raise": (0.8, 1.1),
                "spine_angle": (-5, 5),
            },
        },
        "key_joints": ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow"],
        "difficulty": 2,
    },
    
    3: {  # 五劳七伤往后瞧
        "name": "五劳七伤往后瞧",
        "phases": {
            "中": {
                "spine_angle": (-5, 5),
                "l_shoulder_angle": (80, 100),
                "r_shoulder_angle": (80, 100),
            },
            "后瞧(左)": {
                "spine_angle": (20, 40),
                "l_shoulder_angle": (80, 100),
                "r_shoulder_angle": (80, 100),
            },
            "后瞧(右)": {
                "spine_angle": (-40, -20),
                "l_shoulder_angle": (80, 100),
                "r_shoulder_angle": (80, 100),
            },
        },
        "key_joints": ["spine", "neck"],
        "difficulty": 2,
    },
    
    4: {  # 摇头摆尾去心火
        "name": "摇头摆尾去心火",
        "phases": {
            "中": {
                "spine_angle": (-10, 10),
                "l_hip_angle": (80, 100),
                "r_hip_angle": (80, 100),
            },
            "摇头摆尾(摆动)": {
                "spine_angle": (-45, 45),
                "l_hip_angle": (70, 110),
                "r_hip_angle": (70, 110),
            },
        },
        "key_joints": ["spine", "hip"],
        "difficulty": 4,
    },
    
    5: {  # 攒拳怒目增气力
        "name": "攒拳怒目增气力",
        "phases": {
            "预备": {
                "l_shoulder_angle": (70, 90),
                "r_shoulder_angle": (70, 90),
                "l_elbow_angle": (160, 180),
                "r_elbow_angle": (160, 180),
            },
            "攒拳(左)": {
                "l_shoulder_angle": (60, 80),
                "r_shoulder_angle": (70, 90),
                "l_elbow_angle": (40, 70),
                "r_elbow_angle": (160, 180),
            },
            "攒拳(右)": {
                "l_shoulder_angle": (70, 90),
                "r_shoulder_angle": (60, 80),
                "l_elbow_angle": (160, 180),
                "r_elbow_angle": (40, 70),
            },
        },
        "key_joints": ["l_shoulder", "r_shoulder", "l_elbow", "r_elbow"],
        "difficulty": 2,
    },
    
    6: {  # 背后七颠百病消
        "name": "背后七颠百病消",
        "phases": {
            "颠足": {
                "l_knee_angle": (170, 180),
                "r_knee_angle": (170, 180),
                "l_hip_angle": (90, 110),
                "r_hip_angle": (90, 110),
            },
            "提踵": {
                "l_knee_angle": (155, 170),
                "r_knee_angle": (155, 170),
                "l_hip_angle": (95, 115),
                "r_hip_angle": (95, 115),
            },
        },
        "key_joints": ["l_knee", "r_knee", "l_hip", "r_hip"],
        "difficulty": 1,
    },
    
    7: {  # 收势
        "name": "收势",
        "phases": {
            "站立": {
                "l_shoulder_angle": (70, 90),
                "r_shoulder_angle": (70, 90),
                "l_elbow_angle": (160, 180),
                "r_elbow_angle": (160, 180),
                "l_knee_angle": (170, 180),
                "r_knee_angle": (170, 180),
                "spine_angle": (-5, 5),
            },
            "放松": {
                "l_shoulder_angle": (60, 100),
                "r_shoulder_angle": (60, 100),
                "l_elbow_angle": (150, 180),
                "r_elbow_angle": (150, 180),
                "l_knee_angle": (165, 180),
                "r_knee_angle": (165, 180),
                "spine_angle": (-10, 10),
            },
        },
        "key_joints": ["spine", "shoulder", "knee"],
        "difficulty": 1,
    },
}


def get_template(movement_id: int, phase: str = None) -> Dict:
    """获取指定动作的模板"""
    if movement_id not in MOVEMENT_TEMPLATES:
        raise ValueError(f"Invalid movement_id: {movement_id}")
    
    template = MOVEMENT_TEMPLATES[movement_id]
    if phase is not None:
        if phase not in template["phases"]:
            raise ValueError(f"Invalid phase: {phase} for movement {movement_id}")
        return template["phases"][phase]
    return template


def get_all_phases(movement_id: int) -> List[str]:
    """获取指定动作的所有阶段"""
    if movement_id not in MOVEMENT_TEMPLATES:
        return []
    return list(MOVEMENT_TEMPLATES[movement_id]["phases"].keys())


def get_key_joints(movement_id: int) -> List[str]:
    """获取指定动作的关键关节"""
    if movement_id not in MOVEMENT_TEMPLATES:
        return []
    return MOVEMENT_TEMPLATES[movement_id].get("key_joints", [])


def get_difficulty(movement_id: int) -> int:
    """获取指定动作的难度系数"""
    if movement_id not in MOVEMENT_TEMPLATES:
        return 0
    return MOVEMENT_TEMPLATES[movement_id].get("difficulty", 0)


def is_within_range(value: float, range_tuple: Tuple[float, float]) -> bool:
    """检查值是否在指定范围内"""
    return range_tuple[0] <= value <= range_tuple[1]


def calculate_compliance_score(current_features: Dict[str, float], 
                               template: Dict[str, Tuple[float, float]]) -> float:
    """
    计算当前姿态与模板的符合度 (0-100)
    """
    if not template:
        return 0.0
    
    total_score = 0.0
    count = 0
    
    for joint, target_range in template.items():
        if joint in current_features:
            value = current_features[joint]
            # 计算偏离程度
            if is_within_range(value, target_range):
                # 在范围内，得满分
                score = 100.0
            else:
                # 超出范围，计算偏离惩罚
                center = (target_range[0] + target_range[1]) / 2
                half_range = (target_range[1] - target_range[0]) / 2
                deviation = abs(value - center)
                # 偏离50%以内仍有一定分数
                score = max(0, 100 * (1 - (deviation - half_range) / half_range))
            
            total_score += score
            count += 1
    
    if count == 0:
        return 0.0
    
    return total_score / count


if __name__ == "__main__":
    print("=== 八段锦标准动作模板库测试 ===")
    print()
    
    for mid in range(8):
        template = MOVEMENT_TEMPLATES[mid]
        print(f"动作 {mid+1}: {template['name']}")
        print(f"  难度: {template['difficulty']}")
        print(f"  阶段: {list(template['phases'].keys())}")
        print(f"  关键关节: {template['key_joints']}")
        print()
    
    # 测试符合度计算
    print("=== 符合度测试 ===")
    test_features = {
        "l_shoulder_angle": 155,
        "r_shoulder_angle": 155,
        "l_elbow_angle": 178,
        "r_elbow_angle": 178,
        "l_arm_raise": 0.9,
        "r_arm_raise": 0.9,
        "spine_angle": 1.0,
    }
    
    template = get_template(0, "托天(保持)")
    score = calculate_compliance_score(test_features, template)
    print(f"标准'托天(保持)'姿态符合度: {score:.1f}%")
