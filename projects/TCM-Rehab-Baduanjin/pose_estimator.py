"""
八段锦姿态估计模块
从模拟的骨骼点数据识别动作阶段
"""

import numpy as np
from typing import List, Tuple, Dict

# 骨骼点索引定义 (12个关键点)
BONE_POINTS = {
    'nose': 0,
    'neck': 1,
    'r_shoulder': 2,
    'r_elbow': 3,
    'r_wrist': 4,
    'l_shoulder': 5,
    'l_elbow': 6,
    'l_wrist': 7,
    'r_hip': 8,
    'r_knee': 9,
    'r_ankle': 10,
    'l_hip': 11,
    'l_knee': 12,
    'l_ankle': 13,
    'spine_mid': 14,
}

# 八段锦8个动作名称
BADUANJIN_MOVEMENTS = [
    "双手托天理三焦",   # 0
    "左右开弓似射雕",   # 1
    "调理脾胃须单举",   # 2
    "五劳七伤往后瞧",   # 3
    "摇头摆尾去心火",   # 4
    "攒拳怒目增气力",   # 5
    "背后七颠百病消",   # 6
    "收势",            # 7
]


def compute_angle(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    """
    计算三个点形成的角度 (以p2为顶点)
    """
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    angle = np.arccos(cos_angle) * 180.0 / np.pi
    return angle


def extract_features(skeleton: np.ndarray) -> Dict[str, float]:
    """
    从骨骼点提取关键角度特征
    skeleton: (15, 3) 数组，15个骨骼点，xyz坐标
    """
    features = {}
    
    # 左臂角度
    features['l_shoulder_angle'] = compute_angle(
        skeleton[BONE_POINTS['l_elbow']],
        skeleton[BONE_POINTS['l_shoulder']],
        skeleton[BONE_POINTS['neck']]
    )
    features['l_elbow_angle'] = compute_angle(
        skeleton[BONE_POINTS['l_shoulder']],
        skeleton[BONE_POINTS['l_elbow']],
        skeleton[BONE_POINTS['l_wrist']]
    )
    
    # 右臂角度
    features['r_shoulder_angle'] = compute_angle(
        skeleton[BONE_POINTS['r_elbow']],
        skeleton[BONE_POINTS['r_shoulder']],
        skeleton[BONE_POINTS['neck']]
    )
    features['r_elbow_angle'] = compute_angle(
        skeleton[BONE_POINTS['r_shoulder']],
        skeleton[BONE_POINTS['r_elbow']],
        skeleton[BONE_POINTS['r_wrist']]
    )
    
    # 脊柱角度 (计算与垂直线的夹角)
    spine_vec = skeleton[BONE_POINTS['neck']] - skeleton[BONE_POINTS['spine_mid']]
    features['spine_angle'] = np.arctan2(spine_vec[0], spine_vec[1]) * 180.0 / np.pi
    
    # 左腿角度
    features['l_hip_angle'] = compute_angle(
        skeleton[BONE_POINTS['l_knee']],
        skeleton[BONE_POINTS['l_hip']],
        skeleton[BONE_POINTS['spine_mid']]
    )
    features['l_knee_angle'] = compute_angle(
        skeleton[BONE_POINTS['l_hip']],
        skeleton[BONE_POINTS['l_knee']],
        skeleton[BONE_POINTS['l_ankle']]
    )
    
    # 右腿角度
    features['r_hip_angle'] = compute_angle(
        skeleton[BONE_POINTS['r_knee']],
        skeleton[BONE_POINTS['r_hip']],
        skeleton[BONE_POINTS['spine_mid']]
    )
    features['r_knee_angle'] = compute_angle(
        skeleton[BONE_POINTS['r_hip']],
        skeleton[BONE_POINTS['r_knee']],
        skeleton[BONE_POINTS['r_ankle']]
    )
    
    # 手臂上举高度 (相对于肩部)
    features['l_arm_raise'] = skeleton[BONE_POINTS['l_wrist']][1] - skeleton[BONE_POINTS['l_shoulder']][1]
    features['r_arm_raise'] = skeleton[BONE_POINTS['r_wrist']][1] - skeleton[BONE_POINTS['r_shoulder']][1]
    
    return features


def recognize_movement_stage(skeleton: np.ndarray, movement_id: int) -> Tuple[str, float]:
    """
    识别当前动作的具体阶段
    movement_id: 0-7 对应八段锦8个动作
    返回: (阶段名称, 置信度)
    """
    features = extract_features(skeleton)
    
    if movement_id == 0:  # 双手托天理三焦
        # 阶段: 预备 -> 托天 -> 保持 -> 回落
        if features['l_arm_raise'] > 0.8 and features['r_arm_raise'] > 0.8:
            return "双手托天(保持)", 0.95
        elif features['l_arm_raise'] > 0.4 and features['r_arm_raise'] > 0.4:
            return "双手托天(上举)", 0.85
        else:
            return "双手托天(预备)", 0.70
    
    elif movement_id == 1:  # 左右开弓似射雕
        if features['l_elbow_angle'] < 30:
            return "开弓(左拉)", 0.90
        elif features['r_elbow_angle'] < 30:
            return "开弓(右拉)", 0.90
        else:
            return "开弓(搭箭)", 0.75
    
    elif movement_id == 2:  # 调理脾胃须单举
        if features['l_arm_raise'] > 0.7:
            return "左托天", 0.92
        elif features['r_arm_raise'] > 0.7:
            return "右托天", 0.92
        else:
            return "单举(预备)", 0.70
    
    elif movement_id == 3:  # 五劳七伤往后瞧
        if features['spine_angle'] > 20:
            return "后瞧(左)", 0.88
        elif features['spine_angle'] < -20:
            return "后瞧(右)", 0.88
        else:
            return "后瞧(中)", 0.70
    
    elif movement_id == 4:  # 摇头摆尾去心火
        if abs(features['spine_angle']) > 30:
            return "摇头摆尾(摆动)", 0.85
        else:
            return "摇头摆尾(中)", 0.75
    
    elif movement_id == 5:  # 攒拳怒目增气力
        if features['l_elbow_angle'] < 60:
            return "攒拳(左)", 0.90
        elif features['r_elbow_angle'] < 60:
            return "攒拳(右)", 0.90
        else:
            return "攒拳(预备)", 0.70
    
    elif movement_id == 6:  # 背后七颠百病消
        if features['l_knee_angle'] < 160 and features['r_knee_angle'] < 160:
            return "提踵", 0.92
        else:
            return "颠足", 0.80
    
    else:  # 收势
        return "收势(站立)", 0.95


def generate_skeleton(movement_id: int, stage: str, noise_level: float = 0.0) -> np.ndarray:
    """
    生成模拟骨骼点数据
    用于测试和演示
    """
    # 基础站立姿态 (15个点, 3D坐标)
    skeleton = np.array([
        [0.0, 1.7, 0.0],    # 0: nose
        [0.0, 1.55, 0.0],   # 1: neck
        [0.2, 1.5, 0.1],    # 2: r_shoulder
        [0.4, 1.3, 0.1],    # 3: r_elbow
        [0.6, 1.1, 0.1],    # 4: r_wrist
        [-0.2, 1.5, 0.1],   # 5: l_shoulder
        [-0.4, 1.3, 0.1],   # 6: l_elbow
        [-0.6, 1.1, 0.1],   # 7: l_wrist
        [0.15, 0.9, 0.0],   # 8: r_hip
        [0.15, 0.55, 0.0],  # 9: r_knee
        [0.15, 0.05, 0.0],  # 10: r_ankle
        [-0.15, 0.9, 0.0],  # 11: l_hip
        [-0.15, 0.55, 0.0], # 12: l_knee
        [-0.15, 0.05, 0.0], # 13: l_ankle
        [0.0, 1.1, -0.1],   # 14: spine_mid
    ])
    
    # 根据动作和阶段调整姿态
    if movement_id == 0:  # 双手托天
        if "托天" in stage:
            # 双手上举
            skeleton[BONE_POINTS['l_wrist']] = [-0.3, 2.0, 0.2]
            skeleton[BONE_POINTS['r_wrist']] = [0.3, 2.0, 0.2]
            skeleton[BONE_POINTS['l_elbow']] = [-0.15, 1.7, 0.15]
            skeleton[BONE_POINTS['r_elbow']] = [0.15, 1.7, 0.15]
    
    elif movement_id == 1:  # 左右开弓
        if "左拉" in stage:
            skeleton[BONE_POINTS['l_wrist']] = [-0.8, 1.3, 0.5]
            skeleton[BONE_POINTS['l_elbow']] = [-0.5, 1.2, 0.3]
        elif "右拉" in stage:
            skeleton[BONE_POINTS['r_wrist']] = [0.8, 1.3, 0.5]
            skeleton[BONE_POINTS['r_elbow']] = [0.5, 1.2, 0.3]
    
    elif movement_id == 2:  # 调理脾胃
        if "左托天" in stage:
            skeleton[BONE_POINTS['l_wrist']] = [-0.4, 2.1, 0.3]
            skeleton[BONE_POINTS['l_elbow']] = [-0.2, 1.7, 0.2]
        elif "右托天" in stage:
            skeleton[BONE_POINTS['r_wrist']] = [0.4, 2.1, 0.3]
            skeleton[BONE_POINTS['r_elbow']] = [0.2, 1.7, 0.2]
    
    elif movement_id == 3:  # 五劳七伤往后瞧
        if "左" in stage:
            skeleton[BONE_POINTS['neck']] = [-0.15, 1.55, 0.0]
            skeleton[BONE_POINTS['spine_mid']] = [-0.1, 1.1, -0.1]
        elif "右" in stage:
            skeleton[BONE_POINTS['neck']] = [0.15, 1.55, 0.0]
            skeleton[BONE_POINTS['spine_mid']] = [0.1, 1.1, -0.1]
    
    elif movement_id == 4:  # 摇头摆尾
        if "摆动" in stage:
            skeleton[BONE_POINTS['neck']] = [0.25, 1.55, 0.0]
            skeleton[BONE_POINTS['spine_mid']] = [0.15, 1.1, -0.1]
    
    elif movement_id == 5:  # 攒拳怒目
        if "左" in stage:
            skeleton[BONE_POINTS['l_wrist']] = [-0.5, 1.1, 0.3]
            skeleton[BONE_POINTS['l_elbow']] = [-0.3, 1.15, 0.2]
        elif "右" in stage:
            skeleton[BONE_POINTS['r_wrist']] = [0.5, 1.1, 0.3]
            skeleton[BONE_POINTS['r_elbow']] = [0.3, 1.15, 0.2]
    
    elif movement_id == 6:  # 背后七颠
        if "提踵" in stage:
            skeleton[BONE_POINTS['l_ankle']][1] = 0.15
            skeleton[BONE_POINTS['r_ankle']][1] = 0.15
            skeleton[BONE_POINTS['l_knee']][1] = 0.6
            skeleton[BONE_POINTS['r_knee']][1] = 0.6
    
    # 添加噪声
    if noise_level > 0:
        noise = np.random.randn(15, 3) * noise_level
        skeleton += noise
    
    return skeleton


if __name__ == "__main__":
    print("=== 八段锦姿态估计测试 ===")
    print(f"骨骼点数量: {len(BONE_POINTS)}")
    print(f"动作数量: {len(BADUANJIN_MOVEMENTS)}")
    print()
    
    # 测试每个动作的姿态识别
    for i, name in enumerate(BADUANJIN_MOVEMENTS):
        print(f"动作 {i+1}: {name}")
        skeleton = generate_skeleton(i, "保持" if i != 6 else "提踵")
        features = extract_features(skeleton)
        stage, conf = recognize_movement_stage(skeleton, i)
        print(f"  阶段: {stage}, 置信度: {conf:.2f}")
        print(f"  左臂抬高: {features['l_arm_raise']:.2f}m, 右臂抬高: {features['r_arm_raise']:.2f}m")
        print()
