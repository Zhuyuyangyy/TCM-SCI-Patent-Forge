"""
八段锦动作识别与纠偏系统演示
模拟用户做"双手托天理三焦"动作的完整流程
"""

import numpy as np
import time
from pose_estimator import (
    generate_skeleton, extract_features, recognize_movement_stage,
    BADUANJIN_MOVEMENTS, BONE_POINTS
)
from movement_db import get_template, MOVEMENT_TEMPLATES
from deviation_detector import DeviationDetector, generate_correction_report
from score_evaluator import ScoreEvaluator, generate_score_report
from rehab_prescriber import RehabPrescriber, generate_prescription_report


def simulate_sensor_data(movement_id, stage, noise_level=0.02, deviation_scale=1.0):
    skeleton = generate_skeleton(movement_id, stage)
    noise = np.random.randn(15, 3) * noise_level
    skeleton += noise
    if deviation_scale != 1.0:
        skeleton[BONE_POINTS['l_shoulder']][1] -= 0.1 * (deviation_scale - 1)
        skeleton[BONE_POINTS['l_elbow']][1] -= 0.05 * (deviation_scale - 1)
    return skeleton


def run_demo():
    print("=" * 60)
    print("    八段锦动作识别与纠偏系统 - 演示程序")
    print("=" * 60)
    print()
    
    movement_id = 0
    movement_name = BADUANJIN_MOVEMENTS[movement_id]
    print("【演示动作】: " + movement_name)
    print()
    
    phases = ["预备", "上举", "托天(保持)", "回落"]
    all_deviations = {}
    all_features = []
    timestamps = []
    phase_scores = []
    
    evaluator = ScoreEvaluator(movement_id)
    prescriber = RehabPrescriber(constitution="气虚质")
    
    print("【第一阶段: 姿态识别】")
    print("-" * 40)
    
    for phase in phases:
        print("\n>> 阶段: " + phase)
        skeleton = simulate_sensor_data(movement_id, phase, noise_level=0.01, deviation_scale=1.3)
        detected_stage, confidence = recognize_movement_stage(skeleton, movement_id)
        print("   识别结果: {}, 置信度: {:.2f}".format(detected_stage, confidence))
        features = extract_features(skeleton)
        print("   左臂抬高: {:.2f}m".format(features['l_arm_raise']))
        print("   右臂抬高: {:.2f}m".format(features['r_arm_raise']))
        template = get_template(movement_id, phase)
        print("   标准左臂抬高范围: {}".format(template['l_arm_raise']))
        print("   标准右臂抬高范围: {}".format(template['r_arm_raise']))
        all_features.append(features)
    
    print("\n")
    print("【第二阶段: 偏差检测与纠偏】")
    print("-" * 40)
    
    detector = DeviationDetector(movement_id, "托天(保持)")
    
    for frame in range(5):
        skeleton = simulate_sensor_data(movement_id, "托天(保持)", noise_level=0.02, deviation_scale=1.4)
        features = extract_features(skeleton)
        result = detector.process_frame(features)
        
        if frame == 2:
            all_deviations = result['deviations']
        
        if frame == 0:
            print("\n第{}帧偏差分析:".format(frame + 1))
            print(generate_correction_report(result))
    
    print("\n")
    print("【第三阶段: 质量评分】")
    print("-" * 40)
    
    features_history = []
    for i in range(15):
        skeleton = simulate_sensor_data(movement_id, "托天(保持)", noise_level=0.03, deviation_scale=1.3)
        features = extract_features(skeleton)
        features_history.append(features)
    
    current_deviations = {
        "l_shoulder_angle": -8.5,
        "r_shoulder_angle": -3.2,
        "l_elbow_angle": -6.0,
        "r_elbow_angle": -2.5,
        "l_arm_raise": -0.12,
        "r_arm_raise": -0.05,
        "spine_angle": 3.0,
    }
    
    eval_result = evaluator.evaluate_comprehensive(
        deviations=current_deviations,
        features_history=features_history,
        timestamps=[i * 0.5 for i in range(15)],
    )
    
    print(generate_score_report(eval_result))
    
    print("\n")
    print("【第四阶段: 康复处方推荐】")
    print("-" * 40)
    
    movement_scores = {
        0: eval_result['total_score'],
        1: 68,
        2: 75,
        3: 72,
        4: 60,
    }
    
    prescription = prescriber.recommend_prescription(
        movement_scores=movement_scores,
        age=62,
        has_injury=False,
    )
    
    print(generate_prescription_report(prescription))
    
    print("\n")
    print("【演示完成】")
    print("=" * 60)


def run_test_cases():
    print("\n" + "=" * 60)
    print("    八段锦系统 - 测试用例")
    print("=" * 60)
    
    test_cases = [
        {"name": "测试1: 标准姿态", "movement_id": 0, "stage": "托天(保持)", "deviation_scale": 1.0, "noise_level": 0.005},
        {"name": "测试2: 轻微偏差", "movement_id": 0, "stage": "托天(保持)", "deviation_scale": 1.3, "noise_level": 0.01},
        {"name": "测试3: 明显偏差", "movement_id": 0, "stage": "托天(保持)", "deviation_scale": 2.0, "noise_level": 0.02},
        {"name": "测试4: 摇头摆尾动作", "movement_id": 4, "stage": "摇头摆尾(摆动)", "deviation_scale": 1.5, "noise_level": 0.02},
        {"name": "测试5: 背后七颠", "movement_id": 6, "stage": "提踵", "deviation_scale": 1.2, "noise_level": 0.01},
    ]
    
    for i, tc in enumerate(test_cases, 1):
        print("\n" + "=" * 50)
        print("【{}】".format(tc['name']))
        print("动作: {}".format(BADUANJIN_MOVEMENTS[tc['movement_id']]))
        print("阶段: {}".format(tc['stage']))
        print("-" * 50)
        
        skeleton = simulate_sensor_data(tc['movement_id'], tc['stage'], noise_level=tc['noise_level'], deviation_scale=tc['deviation_scale'])
        stage, conf = recognize_movement_stage(skeleton, tc['movement_id'])
        print("识别阶段: {} (置信度: {:.2f})".format(stage, conf))
        features = extract_features(skeleton)
        detector = DeviationDetector(tc['movement_id'], tc['stage'])
        result = detector.process_frame(features)
        
        print("最大偏差: {:.1f}".format(result['max_deviation']))
        print("稳定性: {:.1f}%".format(result['stability_score']))
        print("评价: {}".format(result['pattern']))
        
        if result['correction_commands']:
            print("纠偏指令:")
            for joint, dev, cmd in result['correction_commands'][:3]:
                print("  - {}".format(cmd))
        
        score_result = ScoreEvaluator(tc['movement_id']).evaluate_comprehensive(result['deviations'], [features] * 10)
        print("质量评分: {:.1f}/100 (等级: {})".format(score_result['total_score'], score_result['grade']))


def main():
    print("\n")
    print("+" + "=" * 58 + "+")
    print("|" + " " * 15 + "八段锦动作识别与纠偏系统" + " " * 14 + "|")
    print("|" + " " * 10 + "Baduanjin Pose Recognition & Correction" + " " * 10 + "|")
    print("+" + "=" * 58 + "+")
    print()
    
    run_demo()
    run_test_cases()
    
    print("\n所有演示和测试完成!")


if __name__ == "__main__":
    main()
