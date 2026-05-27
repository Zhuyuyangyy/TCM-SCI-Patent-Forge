"""
M01 舌象-声纹-面色三模态体质识别系统 demo
基于TCM体质分类的舌象/声纹/面色融合识别
修复版本：MultiModalFusion 支持各模态独立输入
"""
import numpy as np
from tongue_analyzer import TongueAnalyzer
from voice_analyzer import VoiceAnalyzer
from face_analyzer import FaceAnalyzer
from multi_modal_fusion import MultiModalFusion
from confidence_explainer import ConfidenceExplainer

def demo():
    print("=" * 60)
    print("M01 舌象-声纹-面色三模态体质识别系统")
    print("=" * 60)

    # 初始化各模块
    tongue = TongueAnalyzer()
    voice = VoiceAnalyzer()
    face = FaceAnalyzer()
    fusion = MultiModalFusion(
        tongue_dim=8, voice_dim=8, face_dim=6,
        hidden_dims=[64, 32], output_dim=9
    )
    explainer = ConfidenceExplainer()

    constitution_names = ['气虚', '阳虚', '阴虚', '痰湿', '湿热', '血瘀', '气郁', '特禀', '平和']

    # 模拟患者数据
    patient_tongue = tongue.CONSTITUTION_TONGUE_FEATURES['湿热']
    patient_voice = {
        'f0': 180.0,
        'formants': [500, 1500, 2500],
        'lpc': np.array([0.5, -0.3, 0.2, -0.1, 0.05] * 2 + [0.0] * 2),
        'jitter': 0.025,
        'shimmer': 0.035,
    }
    patient_face = {
        'zones': {
            'forehead': {'luminance': 135, 'a': 12, 'b': 35},
            'cheek_l': {'luminance': 128, 'a': 22, 'b': 42},
            'cheek_r': {'luminance': 126, 'a': 24, 'b': 44},
            'nose': {'luminance': 130, 'a': 6, 'b': 22},
        }
    }

    # Step 1: 舌象特征提取
    print("\n[Step 1] 舌象特征提取")
    tongue_feat = tongue.analyze(patient_tongue)
    tongue_probs = fusion.predict_modality(tongue_feat, 'tongue')
    tongue_pred = constitution_names[np.argmax(tongue_probs)]
    tongue_conf = float(tongue_probs[np.argmax(tongue_probs)])
    print(f"  舌色: {[round(x,2) for x in patient_tongue['tongue_color_rgb']]}")
    print(f"  苔色: {[round(x,2) for x in patient_tongue['fur_color_rgb']]}")
    print(f"  裂纹={patient_tongue['crack']}, 齿痕={patient_tongue['teeth_mark']}")
    print(f"  舌象预测: {tongue_pred}质 (置信度: {tongue_conf:.3f})")
    print(f"  舌象特征维度: {tongue_feat.shape}")

    # Step 2: 声纹特征提取
    print("\n[Step 2] 声纹特征提取")
    voice_feat = voice.extract_from_dict(patient_voice)
    voice_probs = fusion.predict_modality(voice_feat, 'voice')
    voice_pred = constitution_names[np.argmax(voice_probs)]
    voice_conf = float(voice_probs[np.argmax(voice_probs)])
    print(f"  F0: {patient_voice['f0']:.1f} Hz")
    print(f"  共振峰: {patient_voice['formants']}")
    print(f"  声纹预测: {voice_pred}质 (置信度: {voice_conf:.3f})")
    print(f"  声纹特征维度: {voice_feat.shape}")

    # Step 3: 面色特征提取
    print("\n[Step 3] 面色特征提取")
    face_feat = face.analyze(patient_face)
    face_probs = fusion.predict_modality(face_feat, 'face')
    face_pred = constitution_names[np.argmax(face_probs)]
    face_conf = float(face_probs[np.argmax(face_probs)])
    cheek_a = patient_face['zones']['cheek_l']['a']
    cheek_b = patient_face['zones']['cheek_l']['b']
    print(f"  面色(颧部): a={cheek_a}, b={cheek_b}")
    print(f"  面色预测: {face_pred}质 (置信度: {face_conf:.3f})")
    print(f"  面色特征维度: {face_feat.shape}")

    # Step 4: 多模态融合
    print("\n[Step 4] 多模态融合预测")
    fusion_probs = fusion.predict_fusion(tongue_feat, voice_feat, face_feat)
    fusion_pred = constitution_names[np.argmax(fusion_probs)]
    fusion_conf = float(fusion_probs[np.argmax(fusion_probs)])
    print(f"  融合特征维度: {tongue_feat.shape[0]}+{voice_feat.shape[0]}+{face_feat.shape[0]}={tongue_feat.shape[0]+voice_feat.shape[0]+face_feat.shape[0]}")
    print(f"  融合预测: {fusion_pred}质 (概率: {fusion_conf:.3f})")
    top3_idx = np.argsort(fusion_probs)[::-1][:3]
    print(f"  Top3: {constitution_names[top3_idx[0]]} {fusion_probs[top3_idx[0]]:.3f}, "
          f"{constitution_names[top3_idx[1]]} {fusion_probs[top3_idx[1]]:.3f}, "
          f"{constitution_names[top3_idx[2]]} {fusion_probs[top3_idx[2]]:.3f}")

    # Step 5: 可信评分与冲突解释
    print("\n[Step 5] 可信评分与冲突解释")
    confidence_score = explainer.compute_confidence_score(
        fusion_probs, tongue_probs, voice_probs, face_probs)
    conflicts = explainer.detect_conflicts(tongue_probs, voice_probs, face_probs)
    confidence_info = explainer.explain_confidence(
        fusion_probs, tongue_probs, voice_probs, face_probs)

    print(f"  可信评分: {confidence_score:.3f} ({'高可信' if confidence_score > 0.7 else '中可信' if confidence_score > 0.4 else '低可信'})")
    print(f"  模态冲突数: {len(conflicts)}")
    if conflicts:
        for c in conflicts[:3]:
            print(f"    - {c}")
    print(f"  解释: {confidence_info}")

    print("\n" + "=" * 60)
    print("Demo Complete")


if __name__ == '__main__':
    demo()
