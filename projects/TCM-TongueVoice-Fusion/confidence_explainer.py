"""
可信评分与冲突解释模块 - Confidence Scoring and Conflict Explanation
多模态预测不一致时，识别哪个模态"说谎"，输出解释
"""

import numpy as np


class ConfidenceExplainer:
    """
    多模态可信评分与冲突解释器
    分析三模态预测一致性，识别异常模态
    """
    
    CONSTITUTIONS = ['平和', '气虚', '阳虚', '阴虚', '痰湿', '湿热', '血瘀', '气郁', '特禀']
    
    # 模态可靠性权重（基于中医诊断学先验）
    MODALITY_WEIGHTS = {
        'tongue': 0.35,   # 舌象权重
        'voice': 0.25,    # 声纹权重
        'face': 0.40      # 面色权重
    }
    
    # 各体质在各模态中的典型特征强度
    CONSTITUTION_MODALITY_STRENGTH = {
        '平和': {'tongue': 0.9, 'voice': 0.85, 'face': 0.9},
        '气虚': {'tongue': 0.8, 'voice': 0.75, 'face': 0.7},
        '阳虚': {'tongue': 0.85, 'voice': 0.8, 'face': 0.85},
        '阴虚': {'tongue': 0.9, 'voice': 0.85, 'face': 0.8},
        '痰湿': {'tongue': 0.75, 'voice': 0.7, 'face': 0.75},
        '湿热': {'tongue': 0.85, 'voice': 0.75, 'face': 0.8},
        '血瘀': {'tongue': 0.9, 'voice': 0.7, 'face': 0.85},
        '气郁': {'tongue': 0.7, 'voice': 0.9, 'face': 0.75},
        '特禀': {'tongue': 0.7, 'voice': 0.65, 'face': 0.7}
    }
    
    def __init__(self):
        self.min_confidence_threshold = 0.3
        self.conflict_threshold = 0.2
    
    def compute_confidence_score(self, fusion_result, tongue_result, voice_result, face_result):
        """
        计算综合可信评分
        
        返回: dict with overall_confidence, modality_confidences, confidence_level
        """
        # 各模态置信度
        tongue_conf = tongue_result['confidence']
        voice_conf = voice_result['confidence']
        face_conf = face_result['confidence']
        
        # 加权综合置信度
        weighted_conf = (
            self.MODALITY_WEIGHTS['tongue'] * tongue_conf +
            self.MODALITY_WEIGHTS['voice'] * voice_conf +
            self.MODALITY_WEIGHTS['face'] * face_conf
        )
        
        # 预测一致性评分
        predictions = [
            tongue_result['constitution'],
            voice_result['constitution'],
            face_result['constitution'],
            fusion_result['constitution']
        ]
        
        unique_predictions = set(predictions)
        agreement_score = 1 - (len(unique_predictions) - 1) / 3
        
        # 综合可信评分
        overall_confidence = 0.6 * weighted_conf + 0.4 * agreement_score
        
        # 置信度等级
        if overall_confidence >= 0.8:
            confidence_level = '高'
        elif overall_confidence >= 0.6:
            confidence_level = '中'
        elif overall_confidence >= 0.4:
            confidence_level = '低'
        else:
            confidence_level = '不可信'
        
        return {
            'overall_confidence': overall_confidence,
            'confidence_level': confidence_level,
            'modality_confidences': {
                'tongue': tongue_conf,
                'voice': voice_conf,
                'face': face_conf
            },
            'agreement_score': agreement_score,
            'unique_predictions': len(unique_predictions)
        }
    
    def detect_conflicts(self, tongue_result, voice_result, face_result):
        """
        检测模态间冲突
        
        返回: dict with has_conflict, conflicting_modalities, conflict_description
        """
        predictions = {
            'tongue': tongue_result['constitution'],
            'voice': voice_result['constitution'],
            'face': face_result['constitution']
        }
        
        unique_preds = set(predictions.values())
        
        if len(unique_preds) <= 1:
            return {
                'has_conflict': False,
                'conflicting_modalities': [],
                'conflict_description': '三模态预测一致'
            }
        
        # 找出冲突的模态对
        conflicting = []
        modalities = list(predictions.keys())
        for i in range(len(modalities)):
            for j in range(i+1, len(modalities)):
                if predictions[modalities[i]] != predictions[modalities[j]]:
                    conflicting.append((modalities[i], modalities[j]))
        
        # 冲突描述
        if len(unique_preds) == 2:
            conflict_type = '两模态一致，一模态分歧'
        else:
            conflict_type = '三模态均不一致'
        
        conflict_description = f"{conflict_type}：舌象={predictions['tongue']}，声纹={predictions['voice']}，面色={predictions['face']}"
        
        return {
            'has_conflict': True,
            'conflicting_modalities': conflicting,
            'conflict_description': conflict_description,
            'predictions': predictions
        }
    
    def identify_lying_modality(self, tongue_result, voice_result, face_result, fusion_result):
        """
        识别哪个模态可能"说谎"（与其他模态不一致）
        
        基于原理：
        1. 找出与其他两个模态预测不同的模态
        2. 考虑各模态的置信度，置信度低的可能不可靠
        3. 考虑各体质在各模态中的典型特征强度
        """
        
        predictions = {
            'tongue': tongue_result['constitution'],
            'voice': voice_result['constitution'],
            'face': face_result['constitution']
        }
        
        confidences = {
            'tongue': tongue_result['confidence'],
            'voice': voice_result['confidence'],
            'face': face_result['confidence']
        }
        
        unique_preds = set(predictions.values())
        
        if len(unique_preds) == 1:
            return {
                'lying_modality': None,
                'lying_probability': 0.0,
                'reason': '三模态预测一致，无需识别'
            }
        
        # 找出与其他不一致的模态
        modality_votes = {}
        for mod, pred in predictions.items():
            if pred not in modality_votes:
                modality_votes[pred] = []
            modality_votes[pred].append(mod)
        
        # 找出占多数的预测
        majority_pred = max(modality_votes.keys(), key=lambda x: len(modality_votes[x]))
        minority_modalities = []
        
        for pred, mods in modality_votes.items():
            if len(mods) == 1 and pred != majority_pred:
                minority_modalities.extend(mods)
        
        # 评估每个少数模态的可靠度
        lying_scores = {}
        
        for mod in minority_modalities:
            # 基础分数（置信度）
            base_score = confidences[mod]
            
            # 调整：根据融合结果的预测判断该模态是否应该被信任
            fusion_pred = fusion_result['constitution']
            target_pred = predictions[mod]
            
            # 如果融合结果支持该模态的预测，降低其"说谎"概率
            if fusion_pred == target_pred:
                adjustment = -0.2
            else:
                adjustment = 0.1
            
            # 如果该模态在这个体质上本来就不太可靠，进一步降低
            strength = self.CONSTITUTION_MODALITY_STRENGTH.get(
                fusion_pred, {}
            ).get(mod, 0.7)
            strength_adjustment = (1 - strength) * 0.15
            
            lying_score = base_score + adjustment - strength_adjustment
            lying_scores[mod] = max(0, min(1, lying_score))
        
        # 找出最可能说谎的模态
        if lying_scores:
            lying_modality = max(lying_scores.keys(), key=lambda x: lying_scores[x])
            lying_probability = lying_scores[lying_modality]
        else:
            lying_modality = None
            lying_probability = 0.0
        
        return {
            'lying_modality': lying_modality,
            'lying_probability': lying_probability,
            'lying_scores': lying_scores,
            'reason': self._generate_lying_reason(lying_modality, lying_probability, 
                                                   minority_modalities, predictions)
        }
    
    def _generate_lying_reason(self, lying_modality, lying_prob, minority_modalities, predictions):
        """生成解释文本"""
        if lying_modality is None or lying_prob < 0.3:
            return '各模态表现一致，未发现明显异常'
        
        mod_names = {'tongue': '舌象', 'voice': '声纹', 'face': '面色'}
        mod_name = mod_names.get(lying_modality, lying_modality)
        pred = predictions.get(lying_modality, '未知')
        
        if lying_prob >= 0.7:
            confidence_word = '高度怀疑'
        elif lying_prob >= 0.5:
            confidence_word = '可能'
        else:
            confidence_word = '轻微怀疑'
        
        return f"{confidence_word} {mod_name} 特征与体质不符（预测为{pred}），建议复核该模态数据"
    
    def explain_confidence(self, fusion_result, tongue_result, voice_result, face_result):
        """
        综合解释：整合可信评分与冲突分析
        
        返回: dict with all analysis results and human-readable explanation
        """
        # 计算可信评分
        confidence_info = self.compute_confidence_score(
            fusion_result, tongue_result, voice_result, face_result
        )
        
        # 检测冲突
        conflict_info = self.detect_conflicts(tongue_result, voice_result, face_result)
        
        # 识别说谎模态
        lying_info = self.identify_lying_modality(
            tongue_result, voice_result, face_result, fusion_result
        )
        
        # 生成综合解释
        explanation = self._generate_comprehensive_explanation(
            confidence_info, conflict_info, lying_info,
            tongue_result, voice_result, face_result
        )
        
        return {
            'confidence_info': confidence_info,
            'conflict_info': conflict_info,
            'lying_info': lying_info,
            'explanation': explanation
        }
    
    def _generate_comprehensive_explanation(self, confidence_info, conflict_info, 
                                           lying_info, tongue_result, voice_result, face_result):
        """生成综合解释文本"""
        lines = []
        
        # 总体评估
        lines.append("=" * 50)
        lines.append("【综合评估】")
        lines.append(f"可信度等级: {confidence_info['confidence_level']}")
        lines.append(f"综合可信评分: {confidence_info['overall_confidence']:.2%}")
        
        # 模态详情
        lines.append("\n【各模态详情】")
        lines.append(f"  舌象: {tongue_result['constitution']} "
                    f"(置信度: {tongue_result['confidence']:.2%})")
        lines.append(f"  声纹: {voice_result['constitution']} "
                    f"(置信度: {voice_result['confidence']:.2%})")
        lines.append(f"  面色: {face_result['constitution']} "
                    f"(置信度: {face_result['confidence']:.2%})")
        
        # 冲突检测
        if conflict_info['has_conflict']:
            lines.append(f"\n【冲突检测】")
            lines.append(f"  {conflict_info['conflict_description']}")
        
        # 说谎模态分析
        if lying_info['lying_modality'] is not None:
            lines.append(f"\n【模态可靠性分析】")
            lines.append(f"  {lying_info['reason']}")
            lines.append(f"  可疑模态: {lying_info['lying_modality']} "
                        f"(可疑度: {lying_info['lying_probability']:.2%})")
        else:
            lines.append(f"\n【模态可靠性分析】")
            lines.append(f"  {lying_info['reason']}")
        
        # 建议
        lines.append("\n【建议】")
        if confidence_info['confidence_level'] == '高':
            lines.append("  融合预测结果可信，可作为诊断参考")
        elif confidence_info['confidence_level'] == '中':
            lines.append("  融合预测结果可参考，建议结合临床信息")
        else:
            lines.append("  各模态预测不一致，建议重新采集数据或人工复核")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def get_confidence_recommendation(self, confidence_info, lying_info):
        """
        获取诊断建议
        """
        overall_conf = confidence_info['overall_confidence']
        confidence_level = confidence_info['confidence_level']
        lying_mod = lying_info['lying_modality']
        
        if overall_conf >= 0.8:
            action = '可直接使用融合预测结果'
        elif overall_conf >= 0.6:
            action = '可参考，建议关注可疑模态'
        elif overall_conf >= 0.4:
            action = '需谨慎，建议重点复核以下模态'
        else:
            action = '建议重新采集数据，人工复核'
        
        suspicious = []
        if lying_mod:
            mod_names = {'tongue': '舌象', 'voice': '声纹', 'face': '面色'}
            suspicious.append(mod_names.get(lying_mod, lying_mod))
        
        return {
            'action': action,
            'suspicious_modalities': suspicious,
            'confidence_level': confidence_level,
            'overall_confidence': overall_conf
        }


def explain_single_case(tongue_data, voice_data, face_data, fusion_result, 
                       tongue_result, voice_result, face_result):
    """
    解释单个案例的可信度
    便捷函数
    """
    explainer = ConfidenceExplainer()
    
    result = explainer.explain_confidence(
        fusion_result, tongue_result, voice_result, face_result
    )
    
    recommendation = explainer.get_confidence_recommendation(
        result['confidence_info'], result['lying_info']
    )
    
    return {
        'explanation': result['explanation'],
        'confidence_info': result['confidence_info'],
        'conflict_info': result['conflict_info'],
        'lying_info': result['lying_info'],
        'recommendation': recommendation
    }


if __name__ == "__main__":
    # 测试冲突解释
    print("=" * 60)
    print("可信评分与冲突解释测试")
    print("=" * 60)
    
    # 模拟冲突案例：舌象和面色支持气虚，声纹支持阴虚
    from tongue_analyzer import TongueAnalyzer
    from voice_analyzer import VoiceAnalyzer
    from face_analyzer import FaceAnalyzer
    
    tongue_analyzer = TongueAnalyzer()
    voice_analyzer = VoiceAnalyzer()
    face_analyzer = FaceAnalyzer()
    
    # 气虚的舌象和面色，阴虚的声纹
    tongue_data = {
        'tongue_color_rgb': [0.75, 0.50, 0.42],  # 气虚舌色
        'fur_color_rgb': [0.88, 0.85, 0.80],
        'crack': 0.2,
        'teeth_mark': 0.7,
        'shape': 0.3,
        'moisture': 0.6
    }
    
    voice_data = {
        'f0': 220,  # 阴虚声纹
        'f1': 550,
        'f2': 1600,
        'f3': 2700,
        'lpc_1': 0.9,
        'lpc_2': -0.4,
        'jitter': 0.025,
        'shimmer': 0.06
    }
    
    face_data = {
        'forehead_luster': 0.4,
        'forehead_color_diff': 0.08,
        'cheek_blood': 0.4,
        'nose_luster': 0.35,
        'nose_color_diff': 0.06,
        'overall_ruddiness': 0.35
    }
    
    # 提取特征和预测
    tongue_feat = tongue_analyzer.analyze(tongue_data)
    voice_feat = voice_analyzer.analyze(voice_data)
    face_feat = face_analyzer.analyze(face_data)
    
    tongue_pred = tongue_analyzer.get_constitution_prediction(tongue_feat)
    voice_pred = voice_analyzer.get_constitution_prediction(voice_feat)
    face_pred = face_analyzer.get_constitution_prediction(face_feat)
    
    fusion_result = {
        'constitution': '气虚',
        'confidence': 0.65,
        'probabilities': {}
    }
    
    tongue_result = {
        'constitution': max(tongue_pred, key=tongue_pred.get),
        'confidence': max(tongue_pred.values()),
        'features': tongue_feat,
        'probabilities': tongue_pred
    }
    
    voice_result = {
        'constitution': max(voice_pred, key=voice_pred.get),
        'confidence': max(voice_pred.values()),
        'features': voice_feat,
        'probabilities': voice_pred
    }
    
    face_result = {
        'constitution': max(face_pred, key=face_pred.get),
        'confidence': max(face_pred.values()),
        'features': face_feat,
        'probabilities': face_pred
    }
    
    # 解释
    explainer = ConfidenceExplainer()
    result = explainer.explain_confidence(
        fusion_result, tongue_result, voice_result, face_result
    )
    
    print("\n测试案例：气虚舌象+面色，声纹似阴虚")
    print(result['explanation'])
