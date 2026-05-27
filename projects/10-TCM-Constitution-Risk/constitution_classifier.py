"""
TCM Constitution Classifier
中医体质分类器（九种体质）
"""
import numpy as np
from typing import Dict, List, Tuple

class ConstitutionClassifier:
    """
    九种体质分类器
    平和质、气虚质、阳虚质、阴虚质、痰湿质、湿热质、血瘀质、气郁质、特禀质
    """
    CONSTITUTIONS = [
        '平和质', '气虚质', '阳虚质', '阴虚质', 
        '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质'
    ]
    
    # 各体质特征问卷权重（简化版）
    QUESTION_WEIGHTS = {
        0: {'气虚质': 0.3, '阳虚质': 0.2, '平和质': 0.1},  # 容易疲乏
        1: {'气虚质': 0.2, '阳虚质': 0.3},                  # 手脚发凉
        2: {'阴虚质': 0.3, '湿热质': 0.2},                  # 手脚心发热
        3: {'痰湿质': 0.3, '湿热质': 0.2},                  # 腹部肥满
        4: {'湿热质': 0.3, '痰湿质': 0.2},                  # 面部油腻
        5: {'血瘀质': 0.3, '气郁质': 0.2},                  # 容易瘀青
        6: {'气郁质': 0.3, '血瘀质': 0.2},                  # 情绪低落
        7: {'特禀质': 0.3, '平和质': -0.1},                  # 容易过敏
        8: {'平和质': 0.3, '气虚质': -0.1},                  # 精力充沛
    }
    
    def __init__(self):
        self.feature_dim = len(self.QUESTION_WEIGHTS)
    
    def classify(self, questionnaire_responses: List[float]) -> Dict:
        """
        根据问卷回答分类体质
        responses: 每题评分(0-5)的列表
        返回: 各体质得分 + 判定结果
        """
        if len(questionnaire_responses) < self.feature_dim:
            # 填充默认值
            questionnaire_responses = list(questionnaire_responses) +                 [0] * (self.feature_dim - len(questionnaire_responses))
        
        scores = {c: 0.0 for c in self.CONSTITUTIONS}
        
        for q_idx, answer in enumerate(questionnaire_responses[:self.feature_dim]):
            if q_idx not in self.QUESTION_WEIGHTS:
                continue
            for constitution, weight in self.QUESTION_WEIGHTS[q_idx].items():
                scores[constitution] += answer * weight / 5.0  # 归一化
        
        # 归一化为概率
        total = sum(max(0, s) for s in scores.values()) + 1e-8
        probs = {c: max(0, s) / total for c, s in scores.items()}
        
        # 判定主型体质
        primary = max(probs, key=probs.get)
        
        return {
            'primary': primary,
            'probabilities': probs,
            'confidence': probs[primary],
            'recommendation': self._get_recommendation(primary)
        }
    
    def _get_recommendation(self, constitution: str) -> str:
        """根据体质给出养生建议"""
        recommendations = {
            '平和质': '保持均衡饮食，适度运动，规律作息',
            '气虚质': '宜食黄芪、党参等补气之品，避免过度劳累',
            '阳虚质': '宜食羊肉、桂圆等温阳之品，避寒保暖',
            '阴虚质': '宜食枸杞、麦冬等滋阴之品，忌辛辣',
            '痰湿质': '宜食薏米、冬瓜等祛湿之品，加强运动',
            '湿热质': '宜食菊花、绿豆等清热之品，忌烟酒',
            '血瘀质': '宜食山楂、玫瑰花等活血之品，适度运动',
            '气郁质': '宜食玫瑰花、陈皮等疏肝之品，调畅情志',
            '特禀质': '规避过敏原，饮食清淡，增强体质'
        }
        return recommendations.get(constitution, '请咨询中医师')
    
    def batch_classify(self, responses_list: List[List[float]]) -> List[Dict]:
        """批量分类"""
        return [self.classify(resp) for resp in responses_list]
