"""
Tongue Diagnosis Agent
舌诊证据提取Agent
"""
from base_agent import BaseTCMAgent
import numpy as np
from typing import Dict, List

class TongueAgent(BaseTCMAgent):
    """
    舌诊专家Agent
    输入: 舌象描述
    输出: 舌象特征 + 证候推断
    """
    def __init__(self):
        super().__init__('TongueAgent', '舌诊专家')
        self.tongue_features = {
            '舌质': ['淡红', '红', '绛红', '紫暗'],
            '舌苔': ['薄白', '黄腻', '白腻', '少苔'],
            '舌形': ['正常', '胖大', '齿痕', '裂纹']
        }
    
    def process(self, patient_data: Dict) -> Dict:
        tongue_description = patient_data.get('tongue', '舌淡红，苔薄白')
        
        # 模拟舌诊分析
        features = self._extract_features(tongue_description)
        syndrome_inference = self._infer_syndrome(features)
        
        evidence = {
            'agent': self.name,
            'input': tongue_description,
            'features': features,
            'syndrome': syndrome_inference,
            'confidence': 0.85,
            'reasoning': self._generate_reasoning(features, syndrome_inference)
        }
        self.add_evidence(evidence)
        return evidence
    
    def _extract_features(self, description: str) -> Dict:
        """从描述中提取舌象特征"""
        features = {}
        for category, values in self.tongue_features.items():
            for value in values:
                if value in description:
                    features[category] = value
                    break
        if '舌质' not in features:
            features['舌质'] = '淡红'  # 默认
        if '舌苔' not in features:
            features['舌苔'] = '薄白'
        return features
    
    def _infer_syndrome(self, features: Dict) -> str:
        """根据舌象推断证候"""
        tongue_body = features.get('舌质', '淡红')
        coating = features.get('舌苔', '薄白')
        
        if tongue_body in ['红', '绛红']:
            if coating == '黄腻':
                return '肝火亢盛'
            return '热证'
        elif tongue_body == '紫暗':
            return '气滞血瘀'
        elif coating == '白腻':
            return '痰湿内阻'
        elif coating == '少苔':
            return '阴虚火旺'
        else:
            return '肝郁气滞'
    
    def _generate_reasoning(self, features: Dict, syndrome: str) -> str:
        return f"舌质{features.get('舌质')}，舌苔{features.get('舌苔')}，符合{syndrome}的舌象特征"
