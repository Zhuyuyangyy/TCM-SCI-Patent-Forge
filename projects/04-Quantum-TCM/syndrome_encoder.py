"""
Syndrome Encoder: 将中医症状编码为量子态参数
"""
import numpy as np

class SyndromeEncoder:
    """
    将症状映射为VQC参数
    症状 → 特征向量 → 量子电路参数
    """
    SYNDROME_BASIS = {
        '肝': ['胁肋胀痛', '情绪抑郁', '暖气频繁', '脉弦', '口苦'],
        '心': ['心悸', '失眠', '多梦', '盗汗', '舌红'],
        '脾': ['腹胀', '食欲不振', '大便溏', '乏力', '舌淡'],
        '肺': ['咳嗽', '气喘', '痰多', '自汗', '舌淡白'],
        '肾': ['腰膝酸软', '耳鸣', '夜尿多', '遗精', '舌红少苔']
    }
    
    SYMPTOM_WEIGHTS = {
        '胁肋胀痛': 0.8, '情绪抑郁': 0.7, '脉弦': 0.9,
        '心悸': 0.8, '失眠': 0.6, '舌红': 0.5,
        '腹胀': 0.7, '食欲不振': 0.6, '大便溏': 0.5,
        '咳嗽': 0.8, '气喘': 0.7, '痰多': 0.6,
        '腰膝酸软': 0.9, '耳鸣': 0.7, '夜尿多': 0.8
    }
    
    def encode_symptoms(self, symptoms):
        """
        将症状列表编码为VQC参数
        """
        # 初始化各脏腑激活
        organ_activation = {org: 0.0 for org in self.SYNDROME_BASIS}
        
        for symptom in symptoms:
            for organ, organ_symptoms in self.SYNDROME_BASIS.items():
                if symptom in organ_symptoms:
                    weight = self.SYMPTOM_WEIGHTS.get(symptom, 0.5)
                    organ_activation[organ] += weight
        
        # 归一化
        total = sum(organ_activation.values())
        if total > 0:
            for k in organ_activation:
                organ_activation[k] /= total
        
        # 转换为量子电路参数
        thetas = []
        for organ, activation in organ_activation.items():
            # RY角 = π * 激活度
            thetas.append(np.pi * activation)
            # RZ角 = 2π * 激活度^2
            thetas.append(2 * np.pi * (activation ** 2))
        
        # 补齐到12个参数（6个量子比特 * 2）
        while len(thetas) < 12:
            thetas.append(0.0)
        
        return np.array(thetas[:12])
