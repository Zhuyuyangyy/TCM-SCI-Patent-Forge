"""
Temporal Medical Record Model
医案时序建模：症状-治疗-转归链
"""
import numpy as np
from collections import defaultdict

class TemporalMedicalRecord:
    """
    时序医案模型
    记录: 时间节点 → 症状变化 → 治疗方案 → 转归结果
    """
    def __init__(self, patient_id: str):
        self.patient_id = patient_id
        self.records = []
    
    def add_record(self, time_point: int, symptoms: list, treatment: str, outcome: str):
        """添加一个时间节点的记录"""
        self.records.append({
            'time': time_point,
            'symptoms': symptoms,
            'treatment': treatment,
            'outcome': outcome
        })
    
    def get_symptom_trajectory(self) -> dict:
        """获取症状演变轨迹"""
        trajectory = defaultdict(list)
        for record in sorted(self.records, key=lambda x: x['time']):
            for symptom in record['symptoms']:
                trajectory[symptom].append({
                    'time': record['time'],
                    'outcome': record['outcome']
                })
        return dict(trajectory)
    
    def predict_treatment_effect(self, new_symptoms: list) -> str:
        """预测治疗效果（简化版）"""
        if not self.records:
            return '无法预测'
        
        last = self.records[-1]
        overlap = len(set(new_symptoms) & set(last['symptoms']))
        
        if overlap > len(new_symptoms) * 0.5:
            return f"参考上次治疗({last['treatment']})，预后:{last['outcome']}"
        return '需调整治疗方案'

class LSTMEncoder:
    """
    简化的LSTM时序编码器
    将医案序列编码为向量
    """
    def __init__(self, vocab_size=1000, embed_dim=64, hidden_dim=128):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.1
    
    def encode(self, symptom_ids: list) -> np.ndarray:
        """将症状ID序列编码为向量"""
        if not symptom_ids:
            return np.zeros(self.embed_dim)
        
        vectors = self.embeddings[symptom_ids[:self.vocab_size]]
        return np.mean(vectors, axis=0)
    
    def encode_with_position(self, symptom_ids: list, time_weights: list) -> np.ndarray:
        """带时间权重的编码"""
        if not symptom_ids:
            return np.zeros(self.embed_dim)
        
        vectors = self.embeddings[symptom_ids[:self.vocab_size]]
        weights = np.array(time_weights[:len(vectors)]).reshape(-1, 1)
        weighted = vectors * weights
        return np.sum(weighted, axis=0) / (np.sum(weights) + 1e-8)
