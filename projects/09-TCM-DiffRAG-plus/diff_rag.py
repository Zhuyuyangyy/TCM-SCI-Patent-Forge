"""
Differential RAG: Personalized Knowledge Graph Retrieval
差异化RAG：结合患者个性化信息检索
"""
import json
from kg_index import KGIndex
from typing import List, Dict, Optional

class PatientProfile:
    """
    患者画像：包含个体化信息
    """
    def __init__(self, patient_id: str, constitution: str = None, 
                 history: List[str] = None, contraindications: List[str] = None,
                 age: int = None, gender: str = None):
        self.patient_id = patient_id
        self.constitution = constitution or '平和质'  # 体质类型
        self.history = history or []
        self.contraindications = contraindications or []  # 禁忌症
        self.age = age
        self.gender = gender
        self.previous_reactions = {}  # 既往用药反应
    
    def to_dict(self) -> Dict:
        return {
            'patient_id': self.patient_id,
            'constitution': self.constitution,
            'history': self.history,
            'contraindications': self.contraindications,
            'age': self.age,
            'gender': self.gender
        }


class DifferentialRAG:
    """
    差异化RAG系统
    检索策略根据患者画像动态调整
    """
    CONSTITUTION_WEIGHTS = {
        '平和质': {'default': 1.0, 'herb_boost': 0.0},
        '气虚质': {'default': 0.8, 'herb_boost': 1.2},  # 适合补气药
        '阳虚质': {'default': 0.7, 'herb_boost': 1.3},  # 适合温阳药
        '阴虚质': {'default': 0.7, 'herb_boost': 1.3},  # 适合滋阴药
        '痰湿质': {'default': 0.8, 'herb_boost': 1.1},  # 适合化痰祛湿
        '湿热质': {'default': 0.8, 'herb_boost': 1.1},  # 适合清热祛湿
        '血瘀质': {'default': 0.8, 'herb_boost': 1.2},  # 适合活血化瘀
        '气郁质': {'default': 0.7, 'herb_boost': 1.3},  # 适合疏肝解郁
        '特禀质': {'default': 0.6, 'herb_boost': 0.5},  # 慎用新药
    }
    
    def __init__(self, kg_index: KGIndex):
        self.kg = kg_index
    
    def retrieve(self, query: str, patient: PatientProfile, top_k: int = 5) -> List[Dict]:
        """
        差异化检索
        1. 基础知识图谱检索
        2. 根据患者体质调整权重
        3. 过滤禁忌药物
        4. 返回个性化结果
        """
        # Step 1: 基础检索
        base_results = self._base_retrieve(query, top_k * 3)
        
        # Step 2: 体质权重调整
        constitution_weights = self.CONSTITUTION_WEIGHTS.get(
            patient.constitution, 
            self.CONSTITUTION_WEIGHTS['平和质']
        )
        
        adjusted_results = []
        for item in base_results:
            item = item.copy()
            base_score = item.get('score', 0.5)
            
            # 根据体质调整
            constitution_factor = constitution_weights['default']
            
            # 根据药物类型进一步调整
            if 'herbs' in item:
                herb_boost = constitution_weights['herb_boost']
                constitution_factor *= (1.0 + (herb_boost - 1.0) * 0.5)
            
            item['score'] = base_score * constitution_factor
            item['constitution_adjustment'] = f"{patient.constitution}体质调整×{constitution_factor:.2f}"
            adjusted_results.append(item)
        
        # Step 3: 过滤禁忌
        filtered_results = []
        for item in adjusted_results:
            if 'herbs' in item:
                herbs = item['herbs']
                contraindications = self._check_contraindications(herbs, patient)
                if contraindications:
                    item['contraindications'] = contraindications
                    item['score'] *= 0.3  # 降低分数
            filtered_results.append(item)
        
        # Step 4: 排序输出
        filtered_results.sort(key=lambda x: -x['score'])
        return filtered_results[:top_k]
    
    def _base_retrieve(self, query: str, top_k: int) -> List[Dict]:
        """基础检索"""
        results = []
        
        # 实体搜索
        entities = self.kg.search_entity(query, top_k=top_k)
        for ent in entities:
            subgraph = self.kg.get_neighborhood(ent['name'], depth=1)
            results.append({
                'type': 'entity',
                'name': ent['name'],
                'score': ent['score'],
                'entity_type': ent['type'],
                'neighborhood': subgraph,
                'reasoning': f"通过实体检索找到: {ent['name']}"
            })
        
        # 关系搜索（简单模糊匹配）
        for subj, pred, obj in self.kg.relations:
            if query.lower() in pred.lower() or query.lower() in obj.lower():
                results.append({
                    'type': 'relation',
                    'subject': subj,
                    'predicate': pred,
                    'object': obj,
                    'score': 0.5,
                    'reasoning': f"关系匹配: {subj}-{pred}->{obj}"
                })
        
        results.sort(key=lambda x: -x['score'])
        return results[:top_k]
    
    def _check_contraindications(self, herbs: List[str], patient: PatientProfile) -> List[str]:
        """检查药物禁忌"""
        warnings = []
        for herb in herbs:
            if herb in patient.contraindications:
                warnings.append(f"{herb}是患者禁忌药物")
            # 体质禁忌检查
            if patient.constitution == '特禀质' and herb in ['海鲜', '花粉类']:
                warnings.append(f"特禀质对{herb}可能过敏")
        return warnings
    
    def generate_response(self, retrieved: List[Dict], patient: PatientProfile) -> str:
        """生成个性化回答"""
        if not retrieved:
            return "未找到相关知识，请补充查询。"
        
        lines = [f"根据{patient.constitution}体质，为您检索到以下信息："]
        
        for i, item in enumerate(retrieved[:3], 1):
            lines.append(f"\n{i}. {item.get('name', item.get('subject', 'N/A'))}")
            if 'reasoning' in item:
                lines.append(f"   依据: {item['reasoning']}")
            if 'score' in item:
                lines.append(f"   相关度: {item['score']:.2f}")
            if 'constitution_adjustment' in item:
                lines.append(f"   个性化: {item['constitution_adjustment']}")
            if 'contraindications' in item:
                lines.append(f"   ⚠️ 禁忌: {'; '.join(item['contraindications'])}")
        
        return '\n'.join(lines)
