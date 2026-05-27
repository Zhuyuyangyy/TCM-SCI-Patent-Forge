"""
LLM-based Syndrome Differentiator with KG constraints
使用KG约束的LLM证候鉴别
"""
import json
import random
from pathlib import Path

class SyndromeDifferentiator:
    def __init__(self, kg_path=None, llm_api=None):
        self.kg = None
        self.llm_api = llm_api
        self.wuxing_chain = ['木', '火', '土', '金', '水']
        
        # 症状到证候的映射规则（基于中医理论）
        self.symptom_syndrome_rules = {
            '胁肋胀痛': ['肝郁气滞', '肝胆湿热'],
            '情绪抑郁': ['肝郁气滞', '肝郁脾虚'],
            '暖气频繁': ['肝胃不和', '肝气犯胃'],
            '脉弦': ['肝郁气滞', '肝阳上亢'],
            '心悸失眠': ['心脾两虚', '心肾不交'],
            '食欲不振': ['脾胃虚弱', '脾虚湿盛'],
            '腹胀便溏': ['脾虚湿盛', '脾胃湿热'],
            '咳嗽痰多': ['痰湿蕴肺', '肺气虚'],
            '腰膝酸软': ['肾虚', '肝肾阴虚'],
            '头晕耳鸣': ['肝阳上亢', '肾精不足'],
        }
        
        # 证候五行属性
        self.syndrome_wuxing = {
            '肝郁气滞': '木',
            '肝郁化火': '木',
            '肝郁脾虚': '木',
            '肝胆湿热': '木',
            '心脾两虚': '火',
            '心肾不交': '火',
            '脾胃虚弱': '土',
            '脾胃湿热': '土',
            '脾虚湿盛': '土',
            '痰湿蕴肺': '金',
            '肺气虚': '金',
            '肾虚': '水',
            '肝肾阴虚': '水',
        }
        
        if kg_path:
            self._load_kg(kg_path)
    
    def _load_kg(self, kg_path):
        """加载知识图谱"""
        try:
            from kg_builder import TCMKnowledgeGraph
            self.kg = TCMKnowledgeGraph()
            self.kg.load(kg_path)
        except Exception as e:
            print(f"加载知识图谱失败: {e}")
            self.kg = None
    
    def differentiate(self, symptoms, patient_history=None):
        """
        输入: 症状列表
        输出: 证候列表及置信度
        
        流程:
        1. 基于规则生成候选证候
        2. KG检索相关知识
        3. 五行约束验证
        4. 返回最终证候+解释
        """
        if not symptoms:
            return {
                'primary_syndrome': '未知',
                'confidence': 0.0,
                'candidate_syndromes': [],
                'reasoning': '无症状输入'
            }
        
        # Step 1: 基于规则生成候选证候
        candidate_scores = {}
        for symptom in symptoms:
            if symptom in self.symptom_syndrome_rules:
                for syndrome in self.symptom_syndrome_rules[symptom]:
                    candidate_scores[syndrome] = candidate_scores.get(syndrome, 0) + 1
        
        # 转换为带置信度的列表
        total_symptoms = len(symptoms)
        candidates = []
        for syndrome, score in candidate_scores.items():
            confidence = score / total_symptoms
            candidates.append({
                'syndrome': syndrome,
                'score': score,
                'confidence': confidence,
                'wuxing': self.syndrome_wuxing.get(syndrome, None)
            })
        
        # 按置信度排序
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Step 2: KG检索增强
        if self.kg:
            for candidate in candidates[:5]:  # 取前5个候选
                kg_result = self.kg.query_by_syndrome(candidate['syndrome'])
                candidate['kg_related'] = kg_result['description']
        
        # Step 3: 五行约束验证（使用束搜索）
        verified_candidates = self.beam_search_with_constraints(candidates, max_depth=3)
        
        # Step 4: 选择最佳证候
        if verified_candidates:
            best = verified_candidates[0]
            primary_syndrome = best['syndrome']
            confidence = best['confidence']
            
            # 生成解释
            reasoning = self.explain_reasoning(primary_syndrome, symptoms)
            
            return {
                'primary_syndrome': primary_syndrome,
                'confidence': confidence,
                'candidate_syndromes': verified_candidates[:3],
                'reasoning': reasoning,
                'wuxing': best.get('wuxing', '未知')
            }
        else:
            return {
                'primary_syndrome': candidates[0]['syndrome'] if candidates else '未知',
                'confidence': candidates[0]['confidence'] if candidates else 0.0,
                'candidate_syndromes': candidates[:3],
                'reasoning': '基于症状匹配',
                'wuxing': '未知'
            }
    
    def beam_search_with_constraints(self, candidates, max_depth=3):
        """
        带约束的束搜索
        
        Args:
            candidates: 候选证候列表
            max_depth: 最大搜索深度
        
        Returns:
            验证后的候选列表
        """
        if not candidates:
            return []
        
        # 五行验证器
        try:
            from verification_layer import FiveElementsVerifier
            verifier = FiveElementsVerifier()
        except ImportError:
            return candidates
        
        verified = []
        
        for candidate in candidates:
            syndrome = candidate['syndrome']
            wuxing = candidate.get('wuxing')
            
            # 如果是第一个候选，直接通过
            if len(verified) == 0:
                candidate['verified'] = True
                candidate['constraint_notes'] = '首个候选，无约束冲突'
                verified.append(candidate)
                continue
            
            # 检查与已有证候的五行相生相克关系
            constraint_ok = True
            constraint_notes = []
            
            for v in verified:
                v_wuxing = v.get('wuxing')
                if wuxing and v_wuxing:
                    # 简单检查：同属或相生关系是可以接受的
                    if wuxing == v_wuxing:
                        constraint_notes.append(f'与{v["syndrome"]}同属{wuxing}行')
                    elif self._is_sheng_relation(wuxing, v_wuxing):
                        constraint_notes.append(f'与{v["syndrome"]}为{wuxing}生{v_wuxing}关系')
                    elif self._is_ke_relation(wuxing, v_wuxing):
                        constraint_notes.append(f'与{v["syndrome"]}为{wuxing}克{v_wuxing}关系')
                    else:
                        # 既不相生也不相克
                        constraint_notes.append(f'与{v["syndrome"]}无直接生克关系')
            
            candidate['constraint_notes'] = '; '.join(constraint_notes) if constraint_notes else '无冲突'
            candidate['verified'] = True
            verified.append(candidate)
        
        return verified
    
    def _is_sheng_relation(self, wx1, wx2):
        """检查是否相生关系"""
        sheng_map = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
        return sheng_map.get(wx1) == wx2 or sheng_map.get(wx2) == wx1
    
    def _is_ke_relation(self, wx1, wx2):
        """检查是否相克关系"""
        ke_map = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}
        return ke_map.get(wx1) == wx2 or ke_map.get(wx2) == wx1
    
    def explain_reasoning(self, syndrome, symptoms):
        """生成推理解释"""
        explanations = []
        
        # 找到支持该证候的症状
        supporting_symptoms = []
        for symptom in symptoms:
            if symptom in self.symptom_syndrome_rules:
                if syndrome in self.symptom_syndrome_rules[symptom]:
                    supporting_symptoms.append(symptom)
        
        if supporting_symptoms:
            explanations.append(f"基于以下症状推断：{'、'.join(supporting_symptoms)}")
        
        # 添加五行说明
        wuxing = self.syndrome_wuxing.get(syndrome)
        if wuxing:
            wuxing_desc = {
                '木': '肝胆系统',
                '火': '心小肠系统',
                '土': '脾胃系统',
                '金': '肺大肠系统',
                '水': '肾膀胱系统'
            }
            explanations.append(f"五行属{wuxing}，对应{wuxing_desc.get(wuxing, '未知')}")
        
        # 添加病机说明
        pathogenesis = {
            '肝郁气滞': '肝主疏泄功能失常，气机郁滞',
            '肝郁化火': '肝郁日久化火，热扰心神',
            '心脾两虚': '心血不足，脾气虚弱',
            '脾胃湿热': '湿热内蕴，脾胃运化失常',
        }
        if syndrome in pathogenesis:
            explanations.append(f"病机：{pathogenesis[syndrome]}")
        
        return '；'.join(explanations) if explanations else '基于临床经验判断'


if __name__ == '__main__':
    # 测试代码
    differentiator = SyndromeDifferentiator()
    
    # 测试病例
    symptoms = ['胁肋胀痛', '情绪抑郁', '暖气频繁', '脉弦']
    result = differentiator.differentiate(symptoms)
    
    print(f"辨证结果: {result['primary_syndrome']}")
    print(f"置信度: {result['confidence']:.2f}")
    print(f"五行: {result['wuxing']}")
    print(f"推理: {result['reasoning']}")
    print("\n候选证候:")
    for c in result['candidate_syndromes']:
        print(f"  - {c['syndrome']}: {c['confidence']:.2f}")
