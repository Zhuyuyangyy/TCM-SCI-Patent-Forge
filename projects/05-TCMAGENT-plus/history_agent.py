"""
Patient History Agent
病史信息采集与分析Agent
"""
from base_agent import BaseTCMAgent
from typing import List, Dict

class HistoryAgent(BaseTCMAgent):
    """
    病史采集Agent
    输入: 症状描述、病程时间、既往史
    输出: 主要症状 + 证候线索
    """
    def __init__(self):
        super().__init__('HistoryAgent', '病史专家')
        
    def process(self, patient_data: Dict) -> Dict:
        symptoms = patient_data.get('symptoms', [])
        duration = patient_data.get('duration', '未知')
        history = patient_data.get('history', '')
        
        # 提取症状特征
        symptom_analysis = self._analyze_symptoms(symptoms)
        syndrome_clues = self._extract_clues(symptoms)
        
        evidence = {
            'agent': self.name,
            'symptoms': symptoms,
            'duration': duration,
            'symptom_analysis': symptom_analysis,
            'syndrome_clues': syndrome_clues,
            'primary_syndrome': syndrome_clues[0] if syndrome_clues else '未分类',
            'confidence': 0.80,
            'reasoning': f"主要症状: {', '.join(symptoms[:3])}，符合{syndrome_clues[0] if syndrome_clues else '未分类'}"
        }
        self.add_evidence(evidence)
        return evidence
    
    def _analyze_symptoms(self, symptoms: List[str]) -> Dict:
        """分析症状特征"""
        categories = {
            '情志': ['抑郁', '焦虑', '易怒', '情绪低落'],
            '消化': ['腹胀', '食欲不振', '恶心', '便秘'],
            '睡眠': ['失眠', '多梦', '早醒', '嗜睡'],
            '疼痛': ['头痛', '胁痛', '腹痛', '腰痛']
        }
        
        result = {}
        for symptom in symptoms:
            for cat, keywords in categories.items():
                if any(kw in symptom for kw in keywords):
                    if cat not in result:
                        result[cat] = []
                    result[cat].append(symptom)
        return result
    
    def _extract_clues(self, symptoms: List[str]) -> List[str]:
        """提取证候线索"""
        clues = []
        symptom_str = ' '.join(symptoms)
        
        if any(s in symptom_str for s in ['胁肋胀痛', '抑郁', '叹气']):
            clues.append('肝郁气滞')
        if any(s in symptom_str for s in ['失眠', '心悸', '多梦']):
            clues.append('心血不足')
        if any(s in symptom_str for s in ['腹胀', '食欲不振', '乏力']):
            clues.append('脾气虚弱')
        if any(s in symptom_str for s in ['腰膝酸软', '耳鸣', '夜尿多']):
            clues.append('肾虚')
        
        if not clues:
            clues.append('需进一步辨证')
        return clues
