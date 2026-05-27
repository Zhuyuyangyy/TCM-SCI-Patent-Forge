"""
Lab Results Agent
化验单分析Agent
"""
from base_agent import BaseTCMAgent
from typing import Dict

class LabAgent(BaseTCMAgent):
    """
    化验单分析Agent
    输入: 化验指标
    输出: 指标解读 + 与中医证候的关联
    """
    def __init__(self):
        super().__init__('LabAgent', '检验专家')
    
    def process(self, patient_data: Dict) -> Dict:
        lab_results = patient_data.get('lab_results', {})
        
        interpretation = self._interpret_results(lab_results)
        tcm_correlation = self._correlate_with_tcm(interpretation)
        
        evidence = {
            'agent': self.name,
            'lab_results': lab_results,
            'interpretation': interpretation,
            'tcm_correlation': tcm_correlation,
            'confidence': 0.75,
            'reasoning': f"异常指标: {interpretation['abnormal_count']}项，与中医关联: {tcm_correlation['primary_correlation']}"
        }
        self.add_evidence(evidence)
        return evidence
    
    def _interpret_results(self, results: Dict) -> Dict:
        """解释化验结果"""
        interpretation = {}
        abnormal = []
        
        for marker, value in results.items():
            if marker == 'ALT' and value > 40:
                abnormal.append(f'ALT升高({value})')
                interpretation[marker] = '肝功能异常'
            elif marker == 'AST' and value > 40:
                abnormal.append(f'AST升高({value})')
                interpretation[marker] = '肝功能异常'
            elif marker == 'Cr' and value > 110:
                abnormal.append(f'肌酐升高({value})')
                interpretation[marker] = '肾功能异常'
            else:
                interpretation[marker] = '正常'
        
        return {
            'interpretations': interpretation,
            'abnormal_count': len(abnormal),
            'abnormal_items': abnormal
        }
    
    def _correlate_with_tcm(self, interpretation: Dict) -> Dict:
        """化验结果与中医证候关联"""
        corr_map = {
            '肝功能异常': '肝郁化火',
            '肾功能异常': '肾虚',
            '血脂异常': '痰湿内阻',
            '血糖异常': '阴虚燥热'
        }
        
        correlations = []
        for marker, finding in interpretation.get('interpretations', {}).items():
            if finding != '正常' and finding in corr_map:
                correlations.append(corr_map[finding])
        
        return {
            'correlations': list(set(correlations)),
            'primary_correlation': correlations[0] if correlations else '无明显关联'
        }
