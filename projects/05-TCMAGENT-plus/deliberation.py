"""
Adversarial Deliberation Module
对抗性协作审议模块
"""
import numpy as np
from typing import List, Dict
from collections import Counter

class DeliberationModule:
    """
    协作审议：
    1. 收集各Agent证据
    2. 检测冲突
    3. 对抗性讨论
    4. 达成共识
    """
    def __init__(self):
        self.conflict_rules = {
            ('肝郁气滞', '肝火亢盛'): "肝郁化火是肝郁气滞的发展，应综合考虑",
            ('脾气虚弱', '痰湿内阻'): "脾虚生湿，湿重也可表现为实证",
        }
    
    def deliberate(self, evidence_list: List[Dict]) -> Dict:
        """
        输入: 各Agent的证据
        输出: 综合诊断结果
        """
        syndromes = [e.get('primary_syndrome') or e.get('syndrome', '未分类') 
                     for e in evidence_list]
        
        # 统计证候投票
        syndrome_votes = {}
        for s in syndromes:
            syndrome_votes[s] = syndrome_votes.get(s, 0) + 1
        
        # 检测冲突
        conflicts = self._detect_conflicts(evidence_list)
        
        # 对抗性消解
        resolved_syndrome = self._resolve_conflicts(syndromes, conflicts)
        
        # 综合置信度
        confidences = [e.get('confidence', 0) for e in evidence_list]
        avg_confidence = np.mean(confidences)
        
        return {
            'syndrome_votes': syndrome_votes,
            'final_syndrome': resolved_syndrome,
            'confidence': avg_confidence,
            'conflicts': conflicts,
            'reasoning': self._generate_reasoning(evidence_list, resolved_syndrome),
            'treatment_principle': self._get_treatment_principle(resolved_syndrome)
        }
    
    def _detect_conflicts(self, evidence_list: List[Dict]) -> List[Dict]:
        """检测证据间的冲突"""
        conflicts = []
        syndromes = [e.get('primary_syndrome') or e.get('syndrome', '') 
                     for e in evidence_list]
        
        for i in range(len(syndromes)):
            for j in range(i+1, len(syndromes)):
                pair = tuple(sorted([syndromes[i], syndromes[j]]))
                if pair in self.conflict_rules:
                    conflicts.append({
                        'pair': pair,
                        'resolution': self.conflict_rules[pair]
                    })
        
        return conflicts
    
    def _resolve_conflicts(self, syndromes: List[str], conflicts: List[Dict]) -> str:
        """消解冲突，确定最终证候"""
        if not conflicts:
            # 无冲突，取票数最多的
            counter = Counter(syndromes)
            return counter.most_common(1)[0][0]
        
        # 有冲突，根据规则综合
        resolved = syndromes[0]
        for conflict in conflicts:
            pair = conflict['pair']
            resolution = conflict['resolution']
            # 简单处理：取第一个非冲突的证候
            for s in syndromes:
                if s not in pair:
                    resolved = s
                    break
        return resolved
    
    def _generate_reasoning(self, evidence_list: List[Dict], final: str) -> str:
        reasons = []
        for e in evidence_list:
            agent = e.get('agent', 'Unknown')
            syndrome = e.get('primary_syndrome') or e.get('syndrome', '')
            reason = e.get('reasoning', '')
            reasons.append(f"{agent}认为{syndrome}。{reason}")
        return ' | '.join(reasons) + f'。综合判定为{final}。'
    
    def _get_treatment_principle(self, syndrome: str) -> str:
        """根据证候确定治法"""
        principles = {
            '肝郁气滞': '疏肝理气',
            '肝火亢盛': '清肝泻火',
            '脾气虚弱': '健脾益气',
            '痰湿内阻': '化痰祛湿',
            '肾虚': '补肾益精',
            '心血不足': '养心安神',
            '阴虚火旺': '滋阴降火'
        }
        return principles.get(syndrome, '辨证论治')
