"""
AgentShield: TCM Multi-Agent Safety Defense System
中医多智能体安全防御系统
"""
import numpy as np
from typing import List, Dict, Tuple, Optional

class SafetyPolicy:
    """
    安全策略定义
    定义各种安全规则的阈值和响应
    """
    MAX_DOSAGE = {
        '麻黄': 10.0,    # g
        '附子': 15.0,
        '乌头': 3.0,
        '马钱子': 0.6,
        '细辛': 3.0,
    }
    
    FORBIDDEN_COMBINATIONS = [
        ('十八反', ['甘草', '海藻', '甘遂', '芫花', '大戟']),
        ('十九畏', ['犀角', '川草乌', '朴硝', '丁香', '母丁香']),
    ]
    
    MIN_AGE_FOR_HERB = {
        '附子': 3,
        '乌头': 5,
        '大黄': 12,
    }
    
    @classmethod
    def check_dosage(cls, herb: str, dosage: float) -> Tuple[bool, str]:
        """检查剂量是否安全"""
        max_dose = cls.MAX_DOSAGE.get(herb, 100.0)
        if dosage > max_dose:
            return False, f"{herb}剂量{dosage}g超过安全上限{max_dose}g"
        return True, "剂量安全"
    
    @classmethod
    def check_interaction(cls, prescription: List[str]) -> List[str]:
        """检查配伍禁忌"""
        warnings = []
        for name, forbidden_list in cls.FORBIDDEN_COMBINATIONS:
            for herb in prescription:
                if herb in forbidden_list:
                    warnings.append(f"违反{name}：{herb}")
        return warnings


class ThreatDetector:
    """
    威胁检测器：检测Agent输出的各类威胁
    """
    THREAT_TYPES = [
        'excessive_dosage',      # 超剂量
        'forbidden_combo',       # 禁忌配伍
        'age_inappropriate',     # 年龄不适
        'contradict_indication', # 与适应症矛盾
        'hallucination',         # 幻觉/虚构内容
    ]
    
    def __init__(self):
        self.safety_policy = SafetyPolicy()
    
    def detect(self, agent_output: Dict, patient_context: Dict) -> List[Dict]:
        """
        检测威胁
        agent_output: Agent的输出
        patient_context: 患者上下文
        返回: 威胁列表
        """
        threats = []
        
        # 检查1: 剂量
        if 'prescription' in agent_output:
            for herb, dosage in agent_output['prescription'].items():
                safe, msg = self.safety_policy.check_dosage(herb, dosage)
                if not safe:
                    threats.append({
                        'type': 'excessive_dosage',
                        'severity': 'high',
                        'herb': herb,
                        'message': msg
                    })
            
            # 检查2: 配伍禁忌
            herbs = list(agent_output['prescription'].keys())
            warnings = self.safety_policy.check_interaction(herbs)
            for w in warnings:
                threats.append({
                    'type': 'forbidden_combo',
                    'severity': 'critical',
                    'message': w
                })
        
        # 检查3: 年龄适宜性
        age = patient_context.get('age', 30)
        if 'prescription' in agent_output:
            for herb in agent_output['prescription']:
                min_age = self.safety_policy.MIN_AGE_FOR_HERB.get(herb, 0)
                if age < min_age:
                    threats.append({
                        'type': 'age_inappropriate',
                        'severity': 'high',
                        'herb': herb,
                        'message': f"患者年龄{age}岁，不宜使用{herb}（需>{min_age}岁）"
                    })
        
        # 检查4: 矛盾适应症
        if 'diagnosis' in agent_output and 'contraindications' in patient_context:
            for diag in agent_output['diagnosis']:
                if diag in patient_context.get('contraindications', []):
                    threats.append({
                        'type': 'contradict_indication',
                        'severity': 'high',
                        'message': f"诊断{diag}与患者禁忌症冲突"
                    })
        
        return threats


class InterventionEngine:
    """
    干预引擎：当检测到威胁时采取行动
    """
    def __init__(self):
        self.block_threshold = 0.7  # 超过此严重度阈值则阻断
    
    def decide_action(self, threats: List[Dict]) -> Dict:
        """
        决定干预动作
        """
        if not threats:
            return {'action': 'allow', 'reason': '无威胁'}
        
        max_severity = max(t.get('severity', 'low') for t in threats)
        severity_rank = {'critical': 3, 'high': 2, 'medium': 1, 'low': 0}
        max_rank = severity_rank.get(max_severity, 0)
        
        if max_rank >= severity_rank['critical']:
            return {
                'action': 'block',
                'reason': f"检测到{max_rank}个严重威胁",
                'blocked_threats': [t['message'] for t in threats if t.get('severity') == 'critical']
            }
        elif max_rank >= severity_rank['high']:
            return {
                'action': 'warn',
                'reason': f"检测到{max_rank}个高危威胁，建议人工复核",
                'warnings': [t['message'] for t in threats]
            }
        else:
            return {
                'action': 'log',
                'reason': '低风险，记录但不阻断'
            }


class Shield:
    """
    AgentShield: 包装任何TCM Agent的安全壳
    """
    def __init__(self, wrapped_agent):
        self.agent = wrapped_agent
        self.detector = ThreatDetector()
        self.intervention = InterventionEngine()
        self.total_requests = 0
        self.blocked_count = 0
        self.warn_count = 0
    
    def process(self, patient_data: Dict, query: str) -> Dict:
        """
        安全处理流程
        1. 调用被包装Agent
        2. 检测输出威胁
        3. 决定干预动作
        4. 执行干预
        """
        self.total_requests += 1
        
        # Step 1: 调用Agent
        agent_output = self.agent.process(patient_data, query)
        
        # Step 2: 威胁检测
        threats = self.detector.detect(agent_output, patient_data)
        
        # Step 3: 决定干预
        decision = self.intervention.decide_action(threats)
        
        # Step 4: 执行干预
        if decision['action'] == 'block':
            self.blocked_count += 1
            return {
                'status': 'blocked',
                'decision': decision,
                'threats': threats,
                'original_output': agent_output
            }
        elif decision['action'] == 'warn':
            self.warn_count += 1
            return {
                'status': 'warning',
                'decision': decision,
                'threats': threats,
                'output': agent_output
            }
        else:
            return {
                'status': 'allowed',
                'output': agent_output
            }
    
    def get_stats(self) -> Dict:
        """获取拦截统计"""
        return {
            'total': self.total_requests,
            'blocked': self.blocked_count,
            'warned': self.warn_count,
            'block_rate': self.blocked_count / max(self.total_requests, 1),
            'warn_rate': self.warn_count / max(self.total_requests, 1)
        }
