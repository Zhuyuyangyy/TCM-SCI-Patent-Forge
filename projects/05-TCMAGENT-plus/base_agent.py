"""
Base Agent for TCM Multi-Agent System
TCM多智能体系统基础Agent类
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseTCMAgent(ABC):
    def __init__(self, name: str, role: str):
        self.name = name
        self.role = role
        self.evidence = []
        
    @abstractmethod
    def process(self, patient_data: Dict) -> Dict:
        """处理数据并返回证据"""
        pass
    
    def add_evidence(self, evidence: Dict):
        self.evidence.append(evidence)
    
    def get_evidence(self) -> List[Dict]:
        return self.evidence
    
    def clear_evidence(self):
        self.evidence = []
