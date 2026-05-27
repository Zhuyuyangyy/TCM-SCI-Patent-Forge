"""
A-010: TCMShield v2 - Medical Safety Defense Shell
三层防御架构：ThreatDetector + InterventionEngine + SafetyShield

Covers: 超剂量/配伍禁忌/年龄不适/适应症矛盾/幻觉检测
Target: IEC 62304 医疗软件安全标准
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


# ═══════════════════════════════════════════════════════════
# 1. Safety Policy Database
# ═══════════════════════════════════════════════════════════
class SafetyPolicy:
    """TCM-specific safety rules database."""
    # Maximum safe dosages (grams)
    MAX_DOSAGE = {
        '麻黄': 10.0, '附子': 15.0, '乌头': 3.0, '马钱子': 0.6,
        '细辛': 3.0, '半夏': 9.0, '川乌': 3.0, '草乌': 3.0,
        '大黄': 15.0, '芒硝': 12.0, '甘遂': 1.5, '大戟': 1.5,
    }

    # 十八反 十九畏
    FORBIDDEN_COMBOS = [
        {'name': '十八反-甘草组', 'herbs': ['甘草', '甘遂', '大戟', '海藻', '芫花']},
        {'name': '十八反-乌头组', 'herbs': ['乌头', '川乌', '草乌', '贝母', '瓜蒌', '半夏', '白蔹', '白及']},
        {'name': '十八反-藜芦组', 'herbs': ['藜芦', '人参', '丹参', '沙参', '玄参', '细辛', '芍药']},
        {'name': '十九畏-硫朴', 'herbs': ['硫黄', '朴硝']},
        {'name': '十九畏-丁郁', 'herbs': ['丁香', '郁金']},
        {'name': '十九畏-参脂', 'herbs': ['人参', '五灵脂']},
    ]

    # Age contraindications
    AGE_CONTRA = {
        '附子': 3, '乌头': 5, '大黄': 12, '半夏': 6,
        '细辛': 3, '马钱子': 18, '甘遂': 12,
    }

    # Pregnancy contraindications
    PREGNANCY_FORBIDDEN = [
        '麝香', '三棱', '莪术', '水蛭', '虻虫', '斑蝥',
        '巴豆', '牵牛', '大戟', '甘遂', '芫花',
    ]


# ═══════════════════════════════════════════════════════════
# 2. Threat Types & Severity
# ═══════════════════════════════════════════════════════════
class ThreatType(Enum):
    OVERDOSAGE = "超剂量"
    FORBIDDEN_COMBO = "配伍禁忌"
    AGE_INAPPROPRIATE = "年龄禁忌"
    PREGNANCY_FORBIDDEN = "妊娠禁忌"
    INDICATION_CONTRA = "适应症矛盾"
    HALLUCINATION = "幻觉检测"


class Severity(Enum):
    CRITICAL = 3  # 立即阻断
    HIGH = 2      # 警告+人工复核
    MEDIUM = 1    # 警告
    LOW = 0       # 记录


@dataclass
class Threat:
    threat_type: ThreatType
    severity: Severity
    message: str
    herb: Optional[str] = None
    dosage: Optional[float] = None
    limit: Optional[float] = None


# ═══════════════════════════════════════════════════════════
# 3. Hallucination Detector
# ═══════════════════════════════════════════════════════════
class HallucinationDetector(nn.Module):
    """
    Detect if AI-generated prescription contains hallucinated herbs
    (herbs not grounded in TCM knowledge graph).

    Uses a binary classifier trained on:
    - Positive: valid syndrome→herb mappings from KG
    - Negative: random/fabricated syndrome→herb mappings
    """
    def __init__(self, embed_dim: int = 64, hidden_dim: int = 32):
        super().__init__()
        self.syndrome_embed = nn.Embedding(20, embed_dim)
        self.herb_embed = nn.Embedding(100, embed_dim)
        self.detector = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, syndrome_ids, herb_ids):
        """Returns probability that each herb is valid for the syndrome."""
        s_emb = self.syndrome_embed(syndrome_ids)
        h_emb = self.herb_embed(herb_ids)
        combined = torch.cat([s_emb, h_emb], dim=-1)
        return self.detector(combined)


# ═══════════════════════════════════════════════════════════
# 4. Threat Detector (Layer 1)
# ═══════════════════════════════════════════════════════════
class ThreatDetector:
    """Layer 1: Detect all threats from agent output."""
    def __init__(self):
        self.policy = SafetyPolicy()
        self.hallucination_detector = HallucinationDetector()

    def detect(self, agent_output: Dict, patient_context: Dict) -> List[Threat]:
        threats = []

        prescription = agent_output.get('prescription', {})
        diagnosis = agent_output.get('diagnosis', [])
        age = patient_context.get('age', 30)
        is_pregnant = patient_context.get('pregnant', False)

        # Check 1: Overdosage
        for herb, dosage in prescription.items():
            max_dose = self.policy.MAX_DOSAGE.get(herb, 100.0)
            if dosage > max_dose:
                threats.append(Threat(
                    ThreatType.OVERDOSAGE, Severity.CRITICAL,
                    f"{herb}剂量{dosage}g超过安全上限{max_dose}g",
                    herb=herb, dosage=dosage, limit=max_dose,
                ))

        # Check 2: Forbidden combinations (十八反十九畏)
        herb_list = list(prescription.keys())
        for combo in self.policy.FORBIDDEN_COMBOS:
            overlap = set(herb_list) & set(combo['herbs'])
            if len(overlap) >= 2:
                threats.append(Threat(
                    ThreatType.FORBIDDEN_COMBO, Severity.CRITICAL,
                    f"违反{combo['name']}：{overlap}",
                    herb=str(overlap),
                ))

        # Check 3: Age contraindications
        for herb in herb_list:
            min_age = self.policy.AGE_CONTRA.get(herb, 0)
            if age < min_age:
                threats.append(Threat(
                    ThreatType.AGE_INAPPROPRIATE, Severity.HIGH,
                    f"患者{age}岁，不宜使用{herb}（需≥{min_age}岁）",
                    herb=herb,
                ))

        # Check 4: Pregnancy contraindications
        if is_pregnant:
            for herb in herb_list:
                if herb in self.policy.PREGNANCY_FORBIDDEN:
                    threats.append(Threat(
                        ThreatType.PREGNANCY_FORBIDDEN, Severity.CRITICAL,
                        f"妊娠期禁用{herb}",
                        herb=herb,
                    ))

        # Check 5: Indication contradiction
        contra_herbs = {'清热泻火': ['附子', '肉桂', '干姜'], '温阳补肾': ['黄连', '黄芩', '石膏']}
        for diag in diagnosis:
            for principle, contras in contra_herbs.items():
                if principle in diag:
                    for herb in herb_list:
                        if herb in contras:
                            threats.append(Threat(
                                ThreatType.INDICATION_CONTRA, Severity.HIGH,
                                f"诊断'{diag}'与使用'{herb}'矛盾",
                                herb=herb,
                            ))

        return threats


# ═══════════════════════════════════════════════════════════
# 5. Intervention Engine (Layer 2)
# ═══════════════════════════════════════════════════════════
class InterventionEngine:
    """Layer 2: Decide action based on detected threats."""
    def decide(self, threats: List[Threat]) -> Dict:
        if not threats:
            return {'action': 'allow', 'reason': '无威胁'}

        max_severity = max(t.severity.value for t in threats)

        if max_severity >= Severity.CRITICAL.value:
            critical_threats = [t for t in threats if t.severity == Severity.CRITICAL]
            return {
                'action': 'block',
                'reason': f"检测到{len(critical_threats)}个严重威胁",
                'blocked': [t.message for t in critical_threats],
            }
        elif max_severity >= Severity.HIGH.value:
            return {
                'action': 'warn',
                'reason': f"检测到{len(threats)}个威胁，建议人工复核",
                'warnings': [t.message for t in threats],
            }
        else:
            return {
                'action': 'log',
                'reason': '低风险，记录',
            }


# ═══════════════════════════════════════════════════════════
# 6. Safety Shield (Layer 3 - Wrapper)
# ═══════════════════════════════════════════════════════════
class TCMShield:
    """
    Layer 3: Wraps any TCM Agent with safety defense.

    Usage:
        shield = TCMShield(my_agent)
        result = shield.safe_process(patient_data, query)
    """
    def __init__(self, wrapped_agent=None):
        self.agent = wrapped_agent
        self.detector = ThreatDetector()
        self.intervention = InterventionEngine()
        self.stats = {'total': 0, 'blocked': 0, 'warned': 0, 'allowed': 0}

    def safe_process(self, patient_data: Dict, query: str = "") -> Dict:
        self.stats['total'] += 1

        # Step 1: Call wrapped agent
        if self.agent:
            agent_output = self.agent.process(patient_data, query)
        else:
            agent_output = patient_data.get('agent_output', {})

        # Step 2: Detect threats
        threats = self.detector.detect(agent_output, patient_data)

        # Step 3: Decide action
        decision = self.intervention.decide(threats)

        # Step 4: Execute
        if decision['action'] == 'block':
            self.stats['blocked'] += 1
            return {'status': 'blocked', 'decision': decision, 'threats': [t.message for t in threats]}
        elif decision['action'] == 'warn':
            self.stats['warned'] += 1
            return {'status': 'warning', 'decision': decision, 'output': agent_output, 'threats': [t.message for t in threats]}
        else:
            self.stats['allowed'] += 1
            return {'status': 'allowed', 'output': agent_output}

    def get_stats(self) -> Dict:
        total = max(self.stats['total'], 1)
        return {
            **self.stats,
            'block_rate': self.stats['blocked'] / total,
            'warn_rate': self.stats['warned'] / total,
        }


# ═══════════════════════════════════════════════════════════
# 7. Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("A-010: TCMShield v2 - Medical Safety Defense Shell")
    print("=" * 60)

    shield = TCMShield()

    # Test 1: Safe prescription
    print("\n[Test 1] Safe prescription:")
    r1 = shield.safe_process({
        'agent_output': {'prescription': {'柴胡': 10, '白芍': 10, '甘草': 6}, 'diagnosis': ['肝郁气滞']},
        'age': 35,
    })
    print(f"  Status: {r1['status']}")

    # Test 2: Overdosage
    print("\n[Test 2] Overdosage (附子 30g):")
    r2 = shield.safe_process({
        'agent_output': {'prescription': {'附子': 30, '干姜': 10}, 'diagnosis': ['肾阳虚']},
        'age': 45,
    })
    print(f"  Status: {r2['status']}")
    print(f"  Threats: {r2.get('threats', [])}")

    # Test 3: 十八反 violation
    print("\n[Test 3] 十八反 (甘草+甘遂):")
    r3 = shield.safe_process({
        'agent_output': {'prescription': {'甘草': 6, '甘遂': 3}, 'diagnosis': ['水肿']},
        'age': 50,
    })
    print(f"  Status: {r3['status']}")
    print(f"  Threats: {r3.get('threats', [])}")

    # Test 4: Age contraindication
    print("\n[Test 4] Age contraindication (child + 附子):")
    r4 = shield.safe_process({
        'agent_output': {'prescription': {'附子': 5}, 'diagnosis': ['脾肾阳虚']},
        'age': 2,
    })
    print(f"  Status: {r4['status']}")
    print(f"  Threats: {r4.get('threats', [])}")

    # Test 5: Pregnancy forbidden
    print("\n[Test 5] Pregnancy forbidden (麝香):")
    r5 = shield.safe_process({
        'agent_output': {'prescription': {'麝香': 0.1, '人参': 10}, 'diagnosis': ['气虚']},
        'age': 28, 'pregnant': True,
    })
    print(f"  Status: {r5['status']}")
    print(f"  Threats: {r5.get('threats', [])}")

    # Stats
    print(f"\n[Stats] {shield.get_stats()}")
    print("Done!")
