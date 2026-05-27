"""
Target 010: MACCM v2 - Multi-Agent Collaborative Consultation Mechanism
Adversarial deliberation with conflict resolution for TCM diagnosis.

Architecture:
  Patient Input → Problem Decomposer → [6 Parallel Expert Agents] →
  Evidence Aggregator → Adversarial Deliberation → Consensus Engine → Final Diagnosis

Agents:
  Agent-A: 病位分析 (Location)
  Agent-B: 病性分析 (Nature: 寒热虚实)
  Agent-C: 病因分析 (Etiology)
  Agent-D: 传变分析 (Progression)
  Agent-E: 古今经验 (Historical Cases)
  Agent-F: 药理安全 (Safety Check)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json


# ═══════════════════════════════════════════════════════════
# 1. Agent State & Evidence Structures
# ═══════════════════════════════════════════════════════════
class AgentRole(Enum):
    LOCATION = "病位分析"
    NATURE = "病性分析"
    ETIOLOGY = "病因分析"
    PROGRESSION = "传变分析"
    EXPERIENCE = "古今经验"
    SAFETY = "药理安全"


@dataclass
class AgentEvidence:
    """Evidence output from a single agent."""
    agent_role: AgentRole
    primary_syndrome: str
    confidence: float
    supporting_symptoms: List[str]
    reasoning_chain: List[str]
    alternative_syndromes: List[Tuple[str, float]]  # (syndrome, conf)
    raw_scores: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'agent': self.agent_role.value,
            'syndrome': self.primary_syndrome,
            'confidence': self.confidence,
            'symptoms': self.supporting_symptoms,
            'reasoning': self.reasoning_chain,
            'alternatives': self.alternative_syndromes,
        }


@dataclass
class DeliberationResult:
    """Final deliberation output."""
    final_syndrome: str
    confidence: float
    treatment_principle: str
    consensus_score: float
    conflicts_detected: List[Dict]
    conflict_resolution: List[str]
    agent_votes: Dict[str, float]
    reasoning_trace: List[str]


# ═══════════════════════════════════════════════════════════
# 2. Expert Agent Base Class
# ═══════════════════════════════════════════════════════════
class ExpertAgent(nn.Module):
    """Base class for all expert agents."""
    def __init__(self, role: AgentRole, input_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.role = role
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.syndrome_head = nn.Linear(hidden_dim, 20)   # 20 syndrome classes
        self.confidence_head = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(features)
        syndrome_logits = self.syndrome_head(h)
        confidence = torch.sigmoid(self.confidence_head(h))
        return syndrome_logits, confidence


class LocationAgent(ExpertAgent):
    """Agent-A: 病位分析 — identifies affected organs (脏腑定位)."""
    def __init__(self):
        super().__init__(AgentRole.LOCATION)
        # TCM organ mapping
        self.organ_map = {
            '肝': ['胁肋胀痛', '情绪抑郁', '脉弦', '目赤'],
            '心': ['心悸', '失眠', '舌尖红', '脉结代'],
            '脾': ['食欲不振', '腹胀', '便溏', '舌淡胖'],
            '肺': ['咳嗽', '气喘', '胸闷', '脉浮'],
            '肾': ['腰膝酸软', '耳鸣', '夜尿多', '脉沉'],
        }

    def analyze_symptoms(self, symptoms: List[str]) -> AgentEvidence:
        organ_scores = defaultdict(float)
        supporting = defaultdict(list)

        for symptom in symptoms:
            for organ, indicators in self.organ_map.items():
                if any(ind in symptom for ind in indicators):
                    organ_scores[organ] += 1.0
                    supporting[organ].append(symptom)

        # Normalize
        total = sum(organ_scores.values()) or 1.0
        for k in organ_scores:
            organ_scores[k] /= total

        if organ_scores:
            primary = max(organ_scores, key=organ_scores.get)
            alternatives = sorted(
                [(k, v) for k, v in organ_scores.items() if k != primary],
                key=lambda x: -x[1]
            )[:3]
        else:
            primary = '未知'
            alternatives = []

        return AgentEvidence(
            agent_role=self.role,
            primary_syndrome=f"{primary}系病变",
            confidence=max(organ_scores.values()) if organ_scores else 0.0,
            supporting_symptoms=supporting.get(primary, []),
            reasoning_chain=[f"症状{sym}提示{s}" for sym in symptoms for s in [self._infer_organ(sym)] if s],
            alternative_syndromes=alternatives,
            raw_scores=dict(organ_scores),
        )

    def _infer_organ(self, symptom: str) -> Optional[str]:
        for organ, indicators in self.organ_map.items():
            if any(ind in symptom for ind in indicators):
                return organ
        return None


class NatureAgent(ExpertAgent):
    """Agent-B: 病性分析 — 寒热虚实 (Cold/Heat/Deficiency/Excess)."""
    def __init__(self):
        super().__init__(AgentRole.NATURE)
        self.nature_keywords = {
            '寒': ['畏寒', '肢冷', '喜温', '舌淡', '脉迟'],
            '热': ['发热', '口渴', '舌红', '脉数', '便秘'],
            '虚': ['乏力', '气短', '自汗', '脉弱', '舌淡'],
            '实': ['胀痛', '拒按', '脉实', '便秘', '舌红'],
        }

    def analyze_symptoms(self, symptoms: List[str]) -> AgentEvidence:
        scores = defaultdict(float)
        for symptom in symptoms:
            for nature, keywords in self.nature_keywords.items():
                if any(kw in symptom for kw in keywords):
                    scores[nature] += 1.0

        total = sum(scores.values()) or 1.0
        for k in scores:
            scores[k] /= total

        # Combine into syndrome patterns
        pattern = self._classify_pattern(scores)
        return AgentEvidence(
            agent_role=self.role,
            primary_syndrome=pattern,
            confidence=max(scores.values()) if scores else 0.0,
            supporting_symptoms=symptoms,
            reasoning_chain=[f"病性: {k}={v:.2f}" for k, v in scores.items()],
            alternative_syndromes=[],
            raw_scores=dict(scores),
        )

    def _classify_pattern(self, scores: Dict[str, float]) -> str:
        hot_cold = '热' if scores.get('热', 0) > scores.get('寒', 0) else '寒'
        xu_shi = '实' if scores.get('实', 0) > scores.get('虚', 0) else '虚'
        return f"{xu_shi}{hot_cold}"


# ═══════════════════════════════════════════════════════════
# 3. Adversarial Deliberation Module
# ═══════════════════════════════════════════════════════════
class AdversarialDeliberation(nn.Module):
    """
    Core innovation: adversarial debate between agents.

    Process:
    1. Collect evidence from all agents
    2. Detect conflicts (agents disagree on syndrome)
    3. For each conflict, generate adversarial arguments
    4. Weighted voting with confidence-based routing
    5. Converge to consensus or flag for human review
    """
    def __init__(self, num_agents: int = 6, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents

        # Agent credibility weights (learned)
        self.agent_weights = nn.Parameter(torch.ones(num_agents) / num_agents)

        # Conflict detector
        self.conflict_detector = nn.Sequential(
            nn.Linear(num_agents * 20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

        # Consensus network
        self.consensus_net = nn.Sequential(
            nn.Linear(num_agents * 20, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20),  # 20 syndrome classes
        )

    def detect_conflicts(
        self, evidences: List[AgentEvidence]
    ) -> List[Dict]:
        """Detect disagreements between agents."""
        conflicts = []
        syndromes = [e.primary_syndrome for e in evidences]

        for i in range(len(syndromes)):
            for j in range(i + 1, len(syndromes)):
                if self._syndromes_conflict(syndromes[i], syndromes[j]):
                    conflicts.append({
                        'agent_i': evidences[i].agent_role.value,
                        'agent_j': evidences[j].agent_role.value,
                        'syndrome_i': syndromes[i],
                        'syndrome_j': syndromes[j],
                        'confidence_i': evidences[i].confidence,
                        'confidence_j': evidences[j].confidence,
                        'type': 'syndrome_disagreement',
                    })
        return conflicts

    def _syndromes_conflict(self, s1: str, s2: str) -> bool:
        """Check if two syndromes are contradictory."""
        conflict_pairs = [
            ('寒', '热'), ('虚', '实'),
            ('阴虚', '阳虚'), ('气虚', '气滞'),
        ]
        for a, b in conflict_pairs:
            if (a in s1 and b in s2) or (b in s1 and a in s2):
                return True
        return False

    def weighted_vote(
        self, evidences: List[AgentEvidence]
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Confidence-weighted voting across all agents.
        Uses learned agent credibility weights.
        """
        syndrome_votes = defaultdict(float)
        weights = F.softmax(self.agent_weights[:len(evidences)], dim=0)

        for i, evidence in enumerate(evidences):
            w = weights[i].item() * evidence.confidence
            syndrome_votes[evidence.primary_syndrome] += w

            for alt_syndrome, alt_conf in evidence.alternative_syndromes:
                syndrome_votes[alt_syndrome] += w * alt_conf * 0.5

        # Normalize
        total = sum(syndrome_votes.values()) or 1.0
        for k in syndrome_votes:
            syndrome_votes[k] /= total

        final = max(syndrome_votes, key=syndrome_votes.get)
        final_conf = syndrome_votes[final]

        return final, final_conf, dict(syndrome_votes)

    def deliberate(
        self, evidences: List[AgentEvidence]
    ) -> DeliberationResult:
        """Full deliberation pipeline."""
        # Step 1: Detect conflicts
        conflicts = self.detect_conflicts(evidences)

        # Step 2: Weighted voting
        final_syndrome, confidence, votes = self.weighted_vote(evidences)

        # Step 3: Conflict resolution
        resolution = []
        if conflicts:
            resolution = self._resolve_conflicts(conflicts, evidences)

        # Step 4: Consensus score
        consensus = 1.0 - len(conflicts) / max(len(evidences) * (len(evidences)-1) / 2, 1)

        # Step 5: Treatment principle
        treatment = self._infer_treatment(final_syndrome)

        # Step 6: Reasoning trace
        trace = self._build_reasoning_trace(evidences, conflicts, resolution, final_syndrome)

        return DeliberationResult(
            final_syndrome=final_syndrome,
            confidence=confidence,
            treatment_principle=treatment,
            consensus_score=consensus,
            conflicts_detected=conflicts,
            conflict_resolution=resolution,
            agent_votes=votes,
            reasoning_trace=trace,
        )

    def _resolve_conflicts(
        self, conflicts: List[Dict], evidences: List[AgentEvidence]
    ) -> List[str]:
        """Resolve conflicts via majority vote + confidence."""
        resolutions = []
        for conflict in conflicts:
            if conflict['confidence_i'] > conflict['confidence_j'] * 1.2:
                resolutions.append(
                    f"采纳{conflict['agent_i']}意见(置信度{conflict['confidence_i']:.2f})"
                    f"，{conflict['agent_j']}意见保留"
                )
            else:
                resolutions.append(
                    f"{conflict['agent_i']}与{conflict['agent_j']}存在分歧，"
                    f"建议人工复核"
                )
        return resolutions

    def _infer_treatment(self, syndrome: str) -> str:
        """Map syndrome to treatment principle."""
        treatment_map = {
            '肝郁气滞': '疏肝理气',
            '心脾两虚': '补益心脾',
            '脾胃湿热': '清热利湿',
            '肾阳虚': '温补肾阳',
            '痰湿蕴肺': '化痰祛湿',
            '气阴两虚': '益气养阴',
        }
        for key, principle in treatment_map.items():
            if key in syndrome:
                return principle
        return '辨证论治'

    def _build_reasoning_trace(
        self, evidences, conflicts, resolutions, final
    ) -> List[str]:
        trace = []
        for e in evidences:
            trace.append(f"[{e.agent_role.value}] 主张: {e.primary_syndrome} (置信度{e.confidence:.2f})")
        if conflicts:
            trace.append(f"[冲突检测] 发现{len(conflicts)}处分歧")
            for r in resolutions:
                trace.append(f"[消解] {r}")
        trace.append(f"[共识] 最终证候: {final}")
        return trace


# ═══════════════════════════════════════════════════════════
# 4. MACCM System (Full Pipeline)
# ═══════════════════════════════════════════════════════════
class MACCMSystem:
    """
    Multi-Agent Collaborative Consultation Mechanism v2.
    Full pipeline: Input → Parallel Analysis → Deliberation → Output
    """
    def __init__(self):
        self.location_agent = LocationAgent()
        self.nature_agent = NatureAgent()
        # Additional agents can be added similarly
        self.deliberation = AdversarialDeliberation(num_agents=6)

    def consult(self, patient_data: Dict) -> DeliberationResult:
        """
        Full consultation pipeline.
        patient_data: {symptoms: [...], tongue: ..., pulse: ..., history: ...}
        """
        symptoms = patient_data.get('symptoms', [])

        # Parallel agent analysis
        evidences = []
        evidences.append(self.location_agent.analyze_symptoms(symptoms))
        evidences.append(self.nature_agent.analyze_symptoms(symptoms))

        # Deliberation
        result = self.deliberation.deliberate(evidences)
        return result


# ═══════════════════════════════════════════════════════════
# 5. Quick Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Target 010: MACCM v2 - Multi-Agent Consultation System")
    print("=" * 60)

    system = MACCMSystem()

    # Test case
    patient = {
        'symptoms': ['胁肋胀痛', '情绪抑郁', '嗳气频繁', '脉弦', '食欲不振'],
        'age': 45,
        'gender': '女',
    }

    print(f"\nPatient symptoms: {patient['symptoms']}")
    print("-" * 40)

    result = system.consult(patient)

    print(f"\nFinal Syndrome: {result.final_syndrome}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Treatment: {result.treatment_principle}")
    print(f"Consensus: {result.consensus_score:.2%}")
    print(f"Conflicts: {len(result.conflicts_detected)}")

    print(f"\nAgent Votes:")
    for syndrome, vote in sorted(result.agent_votes.items(), key=lambda x: -x[1]):
        print(f"  {syndrome}: {vote:.2%}")

    print(f"\nReasoning Trace:")
    for step in result.reasoning_trace:
        print(f"  {step}")

    print("\nDone!")
