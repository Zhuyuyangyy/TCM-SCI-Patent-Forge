"""
B-011: Digital Twin Constitution v2 - Counterfactual Intervention Engine
体质数字孪生 + 反事实推理引擎

9种体质 × 4种干预(运动/饮食/推拿/艾灸) × 反事实模拟
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════
# 1. Constitution State Representation
# ═══════════════════════════════════════════════════════════
@dataclass
class ConstitutionState:
    """9-dimensional constitution state vector."""
    scores: Dict[str, float]  # 9 constitution types
    yin_yang: float = 0.5     # 阴阳平衡指数
    qi_blood: float = 0.5     # 气血充盈指数

    CONSTITUTIONS = [
        '平和质', '气虚质', '阳虚质', '阴虚质',
        '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质'
    ]

    def __post_init__(self):
        if not self.scores:
            self.scores = {c: 1.0/9 for c in self.CONSTITUTIONS}

    def dominant(self) -> Tuple[str, float]:
        dom = max(self.scores, key=self.scores.get)
        return dom, self.scores[dom]

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor([self.scores[c] for c in self.CONSTITUTIONS], dtype=torch.float32)

    @classmethod
    def from_tensor(cls, t: torch.Tensor, yin_yang=0.5, qi_blood=0.5):
        scores = {c: t[i].item() for i, c in enumerate(cls.CONSTITUTIONS)}
        return cls(scores=scores, yin_yang=yin_yang, qi_blood=qi_blood)

    def __str__(self):
        dom, conf = self.dominant()
        return f"State(dominant={dom}, conf={conf:.3f}, yy={self.yin_yang:.2f}, qb={self.qi_blood:.2f})"


# ═══════════════════════════════════════════════════════════
# 2. Constitution Transition Model (Markov-inspired)
# ═══════════════════════════════════════════════════════════
class ConstitutionTransitionModel(nn.Module):
    """
    Learn constitution state transitions under interventions.

    S(t+1) = f(S(t), intervention, duration) + noise

    The model learns transition dynamics from data.
    """
    def __init__(self, state_dim=9, intervention_dim=4, hidden_dim=64):
        super().__init__()
        self.transition_net = nn.Sequential(
            nn.Linear(state_dim + intervention_dim + 1, hidden_dim),  # +1 for duration
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )
        # Yin-Yang and Qi-Blood dynamics
        self.yy_net = nn.Sequential(
            nn.Linear(state_dim + intervention_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )
        self.qb_net = nn.Sequential(
            nn.Linear(state_dim + intervention_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Tanh(),
        )

    def forward(self, state_tensor, intervention_vec, duration):
        """
        state_tensor: (9,) constitution scores
        intervention_vec: (4,) one-hot or continuous intervention
        duration: (1,) days
        """
        inp = torch.cat([state_tensor, intervention_vec, duration.unsqueeze(0)])
        delta = self.transition_net(inp)
        new_state = state_tensor + delta * 0.1  # small step
        new_state = F.softmax(new_state, dim=0)  # normalize to probability simplex

        yy_delta = self.yy_net(torch.cat([state_tensor, intervention_vec])) * 0.01
        qb_delta = self.qb_net(torch.cat([state_tensor, intervention_vec])) * 0.01

        return new_state, yy_delta.squeeze(), qb_delta.squeeze()


# ═══════════════════════════════════════════════════════════
# 3. Counterfactual Engine
# ═══════════════════════════════════════════════════════════
class CounterfactualEngine:
    """
    Run counterfactual simulations: "What if we applied intervention X for Y days?"

    Supports:
    - Single scenario simulation
    - Multi-scenario comparison
    - Optimal intervention recommendation
    """
    INTERVENTIONS = ['运动', '饮食调节', '推拿', '艾灸']
    DISEASE_RISKS = {
        '气虚质': {'呼吸系统': 0.4, '免疫下降': 0.5, '疲劳': 0.6},
        '阳虚质': {'心脑血管': 0.6, '甲状腺': 0.4, '代谢': 0.5},
        '阴虚质': {'失眠': 0.5, '干燥综合征': 0.4, '便秘': 0.4},
        '痰湿质': {'高血脂': 0.6, '脂肪肝': 0.5, '痛风': 0.4},
        '湿热质': {'痤疮': 0.4, '胆囊炎': 0.3, '代谢': 0.5},
        '血瘀质': {'心脑血管': 0.7, '子宫肌瘤': 0.4, '痛经': 0.5},
        '气郁质': {'抑郁': 0.5, '焦虑': 0.5, '失眠': 0.4},
        '特禀质': {'过敏': 0.7, '哮喘': 0.5, '皮炎': 0.4},
        '平和质': {'健康': 0.1, '亚健康': 0.1},
    }

    def __init__(self, transition_model: ConstitutionTransitionModel):
        self.model = transition_model

    def simulate_scenario(
        self,
        initial_state: ConstitutionState,
        intervention: str,
        duration_days: int,
    ) -> ConstitutionState:
        """Simulate constitution change under intervention."""
        self.model.eval()
        state_tensor = initial_state.to_tensor()
        int_idx = self.INTERVENTIONS.index(intervention) if intervention in self.INTERVENTIONS else 0
        int_vec = torch.zeros(4)
        int_vec[int_idx] = 1.0

        current = state_tensor.clone()
        yy = initial_state.yin_yang
        qb = initial_state.qi_blood

        # Step-wise simulation
        steps = max(1, duration_days // 7)  # weekly steps
        for _ in range(steps):
            with torch.no_grad():
                current, yy_delta, qb_delta = self.model(current, int_vec, torch.tensor(float(duration_days)))
                yy += yy_delta.item()
                qb += qb_delta.item()

        return ConstitutionState.from_tensor(current, yy, qb)

    def compare_scenarios(
        self,
        initial_state: ConstitutionState,
        scenarios: List[Tuple[str, int]],  # (intervention, duration)
    ) -> Dict:
        """Compare multiple intervention scenarios."""
        results = {}
        for intervention, duration in scenarios:
            final = self.simulate_scenario(initial_state, intervention, duration)
            dom_init, conf_init = initial_state.dominant()
            dom_final, conf_final = final.dominant()

            # Compute improvement
            balanced_gain = final.scores.get('平和质', 0) - initial_state.scores.get('平和质', 0)
            risk_reduction = self._compute_risk_reduction(initial_state, final)

            results[f"{intervention}_{duration}d"] = {
                'dominant_change': f"{dom_init}→{dom_final}",
                'balanced_gain': balanced_gain,
                'risk_reduction': risk_reduction,
                'final_state': final,
            }
        return results

    def recommend(self, initial_state: ConstitutionState, duration: int = 30) -> str:
        """Recommend best intervention."""
        scenarios = [(i, duration) for i in self.INTERVENTIONS]
        results = self.compare_scenarios(initial_state, scenarios)
        best = max(results.items(), key=lambda x: x[1]['balanced_gain'] - x[1]['risk_reduction'])
        return best[0]

    def _compute_risk_reduction(self, initial: ConstitutionState, final: ConstitutionState) -> float:
        dom_init, _ = initial.dominant()
        dom_final, _ = final.dominant()
        risks_init = self.DISEASE_RISKS.get(dom_init, {})
        risks_final = self.DISEASE_RISKS.get(dom_final, {})
        init_risk = sum(risks_init.values()) / max(len(risks_init), 1)
        final_risk = sum(risks_final.values()) / max(len(risks_final), 1)
        return init_risk - final_risk

    def generate_report(self, initial: ConstitutionState, duration: int = 30) -> str:
        """Generate human-readable report."""
        scenarios = [(i, duration) for i in self.INTERVENTIONS]
        results = self.compare_scenarios(initial, scenarios)

        lines = ["=" * 50, "体质数字孪生 - 干预方案对比报告", "=" * 50]
        lines.append(f"初始状态: {initial}")
        lines.append("-" * 50)
        for name, r in results.items():
            lines.append(f"  [{name}] 主导变化: {r['dominant_change']}, "
                        f"平和质增益: {r['balanced_gain']:.4f}, 风险降低: {r['risk_reduction']:.4f}")
        best = self.recommend(initial, duration)
        lines.append(f"\n推荐方案: {best}")
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════
# 4. Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("B-011: Digital Twin Constitution v2")
    print("=" * 60)

    model = ConstitutionTransitionModel()
    engine = CounterfactualEngine(model)

    # Initial state: 气虚质为主
    initial = ConstitutionState(
        scores={
            '平和质': 0.1, '气虚质': 0.4, '阳虚质': 0.15,
            '阴虚质': 0.05, '痰湿质': 0.1, '湿热质': 0.05,
            '血瘀质': 0.05, '气郁质': 0.05, '特禀质': 0.05,
        },
        yin_yang=0.3, qi_blood=0.4,
    )
    print(f"\n初始状态: {initial}")

    # Compare scenarios
    report = engine.generate_report(initial, duration=30)
    print(f"\n{report}")

    # Recommend
    best = engine.recommend(initial, duration=30)
    print(f"\n最终推荐: {best}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print("Done!")
