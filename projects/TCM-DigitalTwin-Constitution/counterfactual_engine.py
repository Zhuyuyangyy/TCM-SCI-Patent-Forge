#!/usr/bin/env python3
"""D01 Counterfactual Reasoning Engine - Intervention Effect Simulation"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constitution_state as cs


# Disease risk database keyed by constitution type
DISEASE_RISKS = {
    "气虚质": {"呼吸系统疾病": 0.4, "免疫下降": 0.5, "疲劳综合征": 0.6},
    "阳虚质": {"心脑血管病": 0.6, "甲状腺功能低下": 0.4, "代谢综合征": 0.5},
    "阴虚质": {"失眠": 0.5, "干燥综合征": 0.4, "便秘": 0.4},
    "痰湿质": {"高血脂": 0.6, "脂肪肝": 0.5, "痛风": 0.4},
    "湿热质": {"痤疮": 0.4, "胆囊炎": 0.3, "代谢综合征": 0.5},
    "血瘀质": {"心脑血管病": 0.7, "子宫肌瘤": 0.4, "痛经": 0.5},
    "气郁质": {"抑郁症": 0.5, "焦虑症": 0.5, "失眠": 0.4},
    "特禀质": {"过敏性疾病": 0.7, "哮喘": 0.5, "皮炎": 0.4},
    "平和质": {"健康": 0.1, "亚健康": 0.1, "慢病风险": 0.1},
}


# Intervention effect coefficients (simplified TCM model)
# Format: {intervention: {constitution: improvement_delta, risk_delta}}
INTERVENTION_EFFECTS = {
    "运动": {
        "气虚质": {"平和质": 0.3, "湿热质": -0.1, "血瘀质": 0.2},
        "阳虚质": {"平和质": 0.2, "痰湿质": 0.1},
        "阴虚质": {"平和质": 0.1},
        "痰湿质": {"平和质": 0.35, "湿热质": 0.1},
        "湿热质": {"平和质": 0.25, "血瘀质": 0.05},
        "血瘀质": {"平和质": 0.2, "气郁质": 0.1},
        "气郁质": {"平和质": 0.3},
        "特禀质": {"平和质": 0.15},
    },
    "饮食调节": {
        "气虚质": {"平和质": 0.2, "阳虚质": 0.15},
        "阳虚质": {"平和质": 0.3, "气虚质": 0.1},
        "阴虚质": {"平和质": 0.2, "湿热质": -0.05},
        "痰湿质": {"平和质": 0.4, "湿热质": 0.1},
        "湿热质": {"平和质": 0.2, "阴虚质": 0.1},
        "血瘀质": {"平和质": 0.15, "痰湿质": -0.1},
        "气郁质": {"平和质": 0.2},
        "特禀质": {"平和质": 0.1},
    },
    "推拿": {
        "气虚质": {"平和质": 0.15, "血瘀质": 0.1},
        "阳虚质": {"平和质": 0.25, "气虚质": 0.1},
        "阴虚质": {"平和质": 0.15, "气郁质": 0.1},
        "痰湿质": {"平和质": 0.2},
        "湿热质": {"平和质": 0.2, "血瘀质": 0.05},
        "血瘀质": {"平和质": 0.3, "气郁质": 0.1},
        "气郁质": {"平和质": 0.35, "血瘀质": 0.1},
        "特禀质": {"平和质": 0.1},
    },
    "艾灸": {
        "气虚质": {"平和质": 0.3, "阳虚质": 0.15},
        "阳虚质": {"平和质": 0.4, "气虚质": 0.2},
        "阴虚质": {"平和质": 0.1, "湿热质": -0.05},
        "痰湿质": {"平和质": 0.15},
        "湿热质": {"平和质": 0.1, "阳虚质": 0.05},
        "血瘀质": {"平和质": 0.2, "阳虚质": 0.1},
        "气郁质": {"平和质": 0.25, "阳虚质": 0.1},
        "特禀质": {"平和质": 0.15, "阳虚质": 0.05},
    },
}


class CounterfactualScenario:
    """A counterfactual scenario: what if we applied certain interventions?"""

    def __init__(self, name: str, initial_state, interventions: list):
        self.name = name
        self.initial_state = initial_state
        self.interventions = interventions
        self.final_state = None
        self.simulated = False

    def simulate(self):
        """Simulate the effect of interventions on the initial state."""
        self.final_state = type(self.initial_state)(
            constitution_scores=dict(self.initial_state.constitution_scores),
            yin_index=self.initial_state.yin_index, yang_index=self.initial_state.yang_index,
            qi_index=self.initial_state.qi_index, blood_index=self.initial_state.blood_index,
        )
        for (intervention, duration) in self.interventions:
            self._apply_intervention(intervention, duration)
        self.simulated = True

    def _apply_intervention(self, intervention: str, duration: int):
        """Apply a single intervention and update state."""
        effects = INTERVENTION_EFFECTS.get(intervention, {})
        dom = self.final_state.get_dominant_constitution()[0]

        # Get effect on dominant constitution
        delta = effects.get(dom, {})

        for target, change in delta.items():
            if target in self.final_state.constitution_scores:
                self.final_state.constitution_scores[target] = min(
                    1.0, max(0.0, self.final_state.constitution_scores[target] + change * duration / 30)
                )

        # Re-normalize
        total = sum(self.final_state.constitution_scores.values())
        if total > 0:
            for k in self.final_state.constitution_scores:
                self.final_state.constitution_scores[k] /= total


class CounterfactualEngine:
    """Engine for running counterfactual reasoning over constitution states."""

    def __init__(self, baseline_state):
        self.baseline_state = baseline_state
        self.scenarios = {}

    def create_scenario(self, name: str, interventions: list):
        scenario = CounterfactualScenario(name, self.baseline_state, interventions)
        self.scenarios[name] = scenario
        return scenario

    def compare_scenarios(self):
        """Compare all scenarios and return results."""
        results = {}
        for name, scenario in self.scenarios.items():
            if not scenario.simulated:
                scenario.simulate()

            dom_init = scenario.initial_state.get_dominant_constitution()[0]
            dom_final = scenario.final_state.get_dominant_constitution()[0]

            init_risks = DISEASE_RISKS.get(dom_init, {})
            final_risks = DISEASE_RISKS.get(dom_final, {})

            improvement = self._compute_improvement(scenario.initial_state, scenario.final_state)

            results[name] = {
                "dominant_change": (dom_init, dom_final),
                "improvement_score": improvement,
                "initial_state": scenario.initial_state,
                "final_state": scenario.final_state,
                "initial_risks": init_risks,
                "final_risks": final_risks,
            }
        return results

    def _compute_improvement(self, initial, final):
        """Compute improvement score: gain in balanced constitution + reduction in pathological dominance."""
        balanced_gain = final.constitution_scores.get("平和质", 0.0) - initial.constitution_scores.get("平和质", 0.0)
        init_dom = initial.get_dominant_constitution()[1]
        final_dom = final.get_dominant_constitution()[1]
        dominance_reduction = init_dom - final_dom
        return balanced_gain * 0.6 + dominance_reduction * 0.4

    def generate_report(self):
        """Generate text report of all scenarios."""
        if not self.scenarios:
            return "No scenarios created yet."
        results = self.compare_scenarios()
        lines = []
        lines.append("=" * 60)
        lines.append("Constitution Digital Twin - Intervention Counterfactual Report")
        lines.append("=" * 60)
        dom = self.baseline_state.get_dominant_constitution()
        lines.append("Baseline: " + str(self.baseline_state))
        lines.append("Dominant: " + str(dom[0]))
        lines.append("Yin-Yang: " + str(self.baseline_state.get_yin_yang_state()))
        lines.append("Qi-Blood: " + str(self.baseline_state.get_qi_blood_state()))
        lines.append("-" * 60)
        lines.append("Intervention Comparison")
        lines.append("-" * 60)
        for name, result in results.items():
            lines.append("--- " + name + " ---")
            dc = result["dominant_change"]
            lines.append("  Dominant change: " + str(dc[0]) + " -> " + str(dc[1]))
            lines.append("  Improvement: " + str(round(result["improvement_score"], 3)))
            init_bal = result["initial_state"].constitution_scores.get("平和质", 0.0)
            final_bal = result["final_state"].constitution_scores.get("平和质", 0.0)
            lines.append("  Balanced: " + str(round(init_bal, 2)) + " -> " + str(round(final_bal, 2)))
            if result["initial_risks"] and result["final_risks"]:
                lines.append("  Risk changes:")
                for risk in result["initial_risks"].keys():
                    init_risk = result["initial_risks"].get(risk, 0)
                    final_risk = result["final_risks"].get(risk, 0)
                    change = final_risk - init_risk
                    arrow = " DOWN" if change < 0 else (" UP" if change > 0 else " SAME")
                    lines.append("    " + risk + ": " + str(round(init_risk, 2)) + arrow + " " + str(round(final_risk, 2)))
        lines.append("=" * 60)
        lines.append("Recommendation")
        lines.append("=" * 60)
        best = max(results.items(), key=lambda x: x[1]["improvement_score"])
        lines.append("Based on improvement score, recommend: [" + best[0] + "]")
        return "\n".join(lines)


def quick_counterfactual(baseline, interventions, name="Scenario"):
    engine = CounterfactualEngine(baseline)
    scenario = engine.create_scenario(name, interventions)
    scenario.simulate()
    return scenario
