#!/usr/bin/env python3
"""D01 Transition Model & Markov Chain"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import constitution_state as cs


@dataclass
class TransitionMatrix:
    constitution_order = ["气虚质", "阳虚质", "阴虚质", 
                          "痄潿质", "潿热质", "血沇质", 
                          "气郁质", "特宾质", "尚和质"]
    
    def __init__(self):
        self.matrix = self._build_base_matrix()
        self._normalize_rows()
    
    def _build_base_matrix(self) -> np.ndarray:
        n = 9
        matrix = np.zeros((n, n))
        
        self_stability = {
            "气虚质": 0.70, "阳虚质": 0.75, "阴虚质": 0.70,
            "痄潿质": 0.65, "潿热质": 0.60, "血沇质": 0.70,
            "气郁质": 0.65, "特宾质": 0.75, "尚和质": 0.85,
        }
        
        for i, const in enumerate(self.constitution_order):
            matrix[i, i] = self_stability.get(const, 0.70)
        
        # Damp-Heat transitions
        matrix[4, 3] = 0.15  # -> Phlegm-Damp
        matrix[4, 5] = 0.10  # -> Blood Stasis
        matrix[4, 2] = 0.08  # -> Yin Xu
        matrix[4, 8] = 0.07  # -> Balanced
        
        # Phlegm-Damp transitions
        matrix[3, 4] = 0.12  # -> Damp-Heat
        matrix[3, 5] = 0.08  # -> Blood Stasis
        matrix[3, 0] = 0.05  # -> Qi Xu
        matrix[3, 8] = 0.10  # -> Balanced
        
        # Qi Stagnation transitions
        matrix[6, 2] = 0.12  # -> Yin Xu
        matrix[6, 5] = 0.10  # -> Blood Stasis
        matrix[6, 4] = 0.05  # -> Damp-Heat
        matrix[6, 8] = 0.08  # -> Balanced
        
        # Yang Xu transitions
        matrix[1, 0] = 0.10  # -> Qi Xu
        matrix[1, 3] = 0.08  # -> Phlegm-Damp
        matrix[1, 8] = 0.07  # -> Balanced
        
        # Yin Xu transitions
        matrix[2, 6] = 0.10  # -> Qi Stagnation
        matrix[2, 5] = 0.08  # -> Blood Stasis
        matrix[2, 8] = 0.12  # -> Balanced
        
        # Blood Stasis transitions
        matrix[5, 4] = 0.05  # -> Damp-Heat
        matrix[5, 3] = 0.08  # -> Phlegm-Damp
        matrix[5, 8] = 0.07  # -> Balanced
        
        # Qi Xu transitions
        matrix[0, 1] = 0.08  # -> Yang Xu
        matrix[0, 8] = 0.12  # -> Balanced
        matrix[0, 3] = 0.05  # -> Phlegm-Damp
        
        # Special transitions
        matrix[7, 8] = 0.10
        matrix[7, 2] = 0.05
        
        # Balanced transitions (influenced by external factors)
        matrix[8, 0] = 0.03; matrix[8, 1] = 0.03; matrix[8, 2] = 0.03
        matrix[8, 3] = 0.03; matrix[8, 4] = 0.03
        
        return matrix
    
    def _normalize_rows(self):
        for i in range(len(self.matrix)):
            row_sum = self.matrix[i].sum()
            if row_sum > 0:
                self.matrix[i] = self.matrix[i] / row_sum
    
    def get_transition_prob(self, from_c: str, to_c: str) -> float:
        try:
            i = self.constitution_order.index(from_c)
            j = self.constitution_order.index(to_c)
            return self.matrix[i, j]
        except ValueError:
            return 0.0
    
    def predict_next_state(self, current: str, random_seed: Optional[int] = None) -> str:
        if random_seed is not None:
            np.random.seed(random_seed)
        probs = self.matrix[self.constitution_order.index(current)]
        return np.random.choice(self.constitution_order, p=probs)
    
    def get_stationary_distribution(self) -> Dict[str, float]:
        eigvals, eigvecs = np.linalg.eig(self.matrix.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stationary = np.abs(eigvecs[:, idx])
        stationary = stationary / stationary.sum()
        return {self.constitution_order[i]: stationary[i] for i in range(9)}


class MarkovConstitutionChain:
    def __init__(self, transition_matrix: Optional[TransitionMatrix] = None):
        self.transition = transition_matrix or TransitionMatrix()
        self.state_history: List[str] = []
        self.prob_history: List[Dict[str, float]] = []
    
    def initialize(self, initial: str):
        self.state_history = [initial]
        prob = {k: 0.0 for k in self.transition.constitution_order}
        prob[initial] = 1.0
        self.prob_history.append(prob)
    
    def step(self, random_seed: Optional[int] = None) -> str:
        current = self.state_history[-1]
        next_state = self.transition.predict_next_state(current, random_seed)
        self.state_history.append(next_state)
        probs = self.transition.matrix[self.transition.constitution_order.index(current)]
        self.prob_history.append({self.transition.constitution_order[i]: probs[i] for i in range(9)})
        return next_state
    
    def simulate(self, initial: str, n_steps: int, seed: Optional[int] = None) -> List[str]:
        if seed is not None:
            np.random.seed(seed)
        self.initialize(initial)
        for _ in range(n_steps):
            self.step()
        return self.state_history


class InterventionTransitionModifier:
    INTERVENTION_EFFECTS = {
        "运动": {
            "潿热质": {"target": "尚和质", "prob": 0.25, "self_stability_reduce": 0.15},
            "痄潿质": {"target": "尚和质", "prob": 0.30, "self_stability_reduce": 0.15},
            "气郁质": {"target": "尚和质", "prob": 0.20, "self_stability_reduce": 0.10},
        },
        "饮食调节": {
            "潿热质": {"target": "痄潿质", "prob": 0.20, "self_stability_reduce": 0.10},
            "痄潿质": {"target": "尚和质", "prob": 0.25, "self_stability_reduce": 0.10},
            "阴虚质": {"target": "尚和质", "prob": 0.15, "self_stability_reduce": 0.08},
        },
        "推拿": {
            "气郁质": {"target": "尚和质", "prob": 0.30, "self_stability_reduce": 0.15},
            "血沇质": {"target": "尚和质", "prob": 0.25, "self_stability_reduce": 0.12},
            "气虚质": {"target": "尚和质", "prob": 0.20, "self_stability_reduce": 0.10},
        },
        "艸灸": {
            "阳虚质": {"target": "尚和质", "prob": 0.35, "self_stability_reduce": 0.15},
            "气虚质": {"target": "尚和质", "prob": 0.25, "self_stability_reduce": 0.12},
            "痄潿质": {"target": "尚和质", "prob": 0.15, "self_stability_reduce": 0.08},
        },
    }
    
    def __init__(self, base_transition: TransitionMatrix):
        self.base = base_transition
        self.modified_matrix = base_transition.matrix.copy()
    
    def apply_intervention(self, intervention: str) -> np.ndarray:
        self.modified_matrix = self.base.matrix.copy()
        effects = self.INTERVENTION_EFFECTS.get(intervention, {})
        for from_const, effect in effects.items():
            try:
                from_idx = self.base.constitution_order.index(from_const)
                target_idx = self.base.constitution_order.index(effect["target"])
                reduction = effect.get("self_stability_reduce", 0.0)
                self.modified_matrix[from_idx, from_idx] -= reduction
                add_prob = effect["prob"]
                self.modified_matrix[from_idx, target_idx] += add_prob
                row_sum = self.modified_matrix[from_idx].sum()
                if row_sum > 0:
                    self.modified_matrix[from_idx] /= row_sum
            except ValueError:
                continue
        return self.modified_matrix


def compute_state_transition_vector(current: cs.ConstitutionVector, next_constitution: str, 
                                    transition_matrix: TransitionMatrix) -> cs.ConstitutionVector:
    new_scores = current.constitution_scores.copy()
    for const in transition_matrix.constitution_order:
        prob = transition_matrix.get_transition_prob(current.get_dominant_constitution()[0], const)
        new_scores[const] = 0.5 * current.constitution_scores[const] + 0.5 * prob
    
    constitution_effects = {
        "潿热质": {"yin": -0.05, "yang": 0.02, "qi": -0.03, "blood": 0.02},
        "痄潿质": {"yin": 0.02, "yang": -0.03, "qi": -0.05, "blood": 0.03},
        "气郁质": {"yin": -0.03, "yang": 0.02, "qi": -0.08, "blood": -0.02},
        "阳虚质": {"yin": 0.03, "yang": -0.10, "qi": -0.05, "blood": -0.02},
        "阴虚质": {"yin": -0.10, "yang": 0.03, "qi": 0.02, "blood": -0.03},
        "尚和质": {"yin": 0.01, "yang": 0.01, "qi": 0.01, "blood": 0.01},
    }
    effect = constitution_effects.get(next_constitution, {})
    
    return cs.ConstitutionVector(
        patient_id=current.patient_id, age=current.age + 1, gender=current.gender,
        yin_index=max(0, min(1, current.yin_index + effect.get("yin", 0))),
        yang_index=max(0, min(1, current.yang_index + effect.get("yang", 0))),
        qi_index=max(0, min(1, current.qi_index + effect.get("qi", 0))),
        blood_index=max(0, min(1, current.blood_index + effect.get("blood", 0))),
        constitution_scores=new_scores, timestamp=current.timestamp
    )


if __name__ == "__main__":
    tm = TransitionMatrix()
    print("Transition Matrix shape:", tm.matrix.shape)
    print("Stationary distribution:", tm.get_stationary_distribution())
    chain = MarkovConstitutionChain(tm)
    path = chain.simulate("潿热质", 10, seed=42)
    print("Damp-Heat 10-step simulation:", path)
