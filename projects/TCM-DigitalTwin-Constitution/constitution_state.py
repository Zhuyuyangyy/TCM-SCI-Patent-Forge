#!/usr/bin/env python3
"""D01 Constitution State Vector - Yin-Yang / Qi-Blood / Nine Constitutions"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class FiveConstitution(Enum):
    """Nine TCM Constitution Types"""
    QI_XU = "气虚质"
    YANG_XU = "阳虚质"
    YIN_XU = "阴虚质"
    PHLEGM_DAMP = "痄潿质"
    DAMP_HEAT = "潿热质"
    BLOOD_STASIS = "血沇质"
    QI_STAGNATION = "气郁质"
    SPECIAL = "特宾质"
    BALANCED = "尚和质"


class YinYangState(Enum):
    """Yin-Yang State"""
    YIN_XU = "阴虚"
    YANG_XU = "阳虚"
    YIN_YANG_BALANCED = "阴阳平衡"
    YIN_EXCESS = "阴盛"
    YANG_EXCESS = "阳光"


class QiBloodState(Enum):
    """Qi-Blood State"""
    QI_XU = "气虚"
    QI_STAGNATION = "气沇"
    XUE_XU = "血虚"
    XUE_YU = "血沇"
    QI_XUE_BALANCED = "气血平衡"


@dataclass
class ConstitutionVector:
    """Constitution Digital Twin State Vector"""
    patient_id: str = ""
    age: int = 30
    gender: str = "女"
    
    yin_index: float = 0.5
    yang_index: float = 0.5
    qi_index: float = 0.5
    blood_index: float = 0.5
    
    constitution_scores: Dict[str, float] = field(default_factory=lambda: {
        "气虚质": 0.3,
        "阳虚质": 0.3,
        "阴虚质": 0.3,
        "痄潿质": 0.3,
        "潿热质": 0.3,
        "血沇质": 0.3,
        "气郁质": 0.3,
        "特宾质": 0.3,
        "尚和质": 0.3,
    })
    
    timestamp: str = ""
    
    def to_vector(self) -> np.ndarray:
        vec = [self.yin_index, self.yang_index, self.qi_index, self.blood_index]
        vec.extend([self.constitution_scores[k] for k in sorted(self.constitution_scores.keys())])
        return np.array(vec)
    
    @classmethod
    def from_vector(cls, vec: np.ndarray, patient_id: str = "", age: int = 30, gender: str = "女") -> "ConstitutionVector":
        keys = ["气虚质", "阳虚质", "阴虚质", "痄潿质", 
                "潿热质", "血沇质", "气郁质", "特宾质", "尚和质"]
        state = cls(patient_id=patient_id, age=age, gender=gender,
                    yin_index=vec[0], yang_index=vec[1], qi_index=vec[2], blood_index=vec[3])
        for i, k in enumerate(keys):
            state.constitution_scores[k] = vec[4 + i]
        return state
    
    def get_dominant_constitution(self) -> Tuple[str, float]:
        dom = max(self.constitution_scores.items(), key=lambda x: x[1])
        return dom[0], dom[1]
    
    def get_yin_yang_state(self) -> YinYangState:
        diff = self.yin_index - self.yang_index
        if diff < -0.2: return YinYangState.YANG_XU
        elif diff > 0.2: return YinYangState.YIN_XU
        elif self.yin_index > 0.7: return YinYangState.YIN_EXCESS
        elif self.yang_index > 0.7: return YinYangState.YANG_EXCESS
        else: return YinYangState.YIN_YANG_BALANCED
    
    def get_qi_blood_state(self) -> QiBloodState:
        if self.qi_index < 0.3 and self.blood_index < 0.3: return QiBloodState.QI_XU
        elif self.qi_index > 0.7: return QiBloodState.QI_STAGNATION
        elif self.blood_index < 0.3: return QiBloodState.XUE_XU
        elif self.blood_index > 0.7: return QiBloodState.XUE_YU
        else: return QiBloodState.QI_XUE_BALANCED
    
    def similarity(self, other: "ConstitutionVector") -> float:
        v1, v2 = self.to_vector(), other.to_vector()
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / (norm + 1e-8)
    
    def distance(self, other: "ConstitutionVector") -> float:
        return np.linalg.norm(self.to_vector() - other.to_vector())
    
    def __repr__(self) -> str:
        dom = self.get_dominant_constitution()
        return f"<ConstitutionVector: {dom[0]}({dom[1]:.2f}), Yin={self.yin_index:.2f}, Yang={self.yang_index:.2f}>"


class ConstitutionStateBuilder:
    DAMP_HEAT_FEMALE_PARAMS = {
        "yin_index": 0.35, "yang_index": 0.55, "qi_index": 0.45, "blood_index": 0.50,
        "constitution_scores": {
            "气虚质": 0.25, "阳虚质": 0.20, "阴虚质": 0.35,
            "痄潿质": 0.60, "潿热质": 0.85, "血沇质": 0.40,
            "气郁质": 0.30, "特宾质": 0.15, "尚和质": 0.20,
        }
    }
    
    @classmethod
    def create_damp_heat_female(cls, patient_id: str = "DHF001", age: int = 35) -> ConstitutionVector:
        params = cls.DAMP_HEAT_FEMALE_PARAMS.copy()
        return ConstitutionVector(
            patient_id=patient_id, age=age, gender="女",
            yin_index=params["yin_index"], yang_index=params["yang_index"],
            qi_index=params["qi_index"], blood_index=params["blood_index"],
            constitution_scores=params["constitution_scores"],
            timestamp="2026-01-15T10:00:00"
        )
    
    @classmethod
    def create_balanced(cls, patient_id: str = "BL001", age: int = 30) -> ConstitutionVector:
        return ConstitutionVector(
            patient_id=patient_id, age=age, gender="男",
            yin_index=0.50, yang_index=0.50, qi_index=0.50, blood_index=0.50,
            constitution_scores={
                "气虚质": 0.20, "阳虚质": 0.20, "阴虚质": 0.20,
                "痄潿质": 0.20, "潿热质": 0.20, "血沇质": 0.20,
                "气郁质": 0.20, "特宾质": 0.15, "尚和质": 0.80,
            },
            timestamp="2026-01-15T10:00:00"
        )


if __name__ == "__main__":
    state = ConstitutionStateBuilder.create_damp_heat_female()
    print(state)
    print("Dominant:", state.get_dominant_constitution())
    print("Yin-Yang:", state.get_yin_yang_state())
    print("Qi-Blood:", state.get_qi_blood_state())
    print("Vector shape:", state.to_vector().shape)
