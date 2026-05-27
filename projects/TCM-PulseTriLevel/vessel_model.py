"""
TCM-PulseTriLevel: Vessel Elasticity Model
==========================================
Models vascular elasticity parameters and compliance for pulse wave analysis.
"""

import numpy as np
from typing import Tuple, Optional


class VesselModel:
    """Windkessel-inspired vessel model for TCM pulse analysis."""
    
    def __init__(self, elastic_modulus: float = 1.5e6, vessel_radius: float = 0.5,
                 wall_thickness: float = 0.05, blood_density: float = 1.055, length: float = 10.0):
        self.E = elastic_modulus
        self.r = vessel_radius
        self.h = wall_thickness
        self.rho = blood_density
        self.L = length
        
    @property
    def compliance(self) -> float:
        nu = 0.5
        return 3 * np.pi * self.r**3 * self.h / (2 * self.E * (1 - nu**2))
    
    @property
    def distensibility(self) -> float:
        return 1.0 / (self.E * self.h / self.r)
    
    @property
    def pulse_wave_velocity(self) -> float:
        return np.sqrt(self.E * self.h / (2 * self.rho * self.r))
    
    def stiffness_index(self) -> float:
        return self.E * self.h / self.r
    
    def __repr__(self):
        return f"VesselModel(E={self.E:.2e}, r={self.r:.3f}, PWV={self.pulse_wave_velocity:.1f})"


class ThreeLevelVessel:
    """Vessel model with three pressure levels for TCM pulse classification."""
    
    def __init__(self, base_model: Optional[VesselModel] = None):
        self.base = base_model or VesselModel()
        self._pressure_levels = {'fu': 0.0, 'zhong': 50.0, 'chen': 120.0}
        
    @property
    def fu_compliance(self) -> float:
        return self.base.compliance * 1.5
    
    @property
    def zhong_compliance(self) -> float:
        return self.base.compliance * 1.0
    
    @property
    def chen_compliance(self) -> float:
        return self.base.compliance * 0.6
    
    def get_compliance(self, level: str) -> float:
        level_map = {'fu': self.fu_compliance, 'zhong': self.zhong_compliance, 'chen': self.chen_compliance}
        return level_map.get(level.lower(), self.base.compliance)
    
    def get_elastic_modulus(self, level: str) -> float:
        base_E = self.base.E
        level_pressure = self._pressure_levels.get(level.lower(), 0)
        return base_E * (1 + 0.005 * level_pressure)
    
    def __repr__(self):
        return f"ThreeLevelVessel(fu={self.fu_compliance:.2e}, zhong={self.zhong_compliance:.2e}, chen={self.chen_compliance:.2e})"


if __name__ == "__main__":
    vessel = VesselModel()
    print(f"Base vessel: {vessel}")
    print(f"Compliance: {vessel.compliance:.3e}")
    print(f"PWV: {vessel.pulse_wave_velocity:.1f} cm/s")
    
    three_level = ThreeLevelVessel(vessel)
    print(f"Three-level: {three_level}")
