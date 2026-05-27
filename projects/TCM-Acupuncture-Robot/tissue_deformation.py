"""
TCM-Acupuncture-Robot E01
机械臂推拿力度-穴位-形变闭环控制
 tissue_deformation.py - 胡克定律弹性模型
"""

import math
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
from enum import Enum


class TissueType(Enum):
    """组织类型枚举"""
    SKIN = "皮肤"
    FAT = "脂肪"
    MUSCLE = "肌肉"
    TENDON = "肌腱"
    LIGAMENT = "韧带"
    BONE_PERIOSTEUM = "骨膜"
    CARTILAGE = "软骨"
    VISCERA = "脏器"


@dataclass
class TissueLayer:
    """组织层参数"""
    tissue_type: TissueType
    thickness: float           # 厚度 mm
    youngs_modulus: float      # 弹性模量 MPa
    poissons_ratio: float      # 泊松比
    density: float             # 密度 kg/m³
    damping_ratio: float       # 阻尼比
    
    @property
    def stiffness(self) -> float:
        """计算刚度 N/mm"""
        # E (MPa) -> N/mm² = 10⁶ N/m² -> 10⁶ N/m² = 10³ N/mm²
        # 简化: k = E * A / L, 设 A = 1mm²
        return self.youngs_modulus * 1000 / self.thickness


@dataclass
class DeformationResult:
    """形变计算结果"""
    total_deformation: float       # 总形变 mm
    layer_deformations: Dict[str, float]  # 各层形变
    effective_stiffness: float     # 等效刚度 N/mm
    stress: float                  # 应力 MPa
    strain: float                  # 应变
    safety_margin: float           # 安全裕度
    warning_level: str             # 警告级别


class HookeLawModel:
    """胡克定律弹性模型
    
    基于线性弹性理论计算组织形变
    σ = E * ε (应力-应变关系)
    F = k * Δx (力-位移关系)
    """

    # 组织弹性模量参考值 (MPa)
    TISSUE_PROPERTIES = {
        TissueType.SKIN:            {'E': 0.1,   'v': 0.49, 'ρ': 1100},
        TissueType.FAT:             {'E': 0.02,  'v': 0.49, 'ρ': 900},
        TissueType.MUSCLE:          {'E': 0.1,   'v': 0.49, 'ρ': 1050},
        TissueType.TENDON:          {'E': 500,   'v': 0.30, 'ρ': 1100},
        TissueType.LIGAMENT:        {'E': 300,   'v': 0.30, 'ρ': 1100},
        TissueType.BONE_PERIOSTEUM: {'E': 2000,  'v': 0.28, 'ρ': 1900},
        TissueType.CARTILAGE:       {'E': 10,    'v': 0.45, 'ρ': 1100},
        TissueType.VISCERA:         {'E': 0.002, 'v': 0.49, 'ρ': 1000},
    }
    
    # 安全应变阈值
    SAFETY_STRAIN = {
        TissueType.SKIN:            0.30,
        TissueType.FAT:             0.40,
        TissueType.MUSCLE:          0.25,
        TissueType.TENDON:          0.08,
        TissueType.LIGAMENT:        0.08,
        TissueType.BONE_PERIOSTEUM: 0.02,
        TissueType.CARTILAGE:       0.15,
        TissueType.VISCERA:         0.50,
    }

    def __init__(self):
        self._layers: list[TissueLayer] = []
        self._initialize_default_model()

    def _initialize_default_model(self):
        """初始化默认组织模型 (足三里区域)"""
        # 足三里解剖层次: 皮肤 -> 皮下脂肪 -> 肌肉 -> 骨膜
        self.add_layer(TissueLayer(TissueType.SKIN, 2.0, 0.1, 0.49, 1100, 0.05))
        self.add_layer(TissueLayer(TissueType.FAT, 8.0, 0.02, 0.49, 900, 0.08))
        self.add_layer(TissueLayer(TissueType.MUSCLE, 30.0, 0.1, 0.49, 1050, 0.10))
        self.add_layer(TissueLayer(TissueType.BONE_PERIOSTEUM, 3.0, 2000, 0.28, 1900, 0.02))

    def add_layer(self, layer: TissueLayer):
        """添加组织层"""
        self._layers.append(layer)
        
    def clear_layers(self):
        """清空所有层"""
        self._layers = []
        
    def get_layer_count(self) -> int:
        """获取层数"""
        return len(self._layers)

    def calculate_deformation(self, force: float, 
                             contact_area: float = 1.0) -> DeformationResult:
        """计算组织形变
        
        Args:
            force: 作用力 N
            contact_area: 接触面积 mm²
            
        Returns:
            形变计算结果
        """
        if not self._layers:
            return DeformationResult(
                total_deformation=0,
                layer_deformations={},
                effective_stiffness=0,
                stress=0,
                strain=0,
                safety_margin=1.0,
                warning_level="NONE"
            )
        
        # 计算总刚度 (串联模型)
        total_stiffness = 0.0
        for layer in self._layers:
            if layer.stiffness > 0:
                total_stiffness += 1.0 / layer.stiffness
        effective_stiffness = 1.0 / total_stiffness if total_stiffness > 0 else 0.0
        
        # 计算总形变 F = k * Δx
        total_deformation = force / effective_stiffness if effective_stiffness > 0 else 0.0
        
        # 计算各层形变 (按刚度分配)
        layer_deformations = {}
        remaining_force = force
        remaining_deformation = total_deformation
        
        for i, layer in enumerate(self._layers):
            # 按刚度比例分配形变
            if effective_stiffness > 0:
                ratio = layer.stiffness / (sum(l.stiffness for l in self._layers) + 1e-10)
            else:
                ratio = 1.0 / len(self._layers)
                
            layer_def = remaining_deformation * ratio
            layer_deformations[layer.tissue_type.value] = layer_def
            remaining_deformation -= layer_def * 0.1  # 简化分配
        
        # 计算应力 σ = F / A
        stress = force / contact_area  # MPa (N/mm²)
        
        # 计算等效应变
        total_thickness = sum(l.thickness for l in self._layers)
        strain = total_deformation / total_thickness if total_thickness > 0 else 0
        
        # 安全裕度计算
        safety_margin, warning_level = self._calculate_safety(
            strain, stress, force, total_thickness
        )
        
        return DeformationResult(
            total_deformation=total_deformation,
            layer_deformations=layer_deformations,
            effective_stiffness=effective_stiffness,
            stress=stress,
            strain=strain,
            safety_margin=safety_margin,
            warning_level=warning_level
        )

    def calculate_deformation_by_depth(self, force: float, 
                                       depth: float,
                                       surface_area: float = 1.0) -> DeformationResult:
        """按深度计算形变 (用于实时计算)
        
        Args:
            force: 作用力 N
            depth: 压入深度 mm
            surface_area: 接触面积 mm²
        """
        # 根据深度确定参与变形的层
        current_depth = 0.0
        active_layers = []
        
        for layer in self._layers:
            layer_center = current_depth + layer.thickness / 2
            if depth >= layer_center:
                active_layers.append(layer)
            current_depth += layer.thickness
            
        if not active_layers:
            # 还在浅层
            if self._layers:
                first_layer = self._layers[0]
                stiffness = first_layer.stiffness
                deformation = force / stiffness if stiffness > 0 else 0
                
                # 检查浅层安全
                strain = deformation / first_layer.thickness
                safety = self.SAFETY_STRAIN.get(first_layer.tissue_type, 0.3)
                margin = (safety - strain) / safety if safety > 0 else 1.0
                
                return DeformationResult(
                    total_deformation=deformation,
                    layer_deformations={first_layer.tissue_type.value: deformation},
                    effective_stiffness=stiffness,
                    stress=force / surface_area,
                    strain=strain,
                    safety_margin=max(0, margin),
                    warning_level="WARNING" if margin < 0.3 else "NORMAL"
                )
            return DeformationResult(0, {}, 0, 0, 0, 1.0, "NONE")
        
        # 重新计算active layers的等效刚度
        total_inv_k = sum(1.0/l.stiffness for l in active_layers if l.stiffness > 0)
        eq_stiffness = 1.0 / total_inv_k if total_inv_k > 0 else 0
        
        total_def = force / eq_stiffness if eq_stiffness > 0 else 0
        
        # 计算各层形变
        layer_defs = {}
        total_k = sum(l.stiffness for l in active_layers)
        for l in active_layers:
            ratio = l.stiffness / total_k if total_k > 0 else 0
            layer_defs[l.tissue_type.value] = total_def * ratio
        
        # 安全检查
        total_thickness = sum(l.thickness for l in active_layers)
        strain = total_def / total_thickness if total_thickness > 0 else 0
        
        # 多层安全裕度
        min_margin = 1.0
        for l in active_layers:
            l_strain = (layer_defs[l.tissue_type.value] / l.thickness) if l.thickness > 0 else 0
            safety = self.SAFETY_STRAIN.get(l.tissue_type, 0.3)
            margin = (safety - l_strain) / safety if safety > 0 else 1.0
            min_margin = min(min_margin, margin)
        
        return DeformationResult(
            total_deformation=total_def,
            layer_deformations=layer_defs,
            effective_stiffness=eq_stiffness,
            stress=force / surface_area,
            strain=strain,
            safety_margin=max(0, min_margin),
            warning_level="DANGER" if min_margin < 0.2 else ("WARNING" if min_margin < 0.4 else "NORMAL")
        )

    def _calculate_safety(self, strain: float, stress: float, 
                          force: float, thickness: float) -> Tuple[float, str]:
        """计算安全裕度"""
        # 基于应变的安全判断
        max_allowed_strain = 0.25  # 默认肌肉阈值
        strain_ratio = strain / max_allowed_strain if max_allowed_strain > 0 else 0
        
        # 基于力的安全判断
        force_threshold = 30.0  # N
        force_ratio = force / force_threshold if force_threshold > 0 else 0
        
        # 综合安全裕度
        combined_ratio = max(strain_ratio, force_ratio)
        safety_margin = max(0, 1.0 - combined_ratio)
        
        # 确定警告级别
        if safety_margin < 0.1:
            warning = "CRITICAL"
        elif safety_margin < 0.2:
            warning = "DANGER"
        elif safety_margin < 0.4:
            warning = "WARNING"
        else:
            warning = "NORMAL"
            
        return safety_margin, warning

    def get_tissue_impedance(self, depth: float) -> float:
        """获取组织阻抗 (用于力控制)
        
        阻抗控制: F = k * (x - x_d) + b * (v - v_d)
        """
        result = self.calculate_deformation_by_depth(0, depth)  # 假设0N力
        return result.effective_stiffness * 0.1  # 简化为刚度


class MultiLayerViscoelasticModel(HookeLawModel):
    """多层粘弹性模型
    
    考虑组织的粘弹性特性 (应力松弛、蠕变)
    σ(t) = E * ε(t) + η * dε/dt
    """

    def __init__(self):
        super().__init__()
        self._creep_modulus: Dict[TissueType, float] = {
            TissueType.MUSCLE: 0.05,
            TissueType.FAT: 0.02,
            TissueType.SKIN: 0.03,
        }
        self._relaxation_time: Dict[TissueType, float] = {
            TissueType.MUSCLE: 5.0,
            TissueType.FAT: 2.0,
            TissueType.SKIN: 1.0,
        }

    def calculate_viscoelastic_deformation(self, force: float,
                                          time: float,
                                          depth: float) -> DeformationResult:
        """计算粘弹性形变
        
        Args:
            force: 作用力 N
            time: 作用时间 s
            depth: 压入深度 mm
        """
        # 基础弹性形变
        base_result = self.calculate_deformation_by_depth(force, depth)
        
        # 添加蠕变效应 (Kelvin-Voigt模型简化)
        creep_factor = 1.0
        for layer in self._layers:
            if layer.tissue_type in self._creep_modulus:
                tau = self._relaxation_time.get(layer.tissue_type, 1.0)
                creep_factor += self._creep_modulus[layer.tissue_type] * (1 - math.exp(-time/tau))
        
        # 蠕变后的形变
        modified_deformation = base_result.total_deformation * creep_factor
        
        return DeformationResult(
            total_deformation=modified_deformation,
            layer_deformations={k: v * creep_factor for k, v in base_result.layer_deformations.items()},
            effective_stiffness=base_result.effective_stiffness / creep_factor,
            stress=base_result.stress,
            strain=base_result.strain * creep_factor,
            safety_margin=base_result.safety_margin / creep_factor,
            warning_level=base_result.warning_level
        )


def main():
    """测试组织形变模型"""
    print("=" * 60)
    print("胡克定律弹性模型 - 组织形变计算")
    print("=" * 60)
    
    # 创建模型
    model = HookeLawModel()
    
    print(f"\n组织模型: {model.get_layer_count()}层")
    for i, layer in enumerate(model._layers, 1):
        print(f"  层{i}: {layer.tissue_type.value:<15} | 厚度:{layer.thickness:5.1f}mm | "
              f"E:{layer.youngs_modulus:8.2f}MPa | 刚度:{layer.stiffness:8.3f}N/mm")
    
    # 测试不同压力下的形变
    print("\n" + "=" * 60)
    print("压力-形变关系测试")
    print("=" * 60)
    print(f"{'Force(N)':<10} {'Deform(mm)':<12} {'Stiffness':<12} {'Stress(MPa)':<12} "
          f"{'Strain':<10} {'Safety':<10} {'Warning'}")
    print("-" * 80)
    
    forces = [3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    for force in forces:
        result = model.calculate_deformation(force)
        print(f"{force:<10.1f} {result.total_deformation:<12.3f} {result.effective_stiffness:<12.3f} "
              f"{result.stress:<12.4f} {result.strain:<10.4f} {result.safety_margin:<10.2f} "
              f"{result.warning_level}")
    
    # 测试按深度计算
    print("\n" + "=" * 60)
    print("深度依赖形变测试 (12N恒定压力)")
    print("=" * 60)
    print(f"{'Depth(mm)':<10} {'Deform(mm)':<12} {'Safety':<10} {'Warning'}")
    print("-" * 50)
    
    for depth in [2.0, 5.0, 10.0, 15.0, 25.0, 35.0]:
        result = model.calculate_deformation_by_depth(12.0, depth)
        print(f"{depth:<10.1f} {result.total_deformation:<12.3f} "
              f"{result.safety_margin:<10.2f} {result.warning_level}")
    
    # 测试粘弹性模型
    print("\n" + "=" * 60)
    print("粘弹性蠕变效应测试 (12N)")
    print("=" * 60)
    
    visco_model = MultiLayerViscoelasticModel()
    
    print(f"{'Time(s)':<10} {'Deform(mm)':<12} {'Safety':<10}")
    print("-" * 40)
    
    for t in [0.0, 1.0, 5.0, 10.0, 20.0]:
        result = visco_model.calculate_viscoelastic_deformation(12.0, t, 20.0)
        print(f"{t:<10.1f} {result.total_deformation:<12.3f} {result.safety_margin:<10.2f}")


if __name__ == "__main__":
    main()
