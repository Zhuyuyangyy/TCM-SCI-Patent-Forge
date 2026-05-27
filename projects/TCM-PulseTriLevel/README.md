# TCM-PulseTriLevel: P03浮中沉三压力层级脉波建模

## 项目概述

**TCM-PulseTriLevel** 是中医脉诊三压力层级建模项目，基于物理信息神经网络(PINN)实现血管弹性参数反演与浮(Fu)、中(Zhong)、沉(Chen)三压脉波建模分析。

### 核心功能

- **vessel_model.py**: 血管弹性参数与顺应性模型 (Windkessel模型 + Moens-Korteweg方程)
- **pressure_levels.py**: 浮中沉三压建模 (TCM脉诊深度层次)
- **pinna_inverse.py**: PINN反演求解血管参数
- **pulse_simulator.py**: 生成三压脉波信号
- **demo.py**: 完整分析流程演示

## 算法原理

### 1. 血管弹性模型

基于Windkessel模型和Moens-Korteweg方程:

```
顺应性: C = dV/dP
脉波速度: PWV = √(E·h / 2ρ·r)
弹性阻抗: Z = ρ · PWV / πr²
```

### 2. 三压力层级

| 层级 | 中文 | 英文 | 压力范围 | 顺应性特征 |
|------|------|------|----------|------------|
| Fu   | 浮   | Superficial | 0-20 mmHg | 高顺应性 |
| Zhong| 中   | Middle | 20-80 mmHg | 基线顺应性 |
| Chen | 沉   | Deep | 80-200 mmHg | 低顺应性 |

### 3. PINN反演

损失函数:
```
L = L_data + λ_physics · L_physics + λ_param · L_param

L_physics: Windkessel约束 C·dP/dt + P/R = I(t)
L_param: 参数正则化 (E, C, R, ζ)
```

### 4. CFL稳定性条件

```
库朗数: Co = v · Δt / Δx ≤ 1.0 (显式格式)
v = PWV (脉波速度)
```

## 安装

```bash
pip install -r requirements.txt
```

## 使用方法

### 快速演示

```bash
python demo.py
```

### 自定义参数

```bash
# 正常脉搏
python demo.py --signal synthetic --HR 72 --age 45 --condition normal

# 高血压模拟
python demo.py --signal synthetic --HR 80 --age 60 --condition hypertension

# 僵硬动脉
python demo.py --signal synthetic --HR 70 --age 70 --condition stiff
```

### 加载真实数据

```bash
python demo.py --signal recorded --file data/pulse.csv
```

## 文件说明

| 文件 | 功能 |
|------|------|
| `vessel_model.py` | VesselModel类: 血管弹性、顺应性、PWV计算 |
| | ThreeLevelVessel类: 三压力层级血管模型 |
| `pressure_levels.py` | PressureLevelModel类: 三压边界与分类 |
| | PulseWaveformGenerator: 脉波傅里叶合成 |
| | ThreeLevelPulseAnalyzer: 脉波分解与特征提取 |
| `pinna_inverse.py` | PhysicsInformedNeuralNetwork: PINN网络 |
| | InverseVesselEstimator: 血管参数反演 |
| `pulse_simulator.py` | PulseSimulator: 三压脉波仿真 |
| | SignalQualityEnhancer: 信号质量增强 |
| `demo.py` | 完整分析流程 |
| | CFLValidator: CFL条件验证 |

## 输出结果

```
ANALYSIS SUMMARY
=======================================================================
[Vessel Parameters (from PINN)]
  elastic_modulus_Pa: 1.5000e+06
  compliance_mL_per_mmHg: 1.0000e+00
  resistance_mmHg_s_per_mL: 1.0000e+00
  damping_coef: 1.0000e-01

[Elastic Parameters]
  PWV: 534.52 cm/s
  compliance: 1.25e-10 m³/Pa
  distensibility: 1.33e-06 1/Pa

[Three-Level Compliance]
  FU: 1.87e-10 m³/Pa
  ZHONG: 1.25e-10 m³/Pa
  CHEN: 7.50e-11 m³/Pa

[CFL Validation]
  Courant number: 0.2673
  Stable: True
  Safety margin: 3.74
```

## 依赖库

- numpy >= 1.19
- scipy >= 1.6
- torch >= 1.8
- matplotlib >= 3.3

## 参考文献

1. Moens, A.I. (1878). "Over de voortplantingssnelheid van den pols."
2. Korteweg, D.J. (1878). "Uber die Fortpflanzungsgeschwindigkeit des Schalles in de elastischen Röhren."
3. Westerhof, N. et al. (2008). "Arterial stiffness."
4. Raissi, M. et al. (2019). "Physics-informed neural networks."

## 许可

MIT License
