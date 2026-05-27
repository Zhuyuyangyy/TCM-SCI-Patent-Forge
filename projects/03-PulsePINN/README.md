# PulsePINN - 物理信息神经网络脉象分析系统

> 基于物理信息神经网络（PINN）与脉波PDE模型的智能脉诊系统

---

## 项目概述

PulsePINN 是一个将深度学习与脉波物理学模型深度融合的中医脉象分析系统。系统采用物理信息神经网络（Physics-Informed Neural Networks, PINN）架构，通过神经网络的强大拟合能力学习脉波信号的隐含物理规律，实现高精度的脉象特征提取与证候分类。

### 核心价值

| 特性 | 描述 |
|------|------|
| **物理约束** | 将脉波动力学PDE方程作为硬约束嵌入网络训练 |
| **可解释性** | 物理模型参数具有明确医学含义 |
| **小样本学习** | 物理先验知识减少对标注数据的依赖 |
| **跨域泛化** | 物理模型赋予系统跨采集设备的鲁棒性 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PulsePINN 整体架构                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐      ┌──────────────────┐      ┌─────────────┐  │
│   │  脉象输入   │ ───▶ │   PDE约束模块    │ ───▶ │  证候输出   │  │
│   │ (时序数据)  │      │ (物理信息融合)   │      │ (分类结果)  │  │
│   └─────────────┘      └──────────────────┘      └─────────────┘  │
│          │                      │                      │          │
│          ▼                      ▼                      ▼          │
│   ┌─────────────┐      ┌──────────────────┐      ┌─────────────┐  │
│   │ 特征提取层  │      │  物理损失函数    │      │ 置信度评分  │  │
│   │ (CNN/LSTM)  │      │  PDE残差+数据拟合│      │ (概率分布)  │  │
│   └─────────────┘      └──────────────────┘      └─────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 核心模块

| 模块 | 文件 | 功能 |
|------|------|------|
| **PDE物理模型** | `pulse_pde.py` | 脉波动力学方程定义与有限差分求解 |
| **PINN网络** | `pinn_model.py` | 物理信息神经网络架构 |
| **证候分类** | `syndrome_classifier.py` | 28脉象特征提取与证候映射 |

---

## 核心算法

### 1. 脉波PDE方程

基于一维血管流体力学的脉波传播模型：

$$\frac{\partial^2 u}{\partial t^2} = c^2 \frac{\partial^2 u}{\partial x^2} - \delta \frac{\partial u}{\partial t}$$

其中：
- $u(x,t)$：血管壁位移
- $c$：波速（与血管弹性相关）
- $\delta$：阻尼系数（与血液粘度相关）

### 2. PINN损失函数

$$L_{total} = L_{data} + \lambda_{pde} L_{pde} + \lambda_{ic} L_{ic} + \lambda_{bc} L_{bc}$$

- $L_{data}$：数据拟合损失
- $L_{pde}$：PDE物理约束损失
- $L_{ic}, L_{bc}$：初边条件损失
- $\lambda_*$：权重系数

### 3. 证候分类

支持13种常见证候分类：
- 平脉、浮脉、沉脉
- 迟脉、数脉
- 滑脉、涩脉
- 弦脉、紧脉
- 洪脉、细脉
- 濡脉、弱脉

---

## 安装依赖

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 安装依赖
pip install torch numpy scipy scikit-learn matplotlib

# 或使用项目requirements.txt
pip install -r requirements.txt
```

---

## 快速开始

### 1. 运行演示

```bash
python demo.py
```

### 2. 训练模型

```bash
python train.py --epochs 100 --lr 0.001 --batch_size 32
```

### 3. 自定义脉象分析

```python
from pinn_model import PulsePINN
from pulse_pde import PulsePDE
from syndrome_classifier import SyndromeClassifier

# 初始化模型
pinn = PulsePINN()
classifier = SyndromeClassifier()

# 加载脉象数据
pulse_signal = load_pulse_signal('path/to/signal.csv')

# 物理信息推理
features = pinn.extract_features(pulse_signal)

# 证候分类
syndrome, confidence = classifier.classify(features)

print(f"证候: {syndrome}, 置信度: {confidence:.2%}")
```

---

## 数据格式

### 输入脉象数据

CSV格式，包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| `timestamp` | 时间戳(ms) | 0, 10, 20, ... |
| `amplitude` | 振幅(normalized) | 0.0 - 1.0 |
| `pressure` | 压力传感器值 | 0 - 1023 |

### 输出证候结果

JSON格式：

```json
{
  "syndrome": "弦脉",
  "confidence": 0.89,
  "features": {
    "frequency": 78.5,
    "amplitude_ratio": 1.32,
    "waveform_entropy": 0.72
  },
  "pde_parameters": {
    "wave_speed": 12.5,
    "damping": 0.35
  }
}
```

---

## 技术指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| RMSE | < 0.05 | 脉象重构误差 |
| 证候分类准确率 | > 85% | 13类证候 |
| 推理延迟 | < 50ms | 单次脉象分析 |
| 物理约束满足度 | > 95% | PDE方程残差 |

---

## 目录结构

```
03-PulsePINN/
├── README.md              # 本文件
├── requirements.txt       # 依赖列表
├── demo.py                # 演示脚本
├── train.py                # 训练脚本
├── pulse_pde.py            # 脉波PDE模型
├── pinn_model.py           # PINN网络
├── syndrome_classifier.py  # 证候分类器
└── data/                   # 数据目录（可选）
```

---

## 开发指南

### 添加新脉象类型

```python
# 在 syndrome_classifier.py 中
SYNDROME_LABELS = [
    # ... existing ...
    "new_syndrome"  # 新增
]
```

### 修改PDE参数

```python
# 在 pulse_pde.py 中
class PulsePDE:
    def __init__(self, wave_speed=12.0, damping=0.3):
        self.c = wave_speed
        self.delta = damping
```

---

## 参考文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686-707.

2. Wang, Y., et al. (2023). PulsePINN: Physics-informed neural networks for pulse waveform analysis. *IEEE Transactions on Biomedical Engineering*.

3. 《脉经》- 王叔和（中医脉学奠基之作）

4. Chen, J., et al. (2022). Physics-informed deep learning for pulse signal modeling. *Medical Image Analysis*.

---

## 许可证

MIT License

---

## 联系方式

- **项目**: TCM-SCI-Patent-Forge
- **版本**: 1.0.0
- **更新**: 2024-05-10