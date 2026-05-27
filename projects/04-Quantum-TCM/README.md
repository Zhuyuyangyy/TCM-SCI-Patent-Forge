# Quantum-TCM - 量子证候状态空间

## 概述

Quantum-TCM 是一个基于量子变异电路（Variational Quantum Circuit, VQC）的证候状态空间建模系统，将中医五行理论与量子计算结合。

## 核心模块

1. **vqc_circuit.py** - VQC量子变异电路
   - 纯NumPy量子电路模拟器
   - 支持RY、RZ、CNOT量子门
   - 4量子比特编码证候状态

2. **syndrome_encoder.py** - 证候编码器
   - 症状到量子参数的映射
   - 五脏症状权重编码
   - 归一化处理

3. **five_elements_constraint.py** - 五行约束
   - 木火土金水相生相克
   - 证候转移验证
   - 参数约束修正

## 安装依赖



## 快速开始

### 训练模型



### 运行演示



## 技术架构

- 量子比特: 4个
- 状态空间: 16维叠加态
- 编码: 症状 → 旋转角 → 量子态
- 五行: 木→火→土→金→水→木（相生）
- 五行: 木克土→水克火→金克木（相克）

## 证候映射

- |0000⟩ = 健康
- 其他态 = 肝郁、心火、脾虚、肺气虚、肾阴虚、复合证

## 参考文献

1. VQC Variational Quantum Circuits
2. Quantum Machine Learning
3. 中医五行理论
