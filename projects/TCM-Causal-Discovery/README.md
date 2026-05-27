# TCM-Causal-Discovery: 证候-方药-疗效因果发现系统

> 基于PC算法与异质性分析的中医因果路径发现框架

---

## 项目概述

TCM-Causal-Discovery 是一个基于因果推断技术的中医医案分析系统。系统通过PC算法（Peter-Clark Algorithm）从真实医案数据中发现证候-方药-疗效之间的因果关系，识别关键治疗靶点，并进行患者亚组异质性分析，为精准中医用药提供决策支持。

### 核心特性

| 特性 | 描述 |
|------|------|
| **PC算法因果发现** | 从观测数据中自动发现因果骨架和边方向 |
| **异质性分析** | 决策树识别患者亚组与亚组特异性治疗效应 |
| **因果效应估计** | ATE/ITE计算，支持连续和离散结局 |
| **混杂因素识别** | 自动识别和校正混杂变量 |
| **可视化报告** | 因果图可视化与效应分析报告 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TCM-Causal-Discovery 整体架构                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                         数据输入层                                 │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │
│   │  │ 电子病历   │  │ 医案数据   │  │ 临床试验数据            │   │   │
│   │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘   │   │
│   └─────────┼───────────────┼─────────────────────┼───────────────────┘   │
│             └───────────────┴─────────────────────┘                      │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                    因果发现层                                     │   │
│   │  ┌─────────────────────────────────────────────────────────────┐  │   │
│   │  │               CausalGraph (causal_graph.py)                │  │   │
│   │  │                                                          │  │   │
│   │  │    PC算法 ──▶ 条件独立检验 ──▶ 骨架发现 ──▶ 边定向      │  │   │
│   │  │                                                          │  │   │
│   │  │    因果图: 证候1 ──→ 方药1 ──→ 疗效                      │  │   │
│   │  │                 ↓           ↓                            │  │   │
│   │  │               证候2 ──→ 方药2 ──→ 疗效                   │  │   │
│   │  └─────────────────────────────────────────────────────────────┘  │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                    因果效应分析层                                 │   │
│   │  ┌─────────────────────────────────────────────────────────────┐  │   │
│   │  │          InterventionEffect (intervention_effect.py)       │  │   │
│   │  │                                                          │  │   │
│   │  │    ATE估计 ──▶ 剂量-反应关系 ──▶ 分层分析               │  │   │
│   │  │                                                          │  │   │
│   │  │    混杂因素识别 ──▶ 回归调整 ──▶ 效应置信区间           │  │   │
│   │  └─────────────────────────────────────────────────────────────┘  │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                    异质性分析层                                    │   │
│   │  ┌─────────────────────────────────────────────────────────────┐  │   │
│   │  │        HeterogeneousAnalysis (heterogeneous_analysis.py)  │  │   │
│   │  │                                                          │  │   │
│   │  │    决策树分组 ──▶ 亚组效应估计 ──▶ 交互作用检验         │  │   │
│   │  │                                                          │  │   │
│   │  │    [气虚血瘀型] ──→ ATE=8.52 [6.21, 10.83]               │  │   │
│   │  │    [湿热质]       ──→ ATE=5.23 [3.11, 7.35]             │  │   │
│   │  └─────────────────────────────────────────────────────────────┘  │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                      输出层                                        │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │
│   │  │ 因果图     │  │ 效应报告   │  │ 精准用药建议            │   │   │
│   │  │ (可视化)   │  │ (ATE/CI)   │  │ (亚组特异性方案)        │   │   │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. data_simulator.py - 医案数据模拟

```python
class MedicalRecordSimulator:
    """中医医案数据模拟器"""
    
    def generate_dataset(self, n_samples=1000, seed=42):
        """
        生成模拟中医医案数据
        - 4个证候变量 (syn_1-qyn_4)
        - 4个方药变量 (herb_1-herb_4)
        - 疗效评分 (efficacy_score)
        """
        pass
    
    def get_causal_structure(self):
        """返回预设的因果结构"""
        # 证候 → 方药 → 疗效
        pass
```

### 2. causal_graph.py - 因果图构建

```python
class CausalGraphBuilder:
    """基于PC算法的因果图发现"""
    
    def pc_algorithm(self, data, alpha=0.05):
        """
        PC算法主流程：
        1. 条件独立性检验发现骨架
        2. 边定向确定因果方向
        3. 返回因果图结构
        """
        pass
    
    def conditional_independence_test(self, x, y, z_list, data):
        """条件独立性检验（偏相关）"""
        pass
    
    def orient_edges(self, skeleton, sep_sets):
        """边定向规则"""
        pass
    
    def visualize(self, graph):
        """可视化因果图"""
        pass
```

### 3. intervention_effect.py - 因果效应计算

```python
class InterventionEffectEstimator:
    """因果效应估计"""
    
    def compute_ate(self, treatment, outcome, confounders, data):
        """计算平均处理效应 (Average Treatment Effect)"""
        pass
    
    def compute_ite(self, patient_id, treatment, outcome, confounders, data):
        """计算个体处理效应 (Individual Treatment Effect)"""
        pass
    
    def dose_response_relation(self, treatment, outcome, confounders, data):
        """剂量-反应关系分析"""
        pass
    
    def identify_confounders(self, treatment, outcome, all_variables, data):
        """混杂因素识别"""
        pass
```

### 4. heterogeneous_analysis.py - 异质性分析

```python
class HeterogeneousAnalysis:
    """治疗效应异质性分析"""
    
    def decision_tree_subgroups(self, features, treatment, outcome, data):
        """使用决策树识别患者亚组"""
        pass
    
    def subgroup_ate(self, subgroup_definition, treatment, outcome, data):
        """计算亚组特异性处理效应"""
        pass
    
    def interaction_test(self, effect_modifier, treatment, outcome, data):
        """交互作用检验"""
        pass
    
    def report_heterogeneity(self):
        """生成异质性分析报告"""
        pass
```

---

## 安装依赖

```bash
# 核心依赖
pip install numpy scipy scikit-learn matplotlib networkx

# 或使用项目requirements.txt
pip install -r requirements.txt
```

---

## 快速开始

### 运行完整演示

```bash
python demo.py
```

### 输出示例

```
[步骤1] 数据加载
数据集大小: 1000 条医案
显效率: 52.3%

[步骤2] PC算法因果发现
发现 7 条因果边

[步骤3] 关键因果路径
  herb_1_jun → efficacy_score (直接效应)
  herb_2_chen → efficacy_score (直接效应)
  syn_1_qixu → herb_1_jun (证候影响用药)

[步骤4] 异质性分析
  气虚血瘀型: ATE=8.52 (95%CI: [6.21, 10.83])
  湿热质: ATE=5.23 (95%CI: [3.11, 7.35])
```

---

## 数据格式

### 输入医案数据

```csv
syn_1_qixu,syn_2_xueyu,syn_3_shire,syn_4_yinxu,herb_1_jun,herb_2_chen,herb_3_zuo,herb_4_shi,efficacy_score
0,1,0,0,1,1,0,0,78.5
1,0,1,0,1,0,1,0,62.3
...
```

### 因果图输出

```json
{
  "nodes": ["syn_1_qixu", "syn_2_xueyu", "herb_1_jun", "efficacy_score"],
  "edges": [
    {"from": "syn_1_qixu", "to": "herb_1_jun", "type": "causal"},
    {"from": "herb_1_jun", "to": "efficacy_score", "type": "causal"}
  ],
  "adj_matrix": [[0, 1, 0], ...]
}
```

---

## 技术指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 因果边发现召回率 | > 85% | 与预设因果对比 |
| ATE估计误差 | < 10% | 与真实ATE对比 |
| 亚组分类准确率 | > 80% | 决策树分组 |
| 交互作用检验功效 | > 80% | 特定效应修饰 |

---

## 方法论

### PC算法原理

1. **骨架发现**: 对每对节点进行条件独立性检验，若在给定某变量集时条件独立，则删除边
2. **边定向**: 应用v-结构规则和传递性规则定向边方向
3. **等价类**: 输出因果图的马尔可夫等价类

### 条件独立性检验

对于连续变量，使用偏相关检验：
$$r_{XY.Z} = \frac{r_{XY} - r_{XZ}r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}$$

若 $|r_{XY.Z}| < z_{\alpha/2}$，则X⊥Y|Z。

---

## 目录结构

```
TCM-Causal-Discovery/
├── README.md                      # 本文件
├── requirements.txt               # 依赖列表
├── demo.py                        # 完整演示脚本
├── data_simulator.py              # 医案数据模拟
├── causal_graph.py                # 因果图构建
├── intervention_effect.py         # 因果效应计算
├── heterogeneous_analysis.py      # 异质性分析
└── data/                          # 数据目录（可选）
```

---

## 开发指南

### 自定义因果结构

```python
# 在 data_simulator.py 中
class MedicalRecordSimulator:
    def generate_with_custom_structure(self, structure):
        """
        structure: dict, 自定义因果结构
        """
        pass
```

### 添加新的效应估计方法

```python
# 在 intervention_effect.py 中
class CustomEffectEstimator(InterventionEffectEstimator):
    def compute_effect(self, method="IV"):
        # 实现工具变量法或其他方法
        pass
```

---

## 参考文献

1. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

2. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

3. Petersen, J. M., et al. (2022). Heterogeneous treatment effects in causal discovery. *arXiv*.

4. 《中医医案学》- 中医临床经验传承经典

---

## 许可证

MIT License

---

## 联系方式

- **项目**: TCM-SCI-Patent-Forge
- **版本**: 1.0.0
- **更新**: 2024-05-10