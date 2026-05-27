# KGMAL++: 知识图谱终身学习系统

> 基于灾难性遗忘抑制的中医知识图谱增量学习框架

---

## 项目概述

KGMAL++（Knowledge Graph Multi-Agent Lifelong Learning）是一个面向中医领域的知识图谱终身学习系统。系统通过创新的灾难性遗忘抑制技术，实现新知识的持续融入与旧知识的稳定保持，解决了传统知识图谱更新中的"灾难性遗忘"难题。

### 核心特性

| 特性 | 描述 |
|------|------|
| **增量构建** | 无需全量重建，支持三元组级别增量更新 |
| **遗忘抑制** | ExperienceReplay + EWC双重机制保护关键参数 |
| **冲突检测** | 自动识别新旧知识矛盾并提供解决方案 |
| **时序建模** | 支持医疗记录的时间维度建模与分析 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KGMAL++ 整体架构                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                      数据输入层                              │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │  │
│   │  │ 电子病历   │  │ 经典文献   │  │ 临床指南/规则       │  │  │
│   │  └──────┬──────┘  └──────┬──────┘  └──────────┬──────────┘  │  │
│   └─────────┼───────────────┼───────────────────┼────────────────┘  │
│             └───────────────┴───────────────────┘                   │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    知识图谱构建层                            │  │
│   │  ┌─────────────────────────────────────────────────────────┐  │  │
│   │  │              KGMLIncrementalBuilder                     │  │  │
│   │  │  ┌────────────┐  ┌────────────┐  ┌────────────────┐   │  │  │
│   │  │  │ 抽取模块   │  │ 融合模块   │  │ 冲突检测模块   │   │  │  │
│   │  │  └────────────┘  └────────────┘  └────────────────┘   │  │  │
│   │  └─────────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                   终身学习控制层                             │  │
│   │  ┌──────────────────┐     ┌──────────────────┐               │  │
│   │  │  ExperienceReplay │     │   EWCRegularizer  │               │  │
│   │  │  (经验回放)       │     │  (弹性权重巩固)   │               │  │
│   │  └──────────────────┘     └──────────────────┘               │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │                    时序建模层                               │  │
│   │  ┌─────────────────────────────────────────────────────────┐  │  │
│   │  │              TemporalMedicalRecord                      │  │  │
│   │  │  - 时间点症状-治疗-结果建模                              │  │  │
│   │  │  - 症状轨迹分析                                          │  │  │
│   │  │  - 治疗效果预测                                          │  │  │
│   │  └─────────────────────────────────────────────────────────┘  │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 核心模块

### 1. kg_builder.py - 知识图谱构建器

```python
class KGMLIncrementalBuilder:
    """增量式知识图谱构建器"""
    
    def add_triplet(self, subject, predicate, object, timestamp=None):
        """增量添加三元组，无须全量重建"""
        pass
    
    def detect_conflict(self, new_triplet) -> ConflictReport:
        """检测新旧知识冲突"""
        pass
    
    def merge_knowledge(self, source_triplets) -> MergeResult:
        """合并来自不同来源的知识"""
        pass
```

### 2. catastrophic_forgetting.py - 灾难性遗忘抑制

```python
class CatastrophicForgettingSuppressor:
    """双重机制抑制灾难性遗忘"""
    
    def experience_replay(self, buffer_size=1000):
        """经验回放：维护历史样本优先缓冲区"""
        pass
    
    def ewc_regularizer(self, lambda_ewc=1000):
        """弹性权重巩固：保护关键参数"""
        pass
```

### 3. temporal_model.py - 时序医疗记录

```python
class TemporalMedicalRecord:
    """时序医疗记录建模"""
    
    def model_trajectory(self, patient_id, time_range):
        """建模患者症状轨迹"""
        pass
    
    def predict_outcome(self, treatment_plan, patient_state):
        """预测治疗效果"""
        pass
```

---

## 安装依赖

```bash
# 核心依赖
pip install torch rdflib networkx numpy scikit-learn

# 扩展依赖（可选）
pip install spacy  # 用于NLP抽取
pip install neo4j  # 用于图数据库存储

# 或使用项目requirements.txt
pip install -r requirements.txt
```

---

## 快速开始

### 运行演示

```bash
python demo.py
```

### 基本使用

```python
from kg_builder import KGMLIncrementalBuilder
from catastrophic_forgetting import CatastrophicForgettingSuppressor

# 初始化构建器
builder = KGMLIncrementalBuilder()

# 增量添加三元组
builder.add_triplet("肝郁", "导致", "气滞", timestamp="2024-01-01")
builder.add_triplet("气滞", "表现", "胁痛", timestamp="2024-01-01")

# 查询
results = builder.query("肝郁", relation="导致")
print(results)

# 训练（带遗忘抑制）
suppressor = CatastrophicForgettingSuppressor()
new_knowledge = load_new_knowledge()
suppressor.train(new_knowledge, builder.graph)
```

---

## 数据格式

### 三元组格式

```json
{
  "subject": "肝郁",
  "predicate": "导致",
  "object": "气滞",
  "timestamp": "2024-05-10T10:30:00",
  "confidence": 0.95,
  "source": "临床病历"
}
```

### 冲突报告

```json
{
  "conflict_type": "contradiction",
  "triplet_a": {"subject": "肝郁", "predicate": "导致", "object": "气滞"},
  "triplet_b": {"subject": "肝郁", "predicate": "治疗", "object": "补血"},
  "resolution": "根据中医理论，肝郁既可导致气滞（新知识），"
               "补血也可治疗肝郁（旧知识），两者不矛盾。"
}
```

---

## 技术指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| 遗忘抑制率 | > 90% | 旧任务性能保持率 |
| 增量更新速度 | < 100ms | 单个三元组更新 |
| 冲突检测准确率 | > 85% | 自动冲突识别 |
| 时序预测MAE | < 0.15 | 治疗效果预测 |

---

## 算法详解

### ExperienceReplay（经验回放）

```python
class ExperienceReplay:
    def __init__(self, buffer_size=1000, priority='entropy'):
        self.buffer = deque(maxlen=buffer_size)
        self.priority = priority  # 'entropy' or 'fifo'
    
    def push(self, triplet):
        """入栈并维护优先级"""
        priority_score = self.compute_priority(triplet)
        self.buffer.append((triplet, priority_score))
    
    def sample(self, batch_size):
        """按优先级采样回放"""
        # 高优先级样本被采样概率更高
        pass
```

### EWC（弹性权重巩固）

```python
class EWCRegularizer:
    def __init__(self, lambda_ewc=1000):
        self.lambda = lambda_ewc
        self.fisher_info = {}  # 费雪信息矩阵
        self.optimal_params = {}  # 最优参数备份
    
    def compute_fisher(self, model, dataset):
        """计算费雪信息矩阵"""
        # 衡量参数对旧任务的重要性
        pass
    
    def ewc_loss(self, model):
        """计算EWC正则化损失"""
        loss = 0
        for param_name, param in model.named_parameters():
            if param_name in self.fisher_info:
                diff = param - self.optimal_params[param_name]
                loss += (self.fisher_info[param_name] * diff ** 2).sum()
        return self.lambda * loss
```

---

## 目录结构

```
06-KGMAL-plus/
├── README.md                      # 本文件
├── requirements.txt               # 依赖列表
├── demo.py                         # 演示脚本
├── kg_builder.py                  # 知识图谱构建器
├── catastrophic_forgetting.py      # 遗忘抑制模块
├── temporal_model.py               # 时序建模模块
└── data/                           # 数据目录（可选）
```

---

## 开发指南

### 添加新的抽取规则

```python
# 在 kg_builder.py 中扩展
class KGMLIncrementalBuilder:
    def add_extraction_rule(self, rule_name, rule_func):
        """添加新的抽取规则"""
        self.extraction_rules[rule_name] = rule_func
```

### 自定义冲突解决策略

```python
# 继承冲突解决器
class CustomConflictResolver(ConflictResolver):
    def resolve(self, conflict):
        # 实现自定义解决逻辑
        pass
```

---

## 参考文献

1. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks. *PNAS*, 114(13), 3521-3526.

2. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning. *NeurIPS*, 30.

3. Pan, S., et al. (2020). Knowledge graph embedding for temporal knowledge reasoning. *TKDE*.

4. 《中医基础理论》- 十四五规划教材

---

## 许可证

MIT License

---

## 联系方式

- **项目**: TCM-SCI-Patent-Forge
- **版本**: 1.0.0
- **更新**: 2024-05-10