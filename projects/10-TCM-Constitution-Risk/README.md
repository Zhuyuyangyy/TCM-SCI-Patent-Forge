# TCM-Constitution-Risk: 中医体质疾病风险预测系统

> 基于Cox比例风险模型与机器学习的个体化疾病风险预测

---

## 项目概述

TCM-Constitution-Risk 是一个将中医体质辨识与现代生存分析技术深度融合的疾病风险预测系统。系统通过分析不同体质人群的疾病发生发展规律，提供个性化的健康风险评估与预防建议。

### 核心特性

| 特性 | 描述 |
|------|------|
| **九种体质分类** | 精准辨识平和/气虚/阳虚/阴虚/痰湿/湿热/血瘀/气郁/特禀 |
| **Cox风险建模** | 比例风险假设下的生存分析 |
| **Kaplan-Meier曲线** | 生存概率可视化 |
| **Log-Rank检验** | 体质间差异统计检验 |
| **风险因素排序** | 识别关键风险贡献因素 |

---

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TCM-Constitution-Risk 整体架构                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                         输入层                                     │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │
│   │  │ 体质问卷   │  │ 体检数据   │  │ 历史疾病记录           │   │   │
│   │  └──────┬──────┘  └──────┬──────┘  └───────────┬─────────────┘   │   │
│   └─────────┼───────────────┼─────────────────────┼───────────────────┘   │
│             └───────────────┴─────────────────────┘                      │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                    体质分类模块                                    │   │
│   │  ┌─────────────────────────────────────────────────────────────┐  │   │
│   │  │            ConstitutionClassifier (constitution_classifier) │  │   │
│   │  │                                                          │  │   │
│   │  │    输入问卷 ──▶ 特征提取 ──▶ 九分类Softmax ──▶ 体质结果  │  │   │
│   │  │                                                          │  │   │
│   │  │    特征维度: 问卷28项 + 体检指标48项 + 舌脉特征16项       │  │   │
│   │  └─────────────────────────────────────────────────────────────┘  │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                    Cox风险模型层                                  │   │
│   │  ┌─────────────────────────────────────────────────────────────┐  │   │
│   │  │              CoxModel (cox_model.py)                        │  │   │
│   │  │                                                          │  │   │
│   │  │    ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐ │  │   │
│   │  │    │ 风险因素 │──▶│ 风险函数 │──▶│ 生存函数 │──▶│ 风险评分│ │  │   │
│   │  │    │  提取   │   │  估计   │   │  S(t)   │   │  C-index│ │  │   │
│   │  │    └─────────┘   └─────────┘   └─────────┘   └─────────┘ │  │   │
│   │  └─────────────────────────────────────────────────────────────┘  │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                    生存分析模块                                    │   │
│   │  ┌─────────────────────────────────────────────────────────────┐  │   │
│   │  │         SurvivalAnalysis (survival_analysis.py)             │  │   │
│   │  │                                                          │  │   │
│   │  │    Kaplan-Meier ──▶ 生存曲线    Log-Rank ──▶ p值检验    │  │   │
│   │  │                                                          │  │   │
│   │  │    [体质A] ═══════════════════                           │  │   │
│   │  │    [体质B] ════════════════════                          │  │   │
│   │  │    [体质C] ═════════════════                             │  │   │
│   │  └─────────────────────────────────────────────────────────────┘  │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                              │                                              │
│                              ▼                                              │
│   ┌───────────────────────────────────────────────────────────────────┐   │
│   │                      输出层                                        │   │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐   │   │
│   │  │ 风险评估   │  │ 生存预测   │  │ 个性化建议              │   │   │
│   │  │ (5年/10年) │  │ (生存曲线) │  │ (治未病方案)            │   │   │
│   │  └─────────────┘  └─────────────┘  └─────────────────────────┘   │   │
│   └───────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 九种体质分类

| 体质类型 | 特征描述 | 高风险疾病 |
|---------|---------|-----------|
| 平和质 | 阴阳气血调和，体态适中 | 低风险 |
| 气虚质 | 元气不足，容易疲乏 | 感冒、肺病 |
| 阳虚质 | 阳气不足，畏寒怕冷 | 心脑血管病 |
| 阴虚质 | 阴液亏少，口干咽燥 | 失眠、便秘 |
| 痰湿质 | 痰湿凝聚，体形肥胖 | 代谢综合征 |
| 湿热质 | 湿热内蕴，面垢油光 | 痤疮、胆囊炎 |
| 血瘀质 | 血行不畅，肤色晦暗 | 肿瘤、心梗 |
| 气郁质 | 气机郁滞，情志不畅 | 抑郁、乳腺病 |
| 特禀质 | 先天失常，易过敏 | 过敏性疾病 |

---

## 核心模块

### 1. constitution_classifier.py - 体质分类器

```python
class ConstitutionClassifier:
    """基于问卷和体检数据的九种体质分类"""
    
    def __init__(self):
        self.feature_dim = 92  # 28+48+16
        self.num_classes = 9
    
    def extract_features(self, questionnaire_data, physical_data, tongue_pulse_data):
        """提取体质相关特征"""
        pass
    
    def classify(self, features) -> dict:
        """返回九种体质概率分布"""
        # 输出: {平和质: 0.1, 气虚质: 0.6, ...}
        pass
    
    def get_primary_constitution(self, probs) -> str:
        """获取主要体质类型"""
        pass
```

### 2. cox_model.py - Cox比例风险模型

```python
class CoxRiskModel:
    """Cox比例风险模型用于疾病风险预测"""
    
    def __init__(self):
        self.hazard_ratio = {}
        self.baseline_hazard = {}
        self.concordance_index = 0.0
    
    def fit(self, survival_data, features):
        """训练Cox模型"""
        # survival_data: (time, event) + features
        pass
    
    def predict_risk(self, patient_features, time_horizon=5):
        """预测5年/10年风险"""
        # return: risk_score, survival_probability
        pass
    
    def get_risk_factors(self, top_k=10):
        """获取Top-K风险因素"""
        pass
```

### 3. survival_analysis.py - 生存分析

```python
class SurvivalAnalysis:
    """生存分析模块"""
    
    def kaplan_meier(self, survival_data, group_by):
        """Kaplan-Meier生存曲线估计"""
        pass
    
    def log_rank_test(self, group1_data, group2_data):
        """Log-Rank检验比较两组生存差异"""
        pass
    
    def plot_survival_curves(self, constitution_type):
        """绘制特定体质的生存曲线"""
        pass
```

---

## 安装依赖

```bash
# 核心依赖
pip install torch scikit-learn lifelines matplotlib seaborn pandas numpy

# 生存分析额外依赖
pip install lifelines  # Kaplan-Meier, CoxPH

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
from constitution_classifier import ConstitutionClassifier
from cox_model import CoxRiskModel
from survival_analysis import SurvivalAnalysis

# 1. 体质分类
classifier = ConstitutionClassifier()
patient_features = load_patient_data("path/to/data")
constitution_probs = classifier.classify(patient_features)
primary_constitution = classifier.get_primary_constitution(constitution_probs)

# 2. 风险预测
cox_model = CoxRiskModel()
cox_model.fit(survival_dataset, features)
risk_score = cox_model.predict_risk(patient_features, time_horizon=5)

# 3. 生存分析
analysis = SurvivalAnalysis()
km_curves = analysis.kaplan_meier(survival_data, group_by="constitution")
log_rank_p = analysis.log_rank_test(group_a, group_b)

print(f"体质: {primary_constitution}")
print(f"5年风险评分: {risk_score:.2%}")
```

---

## 数据格式

### 输入数据格式

```json
{
  "patient_id": "P001",
  "questionnaire": {
    "fatigue_frequency": 4,
    "cold_intolerance": 3,
    "sweating_easily": 2,
    ...
  },
  "physical_exam": {
    "bmi": 24.5,
    "blood_pressure": [120, 80],
    "heart_rate": 72,
    ...
  },
  "tongue_pulse": {
    "tongue_color": "淡红",
    "tongue_coating": "薄白",
    "pulse_type": "弦"
  },
  "survival_data": {
    "follow_up_years": 5.2,
    "event_occurred": 0
  }
}
```

### 输出结果格式

```json
{
  "constitution": "气郁质",
  "constitution_confidence": 0.78,
  "risk_prediction": {
    "5_year_risk": 0.23,
    "10_year_risk": 0.41,
    "survival_probability_5y": 0.77
  },
  "risk_factors": [
    {"factor": "气郁质", "hazard_ratio": 1.85, "p_value": 0.001},
    {"factor": "血瘀质", "hazard_ratio": 2.12, "p_value": 0.000},
    ...
  ],
  "survival_curves": {
    "平和质": [...],
    "气郁质": [...],
    ...
  },
  "recommendations": [
    "建议进行情志疏导",
    "定期体检心血管指标",
    ...
  ]
}
```

---

## 技术指标

| 指标 | 目标值 | 说明 |
|------|--------|------|
| C-statistic | > 0.75 | 模型区分度 |
| 体质分类准确率 | > 85% | 与专家判断对比 |
| 风险预测MAE | < 0.10 | 预测误差 |
| Log-Rank p值 | < 0.05 | 体质间差异显著性 |

---

## 目录结构

```
10-TCM-Constitution-Risk/
├── README.md                    # 本文件
├── requirements.txt             # 依赖列表
├── demo.py                       # 演示脚本
├── constitution_classifier.py    # 体质分类器
├── cox_model.py                  # Cox风险模型
├── survival_analysis.py          # 生存分析
└── data/                         # 数据目录（可选）
```

---

## 开发指南

### 添加新疾病预测

```python
# 在 cox_model.py 中
class CoxRiskModel:
    def add_disease_model(self, disease_name, training_data):
        """为特定疾病添加预测模型"""
        model = CoxPHFitter()
        model.fit(training_data, duration_col='time', event_col='event')
        self.disease_models[disease_name] = model
```

### 自定义体质问卷

```python
# 在 constitution_classifier.py 中
QUESTIONNAIRE_ITEMS = [
    {"id": "q1", "question": "是否容易疲劳?", "options": ["从不", "偶尔", "经常", "总是"]},
    # 添加更多题目...
]
```

---

## 参考文献

1. Cox, D. R. (1972). Regression models and life-tables. *JRSS-B*, 34(2), 187-220.

2. Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. *JASA*, 53(282), 457-481.

3. Wang, Y., et al. (2023). TCM Constitution and disease risk: A machine learning approach. *JMIR*.

4. 《中医体质分类与判定》- 中华中医药学会标准 (ZYYXH/T157-2009)

5. 《黄帝内经·素问》- "上医治未病"思想

---

## 许可证

MIT License

---

## 联系方式

- **项目**: TCM-SCI-Patent-Forge
- **版本**: 1.0.0
- **更新**: 2024-05-10