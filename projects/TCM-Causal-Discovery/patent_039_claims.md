# Target 039 专利权利要求书 — 证候因果推断SCM框架

## 独立权利要求

**权利要求1.** 一种基于结构因果模型的中医证候-治疗因果推断方法，其特征在于，包括以下步骤：

S1. 构建中医领域因果知识图谱，其中节点包括证候节点、症状节点、治疗节点、患者特征节点和结局节点，边表示节点间的因果关系；

S2. 为因果图中的每个节点定义结构方程X_i = f_i(Pa(X_i), U_i)，其中Pa(X_i)为节点在因果图中的父节点集合，U_i为外生噪声变量；

S3. 基于后门准则识别需要调整的混杂变量集合Z，使得在控制Z后，治疗T与结局Y之间的关联可解释为因果效应；

S4. 采用逆概率加权或倾向性评分分层方法估计平均治疗效应ATE = E[Y|do(T=1)] - E[Y|do(T=0)]；

S5. 通过反事实推理框架估计个体治疗效应ITE_i = Y_i^{T=1} - Y_i^{T=0}，实现"若对该患者采用不同治法，疗效会如何变化"的反事实分析。

## 从属权利要求

**权利要求2.** 根据权利要求1所述的方法，其特征在于，所述步骤S1中的因果知识图谱的边权重通过PC因果发现算法从临床观察数据中自动学习得到，采用条件独立性检验作为边存在性的判据。

**权利要求3.** 根据权利要求1所述的方法，其特征在于，所述步骤S2中的结构方程采用神经网络参数化，使非线性因果关系能够被充分建模。

**权利要求4.** 根据权利要求1所述的方法，其特征在于，所述步骤S5中的反事实推理包括三个阶段：推断阶段（从观测数据反推外生噪声U）、干预阶段（替换治疗节点的结构方程为反事实值）、预测阶段（利用推断的噪声和新的治疗值计算反事实结局）。

**权利要求5.** 一种基于结构因果模型的中医证候-治疗因果推断系统，其特征在于，包括：因果知识图谱构建模块、结构方程学习模块、混杂变量识别模块、因果效应估计模块和反事实推理模块。

---

## 辨证论治的SCM数学表达 (Agent-Paper)

### Structural Causal Model for TCM

Let $\mathcal{G} = (\mathbf{V}, \mathbf{E})$ be a DAG where $\mathbf{V} = \{X_1, \ldots, X_p\}$ includes syndrome features $\mathbf{S} = (S_1, \ldots, S_m)$, treatment $T$, confounders $\mathbf{Z} = (Z_1, \ldots, Z_l)$, and outcome $Y$.

The SCM is defined by a set of structural equations:

$$X_i := f_i(\text{Pa}(X_i), U_i), \quad i = 1, \ldots, p$$

### Do-Calculus for Treatment Effect

The interventional distribution under do-calculus:

$$P(Y | \text{do}(T = t)) = \sum_{\mathbf{z}} P(Y | T = t, \mathbf{Z} = \mathbf{z}) P(\mathbf{Z} = \mathbf{z})$$

### Counterfactual Reasoning (Three-Step Procedure)

**Step 1 (Abduction):** Given observed data $\mathbf{X} = \mathbf{x}$, infer exogenous noise:

$$U_i = X_i - f_i(\text{Pa}(X_i))$$

**Step 2 (Action):** Replace treatment equation with counterfactual value $T = t'$.

**Step 3 (Prediction):** Compute counterfactual outcome:

$$Y_{T=t'} = f_Y(\text{Pa}(Y) \setminus \{T\}, T = t', U_Y)$$

### Individual Treatment Effect

$$\tau_i = Y_i^{T=1} - Y_i^{T=0} = f_Y(\text{Pa}(Y)_i, T=1, U_{Y_i}) - f_Y(\text{Pa}(Y)_i, T=0, U_{Y_i})$$

### Average Treatment Effect via Backdoor Adjustment

$$\text{ATE} = \mathbb{E}_{\mathbf{Z}} \left[ \mathbb{E}[Y | T=1, \mathbf{Z}] - \mathbb{E}[Y | T=0, \mathbf{Z}] \right]$$

### TCM-Specific: Dynamic Syndrome Adjustment

In TCM, treatment is adjusted based on evolving syndrome. This is modeled as a time-indexed SCM:

$$S^{(t+1)} := f_S(S^{(t)}, T^{(t)}, \mathbf{Z}, U_S^{(t)})$$
$$T^{(t+1)} := g(S^{(t+1)}, \text{treatment\_protocol})$$

This captures the iterative nature of 辨证论治 (syndrome differentiation and treatment adjustment).
