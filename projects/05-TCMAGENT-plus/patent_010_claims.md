# Target 010 专利权利要求书 — MACCM多智能体会诊

## 独立权利要求

**权利要求1.** 一种基于多智能体协同的中医对抗性会诊方法，其特征在于，包括以下步骤：

S1. 将输入病例分解为病位、病性、病因、传变四个子问题，分别路由至对应的专家智能体；

S2. 多个专家智能体并行处理各自子问题，输出包含主证候、置信度、替代证候列表和推理链的结构化证据；

S3. 冲突检测模块分析各智能体输出的证候主张，识别存在逻辑矛盾的证候对（如寒热互斥、虚实矛盾）；

S4. 对存在冲突的证候对启动对抗性审议流程：生成对立假设，要求各智能体提供支持/反驳证据；

S5. 基于可学习的智能体可信度权重和证据置信度进行加权投票，收敛至最终证候判断；

S6. 当共识分数低于阈值时，标记为需人工复核并生成分歧分析报告。

## 从属权利要求

**权利要求2.** 根据权利要求1所述的方法，其特征在于，所述步骤S2中的专家智能体包括六类：病位分析智能体（基于脏腑辨证规则库）、病性分析智能体（基于八纲辨证规则库）、病因分析智能体（基于中医病因学知识图谱）、传变分析智能体（基于疾病传化规律知识库）、古今经验智能体（基于10万+经典医案数据库）和药理安全智能体（基于配伍禁忌规则库）。

**权利要求3.** 根据权利要求1所述的方法，其特征在于，所述步骤S4中的对抗性审议采用三层对抗架构：第一层为症状生成对抗网络，评估症状组合的中医逻辑一致性；第二层为证候转换对抗网络，评估证候传变路径的合理性；第三层为对话策略生成网络，生成符合问诊规范的追问。

**权利要求4.** 根据权利要求1所述的方法，其特征在于，所述步骤S5中的可学习可信度权重通过反向传播在训练过程中自动调整，使历史诊断准确率较高的智能体获得更高的投票权重。

**权利要求5.** 一种基于多智能体协同的中医对抗性会诊系统，其特征在于，包括：问题分解引擎、并行证据合成智能体集群、冲突检测模块、对抗性审议模块、加权共识引擎和人工复核接口。

---

## 分布式群体智能形式化 (Agent-Paper)

### Problem Formulation

Let $\mathcal{A} = \{a_1, \ldots, a_K\}$ denote $K$ expert agents. Each agent $a_k$ produces evidence $e_k = (s_k, c_k, \mathbf{r}_k)$ where $s_k$ is the proposed syndrome, $c_k \in [0,1]$ is confidence, and $\mathbf{r}_k$ is the reasoning chain.

### Consensus Convergence

The final syndrome is determined by confidence-weighted voting:

$$s^* = \arg\max_{s \in \mathcal{S}} \sum_{k=1}^{K} w_k \cdot c_k \cdot \mathbb{1}[s_k = s]$$

where $w_k$ are learned agent credibility weights with $\sum_k w_k = 1$ and $\mathcal{S}$ is the syndrome space.

### Conflict Detection

A conflict exists between agents $a_i$ and $a_j$ iff their proposed syndromes are logically contradictory:

$$\text{Conflict}(a_i, a_j) = \mathbb{1}\left[\exists (p, q) \in \mathcal{C}_{\text{oppose}} : p \in s_i \wedge q \in s_j\right]$$

where $\mathcal{C}_{\text{oppose}} = \{(\text{寒}, \text{热}), (\text{虚}, \text{实}), (\text{阴虚}, \text{阳虚}), \ldots\}$ is the set of opposing syndrome pairs.

### Consensus Score

$$\text{Consensus} = 1 - \frac{|\{(i,j) : \text{Conflict}(a_i, a_j)\}|}{\binom{K}{2}}$$

### Adversarial Deliberation as Min-Max Optimization

$$\min_{\theta_G} \max_{\theta_D} \mathbb{E}_{e \sim \mathcal{E}_{\text{agent}}} \left[ \log D_\theta(e) \right] + \mathbb{E}_{\hat{e} \sim G_\theta} \left[ \log(1 - D_\theta(\hat{e})) \right]$$

where $G_\theta$ generates adversarial evidence and $D_\theta$ discriminates genuine vs. synthetic diagnostic reasoning.
