# Target 003 专利权利要求书 — 五行GNN证候转移

## 独立权利要求

**权利要求1.** 一种融合五行生克约束的中医证候转移概率图神经网络建模方法，其特征在于，包括以下步骤：

S1. 将中医证候建模为图结构中的节点，每个节点包含五行属性（木火土金水）、证候类型和症状向量表示；

S2. 将五行相生关系建模为有向正向激活边，边方向为木→火→土→金→水→木，传递正向激活信号；将五行相克关系建模为有向抑制边，边方向为木→土→水→火→金→木，传递负向抑制信号；

S3. 设计五行约束的消息传递机制，在图神经网络的每一层中，节点特征更新同时聚合相生边的正向消息和相克边的负向消息；

S4. 通过多层消息传递和证候转移预测头，输出从当前证候向其他证候转移的概率分布；

S5. 设计包含五行约束的损失函数，在标准分类损失基础上增加五行一致性惩罚项，使预测结果符合五行生克规律。

## 从属权利要求

**权利要求2.** 根据权利要求1所述的方法，其特征在于，所述步骤S3中的消息传递公式为：
h_i' = σ(W_self·h_i + α_sheng·Σ_{j∈Sheng(i)} W_sheng·h_j + α_ke·Σ_{j∈Ke(i)} W_ke·h_j + b)
其中α_sheng和α_ke为可学习的相生/相克消息权重。

**权利要求3.** 根据权利要求1所述的方法，其特征在于，所述步骤S5中的五行一致性惩罚项对违反五行生克规律的证候转移路径施加额外损失，包括：对逆向相克（被克方反克克制方）施加最大惩罚，对既不相生也不相克的跨行转移施加中等惩罚。

**权利要求4.** 根据权利要求1所述的方法，其特征在于，所述图结构还包含症状节点和方剂节点，症状节点通过"indicates"关系连接到证候节点，方剂节点通过"treated_by"关系连接到证候节点，形成完整的症状-证候-方药知识图谱。

**权利要求5.** 一种融合五行生克约束的中医证候转移概率预测系统，其特征在于，包括：五行知识图谱构建模块、图神经网络消息传递模块、五行约束验证模块和证候转移概率输出模块。

---

## 创新防御论证

### 为何不用纯GNN？
| 纯GNN缺陷 | 五行GNN优势 |
|-----------|-----------|
| 所有边等权，无领域知识 | 相生=正向、相克=负向，物理含义明确 |
| 可能学到违反中医理论的转移 | 硬约束确保不出现"逆相克"等违规 |
| 黑盒，不可解释 | 五行属性可追溯每条边的含义 |

### 为何不用纯规则系统？
| 纯规则缺陷 | 五行GNN优势 |
|-----------|-----------|
| 无法处理模糊性和概率 | GNN输出连续概率分布 |
| 规则冲突无法调和 | 可学习权重自动调和 |
| 无法从数据中学习新模式 | 数据驱动+知识约束双保险 |

---

## SCI论文数学形式化

### Five-Elements Message Passing

Let $\mathcal{G} = (\mathcal{V}, \mathcal{E}_{\text{sheng}} \cup \mathcal{E}_{\text{ke}})$ where:
- $\mathcal{E}_{\text{sheng}} = \{(i,j) : \text{Element}(i) \xrightarrow{\text{生}} \text{Element}(j)\}$
- $\mathcal{E}_{\text{ke}} = \{(i,j) : \text{Element}(i) \xrightarrow{\text{克}} \text{Element}(j)\}$

The layer-wise update rule:

$$\mathbf{h}_i^{(\ell+1)} = \sigma\left(\mathbf{W}_{\text{self}}^{(\ell)} \mathbf{h}_i^{(\ell)} + \alpha_s \sum_{j \in \mathcal{N}_{\text{sheng}}(i)} \mathbf{W}_s^{(\ell)} \mathbf{h}_j^{(\ell)} + \alpha_k \sum_{j \in \mathcal{N}_{\text{ke}}(i)} \mathbf{W}_k^{(\ell)} \mathbf{h}_j^{(\ell)} + \mathbf{b}^{(\ell)}\right)$$

where $\alpha_s = \sigma(\hat{\alpha}_s) > 0$ (promoting) and $\alpha_k = \sigma(\hat{\alpha}_k) < 0$ (restraining).

### Transition Probability

$$P(S_{t+1} = s_j | S_t = s_i, \mathbf{x}) = \text{softmax}\left(\mathbf{W}_{\text{trans}} [\mathbf{h}_i^{(L)} \| \mathbf{h}_j^{(L)}]\right)$$

### Five-Elements Consistency Loss

$$\mathcal{L}_{\text{FE}} = \sum_{i,j} P(s_j|s_i) \cdot \mathbb{1}[\text{Element}(i) \not\sim \text{Element}(j)] \cdot \gamma_{ij}$$

where $\gamma_{ij}$ is the penalty weight: $\gamma_{ij} = 2.0$ for 逆相克, $\gamma_{ij} = 1.0$ for unrelated elements, $\gamma_{ij} = 0$ for valid 生克 relations.
