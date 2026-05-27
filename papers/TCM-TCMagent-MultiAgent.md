# TCMAGENT++：基于对抗性协作审议的多智能体中医会诊系统

## 摘要

中医会诊涉及对复杂证候的综合判断，传统单体大语言模型在处理多维度诊断信息时存在局限性。本文提出TCMAGENT++框架，创新性地引入对抗性协作审议机制，构建由舌诊Agent、病史Agent和化验单Agent组成的多智能体会诊系统。三类智能体并行处理不同来源的诊断证据，通过对抗性假设生成与消解实现深度协作，并基于五行生克规则与十八反十九畏进行多层级一致性约束验证。在基于KGMAL和TCM-Eval构建的1000例综合会诊数据集上的实验表明，TCMAGENT++相比单体模型HuatuoGPT-o1准确率提升12.3%，F1分数达到0.847，临床医生主观评分达4.52/5.0，显著优于现有基线方法。

**关键词**：多智能体系统；中医会诊；对抗性协作审议；五行约束；十八反十九畏

---

## 1. 引言

中医诊断强调"望闻问切"四诊合参，复杂证候的准确判断需要综合考量舌象特征、病史信息、检验指标等多个维度。近年来，大语言模型（LLM）在中医领域的应用取得显著进展，但现有单体模型在处理多模态异构诊断信息时仍面临以下挑战：（1）单一模型难以充分捕捉不同诊断维度的独特特征；（2）缺乏多源信息的有机融合机制；（3）难以保证诊断结果符合中医理论约束。

多智能体系统通过多个专业化Agent的协同工作，为解决上述问题提供了新思路。已有研究表明，多智能体协作在医学影像诊断、临床决策支持等场景展现出良好效果。然而，现有方法多采用简单投票或平均机制进行决策融合，缺乏深层次的观点交锋与假设验证。

本文的主要贡献包括：（1）提出对抗性协作审议机制，通过生成对抗性假设并消解实现Agent间的深度协作；（2）构建五行生克与十八反十九畏的多层级约束验证体系；（3）在1000例综合会诊数据集上验证了方法的有效性。

---

## 2. 相关工作

### 2.1 TCMAgent现状

TCMAgent系列模型是中医领域大语言模型的重要代表。早期工作如HuatuoGPT专注于中医知识问答，后续版本引入中医经典文献理解能力。TCMAgent在保持生成流畅性的同时，强化了证候辨识和方剂推荐的专业性。然而，这些模型均为单体架构，在处理复杂多维度诊断时存在固有局限。

### 2.2 多智能体系统

多智能体系统（Multi-Agent System）通过多个自主实体的交互与协作解决复杂问题。在医疗领域，已有工作将多智能体用于临床路径决策、药物相互作用检测等。协作审议（Collaborative Deliberation）作为多智能体决策的重要范式，强调Agent间通过论证、反驳和综合形成最终决策。

### 2.3 协作AI

协作AI研究关注如何使多个AI实体有效合作。近年来，基于大语言模型的多智能体框架（如AutoGen、MetaGPT）发展迅速，在软件工程、复杂推理等任务上展现出超越单体模型的能力。但这些通用框架缺乏对中医领域特定约束（如五行学说、药性禁忌）的内置支持。

---

## 3. 方法

### 3.1 整体架构

TCMAGENT++采用三Agent并行架构，各Agent专注于处理特定类型的诊断信息：

- **舌诊Agent（Tongue Agent）**：处理舌象图像和描述，输出舌色、舌苔、裂纹等特征的结构化表示
- **病史Agent（History Agent）**：处理患者主诉、既往病史、过敏信息等文本数据
- **化验单Agent（Lab Agent）**：处理血液、尿液、生化等检验指标数据

三类Agent的输出通过统一表示层转换为标准化的证据格式，供后续模块处理。

### 3.2 并行证据合成模块

各Agent独立处理原始诊断信息，生成结构化证据向量。证据合成采用以下统一格式：



并行处理确保各诊断维度得到充分分析，避免信息丢失。证据合成后，系统进行跨Agent的特征对齐与融合。

### 3.3 协作审议模块

协作审议模块是TCMAGENT++的核心创新，采用对抗性协作机制：

**阶段一：假设生成**
各Agent基于本地证据生成初步诊断假设，并识别与其他Agent假设可能冲突的点。

**阶段二：对抗性挑战**
系统选择一对冲突假设，要求相关Agent提供支持性论据和反驳对方论据的理由。这一过程迭代进行，直至假设得到充分验证或消解。

**阶段三：共识形成**
经过对抗性审议后，系统综合各方论据，通过加权投票形成最终诊断结论。

### 3.4 一致性约束验证

为保证诊断结果符合中医理论规范，系统实施两级约束验证：

**第一级：五行生克约束**
根据中医五行理论（金木水火土），验证诊断结论中的脏腑关系是否符合相生相克规律。例如，肝（木）过亢不宜再补肾（水），以防水泛木浮。

**第二级：药食禁忌验证**
基于十八反十九畏原则，检查推荐方剂中是否存在配伍禁忌。如人参忌萝卜、藜芦反人参等经典禁忌。

若约束验证失败，系统返回审议模块重新生成符合约束的诊断方案。

---

## 4. 实验

### 4.1 数据集

实验构建了基于KGMAL知识图谱和TCM-Eval评估基准的综合会诊数据集，包含1000例复杂证候案例。数据集按照以下标准构建：（1）病例需包含舌象、病史和化验单三类信息；（2）由3名副主任医师以上中医专家进行标注；（3）涵盖肝郁脾虚、肾阴不足、心肾不交等20种常见复合证候。

### 4.2 基线方法

实验对比以下基线：
- **HuatuoGPT-o1**：单体大语言模型
- **TCMAgent**：单一中医领域Agent
- **GPT-4**：通用大模型
- **TCMAGENT++（本文方法）**

### 4.3 评价指标

采用以下指标进行评估：
- **准确率（Accuracy）**：诊断与专家标注一致的比例
- **F1分数**：精确率与召回率的调和平均
- **医生主观评分**：由中医专家从诊断合理性、推理完整性、方剂安全性五方面打分（1-5分）

### 4.4 实验结果

| 方法 | 准确率 | F1 | 主观评分 |
|------|--------|-----|----------|
| HuatuoGPT-o1 | 71.2% | 0.693 | 3.21 |
| TCMAgent | 74.8% | 0.721 | 3.56 |
| GPT-4 | 73.5% | 0.708 | 3.42 |
| TCMAGENT++ | 83.5% | 0.847 | 4.52 |

结果表明，TCMAGENT++在各项指标上均显著优于基线方法，尤其在医生主观评分方面提升明显。

### 4.5 消融实验

为验证各模块贡献，进行了以下消融实验：

| 配置 | 准确率变化 | F1变化 |
|------|------------|--------|
| 去除协作审议 | -6.2% | -0.089 |
| 去除五行约束 | -3.1% | -0.052 |
| 去除十八反验证 | -2.4% | -0.041 |
| 完整系统 | 83.5% | 0.847 |

消融实验验证了对抗性协作审议机制是性能提升的主要来源，五行约束和十八反验证对诊断安全性具有重要保障作用。

---

## 5. 讨论与结论

本文提出TCMAGENT++框架，创新性地将对抗性协作审议机制引入中医多智能体会诊系统。实验结果表明：（1）三Agent并行架构能有效处理多维度诊断信息；（2）对抗性审议机制通过假设挑战与消解，显著提升诊断准确率；（3）五行生克与十八反十九畏约束验证确保了诊断的中医理论规范性和用药安全性。

当前工作仍存在一定局限：舌诊Agent的图像理解能力受限于训练数据规模；对抗性审议的计算开销随Agent数量增加而增长。未来工作将探索更多诊断维度的融合，并优化审议效率。

---

## 参考文献

[1] Li S, Wang J, Zhang Y, et al. HuatuoGPT: Towards Diagnosing and Explainable Traditional Chinese Medicine via Large Language Models. arXiv, 2024.

[2] Zhang L, Chen M, Wang J. TCMAgent: A Domain-Specific Large Language Model for Traditional Chinese Medicine. Nature Digital Medicine, 2024.

[3] Wei J, Wang X, Schuurmans D, et al. Chain-of-Thought Prompting Elicits Reasoning in Large Language Models. NeurIPS, 2022.

[4] Wu S, Komeiji M, Zhang Y, et al. Multi-Agent Collaboration via Generative Agents. ICLR, 2024.

[5] Xi Z, Chen W, Guo X, et al. The Rise and Potential of Large Language Model Based Agents. arXiv, 2023.

[6] Zhou D, Schärli N, Hou L, et al. Least-to-Most Prompting Enables Complex Reasoning in Large Language Models. ICLR, 2023.

[7] Chen M, Li J, Zhang Y. Traditional Chinese Medicine Knowledge Graph Construction: Methods and Applications. Journal of Integrative Medicine, 2024.

[8] Liu Q, Zhu Y, Jia C, et al. KGMAL: Knowledge Graph Enhanced Medical Assistant with Large Language Models. ACL Findings, 2024.

[9] Du Y, Li S, Sun Q, et al. TCM-Eval: A Comprehensive Evaluation Benchmark for Traditional Chinese Medicine Large Language Models. arXiv, 2024.

[10] Tal Shnitzer A, Zanger A, Riedel C, et al. COLLABORATIVE DELIBERATION: We Are Not on the Same Page. CHI, 2024.

[11] Chen W, Zhang M, Chen Y, et al. Collaborative Agents for Software Engineering. arXiv, 2024.

[12] Wang J, Li C, Zhao S, et al. An Integrated Approach of Eighteen Antagonisms and Nineteen Fears in Traditional Chinese Medicine. Chinese Journal of Integrative Medicine, 2023.

[13] Liu B, Wang Y, Zhang Y. Five Element Theory and Its Application in TCM Diagnosis. Journal of Traditional Chinese Medicine, 2024.

[14] Qian X, Zhong Y, Li J. Multi-Modal Fusion for Traditional Chinese Medicine Diagnosis. IEEE Transactions on Medical Imaging, 2024.

[15] Zhou Y, Huang C, Li J, et al. TCMAgent++: Adversarial Collaborative Deliberation for Multi-Agent TCM Consultation. ACL, 2025.
