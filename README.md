# TCM-SCI-Patent-Forge

> **中医AI科学研究工厂** — 将100个中医AI研究方向转化为SCI论文 + 专利技术交底书

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/your-repo/TCM-SCI-Patent-Forge)](https://github.com/your-repo/TCM-SCI-Patent-Forge/stargazers)

---

## 🎯 项目定位

| 维度 | 内容 |
|------|------|
| **目标** | 高质量SCI论文 × 50 + 发明专利 × 50 |
| **方向** | 具身智能 · 知识图谱 · 多模态感知 · 中药研发 · 基准测评 |
| **创新** | KG-Enhanced Planning · NeuroSymbolic Reasoning · Multi-Agent Collaboration |

---

## 📁 项目结构

```
TCM-SCI-Patent-Forge/
├── papers/                  # 30篇SCI论文草稿（A- AD）
├── patents/                 # 50个专利技术交底书（001-050）
├── projects/                # 15个旗舰项目实现
│   ├── 01-NeuroSymbolic-TCM/   # 神经符号混合推理 ✓论文 ✓专利
│   ├── 02-AudioSense-TCM/      # 闻诊语音情感VAE ✓论文 ✓专利
│   ├── 03-PulsePINN/           # 脉象物理神经混合建模 ✓论文 ✓专利
│   ├── 04-Quantum-TCM/         # 量子中医证候编码 ✓论文
│   ├── 05-TCMAGENT-plus/       # 多智能体协作 ✓论文 ✓专利
│   ├── 06-KGMAL-plus/          # 知识图谱终身学习 ✓论文 ✓专利
│   ├── 07-TCMAug/              # GAN数据增强 ✓论文
│   ├── 08-TCMShield/           # 安全防御系统 ✓论文
│   ├── 09-TCM-DiffRAG-plus/    # 扩散RAG ✓论文
│   ├── 10-TCM-Constitution-Risk/# 体质疾病风险 ✓论文 ✓专利
│   ├── TCM-Acupuncture-Robot/  # 针灸机器人
│   ├── TCM-Causal-Discovery/    # 因果发现
│   ├── TCM-DigitalTwin-Constitution/
│   ├── TCM-Herbal-FireControl/  # 火候判定
│   ├── TCM-PulseTriLevel/      # 脉象三层次
│   ├── TCM-Rehab-Baduanjin/     # 八段锦康复
│   └── TCM-TongueVoice-Fusion/  # 舌脉融合
├── directions_100.json      # 100方向数据库
├── generate_directions.py   # 方向生成脚本
├── MANIFEST.md              # 交付清单
├── OVERVIEW.md              # 项目概览
└── README.md                # 本文件
```

---

## 📊 核心统计

| 资产类型 | 数量 | 状态 |
|---------|------|------|
| 专利技术交底书 | 50 | ✅ 全部完成 |
| SCI论文草稿 | 30 | ✅ 全部完成 |
| 100方向数据库 | 100 | ✅ 已就绪 |
| 旗舰项目 | 15 | 🔄 持续开发 |
| 验证指标 | 50+ | ✅ 已定义 |

---

## 🚀 快速开始

### 环境要求
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 16GB RAM minimum

### 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/TCM-SCI-Patent-Forge.git
cd TCM-SCI-Patent-Forge

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# 安装所有依赖
pip install -r requirements.txt
```

### 运行旗舰项目示例

```bash
# 神经符号混合推理
cd projects/01-NeuroSymbolic-TCM
python demo.py

# 安全防御系统
cd projects/08-TCMShield
python demo.py

# 火候智能判定
cd projects/TCM-Herbal-FireControl
python demo.py
```

---

## 📂 专利交底书总览（50个）

| 批次 | 编号 | 主题 |
|------|------|------|
| 第一批 | 001-010 | NeRF舌象 · 脉象PINN · 五行GNN · 闻诊VAE · AR针灸 |
| 第二批 | 011-020 | 古方换算 · 强化学习针灸 · 数字伴侣 · 指纹DBN |
| 第三批 | 021-030 | 受试者匹配 · 触觉手套 · 长新冠 · 联邦学习 |
| 第四批 | 031-040 | 因果推断 · 五运六气 · 皮肤光泽 · 微表情 |
| 第五批 | 041-050 | 舌苔演化 · 膏方推荐 · 睡眠关联 · 音乐疗法 |

**查看详情**: [MANIFEST.md](MANIFEST.md)

---

## 📑 SCI论文总览（30篇）

| 编号 | 文件 | 对应旗舰项目 | 验证指标 |
|------|------|-------------|---------|
| A | TCM-RAG-Neural-Symbolic-SCI.md | NeuroSymbolic-TCM | Accuracy>89% |
| B | TCM-AudioSense.md | AudioSense-TCM | F1>0.80 |
| C | TCM-PulsePDE.md | PulsePINN | RMSE<0.05 |
| D | TCM-Quantum-Syndrome.md | Quantum-TCM | 准确率>92% |
| E | TCM-Embodied-Robot-Nursing.md | Embodied-TCM | — |
| F | TCM-TCMagent-MultiAgent.md | TCMAGENT++ | F1>0.85 |
| G | TCM-KGMAL-Lifelong.md | KGMAL++ | 遗忘抑制>90% |
| H | TCM-GAN-DataAug.md | TCMAug | FID<50 |
| I | TCM-DiffRAG-Personalized.md | TCM-DiffRAG++ | 准确率>85% |
| J | TCM-Constitution-Disease-Risk.md | TCM-Constitution-Risk | C-statistic>0.75 |
| ... | ... | ... | ... |

**查看详情**: [MANIFEST.md](MANIFEST.md)

---

## 🎓 100方向分类（directions_100.json）

| 分类 | 编号范围 | 数量 |
|------|----------|------|
| 多模态感知 | 01-15 | 15 |
| 知识图谱 | 16-35 | 20 |
| AI-Agent | 36-55 | 20 |
| 中药研发 | 56-75 | 20 |
| 基准测评 | 76-100 | 25 |

---

## 🔧 开发指南

### 项目架构

每个旗舰项目采用统一架构：

```
project-name/
├── README.md              # 项目文档
├── requirements.txt      # 依赖声明
├── setup.py              # 安装脚本（部分项目）
├── demo.py               # 演示脚本
├── main.py               # 主程序入口（部分项目）
├── *.py                  # 核心模块
└── data/                 # 数据目录（部分项目）
```

### 添加新项目

```bash
# 1. 在 projects/ 下创建目录
# 2. 编写核心代码文件
# 3. 创建 demo.py 演示系统
# 4. 更新 MANIFEST.md
# 5. 提交PR
```

### 代码规范

- 遵循 PEP 8
- 使用 type hints
- 编写 docstring
- 单元测试覆盖率 > 80%

---

## 📈 技术栈

| 领域 | 技术 |
|------|------|
| 深度学习 | PyTorch, TensorFlow, JAX |
| 图神经网络 | PyG, DGL, NetworkX |
| 大语言模型 | GPT-4, ChatGLM, LLaMA |
| 知识图谱 | Neo4j, RDFox, GraphDB |
| 量子计算 | Qiskit, Cirq, PennyLane |
| 数据处理 | Pandas, NumPy, Polars |
| 可视化 | Matplotlib, Plotly, D3.js |

---

## 📝 许可证

本项目基于 MIT 许可证开源。详见 [LICENSE](LICENSE) 文件。

---

## 📧 联系方式

- **项目负责人**: ZYY Project Team
- **邮箱**: 1578804454@qq.com
- **主页**: https://github.com/ZhuYuyang/TCM-SCI-Patent-Forge

---

## 🙏 致谢

本项目借鉴了以下开源项目：

- [PyTorch Geometric](https://github.com/pyg-team/pytorch_geometric)
- [HuatuoGPT](https://github.com/symca/HuatuoGPT)
- [TCM-Eval](https://github.com/tcm-eval/benchmark)