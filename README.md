<p align="center">
  <h1 align="center">TCM-SCI-Patent-Forge</h1>
  <p align="center"><em>Automated Research Forge for TCM AI: SCI Papers and Patent Technical Disclosures</em></p>
  <p align="center">
    100 Research Directions + SCI Paper Generation + Patent Drafting
  </p>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/status-active-brightgreen.svg" alt="Status">
</p>

---

## Overview

TCM-SCI-Patent-Forge is a research automation platform that systematically converts 100 identified TCM AI research directions into publishable SCI paper drafts and patent technical disclosure documents. The project addresses a critical bottleneck in TCM AI research: the gap between promising research ideas and their formalization into academic publications and intellectual property filings.

The platform is organized around five thematic research domains -- multimodal perception (15 directions), knowledge graphs (20 directions), AI agents (20 directions), herbal drug discovery (20 directions), and benchmarking and evaluation (25 directions) -- covering the full spectrum of current TCM AI research frontiers. Each direction has been analyzed for novelty, feasibility, and publication potential, with 15 flagship projects receiving full implementation including working code, demo scripts, and benchmark evaluations.

To date, the forge has produced 30 SCI paper drafts and 50 patent technical disclosure documents. The flagship projects span cutting-edge topics including neuro-symbolic hybrid reasoning, audio-based auscultation analysis with variational autoencoders, pulse diagnosis via physics-informed neural networks (PINNs), quantum syndrome encoding, multi-agent collaborative diagnosis, knowledge graph lifelong learning, GAN-based data augmentation, diffusion-based RAG, and embodied acupuncture robotics. Each flagship project follows a standardized structure with README, requirements, demo script, and core implementation modules.

> **Disclaimer**: The generated papers and patents are drafts intended as starting points for further development. They require domain expert review, validation, and refinement before submission to journals or patent offices.

> **Critical Research Limitations / 重要研究局限性:**
>
> - **100% Synthetic Data / 100%合成数据**: ALL experimental results in this project are based on synthetic/simulated data. No real patient clinical data has been used. Performance metrics do NOT reflect real-world clinical performance.
> - **No Real Baselines / 无真实基线对比**: No fair comparison experiments against published state-of-the-art methods have been conducted on shared datasets.
> - **No Ablation Studies / 无消融实验**: Formal ablation studies quantifying individual module contributions have not been completed.
> - **No IRB Approval / 无伦理审批**: No IRB (Institutional Review Board) ethics approval has been obtained as no real patient data was used. Future clinical studies will require proper ethics approval.
> - **Fabricated References Risk / 虚构引用风险**: Some references in the generated papers may be AI-fabricated. Always verify citations through PubMed, CNKI, or Google Scholar before use.
> - **TCMShield F1=1.0 Warning**: The perfect F1 score reported for TCMShield is likely due to data leakage between test case generation rules and the defense rule engine. The 94.3% red-team interception rate is a more realistic performance estimate.

---

## Key Features

- **100 Research Directions Database**: Curated and categorized database of TCM AI research directions across five domains with novelty assessment
- **Automated SCI Paper Generation**: LLM-powered generation of structured SCI paper drafts with abstract, introduction, methodology, expected results, and references
- **Automated Patent Drafting**: Generation of patent technical disclosure documents including claims, technical field, background, summary, and detailed description
- **15 Flagship Project Implementations**: Working code implementations covering neuro-symbolic reasoning, audio analysis, pulse modeling, quantum encoding, and more
- **Benchmark Evaluation Framework**: Standardized benchmark scripts and result files for flagship project validation
- **Batch Generation Pipeline**: Configurable batch processing for generating papers and patents across multiple directions
- **Quality Framework**: Structured quality assessment criteria for generated research outputs
- **Extensible Architecture**: Modular design allowing new research directions and generation templates to be added

---

## Architecture

```
+---------------------------------------------------------------------+
|                     Generation Pipeline                              |
|                                                                     |
|  directions_100.json --> paper_generator.py --> SCI Paper Drafts    |
|                     --> patent_generator.py --> Patent Disclosures   |
|                     --> generate_batch.py   --> Batch Processing     |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
|                     Flagship Projects (15)                          |
|                                                                     |
|  +-------------------+  +-------------------+  +------------------+ |
|  | 01-NeuroSymbolic  |  | 02-AudioSense    |  | 03-PulsePINN     | |
|  | TCM               |  | TCM              |  |                  | |
|  +-------------------+  +-------------------+  +------------------+ |
|  +-------------------+  +-------------------+  +------------------+ |
|  | 04-Quantum-TCM    |  | 05-TCMAGENT+     |  | 06-KGMAL+        | |
|  +-------------------+  +-------------------+  +------------------+ |
|  +-------------------+  +-------------------+  +------------------+ |
|  | 07-TCMAug         |  | 08-TCMShield     |  | 09-DiffRAG+      | |
|  +-------------------+  +-------------------+  +------------------+ |
|  +-------------------+  +-------------------+  +------------------+ |
|  | 10-Constitution   |  | Acupuncture      |  | Causal Discovery | |
|  | Risk              |  | Robot            |  |                  | |
|  +-------------------+  +-------------------+  +------------------+ |
|  +-------------------+  +-------------------+  +------------------+ |
|  | DigitalTwin       |  | Herbal FireCtrl  |  | PulseTriLevel    | |
|  +-------------------+  +-------------------+  +------------------+ |
|  +-------------------+  +-------------------+                      |
|  | Rehab Baduanjin   |  | TongueVoice      |                      |
|  +-------------------+  +-------------------+                      |
+---------------------------------------------------------------------+
                              |
                              v
+---------------------------------------------------------------------+
|                     Research Domains (5)                            |
|                                                                     |
|  Multimodal      Knowledge     AI          Herbal       Benchmark  |
|  Perception      Graphs        Agents      Discovery    & Eval     |
|  (15 dirs)       (20 dirs)     (20 dirs)   (20 dirs)    (25 dirs) |
+---------------------------------------------------------------------+
```

---

## Tech Stack

| Domain | Technology | Purpose |
|--------|-----------|---------|
| **LLM Backend** | DeepSeek API | Paper and patent content generation |
| **HTTP Client** | httpx | Async API calls to LLM services |
| **Data Processing** | Python stdlib, tqdm | Batch processing with progress tracking |
| **Graph Neural Networks** | PyTorch Geometric, DGL | Knowledge graph and GNN-based projects |
| **Physics Simulation** | PyTorch, custom PINNs | Pulse signal physics-informed modeling |
| **Quantum Computing** | Qiskit, Cirq, PennyLane | Quantum syndrome encoding |
| **Knowledge Graphs** | Neo4j, RDFox, NetworkX | Graph storage and querying |
| **Visualization** | Matplotlib, Plotly | Research result visualization |
| **Data Formats** | JSON, YAML, Markdown | Structured data and document interchange |

---

## Quick Start

### Prerequisites

- Python 3.8+
- DeepSeek API key (required for paper/patent generation)
- (Optional) CUDA 11.0+ for GPU-accelerated flagship projects

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/TCM-SCI-Patent-Forge.git
cd TCM-SCI-Patent-Forge

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Generate Papers and Patents

```bash
# Generate a single paper draft
python paper_generator.py --api_key YOUR_DEEPSEEK_KEY

# Generate a single patent disclosure
python patent_generator.py --api_key YOUR_DEEPSEEK_KEY

# Batch generation across multiple directions
python generate_batch.py --api_key YOUR_DEEPSEEK_KEY
```

### Run Flagship Project Demos

```bash
# Neuro-symbolic hybrid reasoning
cd projects/01-NeuroSymbolic-TCM
python demo.py

# Safety defense system
cd projects/08-TCMShield
python demo.py

# Herbal fire control intelligence
cd projects/TCM-Herbal-FireControl
python demo.py
```

---

## Project Structure

```
TCM-SCI-Patent-Forge/
├── papers/                          # 30 SCI paper drafts (A-AD)
├── patents/                         # 50 patent technical disclosures (001-050)
├── projects/                        # 15 flagship project implementations
│   ├── 01-NeuroSymbolic-TCM/        # Neuro-symbolic hybrid reasoning
│   ├── 02-AudioSense-TCM/           # Auscultation audio VAE
│   ├── 03-PulsePINN/                # Pulse physics-informed neural modeling
│   ├── 04-Quantum-TCM/              # Quantum syndrome encoding
│   ├── 05-TCMAGENT-plus/            # Multi-agent collaborative diagnosis
│   ├── 06-KGMAL-plus/               # Knowledge graph lifelong learning
│   ├── 07-TCMAug/                   # GAN data augmentation
│   ├── 08-TCMShield/                # Safety defense system
│   ├── 09-TCM-DiffRAG-plus/         # Diffusion-based RAG
│   ├── 10-TCM-Constitution-Risk/    # Constitution disease risk prediction
│   ├── TCM-Acupuncture-Robot/       # Embodied acupuncture robotics
│   ├── TCM-Causal-Discovery/        # Causal discovery
│   ├── TCM-DigitalTwin-Constitution/# Digital twin constitution modeling
│   ├── TCM-Herbal-FireControl/      # Herbal processing fire control
│   ├── TCM-PulseTriLevel/           # Tri-level pulse analysis
│   ├── TCM-Rehab-Baduanjin/         # Baduanjin rehabilitation guidance
│   └── TCM-TongueVoice-Fusion/      # Tongue-voice multimodal fusion
├── database/                        # Supporting database files
├── tests/                           # Test suite
├── directions_100.json              # 100 research directions database
├── generate_directions.py           # Direction generation script
├── paper_generator.py               # SCI paper generation engine
├── patent_generator.py              # Patent disclosure generation engine
├── generate_batch.py                # Batch generation pipeline
├── config.py                        # Configuration settings
├── benchmark_results.json           # Cross-project benchmark results
├── *_benchmark_results.json         # Per-project benchmark results
├── MANIFEST.md                      # Deliverables manifest
├── OVERVIEW.md                      # Project overview
├── QUALITY_FRAMEWORK.md             # Quality assessment framework
├── EXPERIMENT_REPORT.md             # Experiment results report
├── requirements.txt                 # Python dependencies
├── LICENSE                          # MIT License
└── README.md                        # This file
```

---

## Research Domains

| Domain | Direction Range | Count | Key Topics |
|--------|----------------|-------|------------|
| **Multimodal Perception** | 01-15 | 15 | Tongue NeRF, pulse PINN, audio VAE, facial analysis |
| **Knowledge Graphs** | 16-35 | 20 | Five-element GNN, syndrome KG, formula networks |
| **AI Agents** | 36-55 | 20 | Multi-agent diagnosis, reinforcement learning, digital twins |
| **Herbal Drug Discovery** | 56-75 | 20 | Molecular generation, fire control, pharmacokinetics |
| **Benchmarking & Evaluation** | 76-100 | 25 | Safety evaluation, federated learning, clinical trials |

---

## Flagship Project Highlights

| # | Project | Innovation | Paper | Patent |
|---|---------|-----------|-------|--------|
| 01 | NeuroSymbolic-TCM | Neural-symbolic hybrid reasoning with KG enhancement | Yes | Yes |
| 02 | AudioSense-TCM | Voice-based auscultation with VAE for TCM diagnosis | Yes | Yes |
| 03 | PulsePINN | Physics-informed neural network for pulse signal modeling | Yes | Yes |
| 04 | Quantum-TCM | Quantum computing for syndrome state encoding | Yes | -- |
| 05 | TCMAGENT-plus | Multi-agent collaborative clinical reasoning | Yes | Yes |
| 06 | KGMAL-plus | Knowledge graph multi-task lifelong learning | Yes | Yes |
| 07 | TCMAug | GAN-based TCM data augmentation | Yes | -- |
| 08 | TCMShield | Adversarial defense for TCM AI systems | Yes | -- |
| 09 | TCM-DiffRAG-plus | Diffusion model-based personalized RAG | Yes | -- |
| 10 | Constitution-Risk | Constitution-disease risk prediction with survival analysis | Yes | Yes |

---

## Benchmarks

| Project | Metric | Target | Status |
|---------|--------|--------|--------|
| NeuroSymbolic-TCM | Top-1 Syndrome Accuracy | > 89% | Pending formal evaluation |
| AudioSense-TCM | F1 Score | > 0.80 | Pending formal evaluation |
| PulsePINN | RMSE | < 0.05 | Pending formal evaluation |
| Quantum-TCM | Syndrome Accuracy | > 92% | Pending formal evaluation |
| TCMAGENT-plus | F1 Score | > 0.85 | Pending formal evaluation |
| KGMAL-plus | Forgetting Inhibition | > 90% | Pending formal evaluation |
| TCMAug | FID Score | < 50 | Pending formal evaluation |
| Constitution-Risk | C-statistic | > 0.75 | Pending formal evaluation |

*Benchmark results from automated evaluations are available in `*_benchmark_results.json` files. Formal clinical validation is pending.*

---

## Patent Disclosure Summary

| Batch | Range | Topics |
|-------|-------|--------|
| Batch 1 | 001-010 | NeRF tongue, pulse PINN, five-element GNN, auscultation VAR, AR acupuncture |
| Batch 2 | 011-020 | Classical formula conversion, RL acupuncture, digital companion, fingerprint DBN |
| Batch 3 | 021-030 | Subject matching, haptic gloves, long COVID, federated learning |
| Batch 4 | 031-040 | Causal inference, five movements six qi, skin luster, micro-expression |
| Batch 5 | 041-050 | Tongue coating evolution, paste formula recommendation, sleep correlation, music therapy |

See [MANIFEST.md](MANIFEST.md) for the complete deliverables list.

---

## Research and Publications

This project is part of the **TCM-AI** research ecosystem. The generated SCI paper drafts and patent disclosures cover the following innovation themes:

- **KG-Enhanced Planning**: Knowledge graph-guided clinical decision planning
- **NeuroSymbolic Reasoning**: Hybrid neural and symbolic approaches for TCM diagnosis
- **Multi-Agent Collaboration**: Distributed agent systems for collaborative clinical reasoning
- **Physics-Informed Modeling**: Domain physics integration into neural network architectures
- **Quantum-Classical Hybrid**: Quantum computing applications in syndrome state representation

**Citation:**
```bibtex
@software{tcm_sci_patent_forge,
  title   = {TCM-SCI-Patent-Forge: Automated Research Forge for TCM AI},
  author  = {TCM-SCI-Patent-Forge Contributors},
  year    = {2026},
  url     = {https://github.com/your-org/TCM-SCI-Patent-Forge},
  version = {1.0.0}
}
```

---

## Roadmap

- [x] 100 research directions database
- [x] 30 SCI paper drafts
- [x] 50 patent technical disclosures
- [x] 15 flagship project implementations
- [x] Automated batch generation pipeline
- [ ] Formal benchmark evaluation with clinical datasets
- [ ] Peer review integration for paper quality assessment
- [ ] Patent office format compliance checking
- [ ] Multi-language generation support (Chinese/English)
- [ ] Continuous integration for automated research output updates

---

## License

This project is released under the [MIT License](LICENSE).

---

## Contact

- **Project Lead**: ZYY Project Team
- **Issues**: Please use [GitHub Issues](https://github.com/your-org/TCM-SCI-Patent-Forge/issues) for bug reports and feature requests

---

## Acknowledgments

This project draws inspiration from the open-source TCM AI research community, including projects such as HuatuoGPT, TCM-Eval, and PyTorch Geometric. We acknowledge the classical TCM literature that forms the knowledge foundation for the research directions catalogued in this forge.

---

## TCM Standardization and Clinical Trial References

The following real, verifiable references provide foundational context for TCM AI research. These can be cited as background in papers developed from this project.

### TCM Standards and Pharmacopoeia
- Chinese Pharmacopoeia Commission. *Pharmacopoeia of the People's Republic of China* (2020 Edition). China Medical Science Press, 2020.
- WHO. *WHO International Standard Terminologies on Traditional Medicine in the Western Pacific Region*. WHO, 2007.
- State Administration of TCM. *Guidelines for Diagnosis and Treatment of Common Diseases in TCM*. China Press of Traditional Chinese Medicine.

### TCM AI Reviews and Benchmarks
- Lukman, S., He, Y., & Heng, S. "Computational methods for traditional Chinese medicine: A survey." *Computer Methods and Programs in Biomedicine*, 2007, 88(3): 283-294.
- Zhang, M.M., Zhang, H., & Wang, Y.G. "Traditional Chinese Medicine Zheng in the era of evidence-based medicine." *Journal of Ethnopharmacology*, 2012, 140(3): 595-601.
- Jiang, M., Zhang, C., Cao, H., et al. "The role of Chinese medicine in the treatment of chronic diseases in China." *Planta Medica*, 2011, 77(9): 873-881.

### TCM Clinical Trial Methodology
- Vickers, A.J., Cronin, A.M., Maschino, A.C., et al. "Acupuncture for chronic pain: individual patient data meta-analysis." *Archives of Internal Medicine*, 2012, 172(19): 1444-1453.
- Witt, C.M., Pach, D., Brinkhaus, B., et al. "Safety of acupuncture: results of a prospective observational study with 229,230 patients." *Forschende Komplementarmedizin*, 2009, 16(2): 91-97.
- Manheimer, E., Wieland, S., Kimbrough, E., et al. "Evidence from the Cochrane Collaboration for traditional Chinese medicine therapies." *Journal of Alternative and Complementary Medicine*, 2009, 15(9): 1001-1014.
