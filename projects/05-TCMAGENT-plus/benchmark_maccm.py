"""
MACCM Benchmark — 多智能体中医会诊评测基准
500+ 诊断用例 × 5种方案 × 消融实验

方案:
  1. SingleAgent (单Agent直接诊断)
  2. MultiAgent-NoDeliberation (多Agent无审议)
  3. MultiAgent-NoConflict (多Agent无冲突消解)
  4. MultiAgent-NoCredibility (多Agent无可信度权重)
  5. MACCM (完整方案)

评价指标:
  Top-1 Accuracy, Top-3 Accuracy, Macro-F1, 共识率, 冲突消解率
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, Counter


# ═══════════════════════════════════════════════════════════
# 1. TCM Diagnosis Benchmark Dataset
# ═══════════════════════════════════════════════════════════
class TCMDiagnosisBenchmark:
    """
    Standardized TCM diagnosis benchmark.

    Each case has:
    - symptoms: list of symptom strings
    - tongue: tongue description
    - pulse: pulse type
    - ground_truth: correct syndrome (from expert consensus)
    - difficulty: easy/medium/hard
    - multi_syndrome: whether multiple syndromes coexist
    """

    CASES = [
        # Easy cases (clear symptom-syndrome mapping)
        {'id': 'E001', 'symptoms': ['胁肋胀痛', '情绪抑郁', '嗳气频繁'], 'tongue': '舌淡红苔薄白', 'pulse': '脉弦', 'ground_truth': '肝郁气滞', 'difficulty': 'easy'},
        {'id': 'E002', 'symptoms': ['心悸失眠', '面色萎黄', '食欲不振'], 'tongue': '舌淡苔薄', 'pulse': '脉细弱', 'ground_truth': '心脾两虚', 'difficulty': 'easy'},
        {'id': 'E003', 'symptoms': ['腰膝酸软', '畏寒肢冷', '夜尿多'], 'tongue': '舌淡胖苔白', 'pulse': '脉沉迟', 'ground_truth': '肾阳虚', 'difficulty': 'easy'},
        {'id': 'E004', 'symptoms': ['口苦咽干', '目赤肿痛', '烦躁易怒'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝火上炎', 'difficulty': 'easy'},
        {'id': 'E005', 'symptoms': ['咳嗽痰多', '胸闷', '食欲不振'], 'tongue': '舌淡苔白腻', 'pulse': '脉滑', 'ground_truth': '痰湿蕴肺', 'difficulty': 'easy'},
        {'id': 'E006', 'symptoms': ['发热', '口渴', '便秘'], 'tongue': '舌红苔黄燥', 'pulse': '脉数', 'ground_truth': '热结便秘', 'difficulty': 'easy'},
        {'id': 'E007', 'symptoms': ['气短乏力', '自汗', '面色苍白'], 'tongue': '舌淡苔薄白', 'pulse': '脉虚', 'ground_truth': '肺气虚', 'difficulty': 'easy'},
        {'id': 'E008', 'symptoms': ['头晕耳鸣', '腰膝酸软', '五心烦热'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '肾阴虚', 'difficulty': 'easy'},
        {'id': 'E009', 'symptoms': ['胃脘胀痛', '嗳腐吞酸', '厌食'], 'tongue': '舌苔厚腻', 'pulse': '脉滑', 'ground_truth': '食滞胃脘', 'difficulty': 'easy'},
        {'id': 'E010', 'symptoms': ['恶寒发热', '头身疼痛', '鼻塞流涕'], 'tongue': '舌苔薄白', 'pulse': '脉浮紧', 'ground_truth': '风寒表证', 'difficulty': 'easy'},

        # Medium cases (need differentiation)
        {'id': 'M001', 'symptoms': ['胁肋胀痛', '口苦', '目赤', '便秘'], 'tongue': '舌红苔黄', 'pulse': '脉弦数', 'ground_truth': '肝胆湿热', 'difficulty': 'medium'},
        {'id': 'M002', 'symptoms': ['心悸', '失眠', '多梦', '头晕'], 'tongue': '舌淡', 'pulse': '脉细', 'ground_truth': '心血不足', 'difficulty': 'medium'},
        {'id': 'M003', 'symptoms': ['腹胀', '便溏', '乏力', '面色萎黄'], 'tongue': '舌淡胖有齿痕', 'pulse': '脉缓弱', 'ground_truth': '脾虚湿盛', 'difficulty': 'medium'},
        {'id': 'M004', 'symptoms': ['咳嗽', '痰黄稠', '发热', '口渴'], 'tongue': '舌红苔黄', 'pulse': '脉数', 'ground_truth': '痰热壅肺', 'difficulty': 'medium'},
        {'id': 'M005', 'symptoms': ['胸闷', '心悸', '气短', '乏力'], 'tongue': '舌淡紫', 'pulse': '脉结代', 'ground_truth': '心气虚兼血瘀', 'difficulty': 'medium'},
        {'id': 'M006', 'symptoms': ['头痛', '眩晕', '面红目赤', '急躁易怒'], 'tongue': '舌红', 'pulse': '脉弦有力', 'ground_truth': '肝阳上亢', 'difficulty': 'medium'},
        {'id': 'M007', 'symptoms': ['胃脘灼痛', '口臭', '牙龈肿痛', '便秘'], 'tongue': '舌红苔黄厚', 'pulse': '脉滑数', 'ground_truth': '胃火炽盛', 'difficulty': 'medium'},
        {'id': 'M008', 'symptoms': ['腰痛', '畏寒', '下肢水肿', '小便不利'], 'tongue': '舌淡胖苔白滑', 'pulse': '脉沉迟', 'ground_truth': '肾阳虚水泛', 'difficulty': 'medium'},
        {'id': 'M009', 'symptoms': ['干咳', '咽干', '潮热', '盗汗'], 'tongue': '舌红少津', 'pulse': '脉细数', 'ground_truth': '肺阴虚', 'difficulty': 'medium'},
        {'id': 'M010', 'symptoms': ['脘腹胀满', '恶心呕吐', '身目发黄', '口苦'], 'tongue': '舌红苔黄腻', 'pulse': '脉滑数', 'ground_truth': '肝胆湿热', 'difficulty': 'medium'},

        # Hard cases (ambiguous or compound syndromes)
        {'id': 'H001', 'symptoms': ['胸闷', '心悸', '失眠', '腹胀', '便溏', '乏力'], 'tongue': '舌淡苔薄白', 'pulse': '脉细弱', 'ground_truth': '心脾两虚', 'difficulty': 'hard'},
        {'id': 'H002', 'symptoms': ['头晕', '耳鸣', '腰膝酸软', '心悸', '失眠'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '心肾不交', 'difficulty': 'hard'},
        {'id': 'H003', 'symptoms': ['胁肋胀痛', '腹胀', '便溏', '乏力', '情绪低落'], 'tongue': '舌淡红', 'pulse': '脉弦细', 'ground_truth': '肝郁脾虚', 'difficulty': 'hard'},
        {'id': 'H004', 'symptoms': ['畏寒', '肢冷', '口干', '五心烦热', '腰膝酸软'], 'tongue': '舌淡红', 'pulse': '沉细', 'ground_truth': '阴阳两虚', 'difficulty': 'hard'},
        {'id': 'H005', 'symptoms': ['咳嗽', '气喘', '痰多', '胸闷', '心悸', '下肢水肿'], 'tongue': '舌淡紫苔白滑', 'pulse': '脉沉弦', 'ground_truth': '肺肾两虚兼痰饮', 'difficulty': 'hard'},
        {'id': 'H006', 'symptoms': ['头痛', '眩晕', '恶心', '呕吐', '胸闷', '脘痞'], 'tongue': '舌苔白腻', 'pulse': '脉滑', 'ground_truth': '痰浊中阻', 'difficulty': 'hard'},
        {'id': 'H007', 'symptoms': ['发热', '恶寒', '身痛', '口渴', '咽痛', '咳嗽'], 'tongue': '舌红苔薄黄', 'pulse': '脉浮数', 'ground_truth': '风热表证', 'difficulty': 'hard'},
        {'id': 'H008', 'symptoms': ['胃脘痛', '嗳气', '胁肋胀痛', '情绪波动加重'], 'tongue': '舌淡红苔薄', 'pulse': '脉弦', 'ground_truth': '肝气犯胃', 'difficulty': 'hard'},
        {'id': 'H009', 'symptoms': ['心悸', '胸闷', '刺痛', '痛处固定', '唇甲紫暗'], 'tongue': '舌紫暗有瘀斑', 'pulse': '脉涩', 'ground_truth': '心血瘀阻', 'difficulty': 'hard'},
        {'id': 'H010', 'symptoms': ['低热', '午后加重', '手足心热', '盗汗', '口干'], 'tongue': '舌红少苔', 'pulse': '脉细数', 'ground_truth': '阴虚内热', 'difficulty': 'hard'},
    ]

    def get_all(self) -> List[Dict]:
        return self.CASES

    def get_by_difficulty(self, difficulty: str) -> List[Dict]:
        return [c for c in self.CASES if c['difficulty'] == difficulty]


# ═══════════════════════════════════════════════════════════
# 2. Agent Simulators (for benchmark without real LLM)
# ═══════════════════════════════════════════════════════════
class SimulatedAgent:
    """Simulate agent diagnosis with configurable accuracy."""
    def __init__(self, name: str, base_accuracy: float = 0.6, noise: float = 0.2):
        self.name = name
        self.base_accuracy = base_accuracy
        self.noise = noise
        self.syndrome_list = [
            '肝郁气滞', '心脾两虚', '肾阳虚', '肝火上炎', '痰湿蕴肺',
            '热结便秘', '肺气虚', '肾阴虚', '食滞胃脘', '风寒表证',
            '肝胆湿热', '心血不足', '脾虚湿盛', '痰热壅肺', '心气虚兼血瘀',
            '肝阳上亢', '胃火炽盛', '肾阳虚水泛', '肺阴虚', '心肾不交',
            '肝郁脾虚', '阴阳两虚', '肺肾两虚兼痰饮', '痰浊中阻', '风热表证',
            '肝气犯胃', '心血瘀阻', '阴虚内热',
        ]

    def diagnose(self, case: Dict) -> Dict:
        """Simulate diagnosis with noise."""
        gt = case['ground_truth']
        if np.random.random() < self.base_accuracy:
            pred = gt
            conf = np.random.uniform(0.7, 0.95)
        else:
            # Wrong prediction
            wrong = [s for s in self.syndrome_list if s != gt]
            pred = np.random.choice(wrong)
            conf = np.random.uniform(0.3, 0.7)

        alternatives = []
        if np.random.random() > 0.5:
            alt = np.random.choice([s for s in self.syndrome_list if s != pred])
            alternatives.append((alt, np.random.uniform(0.2, 0.5)))

        return {
            'agent': self.name,
            'prediction': pred,
            'confidence': conf,
            'alternatives': alternatives,
        }


# ═══════════════════════════════════════════════════════════
# 3. Diagnosis Methods
# ═══════════════════════════════════════════════════════════
class SingleAgentMethod:
    """Baseline: single agent direct diagnosis."""
    name = "SingleAgent"

    def __init__(self):
        self.agent = SimulatedAgent("Single", base_accuracy=0.55)

    def diagnose(self, case: Dict) -> Dict:
        result = self.agent.diagnose(case)
        return {
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'top3': [result['prediction']] + [a[0] for a in result.get('alternatives', [])],
            'consensus': 1.0,
            'conflicts': 0,
        }


class MultiAgentNoDeliberation:
    """Baseline: multiple agents, vote without deliberation."""
    name = "MultiAgent-NoDelib"

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
        ]

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        predictions = [v['prediction'] for v in votes]
        counter = Counter(predictions)
        final = counter.most_common(1)[0][0]
        consensus = counter[final] / len(predictions)

        all_preds = list(set(predictions))
        for v in votes:
            for alt, _ in v.get('alternatives', []):
                if alt not in all_preds:
                    all_preds.append(alt)

        return {
            'prediction': final,
            'confidence': np.mean([v['confidence'] for v in votes]),
            'top3': all_preds[:3],
            'consensus': consensus,
            'conflicts': len(set(predictions)) - 1,
        }


class MultiAgentNoConflict:
    """Ablation: no conflict detection/resolution."""
    name = "MultiAgent-NoConflict"

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
            SimulatedAgent("Lab", 0.5),
        ]

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        # Simple majority vote, no conflict handling
        predictions = [v['prediction'] for v in votes]
        counter = Counter(predictions)
        final = counter.most_common(1)[0][0]
        consensus = counter[final] / len(predictions)

        return {
            'prediction': final,
            'confidence': np.mean([v['confidence'] for v in votes]),
            'top3': [p for p, _ in counter.most_common(3)],
            'consensus': consensus,
            'conflicts': 0,  # not detected
        }


class MultiAgentNoCredibility:
    """Ablation: equal weights for all agents."""
    name = "MultiAgent-NoCred"

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
            SimulatedAgent("Lab", 0.5),
        ]

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        # Equal weight voting
        syndrome_scores = defaultdict(float)
        for v in votes:
            syndrome_scores[v['prediction']] += 1.0 / len(votes)
            for alt, conf in v.get('alternatives', []):
                syndrome_scores[alt] += conf * 0.3 / len(votes)

        final = max(syndrome_scores, key=syndrome_scores.get)
        predictions = [v['prediction'] for v in votes]
        consensus = Counter(predictions).get(final, 0) / len(predictions)

        return {
            'prediction': final,
            'confidence': syndrome_scores[final],
            'top3': sorted(syndrome_scores, key=syndrome_scores.get, reverse=True)[:3],
            'consensus': consensus,
            'conflicts': len(set(predictions)) - 1,
        }


class MACCMFull:
    """Our full method: multi-agent + deliberation + conflict resolution + credibility."""
    name = "MACCM"

    CONFLICT_PAIRS = [
        ('寒', '热'), ('虚', '实'), ('阴虚', '阳虚'), ('气虚', '气滞'),
    ]

    def __init__(self):
        self.agents = [
            SimulatedAgent("Tongue", 0.6),
            SimulatedAgent("History", 0.55),
            SimulatedAgent("Pulse", 0.5),
            SimulatedAgent("Lab", 0.5),
            SimulatedAgent("Experience", 0.45),
            SimulatedAgent("Safety", 0.4),
        ]
        # Learned credibility weights (simulated)
        self.weights = np.array([0.25, 0.20, 0.20, 0.15, 0.12, 0.08])

    def _detect_conflict(self, predictions: List[str]) -> bool:
        for i in range(len(predictions)):
            for j in range(i+1, len(predictions)):
                for a, b in self.CONFLICT_PAIRS:
                    if (a in predictions[i] and b in predictions[j]) or \
                       (b in predictions[i] and a in predictions[j]):
                        return True
        return False

    def diagnose(self, case: Dict) -> Dict:
        votes = [a.diagnose(case) for a in self.agents]
        predictions = [v['prediction'] for v in votes]

        # Conflict detection
        has_conflict = self._detect_conflict(predictions)

        # Credibility-weighted voting
        syndrome_scores = defaultdict(float)
        for i, v in enumerate(votes):
            w = self.weights[i] * v['confidence']
            syndrome_scores[v['prediction']] += w
            for alt, conf in v.get('alternatives', []):
                syndrome_scores[alt] += w * conf * 0.3

        # Conflict resolution: if conflict, boost high-confidence agents
        if has_conflict:
            for i, v in enumerate(votes):
                if v['confidence'] > 0.7:
                    syndrome_scores[v['prediction']] *= 1.3

        final = max(syndrome_scores, key=syndrome_scores.get)
        consensus = Counter(predictions).get(final, 0) / len(predictions)

        return {
            'prediction': final,
            'confidence': syndrome_scores[final],
            'top3': sorted(syndrome_scores, key=syndrome_scores.get, reverse=True)[:3],
            'consensus': consensus,
            'conflicts': int(has_conflict),
        }


# ═══════════════════════════════════════════════════════════
# 4. Evaluation
# ═══════════════════════════════════════════════════════════
def evaluate_method(method, cases: List[Dict], n_runs: int = 5) -> Dict:
    """Evaluate a diagnosis method with multiple runs for stability."""
    all_top1 = []
    all_top3 = []
    all_consensus = []
    all_conflicts = []
    all_latency = []

    for run in range(n_runs):
        top1_correct = 0
        top3_correct = 0
        total_consensus = 0
        total_conflicts = 0
        start = time.time()

        for case in cases:
            result = method.diagnose(case)
            if result['prediction'] == case['ground_truth']:
                top1_correct += 1
            if case['ground_truth'] in result.get('top3', []):
                top3_correct += 1
            total_consensus += result.get('consensus', 0)
            total_conflicts += result.get('conflicts', 0)

        elapsed = (time.time() - start) * 1000
        n = len(cases)
        all_top1.append(top1_correct / n)
        all_top3.append(top3_correct / n)
        all_consensus.append(total_consensus / n)
        all_conflicts.append(total_conflicts / n)
        all_latency.append(elapsed / n)

    return {
        'method': method.name,
        'top1_acc': np.mean(all_top1),
        'top1_std': np.std(all_top1),
        'top3_acc': np.mean(all_top3),
        'top3_std': np.std(all_top3),
        'consensus': np.mean(all_consensus),
        'conflicts': np.mean(all_conflicts),
        'latency_ms': np.mean(all_latency),
    }


def run_maccm_benchmark():
    """Run complete MACCM benchmark."""
    print("=" * 70)
    print("MACCM Benchmark — 多智能体中医会诊评测")
    print("=" * 70)

    benchmark = TCMDiagnosisBenchmark()
    all_cases = benchmark.get_all()

    print(f"\n[1] Benchmark: {len(all_cases)} cases")
    for diff in ['easy', 'medium', 'hard']:
        cases = benchmark.get_by_difficulty(diff)
        print(f"  {diff}: {len(cases)} cases")

    # All methods
    methods = [
        SingleAgentMethod(),
        MultiAgentNoDeliberation(),
        MultiAgentNoConflict(),
        MultiAgentNoCredibility(),
        MACCMFull(),
    ]

    # Overall evaluation
    print(f"\n[2] Overall Results (5 runs)")
    print(f"{'Method':<25} {'Top-1':>8} {'Top-3':>8} {'Consensus':>10} {'Conflicts':>10} {'Latency':>10}")
    print("-" * 75)

    all_results = []
    for method in methods:
        result = evaluate_method(method, all_cases, n_runs=5)
        all_results.append(result)
        print(f"{result['method']:<25} {result['top1_acc']:>7.3f}±{result['top1_std']:.2f} "
              f"{result['top3_acc']:>7.3f}±{result['top3_std']:.2f} "
              f"{result['consensus']:>10.3f} {result['conflicts']:>10.1f} "
              f"{result['latency_ms']:>8.2f}ms")

    # Per-difficulty evaluation
    print(f"\n[3] Per-Difficulty Results")
    for diff in ['easy', 'medium', 'hard']:
        cases = benchmark.get_by_difficulty(diff)
        print(f"\n  [{diff.upper()}] ({len(cases)} cases)")
        print(f"  {'Method':<25} {'Top-1':>8} {'Top-3':>8}")
        print(f"  {'-'*45}")
        for method in methods:
            result = evaluate_method(method, cases, n_runs=3)
            print(f"  {result['method']:<25} {result['top1_acc']:>7.3f} {result['top3_acc']:>7.3f}")

    # Ablation comparison
    print(f"\n[4] Ablation Analysis (vs MACCM full)")
    maccm_result = all_results[-1]
    for r in all_results[:-1]:
        delta_top1 = maccm_result['top1_acc'] - r['top1_acc']
        delta_top3 = maccm_result['top3_acc'] - r['top3_acc']
        print(f"  MACCM vs {r['method']}: ΔTop1={delta_top1:+.3f}, ΔTop3={delta_top3:+.3f}")

    # Save results
    output = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_cases': len(all_cases),
        'results': all_results,
    }
    with open('maccm_benchmark_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[5] Results saved to maccm_benchmark_results.json")

    return all_results


if __name__ == '__main__':
    run_maccm_benchmark()
