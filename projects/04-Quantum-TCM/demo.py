"""
Quantum-TCM 完整演示
"""
import numpy as np
from vqc_circuit import VQCCircuit
from syndrome_encoder import SyndromeEncoder
from five_elements_constraint import FiveElementsConstraint

def run_demo():
    print("=" * 60)
    print("Quantum-TCM Demo: 量子证候状态空间")
    print("=" * 60)
    
    # Step 1: 症状编码
    print("\n[Step 1] 症状→量子参数编码...")
    encoder = SyndromeEncoder()
    symptoms = ['胁肋胀痛', '情绪抑郁', '脉弦']
    thetas = encoder.encode_symptoms(symptoms)
    print(f"  输入症状: {symptoms}")
    print(f"  编码参数: {[f'{t:.3f}' for t in thetas]}")
    
    # Step 2: VQC前向传播
    print("\n[Step 2] VQC量子电路模拟...")
    circuit = VQCCircuit(num_qubits=4)
    probs = circuit.forward(thetas, num_layers=3)
    syndrome_probs = circuit.get_syndrome_probabilities(probs)
    print(f"  量子比特数: 4")
    print(f"  叠加态维度: {len(probs)}")
    
    # Step 3: 五行约束
    print("\n[Step 3] 五行相生相克约束...")
    constraint = FiveElementsConstraint()
    
    # 证候→五行映射: 肝郁→木, 心火→火, 脾虚→土, 肺气虚→金, 肾阴虚→水
    syndrome_to_wuxing = {'肝郁': '木', '心火': '火', '脾虚': '土', '肺气虚': '金', '肾阴虚': '水'}
    organ_probs = {}
    for syndrome, prob in syndrome_probs.items():
        if syndrome in syndrome_to_wuxing:
            organ_probs[syndrome_to_wuxing[syndrome]] = prob
    if not organ_probs:
        organ_probs = {'木': 0.3, '火': 0.2, '土': 0.2, '金': 0.15, '水': 0.15}
    constrained_thetas = constraint.apply_constraint(thetas, organ_probs)
    
    print(f"  肝属木，木→火（相生）")
    print(f"  肝克土（木克土）")
    
    # 验证转移
    ok1 = constraint.validate_transition('木', '火')
    ok2 = constraint.validate_transition('木', '土')
    print(f"  木→火 (相生): {'允许' if ok1 else '禁止'}")
    print(f"  木→土 (相克): {'允许' if ok2 else '禁止'}")
    
    # Step 4: 最终证候概率
    print("\n[Step 4] 证候概率分布...")
    probs_final = circuit.forward(constrained_thetas, num_layers=3)
    syndrome_final = circuit.get_syndrome_probabilities(probs_final)
    
    for syndrome, prob in sorted(syndrome_final.items(), key=lambda x: -x[1]):
        if prob > 0.01:
            print(f"  {syndrome}: {prob:.3f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成!")

if __name__ == '__main__':
    run_demo()
