"""
Quantum-TCM VQC训练（模拟）
"""
import numpy as np
from vqc_circuit import VQCCircuit
from syndrome_encoder import SyndromeEncoder
from five_elements_constraint import FiveElementsConstraint

def generate_synthetic_data(n_samples=100):
    """生成合成证候数据"""
    encoder = SyndromeEncoder()
    syndromes = ['肝郁', '心火', '脾虚', '肺气虚', '肾阴虚']
    
    data = []
    for _ in range(n_samples):
        syndrome = np.random.choice(syndromes)
        symptoms = encoder.SYNDROME_BASIS[syndrome][:3]  # 取前3个症状
        thetas = encoder.encode_symptoms(symptoms)
        data.append((thetas, syndrome))
    
    return data

def train(num_epochs=100, num_layers=3):
    print("Quantum-TCM VQC训练 (模拟)")
    circuit = VQCCircuit(num_qubits=4)
    encoder = SyndromeEncoder()
    constraint = FiveElementsConstraint()
    
    data = generate_synthetic_data(n_samples=50)
    
    # 随机初始化参数
    thetas = np.random.randn(12) * 0.1
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for params, target_syndrome in data:
            # 使用encoder编码的theta作为初始参数
            input_thetas = params
            
            # 应用五行约束
            organ_probs = {k: 1.0/5 for k in ['肝', '心', '脾', '肺', '肾']}
            constrained_thetas = constraint.apply_constraint(input_thetas, organ_probs)
            
            # 前向传播
            probs = circuit.forward(constrained_thetas, num_layers)
            syndrome_probs = circuit.get_syndrome_probabilities(probs)
            
            # 简单损失
            target_prob = syndrome_probs.get(target_syndrome, 0.0)
            loss = (1.0 - target_prob) ** 2
            total_loss += loss
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Avg Loss = {total_loss/len(data):.4f}")
    
    print("训练完成!")
    return thetas

if __name__ == '__main__':
    train()
