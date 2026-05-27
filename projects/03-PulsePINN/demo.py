"""
PulsePINN 完整演示
使用NumPy替代torch，纯CPU运行
"""
import numpy as np
from pulse_pde import PulsePDE
from syndrome_classifier import PulseSyndromeClassifier

def run_demo():
    print("=" * 60)
    print("PulsePINN Demo: 物理信息神经网络脉象分析")
    print("=" * 60)
    
    # Step 1: PDE求解器生成合成脉波
    print("\n[Step 1] PDE物理脉波模拟...")
    pde = PulsePDE(c=10.0, delta=0.3, nx=100, nt=200)
    u = pde.solve(T=0.5)
    pulse_signal = u[:, 50]  # 取中间位置的脉波
    print(f"  脉波信号长度: {len(pulse_signal)}")
    print(f"  脉波峰值: {np.max(pulse_signal):.4f}")
    
    # Step 2: 证候分类
    print("\n[Step 2] 脉象证候分类...")
    classifier = PulseSyndromeClassifier()
    result = classifier.classify(pulse_signal)
    print(f"  脉象类型: {result['pulse_type']}")
    print(f"  可能的证候: {', '.join(result['syndromes'])}")
    print(f"  置信度: {result['confidence']:.2%}")
    
    # Step 3: PDE物理验证
    print("\n[Step 3] PDE物理约束演示...")
    print(f"  波速c={pde.c}, 阻尼delta={pde.delta}")
    print(f"  空间网格dx={pde.dx:.4f}")
    print(f"  CFL条件: dt={pde.dt:.6f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成!")

if __name__ == '__main__':
    run_demo()
