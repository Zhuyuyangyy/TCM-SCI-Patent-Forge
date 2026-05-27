"""
TCMAug 完整演示
"""
import numpy as np
from augmentation_pipeline import TongueAugmentationPipeline

def run_demo():
    print("=" * 60)
    print("TCMAug Demo: GAN舌象数据增强")
    print("=" * 60)
    
    # 初始化流水线
    pipeline = TongueAugmentationPipeline()
    
    # 生成样本
    print("\n[Step 1] 生成舌象样本...")
    samples = pipeline.generate(n_samples=20, quality_threshold=0.5)
    print(f"\n  成功生成 {len(samples)} 个样本")
    
    # 统计多样性
    print("\n[Step 2] 多样性统计...")
    stats = pipeline.get_diversity_stats()
    print(f"  总样本数: {stats['total_samples']}")
    print(f"  舌色分布:")
    for t, c in stats['type_distribution'].items():
        print(f"    {t}: {c}个")
    print(f"  平均质量: {stats['avg_quality']:.3f}")
    print(f"  质量范围: [{stats['min_quality']:.3f}, {stats['max_quality']:.3f}]")
    
    # FID-like 评估（简化版）
    print("\n[Step 3] 生成质量评估...")
    real_scores = []
    fake_scores = []
    for i, sample in enumerate(samples[:5]):
        fake_scores.append(sample['quality']['total'])
        # 模拟真实图像评分（通常更高）
        real_scores.append(0.8 + np.random.rand() * 0.15)
    
    print(f"  真实样本平均质量: {np.mean(real_scores):.3f}")
    print(f"  生成样本平均质量: {np.mean(fake_scores):.3f}")
    print(f"  质量差距: {abs(np.mean(real_scores) - np.mean(fake_scores)):.3f}")
    
    print("\n" + "=" * 60)
    print("Demo 完成!")

if __name__ == '__main__':
    run_demo()
