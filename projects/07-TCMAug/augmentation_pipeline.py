"""
Tongue Image Augmentation Pipeline
舌象数据增强完整流水线
"""
import numpy as np
from dcgan_generator import DCGANGGenerator, TongueStyleEncoder
from discriminator import TongueDiscriminator, QualityAssessor

class TongueAugmentationPipeline:
    """
    舌象数据增强流水线
    1. DCGAN生成新样本
    2. 质量评估筛选
    3. 多样性保证
    """
    def __init__(self):
        self.generator = DCGANGGenerator()
        self.discriminator = TongueDiscriminator()
        self.quality_assessor = QualityAssessor()
        self.style_encoder = TongueStyleEncoder()
        self.generated_samples = []
    
    def generate(self, n_samples=100, quality_threshold=0.6):
        """
        生成高质量舌象样本
        n_samples: 目标生成数量
        quality_threshold: 质量阈值
        """
        print(f"开始生成 {n_samples} 个舌象样本...")
        
        all_generated = []
        target_types = ['淡白舌', '红舌', '绛舌', '紫舌']
        
        while len(all_generated) < n_samples:
            for tongue_type in target_types:
                if len(all_generated) >= n_samples:
                    break
                
                # 条件生成
                style_vec = self.style_encoder.encode(tongue_type)
                img = self.generator.generate_with_condition(
                    {'tongue_type': tongue_type}, n=1
                )[0]
                
                # 质量评估
                quality = self.quality_assessor.assess(img)
                
                # 判别器评估
                authenticity = self.discriminator.forward(img)
                
                if quality['total'] >= quality_threshold and authenticity > 0.3:
                    all_generated.append({
                        'image': img,
                        'tongue_type': tongue_type,
                        'quality': quality,
                        'authenticity': authenticity
                    })
                    print(f"  生成: {tongue_type}, 质量:{quality['total']:.3f}, 真实度:{authenticity:.3f}")
        
        self.generated_samples = all_generated[:n_samples]
        return self.generated_samples
    
    def get_diversity_stats(self):
        """统计生成样本的多样性"""
        if not self.generated_samples:
            return {}
        
        type_counts = {}
        for sample in self.generated_samples:
            t = sample['tongue_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        
        quality_scores = [s['quality']['total'] for s in self.generated_samples]
        
        return {
            'total_samples': len(self.generated_samples),
            'type_distribution': type_counts,
            'avg_quality': np.mean(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores)
        }
