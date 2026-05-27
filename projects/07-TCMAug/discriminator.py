"""
DCGAN Discriminator for Tongue Images
判别器：区分真实舌象和生成舌象
"""
import numpy as np

class TongueDiscriminator:
    """
    简化的判别器
    输入: 128x128x3 图像
    输出: 真实度分数 (0-1)
    """
    def __init__(self, img_size=128, channels=3):
        self.img_size = img_size
        self.channels = channels
        self.weights = np.random.randn(1000) * 0.01
    
    def forward(self, img):
        """
        判别图像真伪
        img: (128, 128, 3)
        返回: 真实度 (0-1)
        """
        # 展平 + 简化特征提取
        features = np.mean(img, axis=(0, 1))  # 按通道求均值
        features = np.concatenate([features, np.var(img, axis=(0, 1))])
        
        # 评分
        score = np.dot(features, self.weights[:len(features)])
        score = 1 / (1 + np.exp(-score))  # sigmoid
        
        return float(score)
    
    def evaluate_batch(self, images):
        """评估一批图像"""
        scores = [self.forward(img) for img in images]
        return np.array(scores)


class QualityAssessor:
    """
    舌象质量评估器
    评估生成图像的质量和多样性
    """
    def __init__(self):
        self.quality_criteria = {
            'color_consistency': 0.3,
            'tongue_shape': 0.2,
            'coating_visibility': 0.25,
            'no_artifacts': 0.25
        }
    
    def assess(self, img):
        """
        评估图像质量
        返回: 质量分数 + 详细评分
        """
        scores = {}
        
        # 颜色一致性：各通道方差不应太大
        channel_means = np.mean(img, axis=(0, 1))
        scores['color_consistency'] = 1.0 - np.std(channel_means)
        
        # 舌形：中心区域应有一定规律性
        center = img[32:96, 32:96, :]
        scores['tongue_shape'] = float(np.mean(center) > 0.2)
        
        # 舌苔可见度
        coat_region = img[64:, :, :]
        scores['coating_visibility'] = float(np.std(coat_region) > 0.05)
        
        # 无伪影：边缘不应有异常值
        edges = np.concatenate([
            img[0, :, :].flatten(),
            img[-1, :, :].flatten(),
            img[:, 0, :].flatten(),
            img[:, -1, :].flatten()
        ])
        scores['no_artifacts'] = 1.0 - np.mean(np.abs(edges - np.mean(edges)))
        
        # 加权总分
        total = sum(scores[k] * w for k, w in self.quality_criteria.items())
        
        return {
            'total': float(total),
            'details': {k: float(v) for k, v in scores.items()}
        }
