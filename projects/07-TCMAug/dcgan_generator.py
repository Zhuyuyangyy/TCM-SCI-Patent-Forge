"""
DCGAN Generator for Tongue Image Augmentation
深度卷积GAN舌象数据增强
"""
import numpy as np

class DCGANGGenerator:
    """
    简化DCGAN生成器
    输入: 100维随机噪声
    输出: 128x128x3 舌象图像
    """
    def __init__(self, latent_dim=100, img_size=128, channels=3):
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.channels = channels
        
        # 简化的生成器权重（实际应从文件加载）
        self.weights = self._init_weights()
    
    def _init_weights(self):
        """初始化生成器权重"""
        return np.random.randn(1000) * 0.02
    
    def forward(self, z):
        """
        前向传播
        z: (batch, latent_dim) 噪声向量
        返回: (batch, 128, 128, 3) 图像
        """
        batch_size = z.shape[0] if len(z.shape) > 1 else 1
        z = np.array(z).flatten()
        
        # 模拟多层变换
        h = z.copy()
        for i in range(8):
            h = np.tanh(h @ (np.eye(len(h)) * 0.9) + np.random.randn(len(h)) * 0.1)
        
        # 目标像素数
        target_size = self.img_size * self.img_size * self.channels
        
        # 将潜在向量扩展到目标大小（通过重复或插值）
        if len(h) < target_size:
            # 重复扩展
            repeats = (target_size // len(h)) + 1
            h = np.tile(h, repeats)
        
        # 调整为图像形状
        img = h[:target_size].reshape(
            self.img_size, self.img_size, self.channels
        )
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        
        return np.expand_dims(img, 0) if batch_size == 1 else img
    
    def generate_batch(self, n, seed=None):
        """生成一批图像"""
        if seed is not None:
            np.random.seed(seed)
        z = np.random.randn(n, self.latent_dim)
        return self.forward(z)
    
    def generate_with_condition(self, condition, n=1):
        """
        条件生成：给定舌色/舌苔条件生成
        condition: dict, e.g. {'tongue_color': '淡白', 'coating': '薄白'}
        """
        # 简化的条件编码
        cond_code = hash(str(condition)) % 1000 / 1000.0
        base_noise = np.random.randn(n, self.latent_dim)
        # 注入条件信息到噪声
        cond_vector = np.full((n, self.latent_dim), cond_code * 0.5)
        z = base_noise + cond_vector * 0.3
        return self.forward(z)


class TongueStyleEncoder:
    """
    舌象风格编码器
    将舌象编码为风格向量用于插值
    """
    STYLE_DIM = 32
    
    def __init__(self):
        self.style_basis = {
            '淡白舌': [1, 0, 0, 0] * 8,
            '红舌': [0, 1, 0, 0] * 8,
            '绛舌': [0, 0, 1, 0] * 8,
            '紫舌': [0, 0, 0, 1] * 8,
        }
    
    def encode(self, tongue_type):
        """编码舌象类型为风格向量"""
        basis = self.style_basis.get(tongue_type, [0.25] * 4 * 8)
        noise = np.random.randn(self.STYLE_DIM) * 0.1
        return np.array(basis[:self.STYLE_DIM]) + noise
    
    def interpolate(self, style1, style2, alpha=0.5):
        """在两个风格向量间插值"""
        return alpha * np.array(style1) + (1 - alpha) * np.array(style2)
