"""
舌象特征提取模块 - Tongue Image Feature Extraction
提取舌色RGB/苔色/裂纹/齿痕/舌形特征，输出8维特征向量
纯NumPy实现，简化CNN特征提取
"""

import numpy as np


class TongueAnalyzer:
    """舌象分析器 - 从舌象图像/模拟数据中提取8维特征"""
    
    # 体质相关的舌象特征先验知识（用于模拟）
    CONSTITUTION_TONGUE_FEATURES = {
        '平和': {
            'tongue_color_rgb': [0.85, 0.55, 0.45],  # 淡红
            'fur_color_rgb': [0.92, 0.90, 0.85],     # 薄白苔
            'crack': 0.1,    # 无裂纹
            'teeth_mark': 0.0,  # 无齿痕
            'shape': 0.5,   # 正常
            'moisture': 0.8
        },
        '气虚': {
            'tongue_color_rgb': [0.75, 0.50, 0.42],  # 淡胖舌
            'fur_color_rgb': [0.88, 0.85, 0.80],     # 薄白苔
            'crack': 0.2,
            'teeth_mark': 0.7,  # 齿痕明显
            'shape': 0.3,   # 胖大
            'moisture': 0.6
        },
        '阳虚': {
            'tongue_color_rgb': [0.65, 0.45, 0.38],  # 淡胖舌色暗
            'fur_color_rgb': [0.90, 0.88, 0.82],     # 白润苔
            'crack': 0.15,
            'teeth_mark': 0.8,
            'shape': 0.25,  # 胖大
            'moisture': 0.5
        },
        '阴虚': {
            'tongue_color_rgb': [0.90, 0.50, 0.40],  # 红绛舌
            'fur_color_rgb': [0.70, 0.65, 0.60],     # 少苔/无苔
            'crack': 0.6,
            'teeth_mark': 0.1,
            'shape': 0.6,   # 瘦小
            'moisture': 0.3
        },
        '痰湿': {
            'tongue_color_rgb': [0.72, 0.52, 0.45],  # 淡白腻苔
            'fur_color_rgb': [0.85, 0.80, 0.70],     # 白腻苔
            'crack': 0.1,
            'teeth_mark': 0.5,
            'shape': 0.35,  # 胖大
            'moisture': 0.7
        },
        '湿热': {
            'tongue_color_rgb': [0.82, 0.55, 0.45],  # 红舌
            'fur_color_rgb': [0.75, 0.78, 0.65],     # 黄腻苔
            'crack': 0.4,
            'teeth_mark': 0.2,
            'shape': 0.55,
            'moisture': 0.6
        },
        '血瘀': {
            'tongue_color_rgb': [0.70, 0.40, 0.38],  # 紫暗舌
            'fur_color_rgb': [0.80, 0.78, 0.75],     # 薄白苔
            'crack': 0.7,   # 裂纹明显
            'teeth_mark': 0.2,
            'shape': 0.5,
            'moisture': 0.5
        },
        '气郁': {
            'tongue_color_rgb': [0.78, 0.48, 0.42],  # 淡红偏暗
            'fur_color_rgb': [0.85, 0.82, 0.78],     # 白苔
            'crack': 0.3,
            'teeth_mark': 0.3,
            'shape': 0.55,
            'moisture': 0.65
        },
        '特禀': {
            'tongue_color_rgb': [0.80, 0.52, 0.45],  # 淡红
            'fur_color_rgb': [0.78, 0.75, 0.70],     # 可能花剥苔
            'crack': 0.35,
            'teeth_mark': 0.25,
            'shape': 0.5,
            'moisture': 0.6
        }
    }
    
    def __init__(self):
        self.feature_dim = 8
        
    def _rgb_to_feature(self, rgb):
        """将RGB颜色转换为特征分量"""
        r, g, b = rgb
        # 颜色特征：R-G差值、R-B差值、G-B差值、饱和度
        rg_diff = r - g
        rb_diff = r - b
        gb_diff = g - b
        saturation = (max(rgb) - min(rgb)) / (max(rgb) + 1e-8)
        return [rg_diff, rb_diff, gb_diff, saturation]
    
    def extract_from_image(self, image_data):
        """
        从舌象图像数据中提取特征
        image_data: 模拟的舌象图像数据（numpy数组或模拟特征）
        返回: 8维特征向量
        """
        # 简化处理：image_data为包含舌象特征信息的字典或直接为特征向量
        if isinstance(image_data, dict):
            return self.extract_from_dict(image_data)
        elif isinstance(image_data, np.ndarray):
            # 假设是简化的特征向量或模拟图像
            if image_data.shape[-1] == 8:
                return image_data
            else:
                # 从模拟图像中提取特征
                return self._simple_cnn_features(image_data)
        else:
            raise ValueError("Unsupported image data format")
    
    def _simple_cnn_features(self, image):
        """
        简化的CNN特征提取（纯NumPy实现）
        模拟卷积层+池化层的特征提取过程
        """
        # 简化为直接从图像块提取统计特征
        if len(image.shape) == 3:
            # 计算各通道的均值和标准差作为简化特征
            features = []
            for c in range(min(3, image.shape[2])):
                features.extend([image[:,:,c].mean(), image[:,:,c].std()])
            # 添加空间统计特征
            features.append(image.mean())
            features.append(image.std())
        else:
            features = [image.mean(), image.std(), 0, 0, 0, 0]
        
        return np.array(features[:8])
    
    def extract_from_dict(self, tongue_info):
        """
        从舌象信息字典中提取8维特征
        tongue_info: 包含tongue_color_rgb, fur_color_rgb, crack, teeth_mark, shape等
        """
        features = []
        
        # 1-4. 舌色特征 (4维)
        tongue_rgb = tongue_info.get('tongue_color_rgb', [0.8, 0.5, 0.4])
        tongue_rgb_feat = self._rgb_to_feature(tongue_rgb)
        features.extend(tongue_rgb_feat)
        
        # 5-6. 苔色特征 (2维) - 使用RGB差异特征
        fur_rgb = tongue_info.get('fur_color_rgb', [0.9, 0.9, 0.85])
        fur_rgb_feat = self._rgb_to_feature(fur_rgb)
        features.append(fur_rgb[0] - fur_rgb[1])  # R-G差异
        features.append(sum(fur_rgb) / 3)         # 亮度
        
        # 7. 裂纹特征
        crack = tongue_info.get('crack', 0.0)
        features.append(crack)
        
        # 8. 齿痕特征
        teeth_mark = tongue_info.get('teeth_mark', 0.0)
        features.append(teeth_mark)
        
        # 额外2个特征：舌形和湿润度
        shape = tongue_info.get('shape', 0.5)
        moisture = tongue_info.get('moisture', 0.6)
        features.extend([shape, moisture])
        
        return np.array(features[:8])
    
    def analyze(self, tongue_data):
        """
        主分析函数：提取舌象特征并返回特征向量
        """
        features = self.extract_from_image(tongue_data)
        return features
    
    def get_constitution_prediction(self, features):
        """
        基于舌象特征预测体质倾向
        返回: dict of {体质: 概率}
        """
        predictions = {}
        features = np.array(features)
        
        for constitution, tongue_feat in self.CONSTITUTION_TONGUE_FEATURES.items():
            ref_features = self.extract_from_dict(tongue_feat)
            # 计算余弦相似度作为置信度
            similarity = self._cosine_similarity(features, ref_features)
            predictions[constitution] = max(0.0, min(1.0, similarity))
        
        # 归一化
        total = sum(predictions.values())
        if total > 0:
            predictions = {k: v/total for k, v in predictions.items()}
        
        return predictions
    
    def _cosine_similarity(self, a, b):
        """计算余弦相似度"""
        a = np.array(a).flatten()
        b = np.array(b).flatten()
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.5
        return 0.5 + 0.5 * (dot_product / (norm_a * norm_b))


def create_mock_tongue_data(constitution=None):
    """
    创建模拟舌象数据用于测试
    constitution: 体质类型，返回对应的模拟舌象特征
    """
    analyzer = TongueAnalyzer()
    if constitution and constitution in analyzer.CONSTITUTION_TONGUE_FEATURES:
        return analyzer.CONSTITUTION_TONGUE_FEATURES[constitution]
    else:
        # 返回默认的模拟数据
        return {
            'tongue_color_rgb': [0.80, 0.50, 0.42],
            'fur_color_rgb': [0.85, 0.82, 0.78],
            'crack': 0.3,
            'teeth_mark': 0.2,
            'shape': 0.5,
            'moisture': 0.6
        }


if __name__ == "__main__":
    # 测试舌象分析器
    print("=" * 60)
    print("舌象分析器测试")
    print("=" * 60)
    
    analyzer = TongueAnalyzer()
    
    # 测试不同体质的舌象
    test_constitutions = ['气虚', '阴虚', '湿热', '血瘀']
    
    for constitution in test_constitutions:
        tongue_data = create_mock_tongue_data(constitution)
        features = analyzer.analyze(tongue_data)
        predictions = analyzer.get_constitution_prediction(features)
        
        print(f"\n【{constitution}】舌象特征:")
        print(f"  特征向量: {features.round(3)}")
        print(f"  预测结果: {max(predictions, key=predictions.get)} "
              f"(置信度: {max(predictions.values()):.3f})")
