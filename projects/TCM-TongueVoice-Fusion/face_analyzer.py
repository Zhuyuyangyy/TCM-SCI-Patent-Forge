"""
面色特征提取模块 - Facial Complexion Feature Extraction
面色分区：额/颧/颊/鼻，提取光泽度/微色差/血色，输出6维特征向量
纯NumPy实现
"""

import numpy as np


class FaceAnalyzer:
    """面色分析器 - 从面部图像中提取面色特征"""
    
    # 体质相关的面色特征先验知识
    CONSTITUTION_FACE_FEATURES = {
        '平和': {
            'forehead_luster': 0.7,     # 光泽度
            'forehead_color_diff': 0.05,  # 微色差
            'cheek_blood': 0.7,         # 血色
            'nose_luster': 0.65,
            'nose_color_diff': 0.04,
            'overall_ruddiness': 0.6
        },
        '气虚': {
            'forehead_luster': 0.4,     # 光泽暗淡
            'forehead_color_diff': 0.08,
            'cheek_blood': 0.4,         # 血色淡
            'nose_luster': 0.35,
            'nose_color_diff': 0.06,
            'overall_ruddiness': 0.35
        },
        '阳虚': {
            'forehead_luster': 0.3,     # 苍白无华
            'forehead_color_diff': 0.1,
            'cheek_blood': 0.3,         # 面色晄白
            'nose_luster': 0.28,
            'nose_color_diff': 0.08,
            'overall_ruddiness': 0.25
        },
        '阴虚': {
            'forehead_luster': 0.55,     # 两颧潮红
            'forehead_color_diff': 0.12,  # 色红
            'cheek_blood': 0.8,         # 颧红
            'nose_luster': 0.5,
            'nose_color_diff': 0.1,
            'overall_ruddiness': 0.75
        },
        '痰湿': {
            'forehead_luster': 0.45,    # 油腻
            'forehead_color_diff': 0.06,
            'cheek_blood': 0.5,
            'nose_luster': 0.4,
            'nose_color_diff': 0.05,
            'overall_ruddiness': 0.45
        },
        '湿热': {
            'forehead_luster': 0.6,     # 面垢油光
            'forehead_color_diff': 0.09,
            'cheek_blood': 0.65,
            'nose_luster': 0.55,
            'nose_color_diff': 0.07,
            'overall_ruddiness': 0.6
        },
        '血瘀': {
            'forehead_luster': 0.35,    # 晦暗
            'forehead_color_diff': 0.15,  # 紫暗
            'cheek_blood': 0.35,        # 色暗
            'nose_luster': 0.3,
            'nose_color_diff': 0.12,
            'overall_ruddiness': 0.3
        },
        '气郁': {
            'forehead_luster': 0.5,     # 暗淡
            'forehead_color_diff': 0.1,
            'cheek_blood': 0.45,        # 神情抑郁
            'nose_luster': 0.45,
            'nose_color_diff': 0.08,
            'overall_ruddiness': 0.4
        },
        '特禀': {
            'forehead_luster': 0.55,
            'forehead_color_diff': 0.1,
            'cheek_blood': 0.55,
            'nose_luster': 0.5,
            'nose_color_diff': 0.09,
            'overall_ruddiness': 0.5
        }
    }
    
    def __init__(self):
        self.feature_dim = 6
        
    def extract_from_image(self, image_data):
        """
        从面部图像数据中提取特征
        image_data: 模拟的面部图像数据
        
        返回: 6维特征向量 [额/颧/颊光泽, 额/颧/颊色差, 血色, 鼻光泽, 鼻色差, 总体红润度]
        """
        if isinstance(image_data, dict):
            return self.extract_from_dict(image_data)
        elif isinstance(image_data, np.ndarray):
            return self._simple_image_features(image_data)
        else:
            raise ValueError("Unsupported image data format")
    
    def _simple_image_features(self, image):
        """
        从图像中提取简化的面色特征
        模拟面部分区：额(上)、颧(中)、颊(下)、鼻
        """
        h, w = image.shape[:2]
        
        # 简化分区
        forehead = image[:h//4, :, :]      # 额区
        cheek_left = image[h//2:3*h//4, :w//4, :]   # 左颊
        cheek_right = image[h//2:3*h//4, 3*w//4:, :]  # 右颊
        nose = image[h//4:h//2, w//4:3*w//4, :]      # 鼻区
        
        def zone_features(zone):
            """计算区域的特征：光泽度、颜色均值、颜色标准差"""
            mean_rgb = [zone[:,:,c].mean() for c in range(3)]
            std_rgb = [zone[:,:,c].std() for c in range(3)]
            # 光泽度 = 高频成分的比例
            luster = sum(std_rgb) / 3 / 50  # 归一化
            # 红色度
            ruddiness = mean_rgb[0] / (sum(mean_rgb) + 1e-8)
            return luster, ruddiness, std_rgb[0] / 50
        
        forehead_feat = zone_features(forehead)
        cheek_feat = zone_features((cheek_left + cheek_right) / 2)
        nose_feat = zone_features(nose)
        
        # 组装6维特征
        features = np.array([
            forehead_feat[0],                        # 额光泽度
            abs(forehead_feat[2] - nose_feat[2]),     # 额-鼻色差
            (cheek_feat[1] + forehead_feat[1]) / 2,  # 血色
            nose_feat[0],                            # 鼻光泽度
            nose_feat[2],                            # 鼻红色差
            cheek_feat[1]                            # 总体红润度
        ])
        
        return features
    
    def extract_from_dict(self, face_info):
        """
        从面色信息字典中提取6维特征
        face_info: 包含forehead_luster, forehead_color_diff, cheek_blood, 
                   nose_luster, nose_color_diff, overall_ruddiness
        """
        features = []
        features.append(face_info.get('forehead_luster', 0.5))
        features.append(face_info.get('forehead_color_diff', 0.05))
        features.append(face_info.get('cheek_blood', 0.5))
        features.append(face_info.get('nose_luster', 0.5))
        features.append(face_info.get('nose_color_diff', 0.05))
        features.append(face_info.get('overall_ruddiness', 0.5))
        
        return np.array(features)
    
    def analyze(self, face_data):
        """
        主分析函数：提取面色特征
        """
        return self.extract_from_image(face_data)
    
    def get_constitution_prediction(self, features):
        """
        基于面色特征预测体质倾向
        """
        predictions = {}
        features = np.array(features)
        
        for constitution, face_feat in self.CONSTITUTION_FACE_FEATURES.items():
            ref_features = self.extract_from_dict(face_feat)
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
    
    def _extract_color_features(self, image):
        """
        提取面部颜色特征（辅助函数）
        """
        if len(image.shape) == 3:
            # RGB转HSV-like特征
            r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
            
            # 计算各区域颜色特征
            region_means = [
                image[:image.shape[0]//3, :, :].mean(axis=(0,1)),  # 上
                image[image.shape[0]//3:2*image.shape[0]//3, :, :].mean(axis=(0,1)),  # 中
                image[2*image.shape[0]//3:, :, :].mean(axis=(0,1))  # 下
            ]
            
            # 颜色差异
            color_diff = np.abs(region_means[0] - region_means[2]).mean()
            
            return color_diff
        return 0.05


def create_mock_face_data(constitution=None):
    """
    创建模拟面色数据用于测试
    """
    analyzer = FaceAnalyzer()
    if constitution and constitution in analyzer.CONSTITUTION_FACE_FEATURES:
        return analyzer.CONSTITUTION_FACE_FEATURES[constitution]
    else:
        return {
            'forehead_luster': 0.55,
            'forehead_color_diff': 0.07,
            'cheek_blood': 0.55,
            'nose_luster': 0.5,
            'nose_color_diff': 0.06,
            'overall_ruddiness': 0.5
        }


def generate_mock_face_image(constitution=None, size=(128, 128)):
    """
    生成模拟面部图像用于测试
    """
    h, w = size
    
    # 创建基础图像
    image = np.zeros((h, w, 3))
    
    # 模拟面色特征
    if constitution and constitution in FaceAnalyzer.CONSTITUTION_FACE_FEATURES:
        feat = FaceAnalyzer.CONSTITUTION_FACE_FEATURES[constitution]
        base_r = 0.5 + feat['overall_ruddiness'] * 0.4
        base_g = 0.4 + feat['cheek_blood'] * 0.2
        base_b = 0.35 + feat['forehead_luster'] * 0.1
    else:
        base_r, base_g, base_b = 0.6, 0.45, 0.4
    
    # 填充基础颜色
    image[:,:,0] = base_r + np.random.randn(h, w) * 0.05  # R
    image[:,:,1] = base_g + np.random.randn(h, w) * 0.05  # G
    image[:,:,2] = base_b + np.random.randn(h, w) * 0.05  # B
    
    # 添加面部区域变化
    # 额头区域偏白
    forehead_region = image[:h//4, :, :]
    forehead_region[:,:,0] *= 1.1
    forehead_region[:,:,1] *= 1.05
    forehead_region[:,:,2] *= 1.05
    
    # 颧骨区域偏红
    cheek_region = image[h//2:3*h//4, w//4:3*w//4, :]
    cheek_region[:,:,0] *= 1.15
    
    # 鼻部区域
    nose_region = image[h//4:h//2, w//3:2*w//3, :]
    nose_region[:,:,0] *= 0.95
    nose_region[:,:,2] *= 1.1
    
    # 限制范围
    image = np.clip(image, 0, 1)
    
    return (image * 255).astype(np.uint8)


if __name__ == "__main__":
    # 测试面色分析器
    print("=" * 60)
    print("面色分析器测试")
    print("=" * 60)
    
    analyzer = FaceAnalyzer()
    
    # 测试不同体质的面色
    test_constitutions = ['气虚', '阴虚', '湿热', '血瘀']
    
    for constitution in test_constitutions:
        face_data = create_mock_face_data(constitution)
        features = analyzer.analyze(face_data)
        predictions = analyzer.get_constitution_prediction(features)
        
        print(f"\n【{constitution}】面色特征:")
        print(f"  特征向量: {features.round(3)}")
        print(f"  预测结果: {max(predictions, key=predictions.get)} "
              f"(置信度: {max(predictions.values()):.3f})")
