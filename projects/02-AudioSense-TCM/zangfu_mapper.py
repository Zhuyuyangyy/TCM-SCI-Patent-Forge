"""
Zang-Fu Function Mapper
将VAE潜空间向量映射为脏腑功能评估
"""
import numpy as np

class ZangFuMapper:
    # 五脏: 0=心, 1=肝, 2=脾, 3=肺, 4=肾
    ORGAN_NAMES = ['心', '肝', '脾', '肺', '肾']
    ORGAN_FUNCTIONS = {
        '心': '心主血脉，藏神',
        '肝': '肝主疏泄，藏魂',
        '脾': '脾主运化，藏意',
        '肺': '肺主气，藏魄',
        '肾': '肾藏精，主水'
    }
    
    # 五脏与语音特征的关系
    ORGAN_SPEECH_PATTERNS = {
        '心': {'f0_range': (150, 250), 'rate_range': (2.5, 4.5), 'tone': '清亮'},
        '肝': {'f0_range': (100, 180), 'rate_range': (2.0, 3.5), 'tone': '沉稳'},
        '脾': {'f0_range': (120, 200), 'rate_range': (1.5, 3.0), 'tone': '缓和'},
        '肺': {'f0_range': (130, 220), 'rate_range': (2.0, 4.0), 'tone': '有力'},
        '肾': {'f0_range': (80, 150), 'rate_range': (1.0, 2.5), 'tone': '低沉'}
    }
    
    def __init__(self):
        self.organ_thresholds = {
            'normal': 0.5,
            'deficient': 0.3,
            'excess': 0.7
        }
        
        # 状态解释模板
        self.interpretation_templates = {
            '虚证': {
                '心': '心血不足，神失所养，表现为心悸、失眠、多梦',
                '肝': '肝血亏虚，疏泄失常，表现为眩晕、肢麻、爪甲不荣',
                '脾': '脾气虚弱，运化失职，表现为食少、腹胀、便溏',
                '肺': '肺气不足，宣降失常，表现为气短、懒言、咳喘无力',
                '肾': '肾精不足，肾气亏虚，表现为腰膝酸软、眩晕耳鸣'
            },
            '实证': {
                '心': '心火亢盛，热扰心神，表现为心烦、失眠、口舌生疮',
                '肝': '肝气郁结，肝火上炎，表现为胁痛、烦躁、目赤肿痛',
                '脾': '湿热蕴脾，运化受阻，表现为脘腹胀满、恶心呕吐',
                '肺': '痰热蕴肺，肺失宣降，表现为咳嗽、痰黄、胸闷发热',
                '肾': '湿热下注，肾失气化，表现为尿频、尿急、腰痛'
            },
            '平': {
                '心': '心功能正常，气血充盈',
                '肝': '肝气条达，疏泄正常',
                '脾': '脾气健运，消化吸收良好',
                '肺': '肺气充足，呼吸平稳',
                '肾': '肾精充盛，精力充沛'
            }
        }
    
    def map_to_zangfu(self, latent_vector, organ_probs):
        """
        输入: VAE潜空间向量, 各脏腑概率
        输出: 脏腑功能评估报告
        """
        results = []
        for i, prob in enumerate(organ_probs):
            organ_name = self.ORGAN_NAMES[i]
            status = self._classify_status(prob)
            results.append({
                'organ': organ_name,
                'function': self.ORGAN_FUNCTIONS[organ_name],
                'probability': float(prob),
                'status': status,
                'tcm_interpretation': self._interpret(prob, organ_name)
            })
        return results
    
    def _classify_status(self, prob):
        """根据概率分类状态"""
        if prob < 0.3:
            return '虚证'
        elif prob > 0.7:
            return '实证'
        else:
            return '平'
    
    def _interpret(self, prob, organ):
        """生成具体的TCM解释"""
        status = self._classify_status(prob)
        template = self.interpretation_templates.get(status, {}).get(organ, '')
        return template
    
    def analyze_from_features(self, audio_features):
        """
        从语音特征直接分析脏腑状态
        
        Args:
            audio_features: dict，包含 'f0', 'speech_rate' 等
        
        Returns:
            分析报告
        """
        f0 = audio_features.get('f0', 150)
        speech_rate = audio_features.get('speech_rate', 3.0)
        
        results = []
        
        # 基于启发式规则分析
        for organ, patterns in self.ORGAN_SPEECH_PATTERNS.items():
            f0_min, f0_max = patterns['f0_range']
            rate_min, rate_max = patterns['rate_range']
            
            # 计算匹配度
            f0_score = 1.0 - min(abs(f0 - (f0_min + f0_max) / 2) / ((f0_max - f0_min) / 2), 1.0)
            rate_score = 1.0 - min(abs(speech_rate - (rate_min + rate_max) / 2) / ((rate_max - rate_min) / 2), 1.0)
            
            match_score = (f0_score + rate_score) / 2
            
            results.append({
                'organ': organ,
                'function': self.ORGAN_FUNCTIONS[organ],
                'match_score': float(match_score),
                'status': self._classify_status(match_score),
                'tcm_interpretation': self._interpret(match_score, organ)
            })
        
        # 按匹配度排序
        results.sort(key=lambda x: x['match_score'], reverse=True)
        return results
    
    def generate_comprehensive_report(self, latent_vector, organ_probs, audio_features=None):
        """
        生成综合评估报告
        
        Args:
            latent_vector: VAE潜空间向量
            organ_probs: 各脏腑概率
            audio_features: 原始语音特征（可选）
        
        Returns:
            综合报告字典
        """
        # 基础映射
        basic_report = self.map_to_zangfu(latent_vector, organ_probs)
        
        # 语音特征分析（如果提供）
        feature_analysis = None
        if audio_features:
            feature_analysis = self.analyze_from_features(audio_features)
        
        # 综合判断
        dominant_organ_idx = np.argmax(organ_probs)
        dominant_organ = self.ORGAN_NAMES[dominant_organ_idx]
        
        # 识别主要问题
        issues = []
        for item in basic_report:
            if item['status'] == '虚证':
                issues.append(f"{item['organ']}虚")
            elif item['status'] == '实证':
                issues.append(f"{item['organ']}实")
        
        # 生成建议
        suggestions = self._generate_suggestions(basic_report)
        
        report = {
            'basic_assessment': basic_report,
            'feature_analysis': feature_analysis,
            'dominant_organ': dominant_organ,
            'issues': issues,
            'suggestions': suggestions,
            'latent_vector_summary': {
                'dimension': len(latent_vector),
                'mean': float(np.mean(latent_vector)),
                'std': float(np.std(latent_vector))
            }
        }
        
        return report
    
    def _generate_suggestions(self, assessment):
        """基于评估结果生成建议"""
        suggestions = []
        
        for item in assessment:
            organ = item['organ']
            status = item['status']
            
            if status == '虚证':
                if organ == '心':
                    suggestions.append('建议养心安神，可用归脾汤加减')
                elif organ == '肝':
                    suggestions.append('建议补血柔肝，可用四物汤加减')
                elif organ == '脾':
                    suggestions.append('建议健脾益气，可用四君子汤加减')
                elif organ == '肺':
                    suggestions.append('建议补肺益气，可用补肺汤加减')
                elif organ == '肾':
                    suggestions.append('建议补肾填精，可用六味地黄丸加减')
            
            elif status == '实证':
                if organ == '心':
                    suggestions.append('建议清心泻火，可用导赤散加减')
                elif organ == '肝':
                    suggestions.append('建议疏肝解郁，可用柴胡疏肝散加减')
                elif organ == '脾':
                    suggestions.append('建议健脾祛湿，可用参苓白术散加减')
                elif organ == '肺':
                    suggestions.append('建议宣肺化痰，可用二陈汤加减')
                elif organ == '肾':
                    suggestions.append('建议清热利湿，可用知柏地黄丸加减')
        
        if not suggestions:
            suggestions.append('整体状态良好，建议保持良好的生活习惯')
        
        return suggestions


if __name__ == '__main__':
    # 测试代码
    mapper = ZangFuMapper()
    
    # 模拟VAE输出
    latent_vector = np.random.randn(32).astype(np.float32)
    organ_probs = np.random.rand(5)
    organ_probs = organ_probs / organ_probs.sum()  # 归一化
    
    print("=" * 50)
    print("脏腑功能评估报告")
    print("=" * 50)
    
    # 基础映射
    report = mapper.map_to_zangfu(latent_vector, organ_probs)
    
    for item in report:
        print(f"\n{item['organ']}（{item['status']}）")
        print(f"  功能: {item['function']}")
        print(f"  概率: {item['probability']:.3f}")
        print(f"  解读: {item['tcm_interpretation']}")
    
    # 综合报告
    print("\n" + "=" * 50)
    print("综合评估报告")
    print("=" * 50)
    
    audio_features = {'f0': 180, 'speech_rate': 3.0}
    comprehensive = mapper.generate_comprehensive_report(latent_vector, organ_probs, audio_features)
    
    print(f"主导脏腑: {comprehensive['dominant_organ']}")
    print(f"主要问题: {', '.join(comprehensive['issues'])}")
    print("\n建议:")
    for suggestion in comprehensive['suggestions']:
        print(f"  - {suggestion}")
