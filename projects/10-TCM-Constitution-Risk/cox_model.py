"""
Cox Proportional Hazards Model for Risk Prediction
Cox比例风险模型：体质-疾病风险预测
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from constitution_classifier import ConstitutionClassifier


class CoxModel:
    """
    简化的Cox比例风险模型
    h(t) = h0(t) * exp(β'x)
    
    用于预测特定体质人群的疾病发生风险
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.baseline_hazard = 0.01  # h0(t)
        # 模拟的Cox系数（实际应从数据拟合）
        self.coefficients = {
            '气虚质': 0.8,
            '阳虚质': 0.7,
            '阴虚质': 0.6,
            '痰湿质': 0.9,
            '湿热质': 0.7,
            '血瘀质': 1.0,
            '气郁质': 0.5,
            '特禀质': 1.2,
        }
    
    def predict_hazard_ratio(self, constitution_type: str) -> float:
        """预测风险比(HR)"""
        beta = self.coefficients.get(constitution_type, 0.0)
        return np.exp(beta)
    
    def predict_risk(self, constitution_type: str, time_years: float = 5.0) -> Dict:
        """
        预测风险
        返回: 风险评分、累积风险、风险等级
        """
        hr = self.predict_hazard_ratio(constitution_type)
        
        # 累积风险 = 1 - exp(-H0(t) * HR)
        H0 = self.baseline_hazard * time_years
        cumulative_risk = 1 - np.exp(-H0 * hr)
        
        # 风险等级
        if cumulative_risk > 0.3:
            level = '高风险'
        elif cumulative_risk > 0.15:
            level = '中风险'
        else:
            level = '低风险'
        
        return {
            'constitution': constitution_type,
            'hazard_ratio': hr,
            'time_years': time_years,
            'cumulative_risk': cumulative_risk,
            'risk_level': level,
            'interpretation': self._interpret(constitution_type, time_years, cumulative_risk)
        }
    
    def _interpret(self, constitution: str, time_years: float, risk: float) -> str:
        interpretations = {
            '气虚质': f'气虚质人群{time_years:.0f}年内患慢性疾病风险约{risk:.1%}，建议补气养生',
            '血瘀质': f'血瘀质人群{time_years:.0f}年内患心脑血管疾病风险约{risk:.1%}，建议活血化瘀',
            '特禀质': f'特禀质人群过敏性疾病风险约{risk:.1%}，需重点规避过敏原',
        }
        return interpretations.get(constitution, f'风险约{risk:.1%}，建议咨询医师')
    
    def compare_populations(self, constitutions: List[str], time_years: float = 5.0) -> List[Dict]:
        """比较不同体质人群的风险"""
        results = []
        for c in constitutions:
            r = self.predict_risk(c, time_years)
            results.append(r)
        
        # 按风险排序
        results.sort(key=lambda x: -x['cumulative_risk'])
        return results


class RiskPredictor:
    """
    综合风险预测器
    结合体质分类 + Cox风险模型 + 生活习惯
    """
    def __init__(self):
        self.constitution_clf = ConstitutionClassifier()
        self.cox_model = CoxModel(feature_names=[
            'constitution_type', 'age', 'smoking', 'exercise', 'diet'
        ])
        
        # 生活方式风险因子系数
        self.lifestyle_coefs = {
            'smoking': 0.5,
            'sedentary': 0.4,
            'poor_diet': 0.3,
        }
    
    def predict(self, patient_data: Dict) -> Dict:
        """
        综合风险预测
        输入: {questionnaire: [...], age: int, smoking: bool, exercise: bool, diet: str}
        """
        # Step 1: 体质分类
        constitution_result = self.constitution_clf.classify(
            patient_data.get('questionnaire', [])
        )
        
        # Step 2: Cox风险预测
        risk_result = self.cox_model.predict_risk(
            constitution_result['primary'],
            time_years=patient_data.get('time_horizon', 5)
        )
        
        # Step 3: 生活方式调整
        lifestyle_adj = self._lifestyle_adjustment(patient_data)
        adjusted_risk = risk_result['cumulative_risk'] * lifestyle_adj
        
        # Step 4: 综合建议
        recommendations = self._generate_recommendations(
            constitution_result, risk_result, lifestyle_adj
        )
        
        return {
            'constitution': constitution_result,
            'risk_prediction': {
                **risk_result,
                'lifestyle_adjustment': lifestyle_adj,
                'adjusted_risk': adjusted_risk
            },
            'recommendations': recommendations
        }
    
    def _lifestyle_adjustment(self, data: Dict) -> float:
        """生活方式风险调整因子"""
        adj = 1.0
        if data.get('smoking'):
            adj *= (1 + self.lifestyle_coefs['smoking'])
        if not data.get('exercise'):
            adj *= (1 + self.lifestyle_coefs['sedentary'])
        if data.get('poor_diet'):
            adj *= (1 + self.lifestyle_coefs['poor_diet'])
        return adj
    
    def _generate_recommendations(self, constitution: Dict, risk: Dict, 
                                 lifestyle_adj: float) -> List[str]:
        """生成个性化建议"""
        recs = []
        
        # 体质建议
        recs.append(f"您的体质为{constitution['primary']}，{constitution['recommendation']}")
        
        # 风险建议
        if risk['risk_level'] == '高风险':
            recs.append(f"⚠️ {risk['risk_level']}：{risk['interpretation']}")
        else:
            recs.append(f"{risk['interpretation']}")
        
        # 生活方式建议
        if lifestyle_adj > 1.3:
            recs.append("建议改善生活方式：戒烟、增加运动、调整饮食")
        
        return recs
