"""
TCM-Constitution-Risk 完整演示
"""
import numpy as np
from constitution_classifier import ConstitutionClassifier
from cox_model import RiskPredictor
from survival_analysis import KaplanMeierEstimator, LogRankTest

def run_demo():
    print("=" * 60)
    print("TCM-Constitution-Risk Demo: 体质-疾病风险预测")
    print("=" * 60)
    
    # Step 1: 体质分类
    print("\n[Step 1] 体质分类...")
    clf = ConstitutionClassifier()
    
    # 模拟问卷回答（9题）
    responses = [4, 2, 3, 5, 4, 2, 4, 1, 1]  # 各题目评分
    result = clf.classify(responses)
    
    print(f"  判定体质: {result['primary']}")
    print(f"  置信度: {result['confidence']:.2%}")
    print(f"  体质分布概率:")
    for c, p in sorted(result['probabilities'].items(), key=lambda x: -x[1]):
        if p > 0.05:
            print(f"    {c}: {p:.1%}")
    print(f"  养生建议: {result['recommendation']}")
    
    # Step 2: Cox风险预测
    print("\n[Step 2] Cox比例风险模型预测...")
    predictor = RiskPredictor()
    
    patient_data = {
        'questionnaire': responses,
        'age': 45,
        'smoking': False,
        'exercise': True,
        'poor_diet': False,
        'time_horizon': 10
    }
    
    prediction = predictor.predict(patient_data)
    
    print(f"  体质: {prediction['constitution']['primary']}")
    print(f"  风险比(HR): {prediction['risk_prediction']['hazard_ratio']:.2f}")
    print(f"  10年累积风险: {prediction['risk_prediction']['cumulative_risk']:.1%}")
    print(f"  风险等级: {prediction['risk_prediction']['risk_level']}")
    print(f"  生活习惯调整: ×{prediction['risk_prediction']['lifestyle_adjustment']:.2f}")
    print(f"  调整后风险: {prediction['risk_prediction']['adjusted_risk']:.1%}")
    
    print("\n  综合建议:")
    for rec in prediction['recommendations']:
        print(f"    {rec}")
    
    # Step 3: 生存分析
    print("\n[Step 3] Kaplan-Meier生存分析...")
    km = KaplanMeierEstimator()
    
    # 模拟5年随访数据（平和质 vs 气郁质）
    np.random.seed(42)
    times1 = np.random.exponential(8, 100)  # 平和质
    events1 = np.random.binomial(1, 0.3, 100)
    
    times2 = np.random.exponential(6, 100)  # 气郁质
    events2 = np.random.binomial(1, 0.4, 100)
    
    km_result = km.fit(list(times1) + list(times2), list(events1) + list(events2))
    print(f"  中位生存期: {km_result['median_survival_time']:.1f}年")
    print(f"  5年生存率: {km_result['five_year_survival']:.1%}")
    print(f"  10年生存率: {km_result['ten_year_survival']:.1%}")
    
    # Log-Rank检验
    print("\n[Step 4] Log-Rank检验（平和质 vs 气郁质）...")
    lr_result = LogRankTest.compare(
        list(times1), list(events1),
        list(times2), list(events2)
    )
    print(f"  卡方统计量: {lr_result['chi_square']:.2f}")
    print(f"  p值: {lr_result['p_value']:.4f}")
    print(f"  结论: {lr_result['conclusion']}")
    
    print("\n" + "=" * 60)
    print("Demo 完成!")

if __name__ == '__main__':
    run_demo()
