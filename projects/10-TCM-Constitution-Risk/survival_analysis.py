"""
Survival Analysis for TCM Constitution Studies
中医体质研究的生存分析方法
"""
import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict

class KaplanMeierEstimator:
    """
    Kaplan-Meier生存分析
    用于分析不同体质人群的疾病无病生存期
    """
    def __init__(self):
        self.time_points = []
        self.survival_probs = []
    
    def fit(self, survival_times: List[float], events: List[int]) -> Dict:
        """
        拟合Kaplan-Meier曲线
        survival_times: 生存时间
        events: 事件发生(1=发病/死亡, 0=删失)
        """
        # 排序
        sorted_data = sorted(zip(survival_times, events), key=lambda x: x[0])
        
        n = len(survival_times)
        survival_prob = 1.0
        
        times = [0.0]
        probs = [1.0]
        
        at_risk = n
        for t, e in sorted_data:
            if e == 1:
                survival_prob *= (at_risk - 1) / at_risk
                at_risk -= 1
                times.append(t)
                probs.append(survival_prob)
        
        self.time_points = times
        self.survival_probs = probs
        
        # 计算中位生存期
        median_idx = next((i for i, p in enumerate(probs) if p <= 0.5), len(probs) - 1)
        median_survival = times[median_idx] if median_idx < len(times) else float('inf')
        
        return {
            'median_survival_time': median_survival,
            'survival_curve': list(zip(times, probs)),
            'five_year_survival': self._interpolate_survival(5.0),
            'ten_year_survival': self._interpolate_survival(10.0)
        }
    
    def _interpolate_survival(self, t: float) -> float:
        """插值计算特定时间点的生存概率"""
        for i, time in enumerate(self.time_points):
            if time >= t:
                if i == 0:
                    return 1.0
                # 线性插值
                t1, t2 = self.time_points[i-1], self.time_points[i]
                p1, p2 = self.survival_probs[i-1], self.survival_probs[i]
                return p1 + (p2 - p1) * (t - t1) / (t2 - t1)
        return self.survival_probs[-1] if self.survival_probs else 1.0


class LogRankTest:
    """
    Log-Rank检验：比较两组生存曲线是否有显著差异
    """
    @staticmethod
    def compare(group1_times, group1_events, group2_times, group2_events) -> Dict:
        """
        比较两组生存曲线
        返回: 卡方统计量、p值、结论
        """
        # 合并所有时间点
        all_times = sorted(set(group1_times + group2_times))
        
        # 统计量
        O1, E1 = 0, 0
        for t in all_times:
            n1 = sum(1 for x in group1_times if x >= t)
            n2 = sum(1 for x in group2_times if x >= t)
            d1 = sum(1 for x, e in zip(group1_times, group1_events) if x == t and e == 1)
            d2 = sum(1 for x, e in zip(group2_times, group2_events) if x == t and e == 1)
            
            if n1 + n2 > 0:
                E1 += (n1 / (n1 + n2)) * (d1 + d2)
                O1 += d1
        
        # 卡方统计量
        if E1 > 0 and (O1 - E1) != 0:
            chi2 = ((O1 - E1) ** 2) / E1
        else:
            chi2 = 0.0
        
        # 近似p值（1个自由度）
        p_value = np.exp(-0.5 * chi2)  # 简化近似
        
        return {
            'observed': O1,
            'expected': E1,
            'chi_square': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'conclusion': f"{'两组生存曲线有显著差异' if p_value < 0.05 else '两组生存曲线无显著差异'} (p={p_value:.4f})"
        }
