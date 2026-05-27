"""
TCM-Causal-Discovery: W02 证候-方药-疗效路径因果发现
数据模拟器：生成1000条中医医案数据
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)


def generate_tcm_dataset(n_samples=1000):
    """
    生成模拟中医医案数据集
    """
    syndrome_1 = np.random.binomial(1, 0.45, n_samples)
    syndrome_2 = np.random.binomial(1, 0.50, n_samples)
    syndrome_3 = np.random.binomial(1, 0.40, n_samples)
    syndrome_4 = np.random.binomial(1, 0.35, n_samples)

    herb_1 = (
        (syndrome_3 == 1) & (syndrome_4 == 1) & (np.random.random(n_samples) < 0.70) |
        (syndrome_3 == 1) & (syndrome_4 == 0) & (np.random.random(n_samples) < 0.45) |
        (syndrome_3 == 0) & (syndrome_4 == 1) & (np.random.random(n_samples) < 0.40) |
        (np.random.random(n_samples) < 0.20)
    ).astype(int)

    herb_2 = (
        (syndrome_3 == 1) & (np.random.random(n_samples) < 0.55) |
        (syndrome_3 == 0) & (np.random.random(n_samples) < 0.25)
    ).astype(int)

    herb_3 = (
        (syndrome_4 == 1) & (np.random.random(n_samples) < 0.50) |
        (syndrome_4 == 0) & (np.random.random(n_samples) < 0.20)
    ).astype(int)

    herb_4 = np.random.binomial(1, 0.30, n_samples)

    base_score = 50.0
    syndrome_effect = (
        syndrome_1 * 8.0 + syndrome_2 * 5.0 + syndrome_3 * (-3.0) + syndrome_4 * 4.0
    )
    herb_effect = (
        herb_1 * 12.0 + herb_2 * 8.0 + herb_3 * 6.0 + herb_4 * 2.0
    )
    noise = np.random.normal(0, 5.0, n_samples)
    efficacy_score = base_score + syndrome_effect + herb_effect + noise
    efficacy_score = np.clip(efficacy_score, 0, 100)
    efficacy_binary = (efficacy_score >= 70).astype(int)

    base_date = datetime(2020, 1, 1)
    treatment_days = [random.randint(7, 90) for _ in range(n_samples)]
    treatment_dates = [base_date + timedelta(days=d) for d in treatment_days]

    df = pd.DataFrame({
        'case_id': [f'CASE_{i:04d}' for i in range(1, n_samples + 1)],
        'treatment_date': treatment_dates,
        'syn_1_qixu': syndrome_1,
        'syn_2_xueyu': syndrome_2,
        'syn_3_shire': syndrome_3,
        'syn_4_yinxu': syndrome_4,
        'herb_1_jun': herb_1,
        'herb_2_chen': herb_2,
        'herb_3_zuo': herb_3,
        'herb_4_shi': herb_4,
        'efficacy_score': efficacy_score.round(2),
        'efficacy_significant': efficacy_binary,
        'age': np.random.randint(18, 75, n_samples),
        'gender': np.random.binomial(1, 0.52, n_samples),
        'disease_duration_days': np.random.randint(30, 3650, n_samples),
    })

    return df


def add_heterogeneous_subgroups(df, n_subgroups=3):
    def classify_subgroup(row):
        if row['syn_1_qixu'] == 1 and row['syn_2_xueyu'] == 0:
            return '气虚为主型'
        elif row['syn_1_qixu'] == 0 and row['syn_2_xueyu'] == 1:
            return '血瘀为主型'
        elif row['syn_1_qixu'] == 1 and row['syn_2_xueyu'] == 1:
            return '气虚血瘀型'
        elif row['syn_3_shire'] == 1 and row['syn_4_yinxu'] == 0:
            return '湿热为主型'
        elif row['syn_3_shire'] == 0 and row['syn_4_yinxu'] == 1:
            return '阴虚为主型'
        else:
            return '其他证型'

    df['subgroup'] = df.apply(classify_subgroup, axis=1)
    return df


def save_dataset(df, filepath):
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f"数据集已保存至: {filepath}")
    print(f"样本量: {len(df)}")
    print(f"\n数据概览:")
    print(df.describe())


if __name__ == '__main__':
    df = generate_tcm_dataset(n_samples=1000)
    df = add_heterogeneous_subgroups(df)
    save_dataset(df, 'tcm_medical_records.csv')
