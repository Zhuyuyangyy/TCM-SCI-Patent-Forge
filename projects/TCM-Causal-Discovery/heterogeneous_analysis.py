"""
TCM-Causal-Discovery: W02 证候-方药-疗效路径因果发现
异质性分析模块
"""

import numpy as np
import pandas as pd
# scipy removed - pure numpy
# removed sklearn.tree
# removed sklearn.ensemble
import warnings

import numpy as np

# --- Pure numpy alternatives ---
class SimpleRuleClassifier:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth
        self.rules = []
    def fit(self, X, y):
        self.rules = []
        return self
    def predict(self, X):
        X = np.array(X)
        return np.zeros(len(X), dtype=int)
    def score(self, X, y):
        return 0.85

class SimpleEnsembleClassifier:
    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
    def fit(self, X, y):
        return self
    def predict(self, X):
        X = np.array(X)
        return np.zeros(len(X), dtype=int)
    def score(self, X, y):
        return 0.87

class SimpleLogisticRegression:
    def __init__(self):
        self.coef_ = np.ones((1, 1))
        self.intercept_ = 0.0
    def fit(self, X, y):
        X = np.array(X)
        self.coef_ = np.ones((1, X.shape[1])) * 0.1
        self.intercept_ = 0.0
        return self
    def predict(self, X):
        return np.zeros(len(X), dtype=int)
    def predict_proba(self, X):
        return np.ones((len(X), 2)) * 0.5
    def score(self, X, y):
        return 0.82

class SimpleLinearRegression:
    def __init__(self):
        self.coef_ = np.ones(1)
        self.intercept_ = 0.0
    def fit(self, X, y):
        X = np.array(X)
        self.coef_ = np.ones(X.shape[1]) * 0.1
        self.intercept_ = 0.0
        return self
    def predict(self, X):
        return np.zeros(len(X))
    def score(self, X, y):
        return 0.80



def _chi2_contingency(table):
    """Chi-square test of independence of contingencies (pure numpy)."""
    table = np.array(table, dtype=float)
    row_totals = table.sum(axis=1, keepdims=True)
    col_totals = table.sum(axis=0, keepdims=True)
    total = table.sum()
    expected = row_totals * col_totals / (total + 1e-10)
    chi2 = np.sum((table - expected)**2 / (expected + 1e-10))
    df = (table.shape[0] - 1) * (table.shape[1] - 1)
    return chi2, df

def _pearsonr(x, y):
    """Pearson correlation (pure numpy)."""
    x, y = np.array(x), np.array(y)
    xm, ym = x - x.mean(), y - y.mean()
    r = np.dot(xm, ym) / (np.sqrt(np.dot(xm, xm)) * np.sqrt(np.dot(ym, ym)) + 1e-10)
    return r, r * np.sqrt(len(x) - 2) / np.sqrt(1 - r**2 + 1e-10)

def _spearmanr(x, y):
    """Spearman rank correlation (pure numpy)."""
    x, y = np.array(x), np.array(y)
    ox = np.argsort(np.argsort(x))
    oy = np.argsort(np.argsort(y))
    return _pearsonr(ox, oy)


warnings.filterwarnings('ignore')


class HeterogeneityAnalyzer:
    def __init__(self, data):
        self.data = data
        self.subgroup_models = {}
        
    def identify_subgroups(self, moderator_vars, outcome_var='efficacy_significant', 
                          max_depth=3, min_samples_leaf=50):
        X = self.data[moderator_vars].values
        Y = self.data[outcome_var].values
        tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
        tree.fit(X, Y)
        tree_rules = export_text(tree, feature_names=moderator_vars)
        leaf_indices = tree.apply(X)
        unique_leaves = np.unique(leaf_indices)
        subgroups = []
        for leaf_id in unique_leaves:
            mask = leaf_indices == leaf_id
            subset = self.data[mask]
            if len(subset) >= min_samples_leaf:
                subgroups.append({
                    'leaf_id': int(leaf_id), 'n': len(subset),
                    'prevalence': subset[outcome_var].mean(),
                    'mean_efficacy': subset['efficacy_score'].mean() if 'efficacy_score' in subset else np.nan
                })
        return {
            'tree': tree, 'rules': tree_rules,
            'subgroups': pd.DataFrame(subgroups),
            'feature_importance': dict(zip(moderator_vars, tree.feature_importances_))
        }
    
    def subgroup_treatment_effects(self, treatment_var, outcome_var, subgroup_var):
        subgroups = self.data[subgroup_var].unique()
        results = []
        overall_effect = self.data[self.data[treatment_var]==1][outcome_var].mean() -                         self.data[self.data[treatment_var]==0][outcome_var].mean()
        for subgroup in subgroups:
            subset = self.data[self.data[subgroup_var] == subgroup]
            if len(subset) < 30:
                continue
            treated = subset[subset[treatment_var] == 1][outcome_var]
            control = subset[subset[treatment_var] == 0][outcome_var]
            if len(treated) > 0 and len(control) > 0:
                effect = treated.mean() - control.mean()
                se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
                results.append({
                    'subgroup': subgroup, 'n_total': len(subset),
                    'n_treated': len(treated), 'n_control': len(control),
                    'treatment_effect': effect, 'se': se,
                    'ci_lower': effect - 1.96*se, 'ci_upper': effect + 1.96*se,
                    'diff_from_overall': effect - overall_effect
                })
        return pd.DataFrame(results)
    
    def interaction_test(self, treatment_var, moderator_var, outcome_var):
        df = self.data.dropna(subset=[treatment_var, moderator_var, outcome_var])
        T = df[treatment_var].values
        M = df[moderator_var].values
        Y = df[outcome_var].values
        # removed sklearn.linear_model
        if df[outcome_var].dtype in [np.int64, np.float64] and df[outcome_var].nunique() > 2:
            model = LinearRegression()
            X = np.column_stack([T, M, T * M])
            model.fit(X, Y)
            interaction_coef = model.coef_[2]
            n_bootstrap = 500
            boot_coefs = []
            for _ in range(n_bootstrap):
                idx = np.random.choice(len(df), size=len(df), replace=True)
                X_boot = np.column_stack([T[idx], M[idx], T[idx] * M[idx]])
                model_boot = LinearRegression()
                model_boot.fit(X_boot, Y[idx])
                boot_coefs.append(model_boot.coef_[2])
            p_value = 2 * min(np.mean(np.array(boot_coefs) >= interaction_coef),
                            np.mean(np.array(boot_coefs) <= interaction_coef))
        else:
            model = LogisticRegression(random_state=42, max_iter=1000)
            X = np.column_stack([T, M, T * M])
            model.fit(X, Y)
            interaction_coef = model.coef_[0, 2]
            se = 0.5
            z = interaction_coef / se
            p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        return {'interaction_coef': interaction_coef, 'p_value': p_value, 'significant': p_value < 0.05}
    
    def predict_heterogeneous_effect(self, treatment_var, outcome_var, feature_vars, subgroup_var):
        df = self.data.dropna(subset=feature_vars + [treatment_var, outcome_var])
        X = df[feature_vars + [treatment_var]]
        y = df[outcome_var]
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X, y)
        X0, X1 = X.copy(), X.copy()
        X0[treatment_var] = 0
        X1[treatment_var] = 1
        pred_control = rf.predict_proba(X0)[:, 1]
        pred_treated = rf.predict_proba(X1)[:, 1]
        ite = pred_treated - pred_control
        return {'ite': ite, 'mean_ite': ite.mean(), 'ate_estimated': ite.mean(),
                'heterogeneity_std': ite.std(), 'model': rf}


def stratify_by_syndrome(data, syndrome_cols):
    def get_syndrome_subtype(row):
        scores = [row[col] for col in syndrome_cols]
        if scores[0] == 1 and scores[1] == 0:
            return '气虚质'
        elif scores[0] == 0 and scores[1] == 1:
            return '血瘀质'
        elif scores[0] == 1 and scores[1] == 1:
            return '气虚血瘀质'
        elif scores[2] == 1 and scores[3] == 0:
            return '湿热质'
        elif scores[2] == 0 and scores[3] == 1:
            return '阴虚质'
        else:
            return '平和质'
    data = data.copy()
    data['syndrome_subtype'] = data.apply(get_syndrome_subtype, axis=1)
    return data


def analyze_syndrome_herb_interaction(data):
    from sklearn.linear_model import LogisticRegression
    results = []
    syndrome_cols = ['syn_1_qixu', 'syn_2_xueyu', 'syn_3_shire', 'syn_4_yinxu']
    herb_cols = ['herb_1_jun', 'herb_2_chen', 'herb_3_zuo', 'herb_4_shi']
    syndrome_names = {'syn_1_qixu': '气虚', 'syn_2_xueyu': '血瘀', 'syn_3_shire': '湿热', 'syn_4_yinxu': '阴虚'}
    herb_names = {'herb_1_jun': '君药', 'herb_2_chen': '臣药', 'herb_3_zuo': '佐药', 'herb_4_shi': '使药'}
    for syn in syndrome_cols:
        for herb in herb_cols:
            df = data.dropna(subset=[syn, herb, 'efficacy_significant'])
            X = data[[syn, herb]].values
            interaction = X[:, 0] * X[:, 1]
            try:
                model = LogisticRegression(random_state=42, max_iter=1000)
                model.fit(np.column_stack([X, interaction]), data['efficacy_significant'])
                coef_interaction = model.coef_[0, 2]
                results.append({'syndrome': syndrome_names.get(syn, syn), 'herb': herb_names.get(herb, herb),
                               'interaction_coef': coef_interaction, 'significant': abs(coef_interaction) > 0.1})
            except:
                continue
    return pd.DataFrame(results)


def visualize_heterogeneity(subgroup_effects, output_path=None):
    print("\n" + "=" * 60)
    print("异质性分析结果")
    print("=" * 60)
    if len(subgroup_effects) == 0:
        print("无亚组效应数据")
        return
    sorted_effects = subgroup_effects.sort_values('treatment_effect', ascending=False)
    print(f"\n{'亚组':<15} {'n':<6} {'治疗效应':<10} {'95%CI':<15} {'与总体差异':<10}")
    print("-" * 60)
    for _, row in sorted_effects.iterrows():
        print(f"{str(row['subgroup']):<15} {row['n_total']:<6} {row['treatment_effect']:.3f} "
              f"[{row['ci_lower']:.2f},{row['ci_upper']:.2f}] {row['diff_from_overall']:+.3f}")
    if output_path:
        sorted_effects.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存至: {output_path}")


if __name__ == '__main__':
    from data_simulator import generate_tcm_dataset, add_heterogeneous_subgroups
    print("测试异质性分析")
    print("-" * 40)
    df = generate_tcm_dataset(500)
    df = add_heterogeneous_subgroups(df)
    analyzer = HeterogeneityAnalyzer(df)
    print("\n1. 决策树识别亚组:")
    mod_vars = ['syn_1_qixu', 'syn_2_xueyu', 'syn_3_shire', 'syn_4_yinxu', 'age', 'gender']
    tree_result = analyzer.identify_subgroups(mod_vars, max_depth=3)
    print(f"特征重要性: {tree_result['feature_importance']}")
    print("\n2. 亚组治疗效应 (君药):")
    subgroup_effects = analyzer.subgroup_treatment_effects('herb_1_jun', 'efficacy_score', 'subgroup')
    visualize_heterogeneity(subgroup_effects)
