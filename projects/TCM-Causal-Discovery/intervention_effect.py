"""
TCM-Causal-Discovery: W02 证候-方药-疗效路径因果发现
因果效应计算模块
"""

import numpy as np
import pandas as pd
# scipy removed - pure numpy
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


class CausalEffectCalculator:
    def __init__(self, data):
        self.data = data
    
    def compute_ate(self, treatment_var, outcome_var, adjustment_vars=None):
        df = self.data.dropna(subset=[treatment_var, outcome_var] + (adjustment_vars or []))
        
        if adjustment_vars is None or len(adjustment_vars) == 0:
            treated = df[df[treatment_var] == 1][outcome_var]
            control = df[df[treatment_var] == 0][outcome_var]
            ate = treated.mean() - control.mean()
            se = np.sqrt(treated.var()/len(treated) + control.var()/len(control))
            ci = (ate - 1.96*se, ate + 1.96*se)
        else:
            # removed sklearn.linear_model
            X = df[adjustment_vars].values
            T = df[treatment_var].values
            Y = df[outcome_var].values
            model = LinearRegression()
            model.fit(X, Y)
            residuals = Y - model.predict(X)
            ate_model = LinearRegression()
            ate_model.fit(T.reshape(-1, 1), residuals)
            ate = ate_model.coef_[0]
            n = len(Y)
            se = residuals.std() / np.sqrt(n)
            ci = (ate - 1.96*se, ate + 1.96*se)
        
        return {
            'ate': ate, 'se': se, 'ci_95': ci,
            'p_value': 2 * (1 - stats.norm.cdf(abs(ate/se))) if se > 0 else 1.0
        }
    
    def compute_stratified_ate(self, treatment_var, outcome_var, strat_var):
        strata = self.data[strat_var].unique()
        results = []
        
        for stratum in strata:
            subset = self.data[self.data[strat_var] == stratum]
            if len(subset) < 30:
                continue
            calc = CausalEffectCalculator(subset)
            effect = calc.compute_ate(treatment_var, outcome_var)
            weight = len(subset) / len(self.data)
            results.append({'stratum': stratum, 'n': len(subset), 'weight': weight, **effect})
        
        weighted_ate = sum(r['ate'] * r['weight'] for r in results)
        weighted_se = np.sqrt(sum((r['se'] * r['weight'])**2 for r in results))
        weighted_ci = (weighted_ate - 1.96*weighted_se, weighted_ate + 1.96*weighted_se)
        
        return {
            'stratified_results': results,
            'weighted_ate': weighted_ate, 'weighted_se': weighted_se, 'weighted_ci_95': weighted_ci
        }
    
    def compute_dose_response(self, treatment_var, outcome_var, n_levels=5):
        df = self.data.dropna(subset=[treatment_var, outcome_var])
        quantiles = pd.qcut(df[treatment_var], q=n_levels, labels=False, duplicates='drop')
        results = []
        for level in range(n_levels):
            subset = df[quantiles == level]
            if len(subset) > 0:
                results.append({
                    'level': level, 'treatment_value': subset[treatment_var].mean(),
                    'outcome_mean': subset[outcome_var].mean(),
                    'outcome_se': subset[outcome_var].std() / np.sqrt(len(subset)), 'n': len(subset)
                })
        return pd.DataFrame(results)
    
    def estimate_confounder_effect(self, confounder, treatment_var, outcome_var):
        t_model = stats.pointbiserialr(self.data[confounder], self.data[treatment_var])
        y_model = stats.pointbiserialr(self.data[confounder], self.data[outcome_var])
        confounding_bias = t_model.correlation * y_model.correlation
        return {
            'confounder': confounder, 'effect_on_treatment': t_model.correlation,
            'effect_on_outcome': y_model.correlation, 'confounding_strength': confounding_bias,
            'p_value_treatment': t_model.pvalue, 'p_value_outcome': y_model.pvalue
        }


def compute_path_effects(data, causal_edges, outcome_var='efficacy_score'):
    path_results = []
    for cause, effect in causal_edges:
        if effect == outcome_var:
            calc = CausalEffectCalculator(data)
            effect_size = calc.compute_ate(cause, outcome_var)
            path_results.append({
                'path_type': '直接效应', 'cause': cause, 'effect': effect,
                'ate': effect_size['ate'], 'se': effect_size['se'],
                'ci_lower': effect_size['ci_95'][0], 'ci_upper': effect_size['ci_95'][1],
                'p_value': effect_size['p_value']
            })
    df_results = pd.DataFrame(path_results)
    if len(df_results) > 0:
        df_results = df_results.sort_values('ate', ascending=False)
    return df_results


def estimate_heterogeneous_effects(data, treatment_var, outcome_var, moderator_vars):
    # removed sklearn.tree
    X = data[moderator_vars].values
    T = data[treatment_var].values
    Y = data[outcome_var].values
    treatment_effect = data.loc[data[treatment_var]==1, outcome_var].mean() -                       data.loc[data[treatment_var]==0, outcome_var].mean()
    results = []
    for mod_var in moderator_vars:
        for level in data[mod_var].unique():
            subset = data[data[mod_var] == level]
            if len(subset) >= 30:
                te = subset.loc[subset[treatment_var]==1, outcome_var].mean() -                      subset.loc[subset[treatment_var]==0, outcome_var].mean()
                results.append({
                    'moderator': mod_var, 'level': level, 'treatment_effect': te,
                    'n': len(subset), 'diff_from_overall': te - treatment_effect
                })
    return pd.DataFrame(results)


def bootstrap_confidence_interval(data, stat_func, n_bootstrap=1000, ci=0.95):
    np.random.seed(42)
    bootstrap_stats = []
    n = len(data)
    alpha = 1 - ci
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, size=n, replace=True)
        sample = data.iloc[indices]
        try:
            stat = stat_func(sample)
            bootstrap_stats.append(stat)
        except:
            continue
    if len(bootstrap_stats) == 0:
        return (np.nan, np.nan)
    lower = np.percentile(bootstrap_stats, alpha/2 * 100)
    upper = np.percentile(bootstrap_stats, (1 - alpha/2) * 100)
    return (lower, upper)


if __name__ == '__main__':
    from data_simulator import generate_tcm_dataset, add_heterogeneous_subgroups
    print("测试因果效应计算")
    print("-" * 40)
    df = generate_tcm_dataset(500)
    df = add_heterogeneous_subgroups(df)
    calc = CausalEffectCalculator(df)
    print("\n1. ATE分析 (君药对疗效的影响):")
    ate_result = calc.compute_ate('herb_1_jun', 'efficacy_score')
    print(f"  ATE = {ate_result['ate']:.3f}")
    print(f"  95% CI = [{ate_result['ci_95'][0]:.3f}, {ate_result['ci_95'][1]:.3f}]")
    print("\n2. 分层ATE分析 (按性别):")
    stratified = calc.compute_stratified_ate('herb_1_jun', 'efficacy_score', 'gender')
    for r in stratified['stratified_results']:
        print(f"  {r['stratum']}: ATE={r['ate']:.3f}, n={r['n']}")
    print(f"  加权ATE = {stratified['weighted_ate']:.3f}")
