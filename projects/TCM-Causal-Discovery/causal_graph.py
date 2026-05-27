"""
TCM-Causal-Discovery: W02 证候-方药-疗效路径因果发现
基于PC算法的因果图构建
"""

import numpy as np
import pandas as pd
# scipy removed - pure numpy
from itertools import combinations
import warnings

import numpy as np

def _norm_cdf(x):
    """Standard normal CDF approximation."""
    import math
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


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


class PCAlgorithm:
    """
    PC算法实现 - 用于发现因果骨架并定向边
    """
    
    def __init__(self, alpha=0.05):
        self.alpha = alpha
        self.graph = None
        self.sepsets = {}
        self.edges = set()
        
    def fit(self, data):
        variables = list(data.columns)
        n_vars = len(variables)
        self.graph = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        self.edge_list = []
        
        for depth in range(n_vars):
            print(f"=== PC算法 Depth={depth} ===")
            removed_edges = 0
            
            for i in range(n_vars):
                for j in range(i + 1, n_vars):
                    if self.graph[i, j] == 0:
                        continue
                    
                    neighbors = self._get_neighbors(i, j)
                    
                    if len(neighbors) < depth:
                        continue
                    
                    for cond_set in combinations(neighbors, depth):
                        cond_set = list(cond_set)
                        is_independent = self._conditional_independence_test(
                            data, variables[i], variables[j], cond_set
                        )
                        
                        if is_independent:
                            self.graph[i, j] = 0
                            self.graph[j, i] = 0
                            self.sepsets[(i, j)] = cond_set
                            self.sepsets[(j, i)] = cond_set
                            removed_edges += 1
                            break
            
            print(f"移除边数: {removed_edges}")
            
            if removed_edges == 0 and depth > 0:
                print("收敛: 无新边移除")
                break
        
        self._orient_edges(variables)
        
        edge_list = []
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if self.graph[i, j] == 1:
                    edge_list.append((variables[i], variables[j]))
        
        self.edges = set(edge_list)
        return self.graph, edge_list
    
    def _get_neighbors(self, i, j):
        neighbors = []
        for k in range(self.graph.shape[0]):
            if k != i and k != j and self.graph[i, k] == 1:
                neighbors.append(k)
        return neighbors
    
    def _conditional_independence_test(self, data, var_i, var_j, cond_set):
        """Conditionally independent test: returns True if independent (should remove edge)."""
        import math
        try:
            if len(cond_set) == 0:
                x = np.array(data[var_i], dtype=float)
                y = np.array(data[var_j], dtype=float)
                # Pure Pearson p-value
                xm, ym = x - x.mean(), y - y.mean()
                r = np.dot(xm, ym) / (np.sqrt(np.dot(xm, xm)) * np.sqrt(np.dot(ym, ym)) + 1e-10)
                n = len(x)
                if abs(r) >= 1: r = 0.999
                t_stat = r * math.sqrt(n - 2) / math.sqrt(1 - r**2 + 1e-10)
                # Two-tailed p-value from t-distribution approximation
                p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2))))
                p_value = max(0.0, min(1.0, p_value))
                return p_value > self.alpha
            else:
                # Partial correlation test
                cond_vars = [var_i, var_j] + list(cond_set)
                sub = data[cond_vars].dropna()
                if len(sub) < len(cond_vars) + 5:
                    return False
                try:
                    corr_mat = np.corrcoef(sub.T)
                    if corr_mat.shape[0] < 3:
                        return False
                    prec = np.linalg.inv(corr_mat)
                    n = len(sub)
                    pcorr = -prec[0, 1] / math.sqrt(abs(prec[0, 0] * prec[1, 1]) + 1e-10)
                    if abs(pcorr) >= 1: pcorr = 0.0
                    z = 0.5 * math.log((1 + pcorr) / (1 - pcorr + 1e-10))
                    se = 1.0 / math.sqrt(n - len(cond_vars) - 3 + 1e-10)
                    z_stat = z / se if se > 1e-10 else 0.0
                    p_value = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z_stat) / math.sqrt(2))))
                    p_value = max(0.0, min(1.0, p_value))
                    return p_value > self.alpha
                except Exception:
                    return False
        except Exception:
            return False


    def _orient_edges(self, variables):
        n_vars = len(variables)
        
        for k in range(n_vars):
            neighbors = self._get_neighbors(k, -1)
            if len(neighbors) < 2:
                continue
            
            for i, j in combinations(neighbors, 2):
                if self.graph[i, j] == 0:
                    if (i, j) in self.sepsets:
                        if k not in self.sepsets[(i, j)]:
                            self.graph[k, i] = 0
                            self.graph[k, j] = 0
                    else:
                        self.graph[k, i] = 0
                        self.graph[k, j] = 0
        
        for i in range(n_vars):
            for j in range(n_vars):
                if self.graph[i, j] == 1 and self.graph[j, i] == 0:
                    for k in range(n_vars):
                        if k != i and self.graph[j, k] == 1 and self.graph[k, j] == 1:
                            pass


def build_causal_graph(data, alpha=0.05):
    print("=" * 60)
    print("开始PC算法因果发现")
    print("=" * 60)
    
    pc = PCAlgorithm(alpha=alpha)
    graph, edges = pc.fit(data)
    
    print("\n" + "=" * 60)
    print("因果图构建完成")
    print("=" * 60)
    print(f"\n发现 {len(edges)} 条因果边")
    
    return graph, edges, pc


def identify_key_paths(edges, outcome_var='efficacy_score'):
    key_paths = []
    
    for cause, effect in edges:
        if outcome_var in [cause, effect]:
            path_type = '直接效应' if effect == outcome_var else '间接效应'
            key_paths.append({
                'from': cause,
                'to': effect,
                'path_type': path_type
            })
    
    return key_paths


def visualize_causal_graph(edges, variables, output_path=None):
    print("\n" + "=" * 60)
    print("因果图结构")
    print("=" * 60)
    
    adj = {v: [] for v in variables}
    for e in edges:
        adj[e[0]].append(e[1])
    
    for var in variables:
        if adj[var]:
            print(f"{var} -> {', '.join(adj[var])}")
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("因果图边列表:\n")
            for var in variables:
                if adj[var]:
                    f.write(f"{var} -> {', '.join(adj[var])}\n")
        print(f"\n因果图已保存至: {output_path}")


if __name__ == '__main__':
    from data_simulator import generate_tcm_dataset, add_heterogeneous_subgroups
    
    print("测试PC算法因果发现")
    print("-" * 40)
    
    df = generate_tcm_dataset(500)
    df = add_heterogeneous_subgroups(df)
    
    feature_cols = ['syn_1_qixu', 'syn_2_xueyu', 'syn_3_shire', 'syn_4_yinxu',
                    'herb_1_jun', 'herb_2_chen', 'herb_3_zuo', 'herb_4_shi',
                    'efficacy_score']
    
    data = df[feature_cols]
    graph, edges, pc = build_causal_graph(data, alpha=0.05)
    
    key_paths = identify_key_paths(edges)
    print("\n关键因果路径:")
    for p in key_paths:
        print(f"  {p['from']} -> {p['to']} ({p['path_type']})")
