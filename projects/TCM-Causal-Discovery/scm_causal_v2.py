"""
Target 039: SCM Causal Inference v2 - Structural Causal Model for TCM
Counterfactual inference for "证-治" (Syndrome-Treatment) causal effect estimation.

Core: Pearl's SCM + Do-calculus for TCM treatment effect estimation.
Estimands: ATE, CATE, ITE, Counterfactual (what-if analysis).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# 1. Causal Graph (DAG)
# ═══════════════════════════════════════════════════════════
class CausalGraph:
    """
    Directed Acyclic Graph (DAG) for TCM causal structure.

    Nodes: {X_i} — syndrome features, treatment, outcome, confounders
    Edges: X_i → X_j means X_i causally affects X_j

    TCM-specific structure:
      体质 → 证候 → 治法 → 方药 → 疗效
      ↑          ↑         ↑
      年龄/性别  生活习惯  配伍
    """
    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[Tuple[str, str]] = []  # (cause, effect)
        self.adj: Dict[str, List[str]] = defaultdict(list)       # forward
        self.parents: Dict[str, List[str]] = defaultdict(list)    # backward

    def add_node(self, name: str, node_type: str, **kwargs):
        """Add node. node_type: 'syndrome'|'treatment'|'outcome'|'confounder'|'feature'"""
        self.nodes[name] = {'type': node_type, **kwargs}

    def add_edge(self, cause: str, effect: str):
        """Add directed edge cause → effect."""
        self.edges.append((cause, effect))
        self.adj[cause].append(effect)
        self.parents[effect].append(cause)

    def get_parents(self, node: str) -> List[str]:
        return self.parents.get(node, [])

    def get_children(self, node: str) -> List[str]:
        return self.adj.get(node, [])

    def get_ancestors(self, node: str) -> Set[str]:
        """All ancestors of node (transitive closure of parents)."""
        ancestors = set()
        stack = [node]
        while stack:
            n = stack.pop()
            for p in self.get_parents(n):
                if p not in ancestors:
                    ancestors.add(p)
                    stack.append(p)
        return ancestors

    def get_descendants(self, node: str) -> Set[str]:
        descendants = set()
        stack = [node]
        while stack:
            n = stack.pop()
            for c in self.get_children(n):
                if c not in descendants:
                    descendants.add(c)
                    stack.append(c)
        return descendants

    def is_d_separated(self, x: str, y: str, z: Set[str]) -> bool:
        """
        Check d-separation: X ⊥ Y | Z
        Simplified: check if all paths from X to Y are blocked by Z.
        """
        # BFS to find all paths
        paths = self._find_all_paths(x, y)
        for path in paths:
            if not self._is_path_blocked(path, z):
                return False
        return True

    def _find_all_paths(self, start: str, end: str, max_depth: int = 10) -> List[List[str]]:
        """Find all simple paths from start to end."""
        paths = []
        stack = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) > max_depth:
                continue
            if node == end:
                paths.append(path)
                continue
            for neighbor in self.adj.get(node, []):
                if neighbor not in path:
                    stack.append((neighbor, path + [neighbor]))
            for parent in self.parents.get(node, []):
                if parent not in path:
                    stack.append((parent, path + [parent]))
        return paths

    def _is_path_blocked(self, path: List[str], z: Set[str]) -> bool:
        """Check if a path is blocked by conditioning set Z."""
        for i in range(1, len(path) - 1):
            prev, curr, nxt = path[i-1], path[i], path[i+1]
            # Collider: prev → curr ← nxt
            if curr in self.parents.get(prev, []) and curr in self.parents.get(nxt, []):
                if curr not in z and not self.get_descendants(curr).intersection(z):
                    return True  # blocked by unconditioned collider
            else:
                if curr in z:
                    return True  # blocked by conditioning on non-collider
        return False

    def build_tcm_dag(self):
        """Build standard TCM causal DAG."""
        # Confounders
        for c in ['年龄', '性别', '体质基础', '生活习惯']:
            self.add_node(c, 'confounder')

        # Syndrome features
        for s in ['气虚程度', '血瘀程度', '湿热程度', '阴虚程度', '阳虚程度']:
            self.add_node(s, 'syndrome')

        # Treatment
        for t in ['补气方', '活血方', '清热方', '滋阴方', '温阳方']:
            self.add_node(t, 'treatment')

        # Outcome
        self.add_node('疗效评分', 'outcome')

        # Edges: confounders → syndrome
        for c in ['年龄', '体质基础']:
            for s in ['气虚程度', '血瘀程度', '湿热程度']:
                self.add_edge(c, s)

        # Edges: syndrome → treatment (treatment selection depends on syndrome)
        self.add_edge('气虚程度', '补气方')
        self.add_edge('血瘀程度', '活血方')
        self.add_edge('湿热程度', '清热方')
        self.add_edge('阴虚程度', '滋阴方')
        self.add_edge('阳虚程度', '温阳方')

        # Edges: treatment → outcome
        for t in ['补气方', '活血方', '清热方', '滋阴方', '温阳方']:
            self.add_edge(t, '疗效评分')

        # Edges: syndrome → outcome (prognosis)
        for s in ['气虚程度', '血瘀程度', '湿热程度', '阴虚程度', '阳虚程度']:
            self.add_edge(s, '疗效评分')

        # Edges: confounders → treatment (indication)
        self.add_edge('生活习惯', '补气方')
        self.add_edge('体质基础', '活血方')

        return self


# ═══════════════════════════════════════════════════════════
# 2. Structural Causal Model (SCM)
# ═══════════════════════════════════════════════════════════
class StructuralCausalModel:
    """
    SCM with functional equations for each variable.

    X_i = f_i(Pa(X_i), U_i)

    where Pa(X_i) = parents in DAG, U_i = exogenous noise.

    Supports:
      - Observational distribution P(X)
      - Interventional distribution P(X | do(T=t))
      - Counterfactual distribution P(Y_{T=t'} | X, T=t)
    """
    def __init__(self, graph: CausalGraph):
        self.graph = graph
        self.functions: Dict[str, nn.Module] = {}  # structural equations
        self.noise_dists: Dict[str, Tuple[float, float]] = {}  # (mean, std)

    def add_structural_equation(
        self,
        node: str,
        func: nn.Module,
        noise_std: float = 0.1,
    ):
        """Add structural equation: node = f(parents) + noise"""
        self.functions[node] = func
        self.noise_dists[node] = (0.0, noise_std)

    def build_default_scm(self):
        """Build default linear SCM for TCM."""
        graph = self.graph

        for node_name, node_info in graph.nodes.items():
            parents = graph.get_parents(node_name)
            n_parents = len(parents)

            if node_info['type'] == 'confounder':
                # Exogenous: just noise
                self.add_structural_equation(
                    node_name,
                    nn.Identity(),
                    noise_std=1.0,
                )
            elif node_info['type'] == 'syndrome':
                self.add_structural_equation(
                    node_name,
                    nn.Linear(max(n_parents, 1), 1),
                    noise_std=0.2,
                )
            elif node_info['type'] == 'treatment':
                self.add_structural_equation(
                    node_name,
                    nn.Sequential(
                        nn.Linear(max(n_parents, 1), 16),
                        nn.ReLU(),
                        nn.Linear(16, 1),
                        nn.Sigmoid(),
                    ),
                    noise_std=0.1,
                )
            elif node_info['type'] == 'outcome':
                self.add_structural_equation(
                    node_name,
                    nn.Sequential(
                        nn.Linear(max(n_parents, 1), 32),
                        nn.ReLU(),
                        nn.Linear(32, 1),
                    ),
                    noise_std=0.3,
                )

    def sample_observational(
        self, n_samples: int = 1000
    ) -> Dict[str, torch.Tensor]:
        """Sample from observational distribution P(X)."""
        data = {}
        visited = set()

        # Topological sort (simplified: process by type order)
        order = ['confounder', 'syndrome', 'treatment', 'outcome']
        nodes_by_type = defaultdict(list)
        for name, info in self.graph.nodes.items():
            nodes_by_type[info['type']].append(name)

        for node_type in order:
            for node_name in nodes_by_type[node_type]:
                parents = self.graph.get_parents(node_name)
                noise_mean, noise_std = self.noise_dists.get(node_name, (0, 0.1))
                noise = torch.randn(n_samples, 1) * noise_std + noise_mean

                if not parents:
                    data[node_name] = noise
                else:
                    parent_values = torch.cat(
                        [data[p].reshape(n_samples, -1) for p in parents],
                        dim=-1
                    )
                    func = self.functions.get(node_name)
                    if func is not None:
                        data[node_name] = func(parent_values) + noise
                    else:
                        data[node_name] = noise

        return data

    def do_intervention(
        self,
        data: Dict[str, torch.Tensor],
        intervention_node: str,
        intervention_value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        do-calculus: P(X | do(T=t))
        Replace structural equation for intervention_node with constant t.
        """
        intervened = dict(data)
        intervened[intervention_node] = intervention_value

        # Propagate through descendants in topological order
        descendants = self.graph.get_descendants(intervention_node)
        order = ['syndrome', 'treatment', 'outcome']
        nodes_by_type = defaultdict(list)
        for name in descendants:
            nodes_by_type[self.graph.nodes[name]['type']].append(name)

        for node_type in order:
            for node_name in nodes_by_type[node_type]:
                if node_name == intervention_node:
                    continue
                parents = self.graph.get_parents(node_name)
                if not parents:
                    continue
                parent_values = torch.cat(
                    [intervened[p].reshape(intervened[p].shape[0], -1)
                     for p in parents if p in intervened],
                    dim=-1
                )
                func = self.functions.get(node_name)
                if func is not None:
                    n = parent_values.shape[0]
                    noise_mean, noise_std = self.noise_dists.get(node_name, (0, 0.1))
                    noise = torch.randn(n, 1) * noise_std + noise_mean
                    intervened[node_name] = func(parent_values) + noise

        return intervened

    def counterfactual(
        self,
        observed_data: Dict[str, torch.Tensor],
        intervention_node: str,
        counterfactual_value: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Counterfactual: "What would Y have been if T had been t', given that we observed X=x?"

        Three steps (Pearl's approach):
        1. Abduction: infer exogenous noise U from observed data
        2. Action: replace T equation with counterfactual value
        3. Prediction: propagate to get counterfactual outcome
        """
        # Step 1: Abduction — compute residuals (approximate U)
        inferred_noise = {}
        for node_name in observed_data:
            parents = self.graph.get_parents(node_name)
            if parents and node_name in self.functions:
                parent_values = torch.cat(
                    [observed_data[p].reshape(observed_data[p].shape[0], -1)
                     for p in parents],
                    dim=-1
                )
                predicted = self.functions[node_name](parent_values)
                inferred_noise[node_name] = observed_data[node_name] - predicted
            else:
                inferred_noise[node_name] = observed_data[node_name]

        # Step 2 & 3: Intervention + Prediction
        cf_data = self.do_intervention(
            observed_data, intervention_node, counterfactual_value
        )

        # Use inferred noise for outcome prediction
        outcome_name = '疗效评分'
        if outcome_name in cf_data:
            parents = self.graph.get_parents(outcome_name)
            parent_values = torch.cat(
                [cf_data[p].reshape(cf_data[p].shape[0], -1)
                 for p in parents if p in cf_data],
                dim=-1
            )
            func = self.functions.get(outcome_name)
            if func is not None:
                # Add inferred noise (abducted residual)
                cf_data[outcome_name] = func(parent_values) + inferred_noise.get(
                    outcome_name, torch.zeros_like(cf_data[outcome_name])
                )

        return cf_data


# ═══════════════════════════════════════════════════════════
# 3. Causal Effect Estimators
# ═══════════════════════════════════════════════════════════
class CausalEffectEstimator:
    """
    Estimate causal effects: ATE, CATE, ITE.

    ATE  = E[Y | do(T=1)] - E[Y | do(T=0)]
    CATE = E[Y | do(T=1), X=x] - E[Y | do(T=0), X=x]
    ITE  = Y_i(T=1) - Y_i(T=0) for individual i
    """
    def __init__(self, scm: StructuralCausalModel):
        self.scm = scm

    def estimate_ate(
        self,
        treatment_node: str,
        outcome_node: str = '疗效评分',
        n_samples: int = 5000,
    ) -> Dict[str, float]:
        """Average Treatment Effect."""
        # Sample under do(T=1) and do(T=0)
        obs_data = self.scm.sample_observational(n_samples)

        treated = self.scm.do_intervention(
            obs_data, treatment_node,
            torch.ones(n_samples, 1)
        )
        control = self.scm.do_intervention(
            obs_data, treatment_node,
            torch.zeros(n_samples, 1)
        )

        y1 = treated[outcome_node].mean().item()
        y0 = control[outcome_node].mean().item()
        ate = y1 - y0

        return {
            'ATE': ate,
            'E[Y|do(T=1)]': y1,
            'E[Y|do(T=0)]': y0,
            'n_samples': n_samples,
        }

    def estimate_cate(
        self,
        treatment_node: str,
        conditioning_vars: List[str],
        outcome_node: str = '疗效评分',
        n_samples: int = 5000,
    ) -> Dict[str, float]:
        """Conditional Average Treatment Effect."""
        obs_data = self.scm.sample_observational(n_samples)

        treated = self.scm.do_intervention(
            obs_data, treatment_node,
            torch.ones(n_samples, 1)
        )
        control = self.scm.do_intervention(
            obs_data, treatment_node,
            torch.zeros(n_samples, 1)
        )

        # Stratify by conditioning variables (simplified: median split)
        cate_results = {}
        for var in conditioning_vars:
            if var not in obs_data:
                continue
            median_val = obs_data[var].median()
            high_mask = (obs_data[var].squeeze() > median_val)
            low_mask = ~high_mask

            y1_high = treated[outcome_node][high_mask].mean().item()
            y0_high = control[outcome_node][high_mask].mean().item()
            y1_low = treated[outcome_node][low_mask].mean().item()
            y0_low = control[outcome_node][low_mask].mean().item()

            cate_results[f'CATE_{var}_high'] = y1_high - y0_high
            cate_results[f'CATE_{var}_low'] = y1_low - y0_low

        return cate_results

    def estimate_ite(
        self,
        treatment_node: str,
        outcome_node: str = '疗效评分',
        n_samples: int = 1000,
    ) -> torch.Tensor:
        """Individual Treatment Effect (via counterfactual)."""
        obs_data = self.scm.sample_observational(n_samples)

        # Counterfactual: what if T had been 1 instead of 0?
        cf_treated = self.scm.counterfactual(
            obs_data, treatment_node, torch.ones(n_samples, 1)
        )
        # Counterfactual: what if T had been 0 instead of 1?
        cf_control = self.scm.counterfactual(
            obs_data, treatment_node, torch.zeros(n_samples, 1)
        )

        ite = cf_treated[outcome_node] - cf_control[outcome_node]
        return ite


# ═══════════════════════════════════════════════════════════
# 4. Propensity Score Model
# ═══════════════════════════════════════════════════════════
class PropensityScoreModel(nn.Module):
    """
    Propensity score e(X) = P(T=1|X) for IPW estimation.
    In TCM context: probability of receiving a treatment given syndrome features.
    """
    def __init__(self, n_features: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)

    def ipw_estimate(
        self,
        features: torch.Tensor,
        treatment: torch.Tensor,
        outcome: torch.Tensor,
    ) -> float:
        """
        Inverse Probability Weighting estimator for ATE.
        ATE = E[Y*T/e(X)] - E[Y*(1-T)/(1-e(X))]
        """
        propensity = self.forward(features).detach().clamp(0.01, 0.99)
        e = propensity.squeeze()

        w1 = treatment.squeeze() / e
        w0 = (1 - treatment.squeeze()) / (1 - e)

        ate = (outcome.squeeze() * w1).mean() - (outcome.squeeze() * w0).mean()
        return ate.item()


# ═══════════════════════════════════════════════════════════
# 5. Quick Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Target 039: SCM Causal Inference v2 - TCM Treatment Effects")
    print("=" * 60)

    # Build causal graph
    graph = CausalGraph()
    graph.build_tcm_dag()

    print(f"\nCausal Graph:")
    print(f"  Nodes: {len(graph.nodes)}")
    print(f"  Edges: {len(graph.edges)}")
    print(f"  Node types: {set(n['type'] for n in graph.nodes.values())}")

    # Build SCM
    scm = StructuralCausalModel(graph)
    scm.build_default_scm()

    # Sample observational data
    print("\nSampling observational distribution...")
    obs = scm.sample_observational(1000)
    print(f"  Variables: {list(obs.keys())[:8]}...")
    print(f"  疗效评分 mean: {obs['疗效评分'].mean().item():.4f}")

    # Estimate ATE
    print("\nEstimating ATE (补气方 → 疗效评分)...")
    estimator = CausalEffectEstimator(scm)
    ate_result = estimator.estimate_ate('补气方', n_samples=2000)
    print(f"  ATE = {ate_result['ATE']:.4f}")
    print(f"  E[Y|do(T=1)] = {ate_result['E[Y|do(T=1)]']:.4f}")
    print(f"  E[Y|do(T=0)] = {ate_result['E[Y|do(T=0)]']:.4f}")

    # Estimate CATE
    print("\nEstimating CATE stratified by 气虚程度...")
    cate_result = estimator.estimate_cate('补气方', ['气虚程度'], n_samples=2000)
    for k, v in cate_result.items():
        print(f"  {k}: {v:.4f}")

    # Counterfactual
    print("\nCounterfactual: What if treatment had been different?")
    obs_sample = {k: v[:5] for k, v in obs.items()}
    cf = scm.counterfactual(obs_sample, '补气方', torch.ones(5, 1))
    print(f"  Observed outcome: {obs_sample['疗效评分'].squeeze().tolist()}")
    print(f"  Counterfactual (do(补气方=1)): {cf['疗效评分'].squeeze().tolist()}")

    # d-separation test
    print("\nD-separation tests:")
    print(f"  年龄 ⊥ 疗效评分 | {{气虚程度}}: "
          f"{graph.is_d_separated('年龄', '疗效评分', {'气虚程度'})}")
    print(f"  年龄 ⊥ 补气方 | {{气虚程度}}: "
          f"{graph.is_d_separated('年龄', '补气方', {'气虚程度'})}")

    # Propensity score
    print("\nPropensity Score IPW test:")
    ps_model = PropensityScoreModel(n_features=5)
    features = torch.randn(500, 5)
    treatment = (torch.rand(500) > 0.5).float().unsqueeze(1)
    outcome = torch.randn(500, 1) + treatment * 0.5
    ipw_ate = ps_model.ipw_estimate(features, treatment, outcome)
    print(f"  IPW ATE estimate: {ipw_ate:.4f}")

    print("\nDone!")
