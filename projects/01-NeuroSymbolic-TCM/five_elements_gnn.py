"""
Target 003: Five-Elements GNN for Syndrome Transition Modeling
五行生克约束的图神经网络证候转移概率建模

Innovation:
  - 相生关系 → 正向激活边 (Wood→Fire→Earth→Metal→Water→Wood)
  - 相克关系 → 抑制边 (Wood→Earth→Water→Fire→Metal→Wood)
  - GNN消息传递遵循五行传播规则
  - 可学习的转移强度权重
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict


# ═══════════════════════════════════════════════════════════
# 1. Five-Elements Graph Construction
# ═══════════════════════════════════════════════════════════
class FiveElementsGraph:
    """
    TCM Five-Elements (五行) knowledge graph.

    相生 (Generating/Promoting):
      木(Wood)→火(Fire)→土(Earth)→金(Metal)→水(Water)→木(Wood)

    相克 (Controlling/Restraining):
      木(Wood)→土(Earth)→水(Water)→火(Fire)→金(Metal)→木(Wood)

    乘 (Over-acting): 相克太过
    侮 (Counter-acting): 反向克制
    """
    ELEMENTS = ['木', '火', '土', '金', '水']

    SHENG = {  # 相生: key generates value
        '木': '火', '火': '土', '土': '金', '金': '水', '水': '木'
    }
    KE = {  # 相克: key controls value
        '木': '土', '土': '水', '水': '火', '火': '金', '金': '木'
    }

    # 脏腑-五行映射
    ORGAN_ELEMENT = {
        '肝': '木', '胆': '木',
        '心': '火', '小肠': '火',
        '脾': '土', '胃': '土',
        '肺': '金', '大肠': '金',
        '肾': '水', '膀胱': '水',
    }

    def __init__(self):
        self.nodes: Dict[str, dict] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (src, relation, dst)
        self.adj_sheng: Dict[str, List[str]] = defaultdict(list)
        self.adj_ke: Dict[str, List[str]] = defaultdict(list)

    def add_syndrome(self, name: str, element: str, **kwargs):
        """Add syndrome node with five-element attribute."""
        self.nodes[name] = {'element': element, 'type': 'syndrome', **kwargs}

    def add_symptom(self, name: str, **kwargs):
        self.nodes[name] = {'type': 'symptom', **kwargs}

    def add_prescription(self, name: str, composition: str, **kwargs):
        self.nodes[name] = {'type': 'prescription', 'composition': composition, **kwargs}

    def add_sheng_edge(self, src: str, dst: str, weight: float = 1.0):
        """Add 相生 (generating) edge."""
        self.edges.append((src, '相生', dst))
        self.adj_sheng[src].append(dst)

    def add_ke_edge(self, src: str, dst: str, weight: float = 1.0):
        """Add 相克 (restraining) edge."""
        self.edges.append((src, '相克', dst))
        self.adj_ke[src].append(dst)

    def build_default_graph(self):
        """Build default TCM five-elements syndrome graph."""
        # Syndrome nodes
        syndromes = [
            ('肝郁气滞', '木'), ('肝郁化火', '木'), ('肝阳上亢', '木'),
            ('心火亢盛', '火'), ('心脾两虚', '火'), ('心肾不交', '火'),
            ('脾胃虚弱', '土'), ('脾胃湿热', '土'), ('脾虚湿盛', '土'),
            ('肺气虚', '金'), ('痰湿蕴肺', '金'), ('肺肾阴虚', '金'),
            ('肾阳虚', '水'), ('肾阴虚', '水'), ('肾精不足', '水'),
        ]
        for name, elem in syndromes:
            self.add_syndrome(name, elem)

        # Symptom nodes
        symptoms = [
            '胁肋胀痛', '情绪抑郁', '嗳气频繁', '脉弦',
            '心悸失眠', '口舌生疮', '舌尖红',
            '食欲不振', '腹胀便溏', '舌淡胖',
            '咳嗽痰多', '气喘', '胸闷',
            '腰膝酸软', '耳鸣', '夜尿多', '脉沉',
        ]
        for s in symptoms:
            self.add_symptom(s)

        # Prescription nodes
        prescriptions = [
            ('逍遥散', '柴胡、当归、白芍、白术、茯苓、甘草'),
            ('柴胡疏肝散', '柴胡、陈皮、川芎、香附、枳壳、芍药、甘草'),
            ('归脾汤', '人参、白术、茯苓、甘草、当归、黄芪、龙眼肉'),
            ('六味地黄丸', '熟地黄、山茱萸、山药、泽泻、茯苓、丹皮'),
            ('二陈汤', '半夏、陈皮、茯苓、甘草'),
        ]
        for name, comp in prescriptions:
            self.add_prescription(name, comp)

        # 相生 edges (syndrome transitions along generating cycle)
        self.add_sheng_edge('肝郁气滞', '心火亢盛')  # 木生火
        self.add_sheng_edge('心火亢盛', '脾胃湿热')  # 火生土
        self.add_sheng_edge('脾胃虚弱', '肺气虚')    # 土生金
        self.add_sheng_edge('肺气虚', '肾阳虚')      # 金生水
        self.add_sheng_edge('肾阳虚', '肝郁气滞')    # 水生木

        # 相克 edges (syndrome transitions along restraining cycle)
        self.add_ke_edge('肝郁气滞', '脾胃虚弱')    # 木克土
        self.add_ke_edge('脾胃虚弱', '肾阳虚')      # 土克水 (脾虚及肾)
        self.add_ke_edge('肾阳虚', '心火亢盛')      # 水克火 (水饮凌心)
        self.add_ke_edge('心火亢盛', '肺气虚')      # 火克金 (火灼肺金)
        self.add_ke_edge('肺气虚', '肝郁气滞')      # 金克木 (肺虚肝旺)

        return self

    def get_element(self, node: str) -> Optional[str]:
        info = self.nodes.get(node)
        if info:
            return info.get('element')
        # Try organ mapping
        return self.ORGAN_ELEMENT.get(node)


# ═══════════════════════════════════════════════════════════
# 2. Five-Elements GNN Layer
# ═══════════════════════════════════════════════════════════
class FiveElementsGNNLayer(nn.Module):
    """
    Custom GNN layer that respects five-elements constraints.

    Message passing:
      - 相生 edge: positive message (promotion)
      - 相克 edge: negative message (restraint)
      - Self-loop: identity

    h_i' = σ(W_self * h_i + Σ_{j∈Sheng(i)} α_sheng * W_sheng * h_j
                     + Σ_{j∈Ke(i)} α_ke * W_ke * h_j + b)
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.W_self = nn.Linear(in_dim, out_dim)
        self.W_sheng = nn.Linear(in_dim, out_dim)  # 相生 message
        self.W_ke = nn.Linear(in_dim, out_dim)     # 相克 message

        # Learnable edge weights
        self.alpha_sheng = nn.Parameter(torch.tensor(0.5))
        self.alpha_ke = nn.Parameter(torch.tensor(-0.3))  # negative = restraint

        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        node_features: torch.Tensor,  # (N, in_dim)
        sheng_adj: torch.Tensor,      # (N, N) binary adjacency for 相生
        ke_adj: torch.Tensor,         # (N, N) binary adjacency for 相克
    ) -> torch.Tensor:
        """Forward pass with five-elements message passing."""
        # Self transformation
        h_self = self.W_self(node_features)

        # 相生 messages (positive influence)
        h_sheng = self.W_sheng(node_features)
        m_sheng = torch.matmul(sheng_adj, h_sheng) * torch.sigmoid(self.alpha_sheng)

        # 相克 messages (negative influence)
        h_ke = self.W_ke(node_features)
        m_ke = torch.matmul(ke_adj, h_ke) * torch.sigmoid(self.alpha_ke)

        # Combine
        h = h_self + m_sheng + m_ke
        h = self.norm(h)
        h = F.gelu(h)
        h = self.dropout(h)

        return h


# ═══════════════════════════════════════════════════════════
# 3. Five-Elements Syndrome Transition GNN
# ═══════════════════════════════════════════════════════════
class FiveElementsSyndromeGNN(nn.Module):
    """
    Full GNN model for syndrome transition prediction.

    Input:  Current syndrome node features + symptom evidence
    Output: Transition probability distribution over all syndromes

    Architecture:
      - Symptom encoder: symptom text → feature vector
      - Node embedding: syndrome attributes → node features
      - 3x FiveElementsGNNLayer: message passing with 生克 constraints
      - Transition head: predict next syndrome probabilities
    """
    def __init__(
        self,
        num_syndromes: int = 15,
        num_symptoms: int = 20,
        symptom_dim: int = 32,
        hidden_dim: int = 64,
        num_layers: int = 3,
    ):
        super().__init__()

        # Symptom encoder
        self.symptom_embedding = nn.Embedding(num_symptoms, symptom_dim)

        # Node feature initialization
        # 5 element one-hot + 3 type one-hot + symptom_dim
        node_input_dim = 5 + 3 + symptom_dim
        self.node_init = nn.Linear(node_input_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            FiveElementsGNNLayer(hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])

        # Transition prediction head
        self.transition_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_syndromes),
        )

        # Confidence estimation
        self.confidence_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self.num_syndromes = num_syndromes

    def forward(
        self,
        node_features: torch.Tensor,  # (N, node_input_dim)
        sheng_adj: torch.Tensor,      # (N, N)
        ke_adj: torch.Tensor,         # (N, N)
        current_idx: int,             # index of current syndrome node
        symptom_ids: torch.Tensor,    # (S,) symptom token IDs
    ) -> Dict[str, torch.Tensor]:
        """
        Predict syndrome transition probabilities.
        """
        # Initialize node features
        h = self.node_init(node_features)  # (N, hidden)

        # Add symptom evidence to current syndrome node
        if len(symptom_ids) > 0:
            symptom_emb = self.symptom_embedding(symptom_ids).mean(dim=0)  # (symptom_dim,)
            # Inject symptom info into current syndrome node
            h[current_idx, :symptom_emb.shape[0]] += symptom_emb

        # GNN message passing
        for gnn_layer in self.gnn_layers:
            h = gnn_layer(h, sheng_adj, ke_adj)

        # Extract current syndrome representation
        current_repr = h[current_idx]  # (hidden,)

        # Predict transition to all syndromes
        h_expanded = h.expand(self.num_syndromes, -1)[:h.shape[0]]
        transition_input = torch.cat([current_repr.unsqueeze(0).expand(h_expanded.shape[0], -1), h_expanded], dim=-1)
        transition_logits = self.transition_head(transition_input)  # (N, num_syndromes)

        # Confidence
        confidence = self.confidence_head(current_repr)

        return {
            'transition_logits': transition_logits,
            'transition_probs': F.softmax(transition_logits.mean(dim=0), dim=-1),
            'node_embeddings': h,
            'confidence': confidence,
            'current_repr': current_repr,
        }


# ═══════════════════════════════════════════════════════════
# 4. Five-Elements Transition Loss
# ═══════════════════════════════════════════════════════════
class FiveElementsTransitionLoss(nn.Module):
    """
    Loss = L_transition + λ1 * L_element_constraint + λ2 * L_sheng_consistency

    L_transition: cross-entropy for next syndrome prediction
    L_element_constraint: penalize transitions violating 生克 rules
    L_sheng_consistency: encourage 相生 transitions, penalize 逆相克
    """
    def __init__(
        self,
        graph: FiveElementsGraph,
        lambda_element: float = 0.5,
        lambda_sheng: float = 0.3,
    ):
        super().__init__()
        self.graph = graph
        self.lambda_element = lambda_element
        self.lambda_sheng = lambda_sheng

    def forward(
        self,
        transition_probs: torch.Tensor,  # (num_syndromes,)
        target_syndrome_idx: int,
        syndrome_names: List[str],
    ) -> Dict[str, torch.Tensor]:

        # L_transition: standard cross-entropy
        target = torch.tensor([target_syndrome_idx], device=transition_probs.device)
        L_transition = F.cross_entropy(
            transition_probs.unsqueeze(0), target
        )

        # L_element_constraint: penalize invalid transitions
        current_elem = self.graph.get_element(syndrome_names[0])  # simplified
        element_penalty = torch.tensor(0.0, device=transition_probs.device)

        for i, syn_name in enumerate(syndrome_names):
            target_elem = self.graph.get_element(syn_name)
            if current_elem and target_elem:
                # Check if transition is valid (相生 or 相克)
                is_sheng = self.graph.SHENG.get(current_elem) == target_elem
                is_ke = self.graph.KE.get(current_elem) == target_elem
                is_same = current_elem == target_elem

                if not (is_sheng or is_ke or is_same):
                    # Invalid transition: penalize
                    element_penalty += transition_probs[i]

        L_element = element_penalty

        # L_sheng_consistency: reward 相生 transitions
        sheng_bonus = torch.tensor(0.0, device=transition_probs.device)
        for i, syn_name in enumerate(syndrome_names):
            target_elem = self.graph.get_element(syn_name)
            if current_elem and target_elem:
                if self.graph.SHENG.get(current_elem) == target_elem:
                    # 相生 transition should have higher probability
                    sheng_bonus -= transition_probs[i]  # negative = encourage

        L_sheng = sheng_bonus

        total = L_transition + self.lambda_element * L_element + self.lambda_sheng * L_sheng

        return {
            'total': total,
            'L_transition': L_transition,
            'L_element': L_element,
            'L_sheng': L_sheng,
        }


# ═══════════════════════════════════════════════════════════
# 5. Adjacency Matrix Builder
# ═══════════════════════════════════════════════════════════
def build_adjacency_matrices(
    graph: FiveElementsGraph,
    syndrome_names: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build sheng and ke adjacency matrices from graph."""
    n = len(syndrome_names)
    name_to_idx = {name: i for i, name in enumerate(syndrome_names)}

    sheng_adj = torch.zeros(n, n)
    ke_adj = torch.zeros(n, n)

    for src, rel, dst in graph.edges:
        if src in name_to_idx and dst in name_to_idx:
            i, j = name_to_idx[src], name_to_idx[dst]
            if rel == '相生':
                sheng_adj[i, j] = 1.0
            elif rel == '相克':
                ke_adj[i, j] = 1.0

    # Add self-loops
    sheng_adj += torch.eye(n)
    ke_adj += torch.eye(n)

    return sheng_adj, ke_adj


# ═══════════════════════════════════════════════════════════
# 6. Quick Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Target 003: Five-Elements GNN Syndrome Transition")
    print("=" * 60)

    # Build graph
    graph = FiveElementsGraph()
    graph.build_default_graph()

    print(f"\nGraph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
    print(f"相生 edges: {len(graph.adj_sheng)} sources")
    print(f"相克 edges: {len(graph.adj_ke)} sources")

    # Syndrome list
    syndrome_names = [n for n, info in graph.nodes.items() if info.get('type') == 'syndrome']
    print(f"Syndromes: {syndrome_names}")

    # Build adjacency
    sheng_adj, ke_adj = build_adjacency_matrices(graph, syndrome_names)
    print(f"\n相生 adjacency (non-zero): {sheng_adj.nonzero().shape[0]}")
    print(f"相克 adjacency (non-zero): {ke_adj.nonzero().shape[0]}")

    # Create model
    model = FiveElementsSyndromeGNN(
        num_syndromes=len(syndrome_names),
        num_symptoms=20,
        symptom_dim=32,
        hidden_dim=64,
        num_layers=3,
    )

    # Create node features (simplified: element one-hot + type one-hot + zeros)
    n = len(syndrome_names)
    element_to_idx = {e: i for i, e in enumerate(FiveElementsGraph.ELEMENTS)}
    type_to_idx = {'syndrome': 0, 'symptom': 1, 'prescription': 2}

    node_features = torch.zeros(n, 5 + 3 + 32)
    for i, name in enumerate(syndrome_names):
        elem = graph.get_element(name)
        if elem:
            node_features[i, element_to_idx[elem]] = 1.0
        node_features[i, 5 + 0] = 1.0  # type = syndrome

    # Symptom input
    symptom_ids = torch.tensor([0, 1, 2, 3])  # mock symptom IDs

    # Forward
    out = model(node_features, sheng_adj, ke_adj, current_idx=0, symptom_ids=symptom_ids)

    print(f"\nTransition probabilities from '{syndrome_names[0]}':")
    probs = out['transition_probs']
    for i, (name, prob) in enumerate(zip(syndrome_names, probs.tolist())):
        elem = graph.get_element(name)
        marker = " ←" if i == 0 else ""
        print(f"  {name} ({elem}): {prob:.4f}{marker}")

    print(f"\nConfidence: {out['confidence'].item():.4f}")
    print(f"Node embeddings shape: {out['node_embeddings'].shape}")

    # Loss test
    loss_fn = FiveElementsTransitionLoss(graph)
    target_idx = 3  # arbitrary target
    losses = loss_fn(probs, target_idx, syndrome_names)
    print(f"\nLosses:")
    for k, v in losses.items():
        print(f"  {k}: {v.item():.6f}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")

    # Visualize 生克 relations
    print(f"\n五行相生关系:")
    for elem, generated in FiveElementsGraph.SHENG.items():
        print(f"  {elem} → {generated}")
    print(f"\n五行相克关系:")
    for elem, controlled in FiveElementsGraph.KE.items():
        print(f"  {elem} → {controlled}")

    print("\nDone!")
