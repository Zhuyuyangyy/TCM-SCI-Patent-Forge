"""
B-012: Multi-Modal Fusion v2 - Tongue + Voice + Face → Constitution
三模态融合体质识别：舌象+声纹+面色 → 9类体质

Innovation: Independent sub-networks per modality + cross-modal attention fusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple


# ═══════════════════════════════════════════════════════════
# 1. Modality-Specific Encoders
# ═══════════════════════════════════════════════════════════
class TongueEncoder(nn.Module):
    """舌象编码器: 舌色+舌苔+舌形 → feature vector"""
    def __init__(self, input_dim=64, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class VoiceEncoder(nn.Module):
    """声纹编码器: 基频+共振峰+音色 → feature vector"""
    def __init__(self, input_dim=128, hidden_dim=128, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class FaceEncoder(nn.Module):
    """面色编码器: 五色分布+光泽度 → feature vector"""
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


# ═══════════════════════════════════════════════════════════
# 2. Cross-Modal Attention Fusion
# ═══════════════════════════════════════════════════════════
class CrossModalAttention(nn.Module):
    """
    Cross-modal attention: each modality attends to others.

    Q = W_q * h_i (query from modality i)
    K = W_k * [h_j for j != i] (keys from other modalities)
    V = W_v * [h_j for j != i] (values from other modalities)

    output_i = softmax(Q * K^T / sqrt(d)) * V
    """
    def __init__(self, feature_dim=64, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

    def forward(self, query_feat, context_feats):
        """
        query_feat: (B, D) - single modality
        context_feats: (B, N, D) - other modalities stacked
        """
        Q = self.q_proj(query_feat).unsqueeze(1)  # (B, 1, D)
        K = self.k_proj(context_feats)             # (B, N, D)
        V = self.v_proj(context_feats)             # (B, N, D)

        # Reshape for multi-head
        B = Q.shape[0]
        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        attn = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, V)  # (B, H, 1, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, -1)
        return self.out_proj(out).squeeze(1)


# ═══════════════════════════════════════════════════════════
# 3. Multi-Modal Constitution Classifier
# ═══════════════════════════════════════════════════════════
class MultiModalConstitutionClassifier(nn.Module):
    """
    Full pipeline: Tongue + Voice + Face → Constitution (9 classes)

    Supports:
    - Full fusion (all 3 modalities)
    - Partial fusion (any 2 modalities)
    - Single modality prediction
    """
    CONSTITUTIONS = [
        '平和质', '气虚质', '阳虚质', '阴虚质',
        '痰湿质', '湿热质', '血瘀质', '气郁质', '特禀质'
    ]

    def __init__(self, tongue_dim=64, voice_dim=128, face_dim=32, feature_dim=64):
        super().__init__()
        self.tongue_enc = TongueEncoder(tongue_dim, 128, feature_dim)
        self.voice_enc = VoiceEncoder(voice_dim, 128, feature_dim)
        self.face_enc = FaceEncoder(face_dim, 64, feature_dim)

        # Cross-modal attention (query from one modality attends to others)
        self.cross_attn = CrossModalAttention(feature_dim)

        # Fusion classifier (takes concatenated attended features)
        self.fusion_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 3),
            nn.Sigmoid(),
        )
        self.fusion_classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 9),
        )

        # Single-modality classifiers
        self.tongue_cls = nn.Linear(feature_dim, 9)
        self.voice_cls = nn.Linear(feature_dim, 9)
        self.face_cls = nn.Linear(feature_dim, 9)

    def forward(self, tongue=None, voice=None, face=None):
        """
        Supports flexible modality input:
        - All 3: full fusion
        - Any 1-2: partial with available modalities
        """
        features = {}

        if tongue is not None:
            features['tongue'] = self.tongue_enc(tongue)
        if voice is not None:
            features['voice'] = self.voice_enc(voice)
        if face is not None:
            features['face'] = self.face_enc(face)

        if len(features) == 0:
            raise ValueError("At least one modality required")

        # Ensure all 3 modalities present (zero-pad missing ones)
        B = list(features.values())[0].shape[0]
        D = list(features.values())[0].shape[1]
        zero = torch.zeros(B, D, device=list(features.values())[0].device)
        all_feats = {
            'tongue': features.get('tongue', zero),
            'voice': features.get('voice', zero),
            'face': features.get('face', zero),
        }

        # Single modality shortcut
        if len(features) == 1:
            name, feat = list(features.items())[0]
            logits = getattr(self, f'{name}_cls')(feat)
            return {'logits': logits, 'features': features, 'fusion_type': 'single'}

        # Multi-modal: cross-attention fusion
        attended = []
        for name, feat in all_feats.items():
            others = [f for n, f in all_feats.items() if n != name]
            others_ctx = torch.stack(others, dim=1)  # (B, 2, D)
            attended_feat = self.cross_attn(feat, others_ctx)  # (B, D)
            attended.append(attended_feat)

        # Gated fusion
        fused = torch.cat(attended, dim=-1)  # (B, D*3)
        gate = self.fusion_gate(fused)
        fused = fused * gate
        logits = self.fusion_classifier(fused)

        return {
            'logits': logits,
            'features': features,
            'fused': fused,
            'fusion_type': f'{len(features)}-modal',
        }

    def predict(self, tongue=None, voice=None, face=None):
        out = self.forward(tongue, voice, face)
        probs = F.softmax(out['logits'], dim=-1)
        pred_idx = probs.argmax(dim=-1)
        pred_names = [self.CONSTITUTIONS[i] for i in pred_idx]
        confidences = probs.max(dim=-1).values
        return pred_names, confidences, probs


# ═══════════════════════════════════════════════════════════
# 4. Training Pipeline
# ═══════════════════════════════════════════════════════════
class MultiModalTrainer:
    def __init__(self, model, lr=1e-3):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=100)

    def train_step(self, tongue, voice, face, labels):
        self.model.train()
        self.optimizer.zero_grad()

        out = self.model(tongue, voice, face)
        loss = F.cross_entropy(out['logits'], labels)

        # Auxiliary losses for single-modality branches
        if tongue is not None:
            t_logits = self.model.tongue_cls(self.model.tongue_enc(tongue))
            loss += 0.3 * F.cross_entropy(t_logits, labels)
        if voice is not None:
            v_logits = self.model.voice_cls(self.model.voice_enc(voice))
            loss += 0.3 * F.cross_entropy(v_logits, labels)
        if face is not None:
            f_logits = self.model.face_cls(self.model.face_enc(face))
            loss += 0.3 * F.cross_entropy(f_logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        pred = out['logits'].argmax(dim=-1)
        acc = (pred == labels).float().mean().item()
        return {'loss': loss.item(), 'acc': acc}


# ═══════════════════════════════════════════════════════════
# 5. Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("B-012: Multi-Modal Constitution Classifier v2")
    print("=" * 60)

    B = 8
    tongue = torch.randn(B, 64)
    voice = torch.randn(B, 128)
    face = torch.randn(B, 32)
    labels = torch.randint(0, 9, (B,))

    model = MultiModalConstitutionClassifier()
    trainer = MultiModalTrainer(model)

    # Train
    for epoch in range(20):
        metrics = trainer.train_step(tongue, voice, face, labels)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: loss={metrics['loss']:.4f}, acc={metrics['acc']:.2%}")

    # Full fusion prediction
    names, confs, probs = model.predict(tongue, voice, face)
    print(f"\n[3-modal] Prediction: {names[0]} (conf={confs[0]:.4f})")

    # Single modality
    names_t, confs_t, _ = model.predict(tongue=tongue)
    print(f"[Tongue-only] Prediction: {names_t[0]} (conf={confs_t[0]:.4f})")

    # 2-modal
    names_tv, confs_tv, _ = model.predict(tongue=tongue, voice=voice)
    print(f"[Tongue+Voice] Prediction: {names_tv[0]} (conf={confs_tv[0]:.4f})")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("Done!")
