"""
A-006: AudioSense v2 - Voice-ZangFu Mapping via β-VAE
语音韵律特征 → 脏腑功能状态潜空间 → 五脏分类

Innovation: "听声辨证" — first VAE mapping voice prosody to TCM organ states
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════
# 1. Voice Feature Extractor (Multi-scale Prosody)
# ═══════════════════════════════════════════════════════════
class ProsodyFeatureExtractor(nn.Module):
    """
    Extract TCM-relevant voice features:
    - F0 (基频): 肝在声为呼, 心在声为笑
    - Formant (共振峰): 肺气宣发肃降
    - Energy entropy (音色熵): 咳嗽类型
    - Speech rate (语速): 脾虚湿盛→语速慢
    """
    def __init__(self, raw_input_dim: int = 1024, feature_dim: int = 128):
        super().__init__()
        # Multi-scale feature extraction
        self.f0_encoder = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)
        )  # 基频特征 → 16d
        self.formant_encoder = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32)
        )  # 共振峰特征 → 32d
        self.energy_encoder = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16)
        )  # 能量熵特征 → 16d
        self.rate_encoder = nn.Sequential(
            nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 8)
        )  # 语速特征 → 8d

        # Fusion
        self.fusion = nn.Sequential(
            nn.Linear(16 + 32 + 16 + 8, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, f0, formant, energy, rate) -> torch.Tensor:
        f0_feat = self.f0_encoder(f0)
        formant_feat = self.formant_encoder(formant)
        energy_feat = self.energy_encoder(energy)
        rate_feat = self.rate_encoder(rate)
        combined = torch.cat([f0_feat, formant_feat, energy_feat, rate_feat], dim=-1)
        return self.fusion(combined)


# ═══════════════════════════════════════════════════════════
# 2. β-VAE for Zang-Fu Latent Space
# ═══════════════════════════════════════════════════════════
class ZangFuVAE(nn.Module):
    """
    β-VAE: voice features → latent space → Zang-Fu classification

    Loss = L_recon + β * L_kl + L_cls

    β > 1 encourages disentangled latent representations,
    where each latent dimension may correspond to a specific organ function.
    """
    def __init__(
        self,
        input_dim: int = 128,
        latent_dim: int = 16,
        hidden_dim: int = 256,
        num_organs: int = 5,
        beta: float = 4.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.beta = beta

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        # Zang-Fu classification head (5 organs: 心肝脾肺肾)
        self.zangfu_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_organs),
        )

        # 七情 (7 emotions) auxiliary head
        self.emotion_head = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 7),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        zangfu_logits = self.zangfu_head(z)
        emotion_logits = self.emotion_head(z)
        return recon, mu, logvar, zangfu_logits, emotion_logits

    def loss_function(self, recon, x, mu, logvar, zangfu_logits, zangfu_labels,
                      emotion_logits=None, emotion_labels=None):
        # Reconstruction
        L_recon = F.mse_loss(recon, x, reduction='sum')
        # KL divergence
        L_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # Zang-Fu classification
        L_cls = F.cross_entropy(zangfu_logits, zangfu_labels)
        # Emotion auxiliary (optional)
        L_emo = torch.tensor(0.0, device=x.device)
        if emotion_logits is not None and emotion_labels is not None:
            L_emo = F.cross_entropy(emotion_logits, emotion_labels)

        total = L_recon + self.beta * L_kl + L_cls + 0.5 * L_emo
        return {
            'total': total,
            'L_recon': L_recon,
            'L_kl': L_kl,
            'L_cls': L_cls,
            'L_emo': L_emo,
        }

    def infer_zangfu(self, x):
        """Infer organ state probabilities."""
        with torch.no_grad():
            mu, _ = self.encode(x)
            logits = self.zangfu_head(mu)
            probs = F.softmax(logits, dim=-1)
        return probs, mu


# ═══════════════════════════════════════════════════════════
# 3. Conditional VAE (generate voice given syndrome)
# ═══════════════════════════════════════════════════════════
class ConditionalZangFuVAE(nn.Module):
    """Generate synthetic voice features conditioned on syndrome type."""
    def __init__(self, input_dim=128, latent_dim=16, num_conditions=10):
        super().__init__()
        self.cond_encoder = nn.Embedding(num_conditions, 32)
        self.joint_encoder = nn.Sequential(
            nn.Linear(input_dim + 32, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(64, latent_dim)
        self.fc_logvar = nn.Linear(64, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 32, 128), nn.ReLU(),
            nn.Linear(128, input_dim),
        )

    def forward(self, x, condition):
        cond = self.cond_encoder(condition)
        h = self.joint_encoder(torch.cat([x, cond], dim=-1))
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        recon = self.decoder(torch.cat([z, cond], dim=-1))
        return recon, mu, logvar


# ═══════════════════════════════════════════════════════════
# 4. Training Pipeline
# ═══════════════════════════════════════════════════════════
class AudioSenseTrainer:
    """End-to-end trainer for AudioSense-TCM."""
    ORGAN_NAMES = ['心', '肝', '脾', '肺', '肾']
    EMOTION_NAMES = ['喜', '怒', '忧', '思', '悲', '恐', '惊']

    def __init__(self, feature_extractor, vae, lr=1e-3):
        self.feature_extractor = feature_extractor
        self.vae = vae
        params = list(feature_extractor.parameters()) + list(vae.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def train_step(self, f0, formant, energy, rate, zangfu_labels, emotion_labels=None):
        self.feature_extractor.train()
        self.vae.train()
        self.optimizer.zero_grad()

        features = self.feature_extractor(f0, formant, energy, rate)
        recon, mu, logvar, zangfu_logits, emotion_logits = self.vae(features)

        losses = self.vae.loss_function(
            recon, features, mu, logvar, zangfu_logits, zangfu_labels,
            emotion_logits, emotion_labels,
        )
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.feature_extractor.parameters()) + list(self.vae.parameters()), 1.0
        )
        self.optimizer.step()
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


# ═══════════════════════════════════════════════════════════
# 5. Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("A-006: AudioSense v2 - Voice-ZangFu β-VAE")
    print("=" * 60)

    B = 8
    f0 = torch.randn(B, 64)
    formant = torch.randn(B, 128)
    energy = torch.randn(B, 64)
    rate = torch.randn(B, 32)
    zangfu_labels = torch.randint(0, 5, (B,))
    emotion_labels = torch.randint(0, 7, (B,))

    feat_ext = ProsodyFeatureExtractor()
    vae = ZangFuVAE(latent_dim=16, beta=4.0)
    trainer = AudioSenseTrainer(feat_ext, vae)

    for epoch in range(20):
        losses = trainer.train_step(f0, formant, energy, rate, zangfu_labels, emotion_labels)
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}: total={losses['total']:.4f}, "
                  f"recon={losses['L_recon']:.4f}, kl={losses['L_kl']:.4f}, "
                  f"cls={losses['L_cls']:.4f}")

    # Inference
    features = feat_ext(f0, formant, energy, rate)
    probs, latent = vae.infer_zangfu(features)
    print(f"\nOrgan probabilities (first sample):")
    for i, name in enumerate(AudioSenseTrainer.ORGAN_NAMES):
        print(f"  {name}: {probs[0, i]:.4f}")
    print(f"Latent shape: {latent.shape}")

    total_params = sum(p.numel() for p in feat_ext.parameters()) + sum(p.numel() for p in vae.parameters())
    print(f"\nTotal parameters: {total_params:,}")
    print("Done!")
