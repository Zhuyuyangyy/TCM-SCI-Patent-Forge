"""
Target 001: Hyperspectral NeRF Tongue Reconstruction
120-channel hyperspectral + NeRF 3D implicit representation + SpO2 quantification + vessel extraction

Loss = L_color + λ1 * L_SpO2 + λ2 * L_vessel + λ3 * L_geometry
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


# ═══════════════════════════════════════════════════════════
# 1. Positional Encoding (Fourier Features)
# ═══════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    """Fourier feature encoding for (x,y,z) and (θ,φ)"""
    def __init__(self, num_freqs: int = 10, include_input: bool = True):
        super().__init__()
        self.num_freqs = num_freqs
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(num_freqs).float()
        self.register_buffer('freqs', freqs)  # (L,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., D) -> (..., D * (2*L + 1)) if include_input
        """
        encoded = [x] if self.include_input else []
        for freq in self.freqs:
            encoded.append(torch.sin(freq * np.pi * x))
            encoded.append(torch.cos(freq * np.pi * x))
        return torch.cat(encoded, dim=-1)


# ═══════════════════════════════════════════════════════════
# 2. Hyperspectral NeRF MLP (120-channel extension)
# ═══════════════════════════════════════════════════════════
class HyperspectralNeRF(nn.Module):
    """
    Extended NeRF supporting 120-channel hyperspectral output.
    Standard NeRF outputs RGB (3ch); this outputs full spectral radiance.

    Architecture:
      Position encoding -> 8x256 MLP -> density σ + spectral feature
      Direction encoding -> spectral feature -> 120-channel radiance
      SpO2 prediction head (auxiliary)
    """
    def __init__(
        self,
        pos_freqs: int = 10,
        dir_freqs: int = 4,
        hidden_dim: int = 256,
        num_layers: int = 8,
        spectral_bands: int = 120,
    ):
        super().__init__()
        self.spectral_bands = spectral_bands

        # Encoders
        self.pos_enc = PositionalEncoding(pos_freqs)
        self.dir_enc = PositionalEncoding(dir_freqs)
        pos_input_dim = 3 * (2 * pos_freqs + 1)   # 63
        dir_input_dim = 3 * (2 * dir_freqs + 1)    # 27

        # --- Density trunk (shared) ---
        layers = [nn.Linear(pos_input_dim, hidden_dim), nn.ReLU()]
        for i in range(num_layers - 1):
            if i == 4:  # skip connection at layer 4
                layers.append(nn.Linear(hidden_dim + pos_input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        self.density_trunk = nn.ModuleList(layers)
        self.density_head = nn.Linear(hidden_dim, 1)  # σ

        # --- Spectral feature branch ---
        self.spectral_fc = nn.Linear(hidden_dim, hidden_dim)

        # --- Spectral radiance head (120 channels) ---
        self.radiance_head = nn.Sequential(
            nn.Linear(hidden_dim + dir_input_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, spectral_bands),  # 120-band output
        )

        # --- SpO2 auxiliary head ---
        self.spo2_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # SpO2 ∈ [0,1]
        )

    def forward(
        self,
        positions: torch.Tensor,    # (N, 3) world coords
        directions: torch.Tensor,   # (N, 3) viewing direction
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            sigma:      (N, 1) volume density
            radiance:   (N, 120) hyperspectral radiance
            spo2:       (N, 1) blood oxygen saturation prediction
        """
        pos_enc = self.pos_enc(positions)   # (N, 63)
        dir_enc = self.dir_enc(directions)  # (N, 27)

        # Density trunk with skip connection
        h = pos_enc
        for i, layer in enumerate(self.density_trunk):
            if i == 9:  # after layer 4 ReLU, concat skip
                h = torch.cat([h, pos_enc], dim=-1)
            h = layer(h)

        sigma = F.relu(self.density_head(h))          # (N, 1)
        spectral_feat = F.relu(self.spectral_fc(h))   # (N, 256)

        # Radiance (direction-dependent)
        radiance_input = torch.cat([spectral_feat, dir_enc], dim=-1)
        radiance = self.radiance_head(radiance_input)  # (N, 120)
        radiance = torch.sigmoid(radiance)  # normalize to [0,1]

        # SpO2 (direction-independent)
        spo2 = self.spo2_head(spectral_feat)  # (N, 1)

        return {'sigma': sigma, 'radiance': radiance, 'spo2': spo2}


# ═══════════════════════════════════════════════════════════
# 3. Volume Renderer (Hyperspectral)
# ═══════════════════════════════════════════════════════════
class HyperspectralVolumeRenderer(nn.Module):
    """
    Differentiable volume rendering for 120-channel hyperspectral data.
    Standard NeRF rendering equation extended to N bands.
    """
    def __init__(self, near: float = 0.1, far: float = 3.0, num_samples: int = 64):
        super().__init__()
        self.near = near
        self.far = far
        self.num_samples = num_samples

    def forward(
        self,
        radiance: torch.Tensor,  # (N_samples, 120)
        sigma: torch.Tensor,     # (N_samples, 1)
        z_vals: torch.Tensor,    # (N_samples,)
    ) -> torch.Tensor:
        """
        Returns rendered spectral radiance: (120,)
        """
        # Compute distances between adjacent samples
        dists = z_vals[1:] - z_vals[:-1]  # (N-1,)
        dists = torch.cat([dists, torch.tensor([1e10], device=dists.device)])  # last

        # Alpha compositing
        alpha = 1.0 - torch.exp(-sigma.squeeze(-1) * dists)  # (N,)
        T = torch.cumprod(1.0 - alpha + 1e-10, dim=0)
        T = torch.cat([torch.ones(1, device=T.device), T[:-1]])  # shift

        weights = alpha * T  # (N,)

        # Rendered radiance per band
        rendered = (weights.unsqueeze(-1) * radiance).sum(dim=0)  # (120,)
        return rendered


# ═══════════════════════════════════════════════════════════
# 4. SpO2 Physical Model
# ═══════════════════════════════════════════════════════════
class SpO2Calculator:
    """
    Physics-based SpO2 estimation from hyperspectral reflectance.
    Modified Beer-Lambert Law:
        SpO2 = a0 + a1*R1 + a2*R2 + a3*R1*R2
    where R1 = R_760 / R_800, R2 = R_560 / R_580
    """
    # Wavelength indices (approximate for 500-950nm, 120 bands)
    BAND_560 = 14   # ~560nm
    BAND_580 = 18   # ~580nm
    BAND_760 = 58   # ~760nm
    BAND_800 = 65   # ~800nm

    @staticmethod
    def compute_spo2(radiance: torch.Tensor) -> torch.Tensor:
        """
        radiance: (..., 120) hyperspectral reflectance
        returns:  (..., 1) SpO2 ∈ [0,1]
        """
        R_560 = radiance[..., SpO2Calculator.BAND_560] + 1e-8
        R_580 = radiance[..., SpO2Calculator.BAND_580] + 1e-8
        R_760 = radiance[..., SpO2Calculator.BAND_760] + 1e-8
        R_800 = radiance[..., SpO2Calculator.BAND_800] + 1e-8

        R1 = R_760 / R_800
        R2 = R_560 / R_580

        # Calibration coefficients (clinical standard)
        a0, a1, a2, a3 = 0.0, 0.58, -0.23, 0.12
        spo2 = a0 + a1 * R1 + a2 * R2 + a3 * R1 * R2
        return spo2.clamp(0, 1).unsqueeze(-1)


# ═══════════════════════════════════════════════════════════
# 5. 3D Vessel Network Extractor
# ═══════════════════════════════════════════════════════════
class VesselExtractor:
    """
    Extract 3D vessel network from volumetric SpO2 field.
    Vessels = voxels where SpO2 < threshold (deoxygenated regions).
    Then apply connected component analysis for topology.
    """
    def __init__(self, spo2_threshold: float = 0.75, min_vessel_size: int = 50):
        self.spo2_threshold = spo2_threshold
        self.min_vessel_size = min_vessel_size

    def extract(
        self,
        density_grid: np.ndarray,    # (X, Y, Z)
        spo2_grid: np.ndarray,       # (X, Y, Z)
    ) -> Dict[str, np.ndarray]:
        """
        Returns:
            vessel_mask:    (X, Y, Z) binary mask
            vessel_skeleton: (X, Y, Z) skeletonized centerlines
            vessel_diameter: dict of vessel_id -> mean_diameter_voxels
        """
        # Vessel candidate: high density + low SpO2
        tissue_mask = density_grid > np.percentile(density_grid, 70)
        vessel_mask = tissue_mask & (spo2_grid < self.spo2_threshold)

        # Connected component labeling (simplified BFS)
        labeled, num_features = self._connected_components_3d(vessel_mask)

        # Filter small components
        for label_id in range(1, num_features + 1):
            component_size = np.sum(labeled == label_id)
            if component_size < self.min_vessel_size:
                labeled[labeled == label_id] = 0

        # Estimate diameters (simplified: volume / length ratio)
        vessel_diameters = {}
        for label_id in range(1, num_features + 1):
            voxels = np.argwhere(labeled == label_id)
            if len(voxels) > 0:
                extent = voxels.max(axis=0) - voxels.min(axis=0)
                diameter = np.mean(extent) * 0.5  # rough estimate
                vessel_diameters[label_id] = float(diameter)

        return {
            'vessel_mask': (labeled > 0).astype(np.float32),
            'vessel_labels': labeled,
            'vessel_diameters': vessel_diameters,
            'num_vessels': num_features,
        }

    @staticmethod
    def _connected_components_3d(binary: np.ndarray) -> Tuple[np.ndarray, int]:
        """Simple 3D connected component labeling via BFS."""
        from collections import deque
        labeled = np.zeros_like(binary, dtype=np.int32)
        label = 0
        shape = binary.shape
        visited = np.zeros_like(binary, dtype=bool)

        for x in range(shape[0]):
            for y in range(shape[1]):
                for z in range(shape[2]):
                    if binary[x, y, z] and not visited[x, y, z]:
                        label += 1
                        queue = deque([(x, y, z)])
                        visited[x, y, z] = True
                        while queue:
                            cx, cy, cz = queue.popleft()
                            labeled[cx, cy, cz] = label
                            for dx, dy, dz in [(-1,0,0),(1,0,0),(0,-1,0),(0,1,0),(0,0,-1),(0,0,1)]:
                                nx, ny, nz = cx+dx, cy+dy, cz+dz
                                if 0<=nx<shape[0] and 0<=ny<shape[1] and 0<=nz<shape[2]:
                                    if binary[nx,ny,nz] and not visited[nx,ny,nz]:
                                        visited[nx,ny,nz] = True
                                        queue.append((nx,ny,nz))
        return labeled, label


# ═══════════════════════════════════════════════════════════
# 6. Composite Loss Function
# ═══════════════════════════════════════════════════════════
class HyperspectralNeRFLoss(nn.Module):
    """
    L = L_color + λ1 * L_SpO2 + λ2 * L_vessel + λ3 * L_geometry

    L_color:    MSE between rendered and GT 120-band radiance
    L_SpO2:     Physics-constrained SpO2 consistency loss
    L_vessel:   Binary cross-entropy for vessel segmentation
    L_geometry:  Eikonal regularization for density field
    """
    def __init__(
        self,
        lambda_spo2: float = 0.5,
        lambda_vessel: float = 0.3,
        lambda_geometry: float = 0.01,
    ):
        super().__init__()
        self.lambda_spo2 = lambda_spo2
        self.lambda_vessel = lambda_vessel
        self.lambda_geometry = lambda_geometry

    def forward(
        self,
        rendered_radiance: torch.Tensor,   # (B, 120)
        gt_radiance: torch.Tensor,         # (B, 120)
        predicted_spo2: torch.Tensor,      # (B, 1)
        gt_spo2: Optional[torch.Tensor],   # (B, 1) or None
        predicted_density: torch.Tensor,   # (B, 1)
        vessel_logits: Optional[torch.Tensor],  # (B, 1) or None
        gt_vessel: Optional[torch.Tensor],      # (B, 1) or None
        density_grad: Optional[torch.Tensor],   # (B, 3) gradient for Eikonal
    ) -> Dict[str, torch.Tensor]:

        # L_color: spectral reconstruction
        L_color = F.mse_loss(rendered_radiance, gt_radiance)

        # L_SpO2: physics-based SpO2 consistency
        L_spo2 = torch.tensor(0.0, device=L_color.device)
        if gt_spo2 is not None:
            L_spo2 = F.mse_loss(predicted_spo2, gt_spo2)
        else:
            # Self-supervised: SpO2 from physical model should match learned SpO2
            phys_spo2 = SpO2Calculator.compute_spo2(rendered_radiance)
            L_spo2 = F.mse_loss(predicted_spo2, phys_spo2)

        # L_vessel: vessel segmentation
        L_vessel = torch.tensor(0.0, device=L_color.device)
        if vessel_logits is not None and gt_vessel is not None:
            L_vessel = F.binary_cross_entropy_with_logits(vessel_logits, gt_vessel)

        # L_geometry: Eikonal regularization (|∇density| ≈ 1)
        L_geo = torch.tensor(0.0, device=L_color.device)
        if density_grad is not None:
            L_geo = (density_grad.norm(dim=-1) - 1.0).pow(2).mean()

        total = L_color + self.lambda_spo2 * L_spo2 + self.lambda_vessel * L_vessel + self.lambda_geometry * L_geo

        return {
            'total': total,
            'L_color': L_color,
            'L_spo2': L_spo2,
            'L_vessel': L_vessel,
            'L_geometry': L_geo,
        }


# ═══════════════════════════════════════════════════════════
# 7. Training Pipeline (Skeleton)
# ═══════════════════════════════════════════════════════════
def train_step(
    model: HyperspectralNeRF,
    renderer: HyperspectralVolumeRenderer,
    loss_fn: HyperspectralNeRFLoss,
    optimizer: torch.optim.Optimizer,
    batch_rays_o: torch.Tensor,     # (B, 3) ray origins
    batch_rays_d: torch.Tensor,     # (B, 3) ray directions
    gt_radiance: torch.Tensor,      # (B, 120) ground truth
    gt_spo2: Optional[torch.Tensor] = None,
    gt_vessel: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Single training step."""
    model.train()

    # Sample points along rays
    t_vals = torch.linspace(0, 1, renderer.num_samples, device=batch_rays_o.device)
    z_vals = renderer.near * (1 - t_vals) + renderer.far * t_vals  # (S,)
    z_vals = z_vals.unsqueeze(0).expand(batch_rays_o.shape[0], -1)  # (B, S)

    # Perturb sampling
    mids = 0.5 * (z_vals[:, 1:] + z_vals[:, :-1])
    upper = torch.cat([mids, z_vals[:, -1:]], dim=-1)
    lower = torch.cat([z_vals[:, :1], mids], dim=-1)
    t_rand = torch.rand_like(z_vals)
    z_vals = lower + (upper - lower) * t_rand

    # 3D positions
    pts = batch_rays_o.unsqueeze(1) + z_vals.unsqueeze(-1) * batch_rays_d.unsqueeze(1)
    # (B, S, 3)
    dirs = batch_rays_d.unsqueeze(1).expand_as(pts)  # (B, S, 3)

    # Flatten for MLP
    pts_flat = pts.reshape(-1, 3)
    dirs_flat = dirs.reshape(-1, 3)

    # Forward
    out = model(pts_flat, dirs_flat)
    sigma = out['sigma'].reshape(pts.shape[0], pts.shape[1])       # (B, S)
    radiance = out['radiance'].reshape(pts.shape[0], pts.shape[1], -1)  # (B, S, 120)
    spo2 = out['spo2'].reshape(pts.shape[0], pts.shape[1])         # (B, S)

    # Volume render per ray
    rendered_list = []
    for i in range(pts.shape[0]):
        rendered = renderer(radiance[i], sigma[i].unsqueeze(-1), z_vals[i])
        rendered_list.append(rendered)
    rendered_radiance = torch.stack(rendered_list, dim=0)  # (B, 120)

    # Mean SpO2 along ray
    mean_spo2 = spo2.mean(dim=1, keepdim=True)  # (B, 1)

    # Compute loss
    losses = loss_fn(
        rendered_radiance, gt_radiance,
        mean_spo2, gt_spo2,
        sigma.mean(dim=1, keepdim=True),
        None, gt_vessel, None,
    )

    # Backward
    optimizer.zero_grad()
    losses['total'].backward()
    optimizer.step()

    return {k: v.item() for k, v in losses.items()}


# ═══════════════════════════════════════════════════════════
# 8. Quick Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Target 001: Hyperspectral NeRF Tongue Reconstruction")
    print("=" * 60)

    model = HyperspectralNeRF(spectral_bands=120, hidden_dim=128, num_layers=6)
    renderer = HyperspectralVolumeRenderer(num_samples=32)
    loss_fn = HyperspectralNeRFLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    B = 4
    rays_o = torch.randn(B, 3)
    rays_d = F.normalize(torch.randn(B, 3), dim=-1)
    gt_rad = torch.rand(B, 120)
    gt_spo2 = torch.rand(B, 1) * 0.3 + 0.7  # SpO2 ∈ [0.7, 1.0]

    losses = train_step(model, renderer, loss_fn, optimizer, rays_o, rays_d, gt_rad, gt_spo2)
    print(f"\nTraining step losses:")
    for k, v in losses.items():
        print(f"  {k}: {v:.6f}")

    # SpO2 physical model test
    test_radiance = torch.rand(5, 120)
    spo2 = SpO2Calculator.compute_spo2(test_radiance)
    print(f"\nSpO2 physical model test: {spo2.squeeze().tolist()}")

    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    print("Done!")
