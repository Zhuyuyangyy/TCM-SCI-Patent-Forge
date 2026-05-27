"""
Target 002: Pulse PINN v2 - Physics-Informed Neural Network for TCM Pulse Simulation
Embeds 1D hemodynamic PDE into neural network as loss constraint.

PDE: ∂²u/∂t² = c² ∂²u/∂x² - δ ∂u/∂t + f(x,t)
where:
  u(x,t): pulse wave displacement
  c: wave speed (related to vessel elasticity, 5-15 m/s)
  δ: damping coefficient (viscosity)
  f(x,t): cardiac forcing term

寸关尺 three positions = three observation points along radial artery.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional, List


# ═══════════════════════════════════════════════════════════
# 1. PINN Core Network
# ═══════════════════════════════════════════════════════════
class PulsePINN(nn.Module):
    """
    Physics-Informed Neural Network for pulse wave modeling.

    Input:  (x, t) — spatial position along artery + time
    Output: u(x,t) — pulse wave displacement

    The network is trained with:
      L = L_data + λ_pde * L_pde + λ_bc * L_bc + λ_ic * L_ic

    L_data: MSE on measured pulse signals at 寸关尺
    L_pde:  PDE residual (should be ~0)
    L_bc:   boundary conditions (u=0 at artery ends)
    L_ic:   initial conditions (u(x,0) = given)
    """
    def __init__(
        self,
        hidden_dim: int = 128,
        num_layers: int = 6,
    ):
        super().__init__()

        # Input: (x, t) -> 2D
        layers = [nn.Linear(2, hidden_dim), nn.Tanh()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))  # output: u
        self.net = nn.Sequential(*layers)

        # Learnable physical parameters (initialized from domain knowledge)
        self.log_c = nn.Parameter(torch.tensor(np.log(10.0)))    # wave speed
        self.log_delta = nn.Parameter(torch.tensor(np.log(0.5))) # damping

        # Cardiac forcing network (models heart beat)
        self.forcing_net = nn.Sequential(
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    @property
    def c(self) -> torch.Tensor:
        """Wave speed (positive)"""
        return torch.exp(self.log_c)

    @property
    def delta(self) -> torch.Tensor:
        """Damping coefficient (positive)"""
        return torch.exp(self.log_delta)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 1) spatial position ∈ [0, L]
        t: (N, 1) time ∈ [0, T]
        returns: u(x,t) (N, 1)
        """
        inp = torch.cat([x, t], dim=-1)  # (N, 2)
        return self.net(inp)

    def pde_residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Compute PDE residual: R = ∂²u/∂t² - c² ∂²u/∂x² + δ ∂u/∂t - f(x,t)
        Should be ~0 at collocation points.
        """
        x.requires_grad_(True)
        t.requires_grad_(True)

        u = self.forward(x, t)  # (N, 1)

        # First derivatives
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]  # (N, 1)

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]  # (N, 1)

        # Second derivatives
        u_tt = torch.autograd.grad(
            u_t, t, grad_outputs=torch.ones_like(u_t),
            create_graph=True, retain_graph=True
        )[0]  # (N, 1)

        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]  # (N, 1)

        # Cardiac forcing term
        f = self.forcing_net(torch.cat([x, t], dim=-1))

        # PDE residual: u_tt = c² u_xx - δ u_t + f
        residual = u_tt - self.c**2 * u_xx + self.delta * u_t - f
        return residual

    def compute_all_losses(
        self,
        # Data points (measured at 寸关尺)
        x_data: torch.Tensor,      # (N_d, 1)
        t_data: torch.Tensor,      # (N_d, 1)
        u_data: torch.Tensor,      # (N_d, 1) measured pulse
        # Collocation points (for PDE)
        x_col: torch.Tensor,       # (N_c, 1)
        t_col: torch.Tensor,       # (N_c, 1)
        # Boundary points
        x_bc: torch.Tensor,        # (N_b, 1)
        t_bc: torch.Tensor,        # (N_b, 1)
        # Initial condition points
        x_ic: torch.Tensor,        # (N_i, 1)
        t_ic: torch.Tensor,        # (N_i, 1) all zeros
        u_ic: torch.Tensor,        # (N_i, 1) initial displacement
        # Weights
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ) -> Dict[str, torch.Tensor]:

        # L_data: fit measured pulse
        u_pred = self.forward(x_data, t_data)
        L_data = F.mse_loss(u_pred, u_data)

        # L_pde: PDE residual ~ 0
        residual = self.pde_residual(x_col, t_col)
        L_pde = torch.mean(residual ** 2)

        # L_bc: boundary (u=0 at x=0 and x=L)
        u_bc = self.forward(x_bc, t_bc)
        L_bc = torch.mean(u_bc ** 2)

        # L_ic: initial condition
        u_ic_pred = self.forward(x_ic, t_ic)
        L_ic = F.mse_loss(u_ic_pred, u_ic)

        # Also penalize u_t(x,0) = 0 (initial velocity)
        x_ic_g = x_ic.clone().requires_grad_(True)
        t_ic_g = t_ic.clone().requires_grad_(True)
        u_ic_g = self.forward(x_ic_g, t_ic_g)
        u_t_ic = torch.autograd.grad(
            u_ic_g, t_ic_g,
            grad_outputs=torch.ones_like(u_ic_g),
            create_graph=True
        )[0]
        L_ic_vel = torch.mean(u_t_ic ** 2)

        total = L_data + lambda_pde * L_pde + lambda_bc * L_bc + lambda_ic * (L_ic + L_ic_vel)

        return {
            'total': total,
            'L_data': L_data,
            'L_pde': L_pde,
            'L_bc': L_bc,
            'L_ic': L_ic,
            'L_ic_vel': L_ic_vel,
            'c': self.c.detach(),
            'delta': self.delta.detach(),
        }


# ═══════════════════════════════════════════════════════════
# 2. Pulse Syndrome Classifier (on top of PINN features)
# ═══════════════════════════════════════════════════════════
class PulseSyndromeClassifier(nn.Module):
    """
    Classifies TCM pulse types from PINN-extracted features.
    Takes the learned u(x,t) at 寸关尺 as input features.

    Pulse types: 平脉, 浮脉, 沉脉, 迟脉, 数脉, 滑脉, 涩脉, 弦脉, etc.
    """
    PULSE_TYPES = [
        '平脉', '浮脉', '沉脉', '迟脉', '数脉',
        '滑脉', '涩脉', '弦脉', '紧脉', '洪脉',
        '细脉', '濡脉', '弱脉', '促脉', '结脉', '代脉'
    ]

    def __init__(self, pinn: PulsePINN, num_classes: int = 16):
        super().__init__()
        self.pinn = pinn

        # Feature extractor: sample pulse at 寸关尺 over time
        self.feature_net = nn.Sequential(
            nn.Linear(3 * 64, 256),  # 3 positions × 64 time samples
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Linear(128, num_classes)

    def extract_pulse_features(
        self,
        t_samples: torch.Tensor,  # (64,) time points
        L_artery: float = 0.1,    # artery length (m)
    ) -> torch.Tensor:
        """
        Sample PINN at 寸关尺 positions to create feature vector.
        寸 = 0.3L, 关 = 0.5L, 尺 = 0.7L
        """
        self.pinn.eval()
        with torch.no_grad():
            positions = {
                'cun': 0.3 * L_artery,
                'guan': 0.5 * L_artery,
                'chi': 0.7 * L_artery,
            }
            features = []
            for name, x_val in positions.items():
                x = torch.full_like(t_samples, x_val).unsqueeze(-1)
                t = t_samples.unsqueeze(-1)
                u = self.pinn(x, t).squeeze(-1)  # (64,)
                features.append(u)
            return torch.cat(features, dim=0)  # (192,)

    def forward(self, pulse_features: torch.Tensor) -> torch.Tensor:
        """
        pulse_features: (B, 192) — concatenated 寸关尺 signals
        returns: (B, num_classes) logits
        """
        h = self.feature_net(pulse_features)
        return self.classifier(h)


# ═══════════════════════════════════════════════════════════
# 3. Training Pipeline
# ═══════════════════════════════════════════════════════════
class PulsePINNTrainer:
    """End-to-end trainer for Pulse PINN + Syndrome Classifier."""

    def __init__(
        self,
        pinn: PulsePINN,
        classifier: Optional[PulseSyndromeClassifier] = None,
        lr: float = 1e-3,
        lambda_pde: float = 1.0,
        lambda_bc: float = 10.0,
        lambda_ic: float = 10.0,
    ):
        self.pinn = pinn
        self.classifier = classifier
        self.lambda_pde = lambda_pde
        self.lambda_bc = lambda_bc
        self.lambda_ic = lambda_ic

        params = list(pinn.parameters())
        if classifier:
            params += list(classifier.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )

    def generate_collocation_points(
        self,
        n_col: int = 1000,
        L: float = 0.1,     # artery length (m)
        T: float = 1.0,     # time span (s)
        device: str = 'cpu',
    ) -> Dict[str, torch.Tensor]:
        """Generate random collocation points for PDE residual."""
        x_col = torch.rand(n_col, 1, device=device) * L
        t_col = torch.rand(n_col, 1, device=device) * T

        # Boundary: x=0 and x=L
        n_bc = n_col // 4
        t_bc = torch.rand(n_bc, 1, device=device) * T
        x_bc_left = torch.zeros(n_bc, 1, device=device)
        x_bc_right = torch.full((n_bc, 1), L, device=device)
        x_bc = torch.cat([x_bc_left, x_bc_right], dim=0)
        t_bc = torch.cat([t_bc, t_bc], dim=0)

        # Initial condition: t=0
        n_ic = n_col // 4
        x_ic = torch.rand(n_ic, 1, device=device) * L
        t_ic = torch.zeros(n_ic, 1, device=device)
        # Gaussian initial pulse (simulating cardiac ejection)
        u_ic = 0.5 * torch.exp(-((x_ic - 0.3*L)**2) / (0.01*L**2))

        return {
            'x_col': x_col, 't_col': t_col,
            'x_bc': x_bc, 't_bc': t_bc,
            'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic,
        }

    def train_step(
        self,
        x_data: torch.Tensor,
        t_data: torch.Tensor,
        u_data: torch.Tensor,
        L: float = 0.1,
        T: float = 1.0,
    ) -> Dict[str, float]:
        self.pinn.train()
        self.optimizer.zero_grad()

        col = self.generate_collocation_points(
            n_col=500, L=L, T=T, device=x_data.device
        )

        losses = self.pinn.compute_all_losses(
            x_data=x_data, t_data=t_data, u_data=u_data,
            x_col=col['x_col'], t_col=col['t_col'],
            x_bc=col['x_bc'], t_bc=col['t_bc'],
            x_ic=col['x_ic'], t_ic=col['t_ic'], u_ic=col['u_ic'],
            lambda_pde=self.lambda_pde,
            lambda_bc=self.lambda_bc,
            lambda_ic=self.lambda_ic,
        )

        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.pinn.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


# ═══════════════════════════════════════════════════════════
# 4. Pulse Simulation (for generating synthetic training data)
# ═══════════════════════════════════════════════════════════
class PulseSimulator:
    """
    Generate synthetic pulse data using finite difference method.
    Used to create training data for the PINN.
    """
    def __init__(self, c=10.0, delta=0.5, L=0.1, nx=100, T=1.0):
        self.c = c
        self.delta = delta
        self.L = L
        self.nx = nx
        self.T = T
        self.dx = L / nx
        self.dt = self.dx * 0.4 / c  # CFL condition

    def simulate(self, n_heartbeats: int = 3) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns: (x_grid, t_grid, u_solution) all numpy arrays
        """
        nt = int(self.T / self.dt)
        x = np.linspace(0, self.L, self.nx + 1)
        t = np.linspace(0, self.T, nt + 1)
        u = np.zeros((nt + 1, self.nx + 1))

        # Initial Gaussian pulse
        u[0] = np.exp(-((x - 0.3 * self.L) ** 2) / (0.01 * self.L ** 2))

        r = (self.c * self.dt / self.dx) ** 2
        D = self.delta * self.dt

        # Cardiac forcing: periodic Gaussian pulses
        heartbeat_period = self.T / n_heartbeats

        for n in range(1, nt + 1):
            current_t = n * self.dt
            # Cardiac forcing at proximal end
            heartbeat_phase = (current_t % heartbeat_period) / heartbeat_period
            forcing = np.zeros(self.nx + 1)
            if heartbeat_phase < 0.1:
                forcing[:self.nx // 5] = 0.8 * np.sin(np.pi * heartbeat_phase / 0.1)

            for i in range(1, self.nx):
                u[n, i] = (2*u[n-1, i] - (u[n-2, i] if n >= 2 else u[n-1, i])
                           + r * (u[n-1, i+1] - 2*u[n-1, i] + u[n-1, i-1])
                           - D * (u[n-1, i] - (u[n-2, i] if n >= 2 else u[n-1, i]))
                           + self.dt**2 * forcing[i])

            # Boundary conditions
            u[n, 0] = 0
            u[n, -1] = u[n, -2]

        return x, t, u


# ═══════════════════════════════════════════════════════════
# 5. Quick Verification
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("Target 002: Pulse PINN v2 - Physics-Informed Neural Network")
    print("=" * 60)

    # Generate synthetic data
    sim = PulseSimulator(c=10.0, delta=0.5, L=0.1, nx=50, T=0.5)
    x_np, t_np, u_np = sim.simulate(n_heartbeats=2)
    print(f"Simulation: x={x_np.shape}, t={t_np.shape}, u={u_np.shape}")

    # Sample training data
    n_data = 200
    idx_t = np.random.randint(0, len(t_np), n_data)
    idx_x = np.random.randint(0, len(x_np), n_data)
    x_data = torch.tensor(x_np[idx_x], dtype=torch.float32).unsqueeze(-1)
    t_data = torch.tensor(t_np[idx_t], dtype=torch.float32).unsqueeze(-1)
    u_data = torch.tensor(u_np[idx_t, idx_x], dtype=torch.float32).unsqueeze(-1)

    # Create PINN
    pinn = PulsePINN(hidden_dim=64, num_layers=4)
    trainer = PulsePINNTrainer(pinn, lr=1e-3)

    # Train
    print("\nTraining PINN...")
    for epoch in range(100):
        losses = trainer.train_step(x_data, t_data, u_data, L=0.1, T=0.5)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: total={losses['total']:.6f}, "
                  f"data={losses['L_data']:.6f}, pde={losses['L_pde']:.6f}, "
                  f"c={losses['c']:.2f}, δ={losses['delta']:.4f}")

    # Pulse type classification test
    print("\nPulse Syndrome Classifier test:")
    classifier = PulseSyndromeClassifier(pinn, num_classes=16)
    t_samples = torch.linspace(0, 0.5, 64)
    features = classifier.extract_pulse_features(t_samples, L_artery=0.1)
    print(f"  Feature vector shape: {features.shape}")
    logits = classifier(features.unsqueeze(0))
    print(f"  Classification logits: {logits.shape}")
    pred = PulseSyndromeClassifier.PULSE_TYPES[logits.argmax().item()]
    print(f"  Predicted pulse type: {pred}")

    # Parameter count
    total_params = sum(p.numel() for p in pinn.parameters())
    print(f"\nPINN parameters: {total_params:,}")
    print("Done!")
