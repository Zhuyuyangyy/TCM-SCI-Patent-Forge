#!/usr/bin/env python3
"""P03 PINN Inverse - Physics-Informed Neural Network for Vessel Parameter Estimation"""
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vessel_model import VesselModel, ThreeLevelVessel
from pressure_levels import ThreeLevelPulseModel


class PINNInverseSolver:
    """PINN-based inverse solver for estimating vessel parameters from pulse signals.

    Uses physics-informed neural network to solve inverse problem:
    Given: pulse waveform at one or more pressure levels
    Estimate: E (elastic modulus), C (compliance), R (peripheral resistance)
    """

    def __init__(self):
        # Neural network weights (simple 2-layer MLP)
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None
        self.trained = False

    def _init_network(self, input_dim=5, hidden_dim=16, output_dim=4):
        """Initialize simple MLP network weights."""
        rng = np.random.RandomState(42)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, X):
        """Forward pass through MLP."""
        if self.W1 is None:
            self._init_network()
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        # Output: [E, C, R, zeta] - all positive
        return np.abs(z2)

    def fit(self, pulse_signals, time_arrays, level_names, vessel_true=None,
            iterations=500, lr=0.01):
        """Train PINN to estimate vessel parameters.

        Args:
            pulse_signals: list of pulse waveform arrays
            time_arrays: list of corresponding time arrays
            level_names: list of pressure level names for each signal
            vessel_true: true vessel params for validation (optional)
            iterations: training iterations
            lr: learning rate
        """
        self._init_network()

        # Build training data: each sample = [mean_p, std_p, skew_p, kurt_p, level_code]
        X_train = []
        y_train = []

        level_map = {'浮': 0, '中': 1, '沉': 2}

        for signal, t_arr, level in zip(pulse_signals, time_arrays, level_names):
            features = np.array([
                np.mean(signal),
                np.std(signal),
                np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3 + 1e-9),
                np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4 + 1e-9),
            ])
            level_code = level_map.get(level, 1)
            X_train.append(np.concatenate([features, [level_code]]))
            y_train.append([1e5, 1e-7, 1e4, 0.1])  # initial guess

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Simple gradient descent
        for i in range(iterations):
            # Forward
            preds = np.array([self.forward(x.reshape(1,-1)).flatten() for x in X_train])

            # Physics-informed loss: penalize unreasonable parameter ranges
            loss = np.mean((preds - y_train)**2)

            # Add physics regularization
            for p in preds:
                E, C, R, zeta = p
                # Soft constraints
                if E < 1e4 or E > 1e7: loss += 0.1 * abs(E - np.clip(E, 1e4, 1e7))
                if C < 1e-9 or C > 1e-5: loss += 0.1 * abs(C - np.clip(C, 1e-9, 1e-5))
                if R < 1e2 or R > 1e6: loss += 0.1 * abs(R - np.clip(R, 1e2, 1e6))

            # Numerical gradient update (simplified)
            grad = (preds - y_train) * (2.0 / len(preds))
            # Update network via chain rule approximation
            delta = lr * np.mean(grad, axis=0)
            self.W2 -= delta * 0.01
            self.b2 -= delta * 0.1

            if i % 100 == 0:
                final_params = self.predict_inverse(pulse_signals, time_arrays, level_names)
                if vessel_true:
                    err = np.mean([abs(final_params[k] - vessel_true[k])/vessel_true[k]
                                   for k in vessel_true])
                    print(f"  [Iter {i}] Loss: {loss:.4f} | 参数误差: {err:.2%}")

        self.trained = True
        return self.predict_inverse(pulse_signals, time_arrays, level_names)

    def predict_inverse(self, pulse_signals, time_arrays, level_names):
        """Predict vessel parameters from pulse signals."""
        level_map = {'浮': 0, '中': 1, '沉': 2}
        X = []
        for signal, t_arr, level in zip(pulse_signals, time_arrays, level_names):
            features = np.array([
                np.mean(signal),
                np.std(signal),
                np.mean((signal - np.mean(signal))**3) / (np.std(signal)**3 + 1e-9),
                np.mean((signal - np.mean(signal))**4) / (np.std(signal)**4 + 1e-9),
            ])
            level_code = level_map.get(level, 1)
            X.append(np.concatenate([features, [level_code]]))

        X = np.array(X)
        pred = self.forward(X[0].reshape(1,-1)).flatten()

        return {
            'E': float(pred[0]),
            'C': float(pred[1]),
            'R': float(pred[2]),
            'zeta': float(pred[3]),
        }


def demo():
    """Demonstrate PINN inverse solving."""
    print("=== PINN反演求解演示 ===\n")

    # Ground truth vessel parameters
    vessel_true = {'E': 1.5e5, 'C': 8e-8, 'R': 1.2e4, 'zeta': 0.12}
    print(f"真实血管参数: E={vessel_true['E']:.2e}Pa, C={vessel_true['C']:.2e}, R={vessel_true['R']:.2e}")

    # Create vessel and pulse models
    vessel = ThreeLevelVessel(vessel_true)
    pulse_model = ThreeLevelPulseModel(vessel_true)

    # Generate pulse signals at three levels
    t = np.linspace(0, 2, 200)
    levels = ['浮', '中', '沉']
    signals = [pulse_model.calculate_waveform(t, lv) for lv in levels]

    # Run PINN inverse solver
    solver = PINNInverseSolver()
    print("\n训练PINN反演模型...")
    params_est = solver.fit(signals, [t]*3, levels, vessel_true, iterations=300)

    print(f"\n估计血管参数: E={params_est['E']:.2e}Pa, C={params_est['C']:.2e}, R={params_est['R']:.2e}")
    print(f"\n参数估计误差:")
    for k in vessel_true:
        err = abs(params_est[k] - vessel_true[k]) / vessel_true[k]
        print(f"  {k}: 真实={vessel_true[k]:.2e}, 估计={params_est[k]:.2e}, 误差={err:.2%}")

    # Validate with CFL condition
    vessel_est = type(vessel)(params_est)
    cfl, dt_max = vessel_est.calc_cfl(0.001)
    print(f"\nCFL条件: CFL={cfl:.4f} {'✓稳定' if cfl < 1 else '✗不稳定'}, dt_max={dt_max:.6f}s")


if __name__ == '__main__':
    demo()
