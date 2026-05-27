"""
Physics-Informed Neural Network for Pulse Analysis
物理信息神经网络：用神经网络学习PDE的残差
"""
import torch
import torch.nn as nn
import numpy as np

class PINNPulse(nn.Module):
    """
    PINN for pulse wave PDE
    输入: (x, t) 坐标
    输出: u(x,t) 脉压值
    
    物理约束:
    ∂u/∂t + c ∂u/∂x - ν ∂²u/∂x² = 0
    """
    def __init__(self, hidden_dim=64, num_layers=4):
        super().__init__()
        layers = []
        layers.append(nn.Linear(2, hidden_dim))  # 输入: (x, t)
        layers.append(nn.Tanh())
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        
        # PDE参数
        self.c = nn.Parameter(torch.tensor(10.0))
        self.nu = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x, t):
        """预测u(x,t)"""
        X = torch.cat([x, t], dim=1)
        return self.net(X)
    
    def physics_loss(self, x, t):
        """
        计算PDE物理残差
        ∂u/∂t + c ∂u/∂x - ν ∂²u/∂x² = 0
        """
        x.requires_grad_(True)
        t.requires_grad_(True)
        
        u = self.forward(x, t)
        
        # 一阶导数
        grad_u = torch.autograd.grad(u, [x, t], grad_outputs=torch.ones_like(u), create_graph=True)
        u_t = grad_u[1]
        u_x = grad_u[0]
        
        # 二阶导数
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        
        # PDE残差
        residual = u_t + self.c * u_x - self.nu * u_xx
        return torch.mean(residual ** 2)
    
    def initial_loss(self, x, u_initial):
        """初始条件损失"""
        t_zero = torch.zeros_like(x)
        u_pred = self.forward(x, t_zero)
        return torch.mean((u_pred - u_initial) ** 2)
    
    def boundary_loss(self, x_boundary, t):
        """边界条件损失"""
        u_pred = self.forward(x_boundary, t)
        return torch.mean(u_pred ** 2)  # u=0 at boundary
