"""
PulsePINN 训练
"""
import torch
import numpy as np
from pinn_model import PINNPulse
from pulse_pde import PulsePDE

def train_PINN(epochs=500):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PINNPulse(hidden_dim=64, num_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # 合成训练数据
    nx, nt = 50, 50
    x_data = torch.rand(1000, 1, device=device) * 1.0
    t_data = torch.rand(1000, 1, device=device) * 1.0
    
    # 初始条件
    x_init = torch.linspace(0, 1, 50).reshape(-1, 1).to(device)
    u_init = torch.exp(-((x_init - 0.3) ** 2) / 0.01)
    
    print("开始训练PINN...")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 物理损失
        pde_loss = model.physics_loss(x_data, t_data)
        
        # 初始损失
        init_loss = model.initial_loss(x_init, u_init)
        
        # 总损失
        loss = pde_loss + init_loss
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"Epoch {epoch}: PDE Loss={pde_loss.item():.6f}, Init Loss={init_loss.item():.6f}")
    
    torch.save(model.state_dict(), '/tmp/pinn_pulse.pth')
    print("PINN模型已保存")

if __name__ == '__main__':
    train_PINN()
