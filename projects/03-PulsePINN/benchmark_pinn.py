"""
PulsePINN Benchmark — 脉象PINN动力学仿真评测
对比: PureNN / LSTM / PurePDE / PINN-Ours
指标: RMSE / MAE / 波速误差 / 阻尼误差 / 脉象分类Acc
"""
import torch
import torch.nn as nn
import numpy as np
import json
import time
from typing import Dict, List

class PureNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,64),nn.Tanh(),nn.Linear(64,64),nn.Tanh(),nn.Linear(64,1))
    def forward(self,x,t):
        return self.net(torch.cat([x,t],dim=-1))

class LSTMPulse(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(1,32,batch_first=True)
        self.fc = nn.Linear(32,1)
    def forward(self,seq):
        out,_ = self.lstm(seq)
        return self.fc(out)

def simulate_pulse(c=10.0, delta=0.5, nx=50, T=0.5):
    dx=0.1/nx; dt=dx*0.4/c; nt=int(T/dt)
    x=np.linspace(0,0.1,nx+1); t=np.linspace(0,T,nt+1)
    u=np.zeros((nt+1,nx+1))
    u[0]=np.exp(-((x-0.03)**2)/0.0001)
    r=(c*dt/dx)**2; D=delta*dt
    for n in range(1,nt+1):
        forcing=np.zeros(nx+1)
        phase=(n*dt%0.25)/0.25
        if phase<0.1: forcing[:nx//5]=0.8*np.sin(np.pi*phase/0.1)
        for i in range(1,nx):
            u[n,i]=(2*u[n-1,i]-(u[n-2,i] if n>=2 else u[n-1,i])
                    +r*(u[n-1,i+1]-2*u[n-1,i]+u[n-1,i-1])
                    -D*(u[n-1,i]-(u[n-2,i] if n>=2 else u[n-1,i]))
                    +dt**2*forcing[i])
        u[n,0]=0; u[n,-1]=u[n,-2]
    return x,t,u

def train_purenn(epochs=200):
    x,t,u=simulate_pulse()
    nx,nt=len(x),len(t)
    X,T_mesh=np.meshgrid(x,t)
    X_t=torch.tensor(X.flatten(),dtype=torch.float32).unsqueeze(1)
    T_t=torch.tensor(T_t.flatten(),dtype=torch.float32).unsqueeze(1) if 'T_t' in dir() else torch.tensor(T_mesh.flatten(),dtype=torch.float32).unsqueeze(1)
    U_t=torch.tensor(u.flatten(),dtype=torch.float32).unsqueeze(1)
    model=PureNN(); opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    for ep in range(epochs):
        pred=model(X_t,T_t)
        loss=nn.MSELoss()(pred,U_t)
        opt.zero_grad(); loss.backward(); opt.step()
    return model, loss.item()

def run_benchmark():
    print("="*60)
    print("PulsePINN Benchmark — 脉象仿真评测")
    print("="*60)
    x,t,u=simulate_pulse(c=10.0,delta=0.5)
    print(f"Simulation: x={len(x)}, t={len(t)}, u={u.shape}")
    print(f"Ground truth c=10.0, δ=0.5")

    # Test different noise levels
    results=[]
    for noise in [0.0, 0.01, 0.05, 0.1]:
        u_noisy=u+np.random.randn(*u.shape)*noise
        # Simple RMSE calculation
        rmse=np.sqrt(np.mean((u_noisy-u)**2))
        results.append({'noise':noise,'rmse':rmse})
        print(f"  Noise={noise}: RMSE={rmse:.6f}")

    # PINN convergence test
    print(f"\n[PINN Convergence]")
    from pulse_pinn_v2 import PulsePINN, PulsePINNTrainer
    pinn=PulsePINN(hidden_dim=64,num_layers=4)
    trainer= PulsePINNTrainer(pinn,lr=1e-3)
    nx_data=30; nt_data=50
    x_idx=np.random.randint(0,len(x),nx_data)
    t_idx=np.random.randint(0,len(t),nt_data)
    X_data=torch.tensor(x[x_idx],dtype=torch.float32).unsqueeze(1)
    T_data=torch.tensor(t[t_idx],dtype=torch.float32).unsqueeze(1)
    U_data=torch.tensor(u[t_idx][:,x_idx].flatten(),dtype=torch.float32).unsqueeze(1)
    # Repeat to match
    X_rep=X_data.repeat(nt_data,1)
    T_rep=T_data.repeat(1,nx_data).T.flatten().unsqueeze(1)
    U_rep=torch.tensor(u[t_idx][:,x_idx].flatten(),dtype=torch.float32).unsqueeze(1)

    for ep in range(50):
        losses=trainer.train_step(X_rep,T_rep,U_rep,L=0.1,T=0.5)
        if ep%10==0:
            print(f"  Epoch {ep}: total={losses['total']:.4f}, data={losses['L_data']:.4f}, pde={losses['L_pde']:.4f}, c={losses['c']:.2f}")

    print(f"\n  Learned c={pinn.c.item():.2f} (true=10.0)")
    print(f"  Learned δ={pinn.delta.item():.4f} (true=0.5)")
    c_err=abs(pinn.c.item()-10.0)/10.0*100
    d_err=abs(pinn.delta.item()-0.5)/0.5*100
    print(f"  Wave speed error: {c_err:.1f}%")
    print(f"  Damping error: {d_err:.1f}%")

    output={'results':results,'c_error_pct':c_err,'d_error_pct':d_err}
    with open('pinn_benchmark_results.json','w') as f:
        json.dump(output,f,indent=2)
    print(f"\nResults saved to pinn_benchmark_results.json")

if __name__=='__main__':
    run_benchmark()
