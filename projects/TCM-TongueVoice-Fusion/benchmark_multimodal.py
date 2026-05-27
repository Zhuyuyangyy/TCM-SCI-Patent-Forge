"""
MultiModal Fusion Benchmark — 三模态融合体质识别评测
对比: TongueOnly / VoiceOnly / FaceOnly / EarlyFusion / LateFusion / CrossAttn-Ours
指标: Accuracy / F1 / 模态贡献度 / 缺失模态鲁棒性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from typing import Dict

CONSTITUTIONS=['平和质','气虚质','阳虚质','阴虚质','痰湿质','湿热质','血瘀质','气郁质','特禀质']

def generate_multimodal_data(n=500):
    np.random.seed(42)
    tongue=torch.randn(n,64)
    voice=torch.randn(n,128)
    face=torch.randn(n,32)
    labels=torch.randint(0,9,(n,))
    # Add signal: each modality has some predictive power
    for i in range(n):
        tongue[i,labels[i]%64]+=2.0
        voice[i,labels[i]%128]+=1.5
        face[i,labels[i]%32]+=2.5
    return tongue,voice,face,labels

class SimpleEncoder(nn.Module):
    def __init__(self,in_dim,out_dim=64):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(in_dim,128),nn.ReLU(),nn.Linear(128,out_dim))
    def forward(self,x):
        return self.net(x)

class SingleModalClassifier(nn.Module):
    def __init__(self,in_dim):
        super().__init__()
        self.enc=SimpleEncoder(in_dim,64)
        self.cls=nn.Linear(64,9)
    def forward(self,x):
        return self.cls(self.enc(x))

class EarlyFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(64+128+32,128),nn.ReLU(),nn.Linear(128,9))
    def forward(self,t,v,f):
        return self.net(torch.cat([t,v,f],dim=-1))

class LateFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.t_enc=SimpleEncoder(64,32)
        self.v_enc=SimpleEncoder(128,32)
        self.f_enc=SimpleEncoder(32,32)
        self.cls=nn.Linear(96,9)
    def forward(self,t,v,f):
        return self.cls(torch.cat([self.t_enc(t),self.v_enc(v),self.f_enc(f)],dim=-1))

def train_eval(model,train_data,val_data,epochs=30,lr=1e-3,is_single=False,modality='tongue'):
    opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    t,v,f,y=train_data
    for _ in range(epochs):
        if is_single:
            x={'tongue':t,'voice':v,'face':f}[modality]
            logits=model(x)
        else:
            logits=model(t,v,f)
        loss=F.cross_entropy(logits,y)
        opt.zero_grad(); loss.backward(); opt.step()
    # Eval
    t,v,f,y=val_data
    with torch.no_grad():
        if is_single:
            x={'tongue':t,'voice':v,'face':f}[modality]
            pred=model(x).argmax(dim=1)
        else:
            pred=model(t,v,f).argmax(dim=1)
        acc=(pred==y).float().mean().item()
    return acc

def run_benchmark():
    print("="*60)
    print("MultiModal Fusion Benchmark — 三模态融合评测")
    print("="*60)
    t,v,f,y=generate_multimodal_data(500)
    split=400
    train=(t[:split],v[:split],f[:split],y[:split])
    val=(t[split:],v[split:],f[split:],y[split:])

    methods={
        'TongueOnly': ('single','tongue'),
        'VoiceOnly': ('single','voice'),
        'FaceOnly': ('single','face'),
        'EarlyFusion': ('multi','early'),
        'LateFusion': ('multi','late'),
    }

    results={}
    for name,(ftype,mod) in methods.items():
        if ftype=='single':
            dim={'tongue':64,'voice':128,'face':32}[mod]
            model=SingleModalClassifier(dim)
            acc=train_eval(model,train,val,is_single=True,modality=mod)
        elif mod=='early':
            model=EarlyFusion()
            acc=train_eval(model,train,val)
        else:
            model=LateFusion()
            acc=train_eval(model,train,val)
        results[name]=acc
        print(f"  {name:<15} Acc={acc:.3f}")

    # Missing modality robustness
    print(f"\n[Missing Modality Robrobustness]")
    from multimodal_fusion_v2 import MultiModalConstitutionClassifier
    model=MultiModalConstitutionClassifier()
    trainer=__import__('multimodal_fusion_v2',fromlist=['MultiModalTrainer']).MultiModalTrainer(model)
    for ep in range(20):
        trainer.train_step(t[:split],v[:split],f[:split],y[:split])

    # Full 3-modal
    names,confs,_=model.predict(t[split:],v[split:],f[split:])
    acc3=sum(1 for i,n in enumerate(names) if CONSTITUTIONS[y[split+i]]==n)/len(names)
    print(f"  3-modal (CrossAttn): {acc3:.3f}")

    # 2-modal
    names_t,_ ,_=model.predict(tongue=t[split:],voice=v[split:])
    acc2=sum(1 for i,n in enumerate(names_t) if CONSTITUTIONS[y[split+i]]==n)/len(names_t)
    print(f"  2-modal (T+V): {acc2:.3f}")

    # 1-modal
    names_1,_,_=model.predict(tongue=t[split:])
    acc1=sum(1 for i,n in enumerate(names_1) if CONSTITUTIONS[y[split+i]]==n)/len(names_1)
    print(f"  1-modal (T): {acc1:.3f}")

    results['CrossAttn_3modal']=acc3
    results['CrossAttn_2modal']=acc2
    results['CrossAttn_1modal']=acc1

    with open('multimodal_benchmark_results.json','w') as f:
        json.dump(results,f,indent=2)
    print(f"\nResults saved to multimodal_benchmark_results.json")

if __name__=='__main__':
    run_benchmark()
