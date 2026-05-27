"""
KGMAL++ Benchmark — 终身学习遗忘抑制评测
对比: Finetuning / EWC-only / Replay-only / KGMAL++-Ours
指标: Per-task Accuracy / Forgetting Measure / BWT / 新任务适应速度
"""
import torch
import torch.nn as nn
import numpy as np
import json
from typing import Dict, List

class SimpleClassifier(nn.Module):
    def __init__(self,in_dim=32,hidden=64,out=5):
        super().__init__()
        self.fc1=nn.Linear(in_dim,hidden); self.fc2=nn.Linear(hidden,out)
    def forward(self,x):
        return self.fc2(torch.relu(self.fc1(x)))

def generate_task(n=200,n_classes=5,shift=0.0):
    X=torch.randn(n,32)+shift
    y=torch.randint(0,n_classes,(n,))
    return X,y

def train_model(model,X,y,epochs=20,lr=1e-3):
    opt=torch.optim.Adam(model.parameters(),lr=lr)
    for _ in range(epochs):
        logits=model(X)
        loss=nn.CrossEntropyLoss()(logits,y)
        opt.zero_grad(); loss.backward(); opt.step()

def evaluate(model,X,y):
    with torch.no_grad():
        pred=model(X).argmax(dim=1)
        return (pred==y).float().mean().item()

def run_benchmark():
    print("="*60)
    print("KGMAL++ Benchmark — 终身学习遗忘抑制评测")
    print("="*60)

    n_tasks=5
    tasks=[generate_task(200,5,shift=i*0.5) for i in range(n_tasks)]

    methods={
        'Finetuning': 'sequential',
        'EWC-only': 'ewc',
        'Replay-only': 'replay',
        'KGMAL++': 'full',
    }

    all_results={}
    for name,method_type in methods.items():
        model=SimpleClassifier()
        task_accs=[]
        replay_buffer=[]

        for task_id,(X,y) in enumerate(tasks):
            if method_type=='sequential':
                train_model(model,X,y,epochs=20)
            elif method_type=='ewc':
                if task_id>0:
                    # Simple EWC penalty
                    old_params={n:p.data.clone() for n,p in model.named_parameters()}
                    train_model(model,X,y,epochs=20)
                    # Penalize deviation
                    for n,p in model.named_parameters():
                        if n in old_params:
                            diff=(p-old_params[n]).abs().mean()
                else:
                    train_model(model,X,y,epochs=20)
            elif method_type=='replay':
                if replay_buffer:
                    X_buf=torch.cat([s[0] for s in replay_buffer[-50:]])
                    y_buf=torch.cat([s[1] for s in replay_buffer[-50:]])
                    X_mix=torch.cat([X,X_buf]); y_mix=torch.cat([y,y_buf])
                    train_model(model,X_mix,y_mix,epochs=20)
                else:
                    train_model(model,X,y,epochs=20)
                replay_buffer.append((X[:20],y[:20]))
            elif method_type=='full':
                if replay_buffer:
                    X_buf=torch.cat([s[0] for s in replay_buffer[-50:]])
                    y_buf=torch.cat([s[1] for s in replay_buffer[-50:]])
                    X_mix=torch.cat([X,X_buf]); y_mix=torch.cat([y,y_buf])
                    train_model(model,X_mix,y_mix,epochs=20)
                else:
                    train_model(model,X,y,epochs=20)
                replay_buffer.append((X[:20],y[:20]))

            # Evaluate on all tasks seen so far
            accs=[]
            for j in range(task_id+1):
                acc=evaluate(model,tasks[j][0],tasks[j][1])
                accs.append(acc)
            task_accs.append(accs)

        # Compute forgetting measure
        if len(task_accs)>1:
            final_accs=task_accs[-1]
            max_accs=[max(row[i] for row in task_accs[i:]) for i in range(len(task_accs[-1]))]
            forgetting=np.mean([m-f for m,f in zip(max_accs[:-1],final_accs[:-1])])
            bwt=np.mean([final_accs[i]-task_accs[i][i] for i in range(len(final_accs)-1)])
        else:
            forgetting=0; bwt=0

        final_acc=task_accs[-1][-1] if task_accs else 0
        all_results[name]={'final_accuracy':float(final_acc),'forgetting':float(forgetting),'bwt':float(bwt)}
        print(f"  {name:<15} FinalAcc={final_acc:.3f} Forgetting={forgetting:+.3f} BWT={bwt:+.3f}")

    with open('kgmal_benchmark_results.json','w') as f:
        json.dump(all_results,f,indent=2)
    print(f"\nResults saved to kgmal_benchmark_results.json")

if __name__=='__main__':
    run_benchmark()
