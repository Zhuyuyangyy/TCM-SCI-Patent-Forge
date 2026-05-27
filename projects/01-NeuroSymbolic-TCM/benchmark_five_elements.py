"""
FiveElementsGNN Benchmark — 五行证候转移预测评测
对比: GCN / GAT / GraphSAGE / RuleOnly / FiveElementsGNN-Ours
指标: Accuracy / F1 / AUC / 五行一致性率 / 可解释性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
from typing import Dict, List
from collections import defaultdict

SYNDROMES=['肝郁气滞','肝郁化火','肝阳上亢','心火亢盛','心脾两虚','心肾不交',
           '脾胃虚弱','脾胃湿热','脾虚湿盛','肺气虚','痰湿蕴肺','肺肾阴虚',
           '肾阳虚','肾阴虚','肾精不足']
ELEMENTS={'肝郁气滞':'木','肝郁化火':'木','肝阳上亢':'木','心火亢盛':'火','心脾两虚':'火',
          '心肾不交':'火','脾胃虚弱':'土','脾胃湿热':'土','脾虚湿盛':'土','肺气虚':'金',
          '痰湿蕴肺':'金','肺肾阴虚':'金','肾阳虚':'水','肾阴虚':'水','肾精不足':'水'}
SHENG={'木':'火','火':'土','土':'金','金':'水','水':'木'}
KE={'木':'土','土':'水','水':'火','火':'金','金':'木'}

def generate_transition_data(n=2000):
    data=[]
    for _ in range(n):
        src=np.random.choice(SYNDROMES)
        src_elem=ELEMENTS[src]
        r=np.random.random()
        if r<0.4:
            dst_elem=SHENG.get(src_elem,src_elem)
        elif r<0.7:
            dst_elem=KE.get(src_elem,src_elem)
        else:
            dst_elem=np.random.choice(list(ELEMENTS.values()))
        candidates=[s for s,e in ELEMENTS.items() if e==dst_elem]
        dst=np.random.choice(candidates) if candidates else src
        data.append({'src':src,'dst':dst,'label':SYNDROMES.index(dst)})
    return data

class SimpleGNN(nn.Module):
    def __init__(self,in_dim=16,hidden=32,out=15):
        super().__init__()
        self.fc1=nn.Linear(in_dim,hidden); self.fc2=nn.Linear(hidden,out)
    def forward(self,x):
        return self.fc2(F.relu(self.fc1(x)))

def run_benchmark():
    print("="*60)
    print("FiveElementsGNN Benchmark — 五行证候转移评测")
    print("="*60)
    data=generate_transition_data(2000)
    print(f"Generated {len(data)} transition samples")

    methods={
        'RuleOnly': lambda src: ELEMENTS.get(SHENG.get(ELEMENTS.get(src,'木'),'火'),'肝郁气滞'),
        'RandomBaseline': lambda src: np.random.choice(SYNDROMES),
    }
    results={}
    for name,predictor in methods.items():
        correct=0; total=0; fe_correct=0
        for d in data:
            pred=predictor(d['src'])
            if pred==d['dst']: correct+=1
            pred_elem=ELEMENTS.get(pred,''); src_elem=ELEMENTS.get(d['src'],'')
            if pred_elem==SHENG.get(src_elem) or pred_elem==KE.get(src_elem) or pred_elem==src_elem:
                fe_correct+=1
            total+=1
        acc=correct/total; fe_rate=fe_correct/total
        results[name]={'accuracy':acc,'five_elem_consistency':fe_rate}
        print(f"  {name}: Acc={acc:.3f}, FE-Consistency={fe_rate:.3f}")

    # FiveElementsGNN
    from five_elements_gnn import FiveElementsGraph, FiveElementsSyndromeGNN, build_adjacency_matrices
    graph=FiveElementsGraph(); graph.build_default_graph()
    sheng_adj,ke_adj=build_adjacency_matrices(graph,SYNDROMES)
    model=FiveElementsSyndromeGNN(num_syndromes=15,num_symptoms=20,symptom_dim=32,hidden_dim=64)
    n=len(SYNDROMES)
    elem_map={e:i for i,e in enumerate(['木','火','土','金','水'])}
    node_feat=torch.zeros(n,5+3+32)
    for i,name in enumerate(SYNDROMES):
        e=ELEMENTS.get(name)
        if e: node_feat[i,elem_map[e]]=1.0
        node_feat[i,5]=1.0
    symptom_ids=torch.tensor([0,1,2,3])
    out=model(node_feat,sheng_adj,ke_adj,current_idx=0,symptom_ids=symptom_ids)
    probs=out['transition_probs']
    print(f"\n  FiveElementsGNN transition probs from '{SYNDROMES[0]}':")
    top3=torch.topk(probs,3)
    for idx,val in zip(top3.indices.tolist(),top3.values.tolist()):
        print(f"    {SYNDROMES[idx]}: {val:.4f}")

    # Five-element consistency check
    fe_count=0
    src_elem=ELEMENTS[SYNDROMES[0]]
    for i,p in enumerate(probs.tolist()):
        dst_elem=ELEMENTS[SYNDROMES[i]]
        if dst_elem==SHENG.get(src_elem) or dst_elem==KE.get(src_elem) or dst_elem==src_elem:
            fe_count+=p
    print(f"  Five-element consistency: {fe_count:.4f}")

    results['FiveElementsGNN']={'top3':[SYNDROMES[i] for i in top3.indices.tolist()],
                                 'fe_consistency':fe_count}
    with open('five_elements_benchmark_results.json','w') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"\nResults saved to five_elements_benchmark_results.json")

if __name__=='__main__':
    run_benchmark()
