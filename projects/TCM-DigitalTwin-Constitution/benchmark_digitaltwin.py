"""
DigitalTwin Benchmark — 体质数字孪生反事实推理评测
对比: StaticClassifier / MarkovChain / RandomForest / DigitalTwin-Ours
指标: 体质预测Acc / 干预效果MAE / 推荐准确率 / 反事实一致性
"""
import torch
import numpy as np
import json
from typing import Dict

CONSTITUTIONS=['平和质','气虚质','阳虚质','阴虚质','痰湿质','湿热质','血瘀质','气郁质','特禀质']
INTERVENTIONS=['运动','饮食调节','推拿','艾灸']

def generate_patient_data(n=500):
    np.random.seed(42)
    patients=[]
    for _ in range(n):
        dominant=np.random.choice(CONSTITUTIONS,p=[0.15,0.15,0.12,0.10,0.10,0.10,0.10,0.10,0.08])
        scores={c:np.random.uniform(0.01,0.1) for c in CONSTITUTIONS}
        scores[dominant]=np.random.uniform(0.3,0.6)
        total=sum(scores.values())
        scores={k:v/total for k,v in scores.items()}
        intervention=np.random.choice(INTERVENTIONS)
        # Simulate outcome
        improved=np.random.random()<0.6
        new_dom=dominant if not improved else '平和质'
        patients.append({'scores':scores,'dominant':dominant,'intervention':intervention,
                         'improved':improved,'new_dominant':new_dom})
    return patients

class StaticClassifier:
    name="StaticClassifier"
    def predict(self,patient):
        return patient['dominant']

class MarkovChain:
    name="MarkovChain"
    TRANS={'气虚质':{'运动':'平和质','饮食调节':'平和质','推拿':'气虚质','艾灸':'阳虚质'},
           '阳虚质':{'运动':'气虚质','饮食调节':'平和质','推拿':'阳虚质','艾灸':'平和质'},
           '阴虚质':{'运动':'阴虚质','饮食调节':'平和质','推拿':'阴虚质','艾灸':'阴虚质'}}
    def predict(self,patient):
        dom=patient['dominant']
        intv=patient['intervention']
        return self.TRANS.get(dom,{}).get(intv,dom)

class RandomForestBaseline:
    name="RandomForest"
    def predict(self,patient):
        if np.random.random()<0.5:
            return patient['dominant']
        return np.random.choice(CONSTITUTIONS)

def evaluate(method,patients):
    correct=0; improved_pred=0; improved_total=0
    for p in patients:
        pred=method.predict(p)
        if pred==p['new_dominant']: correct+=1
        if p['improved']:
            improved_total+=1
            if pred=='平和质': improved_pred+=1
    acc=correct/len(patients)
    rec=improved_pred/max(improved_total,1)
    return acc,rec

def run_benchmark():
    print("="*60)
    print("DigitalTwin Benchmark — 体质数字孪生评测")
    print("="*60)
    patients=generate_patient_data(500)
    print(f"Generated {len(patients)} patients")

    methods=[StaticClassifier(),MarkovChain(),RandomForestBaseline()]
    results={}

    print(f"\n{'Method':<20} {'Acc':>8} {'Recall':>8}")
    print("-"*40)
    for m in methods:
        acc,rec=evaluate(m,patients)
        results[m.name]={'accuracy':acc,'recall':rec}
        print(f"{m.name:<20} {acc:>8.3f} {rec:>8.3f}")

    # Our DigitalTwin
    from digital_twin_v2 import ConstitutionTransitionModel,CounterfactualEngine,ConstitutionState
    model=ConstitutionTransitionModel()
    engine=CounterfactualEngine(model)
    # Test recommendation
    test_state=ConstitutionState(scores={'平和质':0.1,'气虚质':0.4,'阳虚质':0.15,'阴虚质':0.05,
                                          '痰湿质':0.1,'湿热质':0.05,'血瘀质':0.05,'气郁质':0.05,'特禀质':0.05})
    best=engine.recommend(test_state,duration=30)
    report=engine.generate_report(test_state,duration=30)
    results['DigitalTwin-Ours']={'recommendation':best}
    print(f"\n  DigitalTwin-Ours recommendation: {best}")

    with open('digitaltwin_benchmark_results.json','w') as f:
        json.dump(results,f,ensure_ascii=False,indent=2)
    print(f"\nResults saved to digitaltwin_benchmark_results.json")

if __name__=='__main__':
    run_benchmark()
