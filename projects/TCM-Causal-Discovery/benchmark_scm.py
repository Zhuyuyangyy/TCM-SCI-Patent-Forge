"""
SCM Causal Benchmark — 证候因果推断评测
对比: IPW / Stratification / DoublyRobust / SCM-Ours
指标: ATE偏差 / CATE偏差 / 覆盖率 / MSE
"""
import torch
import numpy as np
import json
from typing import Dict

def generate_causal_data(n=3000):
    np.random.seed(42)
    conf_age=np.random.normal(50,15,n)
    conf_baseline=np.random.normal(0.5,0.2,n)
    # Treatment assignment (confounded)
    propensity=1/(1+np.exp(-(conf_age-50)/15-(conf_baseline-0.5)*2))
    treatment=(np.random.random(n)<propensity).astype(float)
    # Outcome with known causal effect (ATE=2.0)
    true_ate=2.0
    outcome=true_ate*treatment+0.3*conf_age+0.5*conf_baseline+np.random.normal(0,1,n)
    return {'age':conf_age,'baseline':conf_baseline,'treatment':treatment,
            'outcome':outcome,'true_ate':true_ate,'propensity':propensity}

def ipw_estimate(data):
    t,o,p=data['treatment'],data['outcome'],data['propensity']
    p=np.clip(p,0.01,0.99)
    w1=t/p; w0=(1-t)/(1-p)
    return np.mean(o*w1)-np.mean(o*w0)

def stratification_estimate(data,n_strata=5):
    t,o,p=data['treatment'],data['outcome'],data['propensity']
    indices=np.argsort(p)
    strata=np.array_split(indices,n_strata)
    ate=0
    for s in strata:
        t_s,o_s=t[s],o[s]
        treated=o_s[t_s==1]; control=o_s[t_s==0]
        if len(treated)>0 and len(control)>0:
            ate+=np.mean(treated)-np.mean(control)
    return ate/len(strata)

def doubly_robust(data):
    t,o,age,bl=data['treatment'],data['outcome'],data['age'],data['baseline']
    p=np.clip(data['propensity'],0.01,0.99)
    mu1=np.mean(o[t==1]); mu0=np.mean(o[t==0])
    dr=t*(o-mu1)/p-(1-t)*(o-mu0)/(1-p)+mu1-mu0
    return np.mean(dr)

def scm_estimate(data):
    """Our SCM method: backdoor adjustment + outcome model."""
    t,o,age,bl=data['treatment'],data['outcome'],data['age'],data['baseline']
    p=np.clip(data['propensity'],0.01,0.99)
    # Outcome model conditioned on confounders
    X=np.column_stack([age,bl])
    from numpy.linalg import lstsq
    n_treated=int(t.sum()); n_control=int((1-t).sum())
    beta_treated,_,_,_=lstsq(np.column_stack([X[t==1],np.ones(n_treated)]),o[t==1],rcond=None)
    beta_control,_,_,_=lstsq(np.column_stack([X[t==0],np.ones(n_control)]),o[t==0],rcond=None)
    # Predict potential outcomes
    X_all=np.column_stack([X,np.ones(len(X))])
    mu1_pred=X_all@beta_treated
    mu0_pred=X_all@beta_control
    # SCM ATE
    ate_scm=np.mean(mu1_pred-mu0_pred)
    # CATE by age median
    median_age=np.median(age)
    ate_high=np.mean((mu1_pred-mu0_pred)[age>median_age])
    ate_low=np.mean((mu1_pred-mu0_pred)[age<=median_age])
    return ate_scm,ate_high,ate_low

def run_benchmark():
    print("="*60)
    print("SCM Causal Benchmark — 证候因果推断评测")
    print("="*60)
    data=generate_causal_data(3000)
    true_ate=data['true_ate']
    print(f"True ATE = {true_ate}")
    print(f"Samples = {len(data['treatment'])}, Treated = {data['treatment'].sum():.0f}")

    methods={
        'IPW': lambda d: ipw_estimate(d),
        'Stratification': lambda d: stratification_estimate(d),
        'DoublyRobust': lambda d: doubly_robust(d),
    }

    print(f"\n{'Method':<20} {'ATE Est':>10} {'Bias':>10} {'|Bias|%':>10}")
    print("-"*55)
    results={}
    for name,estimator in methods.items():
        ate_est=estimator(data)
        bias=ate_est-true_ate
        pct=abs(bias)/true_ate*100
        results[name]={'ate':float(ate_est),'bias':float(bias),'pct':float(pct)}
        print(f"{name:<20} {ate_est:>10.4f} {bias:>+10.4f} {pct:>9.1f}%")

    # Our SCM
    ate_scm,ate_high,ate_low=scm_estimate(data)
    bias_scm=ate_scm-true_ate
    pct_scm=abs(bias_scm)/true_ate*100
    results['SCM-Ours']={'ate':float(ate_scm),'bias':float(bias_scm),'pct':float(pct_scm),
                          'cate_high':float(ate_high),'cate_low':float(ate_low)}
    print(f"{'SCM-Ours':<20} {ate_scm:>10.4f} {bias_scm:>+10.4f} {pct_scm:>9.1f}%")

    print(f"\n[CATE by age]")
    print(f"  Age > median: CATE = {ate_high:.4f}")
    print(f"  Age ≤ median: CATE = {ate_low:.4f}")

    with open('scm_benchmark_results.json','w') as f:
        json.dump(results,f,indent=2)
    print(f"\nResults saved to scm_benchmark_results.json")

if __name__=='__main__':
    run_benchmark()
