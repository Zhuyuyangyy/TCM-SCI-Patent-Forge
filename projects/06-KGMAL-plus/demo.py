# KGMAL++ Demo: Lifelong Incremental TCM Knowledge Learning
from kg_builder import KGMLIncrementalBuilder
from temporal_model import TemporalMedicalRecord, LSTMEncoder
from catastrophic_forgetting import ExperienceReplay, EWCRegularizer

def run_demo():
    print("=" * 60)
    print("KGMAL++ Demo: Lifelong Incremental TCM Knowledge Learning")
    print("=" * 60)
    
    # Step 1
    print()
    print("[Step 1] Incremental Knowledge Graph Construction...")
    kg = KGMLIncrementalBuilder()
    
    kg.add_triplet("肝郁气滞", "导致", "胁肋胀痛")
    kg.add_triplet("肝郁气滞", "治法", "疏肝理气")
    kg.add_triplet("柴胡", "功效", "疏肝解郁")
    kg.add_triplet("柴胡", "归经", "肝、胆")
    
    print(f"  Initial nodes: {len(kg.nodes)}")
    print(f"  Initial edges: {len(kg.edges)}")
    
    new_knowledge = [
        {"subject": "肝郁化火", "predicate": "发展自", "object": "肝郁气滞"},
        {"subject": "肝火亢盛", "predicate": "导致", "object": "口苦"},
        {"subject": "栀子", "predicate": "功效", "object": "清肝泻火"}
    ]
    
    result = kg.incremental_update(new_knowledge)
    print(f"  Added: {result['added']}, Conflicts: {len(result['conflicts'])}")
    print(f"  After update - nodes: {len(kg.nodes)}")
    
    # Step 2
    print()
    print("[Step 2] Temporal Medical Record Modeling...")
    patient = TemporalMedicalRecord("P001")
    patient.add_record(1, ["胁肋胀痛", "情绪抑郁"], "柴胡疏肝散", "部分缓解")
    patient.add_record(2, ["胁肋隐痛", "失眠"], "逍遥丸+酸枣仁", "明显好转")
    patient.add_record(3, ["偶有胁痛", "情绪改善", "食欲增加"], "继续逍遥丸", "基本痊愈")
    
    traj = patient.get_symptom_trajectory()
    print(f"  Symptom trajectory: {list(traj.keys())}")
    pred = patient.predict_treatment_effect(["胁肋胀痛", "情绪抑郁", "失眠"])
    print(f"  Treatment prediction: {pred}")
    
    # Step 3
    print()
    print("[Step 3] Catastrophic Forgetting Suppression...")
    replay = ExperienceReplay(capacity=100)
    for i in range(50):
        replay.add({"syndrome": f"Syndrome{i}", "sample_id": i}, priority=1.0 + i*0.1)
    
    batch = replay.sample_uniform(10)
    print(f"  Experience replay: sampled {len(batch)} historical samples")
    
    ewc = EWCRegularizer(lambda_ewc=1000)
    print(f"  EWC coefficient: {ewc.lambda_ewc}")
    
    print()
    print("=" * 60)
    print("Demo Complete!")

if __name__ == "__main__":
    run_demo()
