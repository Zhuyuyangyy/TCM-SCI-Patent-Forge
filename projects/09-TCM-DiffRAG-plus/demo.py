
"""TCM-DiffRAG++ 完整演示"""
from kg_index import KGIndex
from diff_rag import DifferentialRAG, PatientProfile

def build_demo_kg():
    kg = KGIndex()
    kg.add_entity("肝郁气滞", "syndrome", {"description": "肝气郁结，胸胁胀痛"}, ["肝郁"])
    kg.add_entity("柴胡", "herb", {"efficacy": "疏肝解郁", "dosage": "3-10g"}, ["柴胡根"])
    kg.add_entity("逍遥丸", "formula", {"composition": "柴胡+白芍+当归+白术+茯苓+甘草"}, ["XiaoYaoWan"])
    kg.add_entity("当归", "herb", {"efficacy": "补血活血", "dosage": "6-15g"})
    kg.add_entity("白芍", "herb", {"efficacy": "养血柔肝", "dosage": "6-15g"})
    kg.add_entity("茯苓", "herb", {"efficacy": "健脾利湿", "dosage": "10-15g"})
    kg.add_entity("肝火亢盛", "syndrome", {"description": "肝郁化火，急躁易怒"}, ["肝火旺"])
    kg.add_relation("肝郁气滞", "治法", "疏肝理气")
    kg.add_relation("肝郁气滞", "代表方", "逍遥丸")
    kg.add_relation("逍遥丸", "君药", "柴胡")
    kg.add_relation("逍遥丸", "臣药", "白芍")
    kg.add_relation("逍遥丸", "臣药", "当归")
    kg.add_relation("逍遥丸", "佐药", "茯苓")
    kg.add_relation("柴胡", "功效", "疏肝解郁")
    kg.add_relation("柴胡", "归经", "肝、胆")
    kg.add_relation("当归", "功效", "补血活血")
    kg.add_relation("白芍", "功效", "养血柔肝")
    kg.add_relation("肝郁气滞", "可发展为", "肝火亢盛")
    kg.add_relation("肝火亢盛", "治法", "清肝泻火")
    return kg

def run_demo():
    print("=" * 60)
    print("TCM-DiffRAG++ Demo: 差异化知识图谱RAG")
    print("=" * 60)
    print()
    print("[Step 1] 构建中医知识图谱...")
    kg = build_demo_kg()
    print(f"  实体数: {len(kg.entities)}, 关系数: {len(kg.relations)}")
    rag = DifferentialRAG(kg)
    patients = [
        PatientProfile("P001", constitution="气郁质", age=35, contraindications=["川乌"]),
        PatientProfile("P002", constitution="平和质", age=45),
        PatientProfile("P003", constitution="特禀质", age=28, contraindications=["人参"]),
    ]
    query = "肝郁气滞"
    print()
    print(f"[Step 2] 查询: {query}")
    for patient in patients:
        print()
        print(f"--- 患者: {patient.patient_id} ({patient.constitution}) ---")
        results = rag.retrieve(query, patient, top_k=3)
        response = rag.generate_response(results, patient)
        print(response)
    print()
    print("[Step 3] 子图检索: 逍遥丸邻域...")
    subgraph = kg.get_neighborhood("逍遥丸", depth=2)
    node_names = [n["id"] for n in subgraph["nodes"]]
    print(f"  节点: {node_names}")
    edge_strs = []
    for e in subgraph["edges"]:
        edge_strs.append(e["from"] + "-" + e["rel"] + "->" + e["to"])
    print(f"  边: {edge_strs}")
    print()
    print("[Step 4] 路径查询: 肝郁气滞 -> 肝火亢盛...")
    paths = kg.query_path("肝郁气滞", "肝火亢盛", max_hops=3)
    for path in paths:
        path_str = " -> ".join([s + "(" + r + ")" + o for s, r, o in path])
        print(f"  {path_str}")
    print()
    print("=" * 60)
    print("Demo 完成!")

if __name__ == "__main__":
    run_demo()
