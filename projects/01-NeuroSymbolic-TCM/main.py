"""
NeuroSymbolic-TCM 主程序
神经符号混合推理系统
"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from kg_builder import TCMKnowledgeGraph
from syndrome_differentiator import SyndromeDifferentiator
from verification_layer import FiveElementsVerifier

def main():
    print("=" * 60)
    print("NeuroSymbolic-TCM 神经符号混合推理系统")
    print("=" * 60)
    
    # 1. 构建知识图谱
    print("\n[Step 1] 初始化知识图谱...")
    kg = TCMKnowledgeGraph()
    kg_path = 'data/tcm_kg.json'
    
    # 尝试加载知识图谱，如果不存在则创建默认
    if not Path(kg_path).exists():
        print(f"  知识图谱文件不存在，创建默认 KG...")
        kg._create_default_kg()
        kg.save(kg_path)
        print(f"  已保存到 {kg_path}")
    else:
        kg.load(kg_path)
    
    print(f"  KG加载完成: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    
    # 2. 初始化证候鉴别器
    print("\n[Step 2] 初始化证候鉴别器...")
    differentiator = SyndromeDifferentiator(kg_path=kg_path)
    print("  证候鉴别器就绪")
    
    # 3. 初始化验证器
    print("\n[Step 3] 初始化五行验证器...")
    verifier = FiveElementsVerifier()
    print("  五行验证器就绪")
    
    # 4. 演示病例
    print("\n[Step 4] 运行病例辨证...")
    symptoms = ['胁肋胀痛', '情绪抑郁', '暖气频繁', '脉弦']
    print(f"  症状输入: {' + '.join(symptoms)}")
    
    result = differentiator.differentiate(symptoms)
    print(f"\n  === 辨证结果 ===")
    print(f"  主证: {result['primary_syndrome']}")
    print(f"  置信度: {result['confidence']:.2%}")
    print(f"  五行: {result.get('wuxing', '未知')}")
    print(f"  推理: {result['reasoning']}")
    
    if result.get('candidate_syndromes'):
        print("\n  候选证候:")
        for c in result['candidate_syndromes']:
            print(f"    - {c['syndrome']}: {c['confidence']:.2%}")
    
    # 5. 五行约束验证
    print("\n[Step 5] 五行约束验证...")
    test_transitions = [
        ('肝郁气滞', '肝郁化火'),  # 木→木 同属
        ('肝郁气滞', '心脾两虚'),  # 木→火 相生
        ('心脾两虚', '脾胃湿热'),  # 火→土 相生
    ]
    
    for from_syn, to_syn in test_transitions:
        ok, reason = verifier.verify_transition(from_syn, to_syn)
        status = "✓" if ok else "✗"
        print(f"  {status} {from_syn} → {to_syn}: {reason}")
    
    # 6. 知识图谱查询演示
    print("\n[Step 6] 知识图谱查询...")
    query_result = kg.query_by_syndrome('肝郁气滞')
    print(f"  查询结果: {query_result['description']}")
    
    # 显示相关节点
    print("  相关症状:")
    for nid, node in query_result['nodes']:
        if node['type'] == 'symptom':
            print(f"    - {node['name']}")
    
    print("  相关方剂:")
    for nid, node in query_result['nodes']:
        if node['type'] == 'prescription':
            print(f"    - {node['name']}")
            if 'properties' in node and 'composition' in node['properties']:
                print(f"      组成: {node['properties']['composition']}")
    
    print("\n" + "=" * 60)
    print("程序运行完成!")


if __name__ == '__main__':
    main()
