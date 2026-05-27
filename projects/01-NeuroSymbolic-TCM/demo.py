"""
NeuroSymbolic-TCM 完整演示
运行合成数据测试
"""
import sys
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from kg_builder import TCMKnowledgeGraph
from syndrome_differentiator import SyndromeDifferentiator
from verification_layer import FiveElementsVerifier

def run_demo():
    print("=" * 60)
    print("NeuroSymbolic-TCM Demo: 神经符号混合推理系统")
    print("=" * 60)
    
    # Step 1: 构建KG
    print("\n[Step 1] 构建中医知识图谱...")
    kg = TCMKnowledgeGraph()
    # 添加核心节点和边
    kg.add_node('n1', 'syndrome', '肝郁气滞', {'properties': '主证', 'wuxing': '木'})
    kg.add_node('n2', 'symptom', '胁肋胀痛', {})
    kg.add_node('n3', 'symptom', '情绪抑郁', {})
    kg.add_node('n4', 'syndrome', '肝郁化火', {'properties': '变证', 'wuxing': '木'})
    kg.add_node('n5', 'symptom', '暖气频繁', {})
    kg.add_node('n6', 'symptom', '脉弦', {})
    kg.add_node('n7', 'prescription', '逍遥散', {'composition': '柴胡、当归、白芍、白术、茯苓、甘草', 'roles': {'柴胡': 'jun', '当归': 'chen', '白芍': 'chen', '白术': 'zuo', '茯苓': 'zuo', '甘草': 'shi'}})
    
    kg.add_edge('n2', 'indicates', 'n1')
    kg.add_edge('n3', 'indicates', 'n1')
    kg.add_edge('n5', 'indicates', 'n1')
    kg.add_edge('n6', 'indicates', 'n1')
    kg.add_edge('n1', 'can_transform_to', 'n4')
    kg.add_edge('n1', 'treated_by', 'n7')
    
    demo_kg_path = '/tmp/tcm_kg_demo.json'
    kg.save(demo_kg_path)
    print(f"  KG构建完成: {len(kg.nodes)} nodes, {len(kg.edges)} edges")
    print(f"  已保存到: {demo_kg_path}")
    
    # Step 2: 证候鉴别
    print("\n[Step 2] 证候鉴别...")
    diff = SyndromeDifferentiator()
    result = diff.differentiate(['胁肋胀痛', '情绪抑郁', '暖气频繁', '脉弦'])
    print(f"  主证: {result['primary_syndrome']}")
    print(f"  置信度: {result['confidence']:.2f}")
    print(f"  五行: {result.get('wuxing', '未知')}")
    print(f"  推理: {result['reasoning']}")
    
    if result.get('candidate_syndromes'):
        print("\n  候选证候详情:")
        for c in result['candidate_syndromes']:
            print(f"    - {c['syndrome']}: {c['confidence']:.2f} (五行: {c.get('wuxing', '?')})")
            if c.get('constraint_notes'):
                print(f"      约束检查: {c['constraint_notes']}")
    
    # Step 3: 五行验证
    print("\n[Step 3] 五行约束验证...")
    verifier = FiveElementsVerifier()
    
    test_cases = [
        ('木', '火', '肝属木 → 心属火'),
        ('木', '土', '肝属木 → 脾属土'),
        ('火', '金', '心属火 → 肺属金'),
    ]
    
    for from_wx, to_wx, desc in test_cases:
        # 直接用五行属性验证
        ok, reason = verifier.verify_transition(
            {'木': '肝郁', '火': '心火', '土': '脾虚', '金': '肺燥', '水': '肾寒'}.get(from_wx, from_wx),
            {'木': '肝郁', '火': '心火', '土': '脾虚', '金': '肺燥', '水': '肾寒'}.get(to_wx, to_wx)
        )
        status = "✓ 通过" if ok else "✗ 失败"
        print(f"  {status}: {desc} - {reason}")
    
    # Step 4: 知识图谱查询演示
    print("\n[Step 4] 知识图谱深度查询...")
    query_result = kg.query_by_syndrome('肝郁气滞')
    print(f"  查询 '肝郁气滞' 相关节点和边:")
    print(f"  {query_result['description']}")
    
    print("\n  相关边关系:")
    for edge in query_result['edges']:
        print(f"    {edge['src_name']} --[{edge['rel']}]--> {edge['dst_name']}")
    
    # Step 5: 君臣佐使配伍演示
    print("\n[Step 5] 君臣佐使配伍理论演示...")
    weights = kg.get_jun_chen_zuo_shi_weights('逍遥散')
    print("  逍遥散配伍权重:")
    for role, weight in weights.items():
        role_desc = {'jun': '君(主药)', 'chen': '臣(辅药)', 'zuo': '佐(佐药)', 'shi': '使(使药)'}
        print(f"    {role_desc[role]}: {weight:.0%}")
    
    # Step 6: KG保存和重新加载测试
    print("\n[Step 6] KG持久化测试...")
    kg2 = TCMKnowledgeGraph()
    kg2.load(demo_kg_path)
    print(f"  重新加载KG: {len(kg2.nodes)} nodes, {len(kg2.edges)} edges")
    
    # 验证加载正确性
    assert len(kg2.nodes) == len(kg.nodes), "节点数量不匹配"
    assert len(kg2.edges) == len(kg.edges), "边数量不匹配"
    print("  ✓ KG持久化验证通过")
    
    print("\n" + "=" * 60)
    print("Demo 完成!")
    print("\nNeuroSymbolic-TCM 系统演示总结:")
    print("  1. 知识图谱: 支持症状→证候→治法→方药的SPO三元组构建")
    print("  2. 证候鉴别: 基于规则+KG约束的智能辨证")
    print("  3. 五行验证: 木火土金水相生相克硬约束")
    print("  4. 君臣佐使: 0.5:0.3:0.15:0.05权重注意力机制")
    print("=" * 60)


if __name__ == '__main__':
    run_demo()
