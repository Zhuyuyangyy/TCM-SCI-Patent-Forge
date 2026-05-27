"""
TCM Knowledge Graph Builder
SPO三元组构建器：症状→证候→治法→方药
"""
import json
from pathlib import Path
from collections import defaultdict

class TCMKnowledgeGraph:
    def __init__(self):
        self.nodes = {}  # node_id -> {type, name, properties}
        self.edges = []  # [(src, rel, dst)]
        self.adjacency = defaultdict(list)
    
    def add_node(self, node_id, node_type, name, properties=None):
        """添加节点到知识图谱"""
        self.nodes[node_id] = {
            'type': node_type,
            'name': name,
            'properties': properties or {}
        }
    
    def add_edge(self, src, rel, dst):
        """添加边到知识图谱"""
        self.edges.append((src, rel, dst))
        self.adjacency[src].append((rel, dst))
        # 双向索引
        self.adjacency[dst].append((rel, src))
    
    def build_from_text(self, text_content):
        """从文本中抽取SPO三元组"""
        # 基于规则的三元组抽取（简化版）
        lines = text_content.strip().split('\n')
        for line in lines:
            parts = line.split('→')
            if len(parts) == 3:
                src, rel, dst = parts[0].strip(), parts[1].strip(), parts[2].strip()
                node_ids = [src, dst]
                for nid in node_ids:
                    if nid not in self.nodes:
                        # 自动推断节点类型
                        node_type = self._infer_node_type(nid)
                        self.add_node(nid, node_type, nid)
                self.add_edge(src, rel, dst)
    
    def _infer_node_type(self, name):
        """根据名称推断节点类型"""
        syndrome_keywords = ['证', '症', '郁', '虚', '实', '热', '寒', '湿', '燥']
        symptom_keywords = ['痛', '胀', '闷', '呕', '泻', '秘', '咳', '喘', '悸', '眩']
        herb_keywords = ['汤', '散', '丸', '膏', '药']
        
        for kw in syndrome_keywords:
            if kw in name:
                return 'syndrome'
        for kw in symptom_keywords:
            if kw in name:
                return 'symptom'
        for kw in herb_keywords:
            if kw in name:
                return 'prescription'
        return 'unknown'
    
    def save(self, path):
        """保存为JSON"""
        data = {
            'nodes': self.nodes,
            'edges': self.edges
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path):
        """加载"""
        if not Path(path).exists():
            # 如果文件不存在，创建默认数据
            self._create_default_kg()
            self.save(path)
            return
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.nodes = data.get('nodes', {})
        self.edges = data.get('edges', [])
        
        # 重建邻接表
        self.adjacency = defaultdict(list)
        for src, rel, dst in self.edges:
            self.adjacency[src].append((rel, dst))
            self.adjacency[dst].append((rel, src))
    
    def _create_default_kg(self):
        """创建默认知识图谱"""
        # 节点定义
        nodes = [
            ('n1', 'syndrome', '肝郁气滞', {'properties': '主证', 'wuxing': '木'}),
            ('n2', 'syndrome', '肝郁化火', {'properties': '变证', 'wuxing': '木'}),
            ('n3', 'syndrome', '心脾两虚', {'properties': '虚证', 'wuxing': '火'}),
            ('n4', 'syndrome', '脾胃湿热', {'properties': '实证', 'wuxing': '土'}),
            ('n5', 'syndrome', '肺肾阴虚', {'properties': '虚证', 'wuxing': '金'}),
            ('n6', 'symptom', '胁肋胀痛', {}),
            ('n7', 'symptom', '情绪抑郁', {}),
            ('n8', 'symptom', '嗳气频繁', {}),
            ('n9', 'symptom', '脉弦', {}),
            ('n10', 'symptom', '心悸失眠', {}),
            ('n11', 'symptom', '食欲不振', {}),
            ('n12', 'prescription', '逍遥散', {'composition': '柴胡、当归、白芍、白术、茯苓、甘草'}),
            ('n13', 'prescription', '柴胡疏肝散', {'composition': '柴胡、陈皮、川芎、香附、枳壳、芍药、甘草'}),
        ]
        
        for nid, ntype, name, props in nodes:
            self.add_node(nid, ntype, name, props)
        
        # 边定义
        edges = [
            ('n6', 'indicates', 'n1'),
            ('n7', 'indicates', 'n1'),
            ('n8', 'indicates', 'n1'),
            ('n9', 'indicates', 'n1'),
            ('n1', 'can_transform_to', 'n2'),  # 肝郁→肝郁化火
            ('n10', 'indicates', 'n3'),
            ('n11', 'indicates', 'n4'),
            ('n1', 'treated_by', 'n12'),
            ('n1', 'treated_by', 'n13'),
            ('n2', 'treated_by', 'n13'),
        ]
        
        for src, rel, dst in edges:
            self.add_edge(src, rel, dst)
    
    def query_by_syndrome(self, syndrome):
        """查询某证候相关的所有节点和边"""
        # 找到证候节点
        syndrome_node = None
        for nid, node in self.nodes.items():
            if node['type'] == 'syndrome' and node['name'] == syndrome:
                syndrome_node = nid
                break
        
        if not syndrome_node:
            return {'nodes': [], 'edges': [], 'description': f'未找到证候: {syndrome}'}
        
        # BFS查找相关节点
        related_nodes = {syndrome_node: {'distance': 0}}
        related_edges = []
        queue = [syndrome_node]
        
        while queue:
            current = queue.pop(0)
            current_dist = related_nodes[current]['distance']
            
            if current_dist >= 2:
                continue
            
            for rel, neighbor in self.adjacency[current]:
                if neighbor not in related_nodes:
                    related_nodes[neighbor] = {'distance': current_dist + 1}
                    queue.append(neighbor)
                
                edge_info = {
                    'src': current,
                    'rel': rel,
                    'dst': neighbor,
                    'src_name': self.nodes[current]['name'],
                    'dst_name': self.nodes[neighbor]['name']
                }
                if edge_info not in related_edges:
                    related_edges.append(edge_info)
        
        return {
            'nodes': [(nid, self.nodes[nid]) for nid in related_nodes],
            'edges': related_edges,
            'description': f'找到 {len(related_nodes)} 个相关节点，{len(related_edges)} 条边'
        }
    
    def get_jun_chen_zuo_shi_weights(self, prescription):
        """获取君臣佐使权重"""
        # 君臣佐使配伍权重
        weights = {
            'jun': 0.5,   # 君药（主药）
            'chen': 0.3,  # 臣药（辅药）
            'zuo': 0.15,  # 佐药
            'shi': 0.05   # 使药
        }
        return weights
    
    def verify_wuxing_constraint(self, syndrome_from, syndrome_to):
        """验证五行相生相克约束"""
        from verification_layer import FiveElementsVerifier
        verifier = FiveElementsVerifier()
        return verifier.verify_transition(syndrome_from, syndrome_to)
    
    def get_all_syndromes(self):
        """获取所有证候节点"""
        return [(nid, n) for nid, n in self.nodes.items() if n['type'] == 'syndrome']
    
    def get_all_symptoms(self):
        """获取所有症状节点"""
        return [(nid, n) for nid, n in self.nodes.items() if n['type'] == 'symptom']
    
    def get_all_prescriptions(self):
        """获取所有方剂节点"""
        return [(nid, n) for nid, n in self.nodes.items() if n['type'] == 'prescription']


if __name__ == '__main__':
    # 测试代码
    kg = TCMKnowledgeGraph()
    kg._create_default_kg()
    print(f"节点数: {len(kg.nodes)}")
    print(f"边数: {len(kg.edges)}")
    
    # 测试查询
    result = kg.query_by_syndrome('肝郁气滞')
    print(f"\n查询结果: {result['description']}")
    for nid, node in result['nodes']:
        print(f"  - {node['name']} ({node['type']})")
