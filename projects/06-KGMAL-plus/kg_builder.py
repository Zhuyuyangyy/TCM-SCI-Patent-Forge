"""
Knowledge Graph Builder with Incremental Learning
支持增量学习的中医知识图谱构建器
"""
import json
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Set, Tuple

class KGNode:
    def __init__(self, node_id: str, node_type: str, name: str, properties: Dict = None):
        self.id = node_id
        self.type = node_type
        self.name = name
        self.properties = properties or {}
        self.attributes = set()

class KGMLIncrementalBuilder:
    """
    增量式知识图谱构建器
    支持:
    - 增量添加三元组（无需全量重建）
    - 冲突检测
    - 时序医案建模
    """
    def __init__(self):
        self.nodes: Dict[str, KGNode] = {}
        self.edges: List[Tuple[str, str, str]] = []
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.entity_to_id: Dict[str, str] = {}
        self.id_counter = 0
    
    def add_entity(self, name: str, entity_type: str, properties: Dict = None) -> str:
        """添加实体，返回实体ID"""
        if name in self.entity_to_id:
            node_id = self.entity_to_id[name]
            if properties:
                self.nodes[node_id].properties.update(properties)
            return node_id
        
        node_id = f"n_{self.id_counter}"
        self.id_counter += 1
        self.nodes[node_id] = KGNode(node_id, entity_type, name, properties)
        self.entity_to_id[name] = node_id
        return node_id
    
    def add_triplet(self, subject: str, predicate: str, obj: str, 
                    subject_type: str = 'entity', obj_type: str = 'entity'):
        """添加三元组"""
        sub_id = self.add_entity(subject, subject_type)
        obj_id = self.add_entity(obj, obj_type)
        self.edges.append((sub_id, predicate, obj_id))
        self.adjacency[sub_id].append((predicate, obj_id))
    
    def incremental_update(self, new_triplets: List[Dict]):
        """
        增量更新：添加新三元组，检测冲突
        new_triplets: [{'subject': ..., 'predicate': ..., 'object': ...}, ...]
        """
        conflicts = []
        for triplet in new_triplets:
            conflict = self._check_conflict(
                triplet['subject'], triplet['predicate'], triplet['object']
            )
            if conflict:
                conflicts.append(conflict)
            else:
                self.add_triplet(
                    triplet['subject'], 
                    triplet['predicate'], 
                    triplet['object']
                )
        
        return {
            'added': len(new_triplets) - len(conflicts),
            'conflicts': conflicts
        }
    
    def _check_conflict(self, subject: str, predicate: str, obj: str) -> Dict:
        """检测与现有知识的冲突"""
        opposites = {
            '相生': '相克',
            '相克': '相生',
            '虚证': '实证',
            '寒证': '热证'
        }
        
        for edge in self.edges:
            src, rel, dst = edge
            if (self.nodes[src].name == subject and 
                self.nodes[dst].name == obj):
                if rel in opposites and opposites[rel] == predicate:
                    return {
                        'existing': (self.nodes[src].name, rel, self.nodes[dst].name),
                        'new': (subject, predicate, obj),
                        'type': 'contradiction'
                    }
        return None
    
    def save(self, path: str):
        data = {
            'nodes': {k: {'type': v.type, 'name': v.name, 'properties': v.properties} 
                      for k, v in self.nodes.items()},
            'edges': [(self.nodes[s].name, p, self.nodes[o].name) 
                      for s, p, o in self.edges]
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
