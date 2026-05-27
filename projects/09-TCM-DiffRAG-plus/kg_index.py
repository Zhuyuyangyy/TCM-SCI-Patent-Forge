"""
Knowledge Graph Index for RAG
为RAG系统构建知识图谱索引
"""
import json
from collections import defaultdict
from typing import List, Dict, Set, Tuple

class KGIndex:
    """
    知识图谱索引
    支持:
    - 实体检索
    - 关系路径查询
    - 子图检索
    """
    def __init__(self):
        self.entities: Dict[str, Dict] = {}  # name -> {type, properties}
        self.relations: List[Tuple[str, str, str]] = []  # (subj, rel, obj)
        self.adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        self.entity_mentions: Dict[str, List[str]] = defaultdict(list)  # 实体别名
    
    def add_entity(self, name: str, entity_type: str, properties: Dict = None, aliases: List[str] = None):
        """添加实体"""
        self.entities[name] = {
            'type': entity_type,
            'properties': properties or {},
            'aliases': aliases or []
        }
        # 别名索引
        for alias in (aliases or []):
            self.entity_mentions[alias].append(name)
        self.entity_mentions[name].append(name)
    
    def add_relation(self, subject: str, predicate: str, obj: str):
        """添加关系"""
        if subject not in self.entities:
            self.add_entity(subject, 'unknown')
        if obj not in self.entities:
            self.add_entity(obj, 'unknown')
        self.relations.append((subject, predicate, obj))
        self.adjacency[subject].append((predicate, obj))
    
    def search_entity(self, query: str, top_k: int = 5) -> List[Dict]:
        """搜索实体（模糊匹配）"""
        query_lower = query.lower()
        results = []
        
        for name, entity in self.entities.items():
            score = 0.0
            if query_lower in name.lower():
                score = 1.0
            elif any(query_lower in alias.lower() for alias in entity.get('aliases', [])):
                score = 0.8
            
            if score > 0:
                results.append({
                    'name': name,
                    'type': entity['type'],
                    'score': score,
                    'properties': entity['properties']
                })
        
        results.sort(key=lambda x: -x['score'])
        return results[:top_k]
    
    def get_neighborhood(self, entity: str, depth: int = 1, rel_type: str = None) -> Dict:
        """
        获取实体的邻域子图
        """
        visited = set()
        to_visit = [(entity, 0)]
        subgraph = {'nodes': [], 'edges': []}
        node_ids = set()
        
        while to_visit:
            curr, d = to_visit.pop(0)
            if curr in visited or d > depth:
                continue
            visited.add(curr)
            
            if curr not in node_ids:
                node_ids.add(curr)
                node_data = self.entities.get(curr, {'type': 'unknown', 'properties': {}})
                subgraph['nodes'].append({
                    'id': curr,
                    **node_data
                })
            
            for pred, obj in self.adjacency.get(curr, []):
                if rel_type and pred != rel_type:
                    continue
                subgraph['edges'].append({'from': curr, 'rel': pred, 'to': obj})
                if obj not in visited:
                    to_visit.append((obj, d + 1))
                    if obj not in node_ids:
                        node_ids.add(obj)
                        obj_data = self.entities.get(obj, {'type': 'unknown', 'properties': {}})
                        subgraph['nodes'].append({'id': obj, **obj_data})
        
        return subgraph
    
    def query_path(self, start: str, end: str, max_hops: int = 3) -> List[List[Tuple[str, str, str]]]:
        """查询两实体间的路径"""
        paths = []
        
        def dfs(current: str, goal: str, path: List, visited: Set):
            if len(path) > max_hops:
                return
            if current == goal:
                paths.append(path.copy())
                return
            for pred, obj in self.adjacency.get(current, []):
                if obj not in visited:
                    visited.add(obj)
                    path.append((current, pred, obj))
                    dfs(obj, goal, path, visited)
                    path.pop()
                    visited.remove(obj)
        
        dfs(start, start, [], {start})
        return paths
    
    def save(self, path: str):
        data = {
            'entities': self.entities,
            'relations': self.relations
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.entities = data['entities']
        self.relations = data['relations']
        for subj, pred, obj in self.relations:
            self.adjacency[subj].append((pred, obj))
