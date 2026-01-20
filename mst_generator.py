#!/usr/bin/env python3
"""
MST-based Training Data Generator for Spatial Reasoning

This module constructs Minimum Spanning Trees (MST) from valid object pairs
to generate training and evaluation data for spatial reasoning tasks.

Key Insight:
- With N objects, only N-1 edges (forming a tree) are needed to uniquely 
  determine all relative positions in 3D space.
- Training (SFT): Use MST edges (e.g., A-B, B-C, C-D)
- Evaluation: Use non-MST edges (e.g., A-C, B-D, A-D)

Usage:
    python mst_generator.py --scenes_root /path/to/InteriorGS --scene_id 0267_840790
    python mst_generator.py --scenes_root /path/to/InteriorGS --output mst_data.json
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

try:
    from object_selector import ObjectSelector, SceneObject
    from config import ObjectSelectionConfig
except ImportError:
    from .object_selector import ObjectSelector, SceneObject
    from .config import ObjectSelectionConfig


@dataclass
class Edge:
    """Represents an edge in the object graph."""
    obj_a: SceneObject
    obj_b: SceneObject
    distance: float  # Weight: distance between object centers
    
    def __lt__(self, other):
        return self.distance < other.distance
    
    def to_dict(self) -> Dict[str, Any]:
        # Compute vector from obj_a center to obj_b center
        vector_a_to_b = [
            round(self.obj_b.center[0] - self.obj_a.center[0], 4),
            round(self.obj_b.center[1] - self.obj_a.center[1], 4),
            round(self.obj_b.center[2] - self.obj_a.center[2], 4),
        ]
        return {
            'obj_a_id': self.obj_a.id,
            'obj_a_label': self.obj_a.label,
            'obj_a_center': [round(c, 4) for c in self.obj_a.center],
            'obj_b_id': self.obj_b.id,
            'obj_b_label': self.obj_b.label,
            'obj_b_center': [round(c, 4) for c in self.obj_b.center],
            'vector_a_to_b': vector_a_to_b,  # [dx, dy, dz] from A to B
            'distance': round(self.distance, 4),
        }


@dataclass
class ObjectGraph:
    """Graph structure for objects and their valid pair relationships."""
    objects: List[SceneObject] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    adjacency: Dict[str, List[Edge]] = field(default_factory=lambda: defaultdict(list))
    
    def add_edge(self, obj_a: SceneObject, obj_b: SceneObject, distance: float):
        edge = Edge(obj_a, obj_b, distance)
        self.edges.append(edge)
        self.adjacency[obj_a.id].append(edge)
        self.adjacency[obj_b.id].append(edge)
    
    @property
    def num_vertices(self) -> int:
        return len(self.objects)
    
    @property
    def num_edges(self) -> int:
        return len(self.edges)


class UnionFind:
    """Union-Find (Disjoint Set Union) data structure for Kruskal's algorithm."""
    
    def __init__(self, elements: List[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}
    
    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]
    
    def union(self, x: str, y: str) -> bool:
        """Union two sets. Returns True if they were in different sets."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        # Union by rank
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        return True


class MSTGenerator:
    """Generates Minimum Spanning Trees from valid object pairs."""
    
    def __init__(self, config: ObjectSelectionConfig = None):
        self.config = config or ObjectSelectionConfig()
        self.selector = ObjectSelector(self.config)
    
    def build_object_graph(self, scene_path: Path) -> ObjectGraph:
        """
        Build a graph where vertices are objects and edges are valid pairs.
        
        Args:
            scene_path: Path to scene folder
            
        Returns:
            ObjectGraph with all valid object pairs as edges
        """
        # Get all valid pairs (from all objects, as per our updated logic)
        valid_pairs = self.selector.select_object_pairs(
            scene_path, objects=None, use_all_objects=True, max_pairs=10000
        )
        
        # Collect all unique objects that appear in valid pairs
        objects_in_pairs: Dict[str, SceneObject] = {}
        for obj_a, obj_b in valid_pairs:
            objects_in_pairs[obj_a.id] = obj_a
            objects_in_pairs[obj_b.id] = obj_b
        
        graph = ObjectGraph()
        graph.objects = list(objects_in_pairs.values())
        
        # Add edges with distance as weight
        for obj_a, obj_b in valid_pairs:
            distance = self.selector.center_distance(obj_a, obj_b)
            graph.add_edge(obj_a, obj_b, distance)
        
        return graph
    
    def find_mst_kruskal(self, graph: ObjectGraph) -> List[Edge]:
        """
        Find Minimum Spanning Tree using Kruskal's algorithm.
        
        Args:
            graph: ObjectGraph with edges
            
        Returns:
            List of edges forming the MST
        """
        if not graph.edges:
            return []
        
        # Sort edges by weight (distance)
        sorted_edges = sorted(graph.edges)
        
        # Initialize Union-Find
        object_ids = [obj.id for obj in graph.objects]
        uf = UnionFind(object_ids)
        
        mst_edges = []
        for edge in sorted_edges:
            if uf.union(edge.obj_a.id, edge.obj_b.id):
                mst_edges.append(edge)
                # MST is complete when we have N-1 edges
                if len(mst_edges) == len(object_ids) - 1:
                    break
        
        return mst_edges
    
    def build_room_graphs(self, scene_path: Path) -> Dict[int, ObjectGraph]:
        """
        Build separate graphs for each room.
        
        Args:
            scene_path: Path to scene folder
            
        Returns:
            Dictionary mapping room_index to ObjectGraph
        """
        # Get all valid pairs
        valid_pairs = self.selector.select_object_pairs(
            scene_path, objects=None, use_all_objects=True, max_pairs=10000
        )
        
        # Group by room
        room_objects: Dict[int, Dict[str, SceneObject]] = defaultdict(dict)
        room_edges: Dict[int, List[Tuple[SceneObject, SceneObject, float]]] = defaultdict(list)
        
        for obj_a, obj_b in valid_pairs:
            # Only include pairs where both objects are in the same known room
            if obj_a.room_index is not None and obj_a.room_index == obj_b.room_index:
                room_idx = obj_a.room_index
                room_objects[room_idx][obj_a.id] = obj_a
                room_objects[room_idx][obj_b.id] = obj_b
                distance = self.selector.center_distance(obj_a, obj_b)
                room_edges[room_idx].append((obj_a, obj_b, distance))
        
        # Build graphs for each room
        room_graphs = {}
        for room_idx in room_objects:
            graph = ObjectGraph()
            graph.objects = list(room_objects[room_idx].values())
            for obj_a, obj_b, dist in room_edges[room_idx]:
                graph.add_edge(obj_a, obj_b, dist)
            room_graphs[room_idx] = graph
        
        return room_graphs
    
    def generate_room_mst_data(self, scene_path: Path) -> Dict[str, Any]:
        """
        Generate MST data separately for each room.
        
        This is useful when you want to train spatial reasoning within rooms,
        where objects in the same room should form a coherent spatial graph.
        
        Returns:
            Dictionary with per-room MST data
        """
        scene_name = scene_path.name
        room_graphs = self.build_room_graphs(scene_path)
        
        if not room_graphs:
            return {
                'scene_id': scene_name,
                'error': 'No valid room-based pairs found',
            }
        
        all_mst_edges = []
        all_non_mst_edges = []
        room_details = []
        
        for room_idx, graph in room_graphs.items():
            if not graph.edges:
                continue
            
            mst_edges = self.find_mst_kruskal(graph)
            mst_edge_set = {(e.obj_a.id, e.obj_b.id) for e in mst_edges}
            mst_edge_set.update({(e.obj_b.id, e.obj_a.id) for e in mst_edges})
            
            non_mst_edges = [e for e in graph.edges 
                           if (e.obj_a.id, e.obj_b.id) not in mst_edge_set]
            
            all_mst_edges.extend(mst_edges)
            all_non_mst_edges.extend(non_mst_edges)
            
            room_details.append({
                'room_index': room_idx,
                'num_objects': graph.num_vertices,
                'num_edges': graph.num_edges,
                'mst_edges': len(mst_edges),
                'non_mst_edges': len(non_mst_edges),
                'object_labels': [obj.label for obj in graph.objects],
            })
        
        result = {
            'scene_id': scene_name,
            'mode': 'per_room',
            'num_rooms': len(room_graphs),
            'total_mst_edges': len(all_mst_edges),
            'total_non_mst_edges': len(all_non_mst_edges),
            'training_data': {
                'description': 'MST edges per room - teach model these relationships',
                'edges': [e.to_dict() for e in all_mst_edges],
            },
            'evaluation_data': {
                'description': 'Non-MST edges - test inference within rooms',
                'edges': [e.to_dict() for e in all_non_mst_edges],
            },
            'room_details': room_details,
        }
        
        return result
    
    def find_connected_components(self, graph: ObjectGraph) -> List[List[SceneObject]]:
        """Find all connected components in the graph."""
        if not graph.objects:
            return []
        
        visited = set()
        components = []
        
        obj_map = {obj.id: obj for obj in graph.objects}
        
        for obj in graph.objects:
            if obj.id in visited:
                continue
            
            # BFS to find component
            component = []
            queue = [obj.id]
            
            while queue:
                curr_id = queue.pop(0)
                if curr_id in visited:
                    continue
                
                visited.add(curr_id)
                component.append(obj_map[curr_id])
                
                # Add neighbors
                for edge in graph.adjacency[curr_id]:
                    neighbor_id = edge.obj_b.id if edge.obj_a.id == curr_id else edge.obj_a.id
                    if neighbor_id not in visited:
                        queue.append(neighbor_id)
            
            components.append(component)
        
        return components
    
    def generate_mst_data(self, scene_path: Path) -> Dict[str, Any]:
        """
        Generate MST-based training and evaluation data for a scene.
        
        Returns:
            Dictionary containing:
            - scene_id: Scene identifier
            - graph_stats: Graph statistics
            - mst_edges: Edges in the MST (for training)
            - non_mst_edges: Edges not in MST (for evaluation)
            - connected_components: Info about graph connectivity
        """
        scene_name = scene_path.name
        
        # Build graph from valid pairs
        graph = self.build_object_graph(scene_path)
        
        if not graph.edges:
            return {
                'scene_id': scene_name,
                'error': 'No valid pairs found',
                'graph_stats': {'vertices': 0, 'edges': 0}
            }
        
        # Find connected components
        components = self.find_connected_components(graph)
        
        # Find MST for each connected component
        all_mst_edges = []
        
        # For disconnected graphs, we find MST for each component
        # But first, let's find the global MST (will give forest for disconnected graph)
        mst_edges = self.find_mst_kruskal(graph)
        mst_edge_set = {(e.obj_a.id, e.obj_b.id) for e in mst_edges}
        mst_edge_set.update({(e.obj_b.id, e.obj_a.id) for e in mst_edges})
        
        # Separate MST and non-MST edges
        non_mst_edges = []
        for edge in graph.edges:
            key = (edge.obj_a.id, edge.obj_b.id)
            if key not in mst_edge_set:
                non_mst_edges.append(edge)
        
        # Calculate statistics
        total_weight_mst = sum(e.distance for e in mst_edges)
        total_weight_non_mst = sum(e.distance for e in non_mst_edges)
        
        # Build result
        result = {
            'scene_id': scene_name,
            'graph_stats': {
                'total_objects_in_scene': len(self.selector.get_all_parsed_objects(scene_path)),
                'objects_in_valid_pairs': graph.num_vertices,
                'total_valid_pairs': graph.num_edges,
                'connected_components': len(components),
                'largest_component_size': max(len(c) for c in components) if components else 0,
            },
            'mst_stats': {
                'mst_edges': len(mst_edges),
                'non_mst_edges': len(non_mst_edges),
                'total_mst_distance': round(total_weight_mst, 4),
                'avg_mst_edge_distance': round(total_weight_mst / len(mst_edges), 4) if mst_edges else 0,
            },
            'training_data': {
                'description': 'MST edges - teach model these N-1 relationships',
                'edges': [e.to_dict() for e in mst_edges],
            },
            'evaluation_data': {
                'description': 'Non-MST edges - test if model can infer these relationships',
                'edges': [e.to_dict() for e in non_mst_edges],
            },
            'connected_components_detail': [
                {
                    'component_id': i,
                    'size': len(comp),
                    'objects': [{'id': obj.id, 'label': obj.label} for obj in comp]
                }
                for i, comp in enumerate(components)
            ]
        }
        
        return result
    
    def generate_complete_mst_data(self, scene_path: Path) -> Dict[str, Any]:
        """
        Generate complete MST-based data with three categories:
        1. MST edges (training) - valid pairs in the MST
        2. Non-MST valid edges (inference with image) - valid pairs not in MST
        3. Non-MST invalid edges (blind evaluation) - all other possible pairs
        
        Returns:
            Dictionary with three separate edge lists for different use cases
        """
        from itertools import combinations
        
        scene_name = scene_path.name
        
        # Get all objects in the scene
        all_objects = self.selector.get_all_parsed_objects(scene_path)
        all_obj_map = {obj.id: obj for obj in all_objects}
        
        # Get valid pairs
        valid_pairs = self.selector.select_object_pairs(
            scene_path, objects=None, use_all_objects=True, max_pairs=100000
        )
        valid_pair_set = {(a.id, b.id) for a, b in valid_pairs}
        valid_pair_set.update({(b.id, a.id) for a, b in valid_pairs})
        
        # Build graph from valid pairs
        graph = self.build_object_graph(scene_path)
        
        if not graph.edges:
            return {
                'scene_id': scene_name,
                'error': 'No valid pairs found',
            }
        
        # Find MST
        mst_edges = self.find_mst_kruskal(graph)
        mst_edge_set = {(e.obj_a.id, e.obj_b.id) for e in mst_edges}
        mst_edge_set.update({(e.obj_b.id, e.obj_a.id) for e in mst_edges})
        
        # Categorize all possible edges
        # Category 1: MST edges (training)
        training_edges = mst_edges
        
        # Category 2: Non-MST valid edges (inference with image)
        inference_edges = []
        for edge in graph.edges:
            key = (edge.obj_a.id, edge.obj_b.id)
            if key not in mst_edge_set:
                inference_edges.append(edge)
        
        # Category 3: Non-MST invalid edges (blind evaluation)
        # These are all possible pairs that are NOT valid pairs
        blind_edges = []
        objects_in_graph = {obj.id: obj for obj in graph.objects}
        
        # Only consider pairs within the same connected component for meaningful evaluation
        # Get objects that are in the graph (have at least one valid pair)
        for obj_a, obj_b in combinations(all_objects, 2):
            pair_key = (obj_a.id, obj_b.id)
            
            # Skip if it's already a valid pair
            if pair_key in valid_pair_set:
                continue
            
            # Check why this pair is invalid
            is_valid, rejection_reason = self.selector.filter_object_pair(obj_a, obj_b)
            
            if not is_valid:
                distance = self.selector.center_distance(obj_a, obj_b)
                blind_edges.append({
                    'obj_a_id': obj_a.id,
                    'obj_a_label': obj_a.label,
                    'obj_b_id': obj_b.id,
                    'obj_b_label': obj_b.label,
                    'distance': round(distance, 4),
                    'rejection_reason': rejection_reason,
                    'obj_a_room': obj_a.room_index,
                    'obj_b_room': obj_b.room_index,
                })
        
        # Calculate statistics
        total_possible_pairs = len(all_objects) * (len(all_objects) - 1) // 2
        
        result = {
            'scene_id': scene_name,
            'statistics': {
                'total_objects': len(all_objects),
                'objects_in_valid_pairs': graph.num_vertices,
                'total_possible_pairs': total_possible_pairs,
                'valid_pairs': graph.num_edges,
                'invalid_pairs': total_possible_pairs - graph.num_edges,
            },
            'edge_counts': {
                'mst_edges': len(training_edges),
                'non_mst_valid_edges': len(inference_edges),
                'non_mst_invalid_edges': len(blind_edges),
            },
            # Category 1: Training data (MST edges)
            'training_mst': {
                'description': 'MST edges for SFT training - can render images',
                'count': len(training_edges),
                'edges': [e.to_dict() for e in training_edges],
            },
            # Category 2: Inference with image (non-MST valid edges)
            'eval_with_image': {
                'description': 'Non-MST valid edges for inference - can render images',
                'count': len(inference_edges),
                'edges': [e.to_dict() for e in inference_edges],
            },
            # Category 3: Blind evaluation (non-MST invalid edges)
            'eval_blind': {
                'description': 'Invalid pairs for blind evaluation - cannot render images together',
                'count': len(blind_edges),
                'edges': blind_edges,  # Already in dict format with rejection_reason
            },
        }
        
        return result
    
    def save_complete_mst_data(self, scene_path: Path, output_dir: Path) -> Dict[str, Path]:
        """
        Generate and save MST data to separate files.
        
        Creates three files:
        - {scene_id}_train_mst.json: MST edges for training
        - {scene_id}_eval_with_image.json: Non-MST valid edges for inference
        - {scene_id}_eval_blind.json: Invalid pairs for blind evaluation
        
        Returns:
            Dictionary with paths to the created files
        """
        data = self.generate_complete_mst_data(scene_path)
        scene_id = data['scene_id']
        
        if 'error' in data:
            print(f"Error: {data['error']}")
            return {}
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save training data
        train_path = output_dir / f"{scene_id}_train_mst.json"
        with open(train_path, 'w') as f:
            json.dump({
                'scene_id': scene_id,
                'split': 'train',
                'description': data['training_mst']['description'],
                'count': data['training_mst']['count'],
                'edges': data['training_mst']['edges'],
            }, f, indent=2)
        
        # Save eval with image data
        eval_image_path = output_dir / f"{scene_id}_eval_with_image.json"
        with open(eval_image_path, 'w') as f:
            json.dump({
                'scene_id': scene_id,
                'split': 'eval_with_image',
                'description': data['eval_with_image']['description'],
                'count': data['eval_with_image']['count'],
                'edges': data['eval_with_image']['edges'],
            }, f, indent=2)
        
        # Save blind eval data
        eval_blind_path = output_dir / f"{scene_id}_eval_blind.json"
        with open(eval_blind_path, 'w') as f:
            json.dump({
                'scene_id': scene_id,
                'split': 'eval_blind',
                'description': data['eval_blind']['description'],
                'count': data['eval_blind']['count'],
                'edges': data['eval_blind']['edges'],
            }, f, indent=2)
        
        return {
            'train': train_path,
            'eval_with_image': eval_image_path,
            'eval_blind': eval_blind_path,
        }

    def generate_mst_questions(
        self, 
        scene_path: Path,
        question_type: str = 'object_pair_distance_vector'
    ) -> Tuple[List[Dict], List[Dict]]:
        """
        Generate question-answer pairs split into training (MST) and evaluation (non-MST).
        
        Args:
            scene_path: Path to scene folder
            question_type: Type of question to generate
            
        Returns:
            Tuple of (training_questions, evaluation_questions)
        """
        mst_data = self.generate_mst_data(scene_path)
        
        if 'error' in mst_data:
            return [], []
        
        training_questions = []
        evaluation_questions = []
        
        # Generate questions for MST edges (training)
        for edge_info in mst_data['training_data']['edges']:
            q = {
                'obj_a_id': edge_info['obj_a_id'],
                'obj_a_label': edge_info['obj_a_label'],
                'obj_b_id': edge_info['obj_b_id'],
                'obj_b_label': edge_info['obj_b_label'],
                'distance': edge_info['distance'],
                'split': 'train',
                'is_mst_edge': True,
                'question_type': question_type,
            }
            training_questions.append(q)
        
        # Generate questions for non-MST edges (evaluation)
        for edge_info in mst_data['evaluation_data']['edges']:
            q = {
                'obj_a_id': edge_info['obj_a_id'],
                'obj_a_label': edge_info['obj_a_label'],
                'obj_b_id': edge_info['obj_b_id'],
                'obj_b_label': edge_info['obj_b_label'],
                'distance': edge_info['distance'],
                'split': 'eval',
                'is_mst_edge': False,
                'question_type': question_type,
            }
            evaluation_questions.append(q)
        
        return training_questions, evaluation_questions


def print_mst_summary(result: Dict[str, Any]):
    """Print a formatted summary of MST analysis."""
    print("\n" + "=" * 70)
    print(f"MST ANALYSIS: {result['scene_id']}")
    print("=" * 70)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    stats = result['graph_stats']
    mst_stats = result['mst_stats']
    
    print(f"\n📊 Graph Statistics:")
    print(f"   Total objects in scene: {stats['total_objects_in_scene']}")
    print(f"   Objects in valid pairs: {stats['objects_in_valid_pairs']}")
    print(f"   Total valid pairs (edges): {stats['total_valid_pairs']}")
    print(f"   Connected components: {stats['connected_components']}")
    print(f"   Largest component: {stats['largest_component_size']} objects")
    
    print(f"\n🌳 MST Statistics:")
    print(f"   MST edges (training): {mst_stats['mst_edges']}")
    print(f"   Non-MST edges (eval): {mst_stats['non_mst_edges']}")
    print(f"   Total MST distance: {mst_stats['total_mst_distance']:.2f}m")
    print(f"   Avg MST edge distance: {mst_stats['avg_mst_edge_distance']:.2f}m")
    
    # Training/eval ratio
    total_edges = mst_stats['mst_edges'] + mst_stats['non_mst_edges']
    if total_edges > 0:
        train_pct = mst_stats['mst_edges'] / total_edges * 100
        eval_pct = mst_stats['non_mst_edges'] / total_edges * 100
        print(f"\n📈 Train/Eval Split:")
        print(f"   Training (MST edges): {mst_stats['mst_edges']} ({train_pct:.1f}%)")
        print(f"   Evaluation (non-MST): {mst_stats['non_mst_edges']} ({eval_pct:.1f}%)")
    
    print(f"\n🔗 Connected Components:")
    for comp in result['connected_components_detail'][:5]:  # Show first 5
        obj_labels = [o['label'] for o in comp['objects'][:5]]
        suffix = f"... (+{len(comp['objects'])-5} more)" if len(comp['objects']) > 5 else ""
        print(f"   Component {comp['component_id']}: {comp['size']} objects")
        print(f"      [{', '.join(obj_labels)}{suffix}]")
    
    if len(result['connected_components_detail']) > 5:
        print(f"   ... and {len(result['connected_components_detail']) - 5} more components")
    
    print(f"\n📝 Sample MST Edges (Training):")
    for edge in result['training_data']['edges'][:5]:
        print(f"   {edge['obj_a_label']} ({edge['obj_a_id']}) ↔ {edge['obj_b_label']} ({edge['obj_b_id']}): {edge['distance']:.2f}m")
    
    print(f"\n📝 Sample Non-MST Edges (Evaluation):")
    for edge in result['evaluation_data']['edges'][:5]:
        print(f"   {edge['obj_a_label']} ({edge['obj_a_id']}) ↔ {edge['obj_b_label']} ({edge['obj_b_id']}): {edge['distance']:.2f}m")
    
    print("\n" + "=" * 70)


def print_room_mst_summary(result: Dict[str, Any]):
    """Print a formatted summary of per-room MST analysis."""
    print("\n" + "=" * 70)
    print(f"PER-ROOM MST ANALYSIS: {result['scene_id']}")
    print("=" * 70)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    print(f"\n📊 Overview:")
    print(f"   Mode: per_room")
    print(f"   Number of rooms: {result['num_rooms']}")
    print(f"   Total MST edges (training): {result['total_mst_edges']}")
    print(f"   Total non-MST edges (eval): {result['total_non_mst_edges']}")
    
    total = result['total_mst_edges'] + result['total_non_mst_edges']
    if total > 0:
        print(f"\n📈 Train/Eval Split:")
        print(f"   Training: {result['total_mst_edges']} ({result['total_mst_edges']/total*100:.1f}%)")
        print(f"   Evaluation: {result['total_non_mst_edges']} ({result['total_non_mst_edges']/total*100:.1f}%)")
    
    print(f"\n🏠 Room Details:")
    for room in result.get('room_details', [])[:10]:
        labels = room['object_labels'][:5]
        suffix = f"... (+{len(room['object_labels'])-5} more)" if len(room['object_labels']) > 5 else ""
        print(f"   Room {room['room_index']}: {room['num_objects']} objects, {room['mst_edges']} MST / {room['non_mst_edges']} non-MST edges")
        print(f"      [{', '.join(labels)}{suffix}]")
    
    if len(result.get('room_details', [])) > 10:
        print(f"   ... and {len(result['room_details']) - 10} more rooms")
    
    print(f"\n📝 Sample MST Edges (Training):")
    for edge in result['training_data']['edges'][:5]:
        print(f"   {edge['obj_a_label']} ({edge['obj_a_id']}) ↔ {edge['obj_b_label']} ({edge['obj_b_id']}): {edge['distance']:.2f}m")
    
    print(f"\n📝 Sample Non-MST Edges (Evaluation):")
    for edge in result['evaluation_data']['edges'][:5]:
        print(f"   {edge['obj_a_label']} ({edge['obj_a_id']}) ↔ {edge['obj_b_label']} ({edge['obj_b_id']}): {edge['distance']:.2f}m")
    
    print("\n" + "=" * 70)


def print_complete_mst_summary(result: Dict[str, Any]):
    """Print a formatted summary of complete MST analysis with three categories."""
    print("\n" + "=" * 70)
    print(f"COMPLETE MST ANALYSIS: {result['scene_id']}")
    print("=" * 70)
    
    if 'error' in result:
        print(f"Error: {result['error']}")
        return
    
    stats = result['statistics']
    counts = result['edge_counts']
    
    print(f"\n📊 Scene Statistics:")
    print(f"   Total objects: {stats['total_objects']}")
    print(f"   Objects in valid pairs: {stats['objects_in_valid_pairs']}")
    print(f"   Total possible pairs: {stats['total_possible_pairs']}")
    print(f"   Valid pairs: {stats['valid_pairs']}")
    print(f"   Invalid pairs: {stats['invalid_pairs']}")
    
    print(f"\n📈 Edge Categories:")
    total = counts['mst_edges'] + counts['non_mst_valid_edges'] + counts['non_mst_invalid_edges']
    
    print(f"   🌳 MST edges (training):           {counts['mst_edges']:6d} ({counts['mst_edges']/total*100:5.1f}%)")
    print(f"   📷 Non-MST valid (with image):     {counts['non_mst_valid_edges']:6d} ({counts['non_mst_valid_edges']/total*100:5.1f}%)")
    print(f"   🔮 Non-MST invalid (blind eval):   {counts['non_mst_invalid_edges']:6d} ({counts['non_mst_invalid_edges']/total*100:5.1f}%)")
    
    print(f"\n📝 Sample MST Edges (Training - can render image):")
    for edge in result['training_mst']['edges'][:3]:
        print(f"   {edge['obj_a_label']} ↔ {edge['obj_b_label']}: {edge['distance']:.2f}m")
    
    print(f"\n📝 Sample Non-MST Valid Edges (Inference - can render image):")
    for edge in result['eval_with_image']['edges'][:3]:
        print(f"   {edge['obj_a_label']} ↔ {edge['obj_b_label']}: {edge['distance']:.2f}m")
    
    print(f"\n📝 Sample Non-MST Invalid Edges (Blind eval - no image):")
    for edge in result['eval_blind']['edges'][:5]:
        print(f"   {edge['obj_a_label']} ↔ {edge['obj_b_label']}: {edge['distance']:.2f}m")
        print(f"      Reason: {edge['rejection_reason']} (rooms: {edge['obj_a_room']} vs {edge['obj_b_room']})")
    
    print("\n" + "=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description='Generate MST-based training data for spatial reasoning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--scenes_root', type=str, required=True,
                        help='Root directory containing InteriorGS scenes')
    parser.add_argument('--scene_id', type=str, default=None,
                        help='Specific scene to analyze (default: all scenes)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory for saving split files (when using --mode complete)')
    parser.add_argument('--max_scenes', type=int, default=10,
                        help='Maximum number of scenes to process (default: 10)')
    parser.add_argument('--mode', type=str, default='complete',
                        choices=['global', 'per_room', 'complete'],
                        help='MST mode: global, per_room, or complete (with 3 categories)')
    
    args = parser.parse_args()
    
    scenes_root = Path(args.scenes_root)
    generator = MSTGenerator()
    
    if args.scene_id:
        # Single scene
        scene_path = scenes_root / args.scene_id
        if not scene_path.exists():
            print(f"Error: Scene not found: {scene_path}")
            return
        
        if args.mode == 'complete':
            result = generator.generate_complete_mst_data(scene_path)
            print_complete_mst_summary(result)
            
            # Save to separate files if output directory specified
            if args.output:
                output_dir = Path(args.output)
                saved_files = generator.save_complete_mst_data(scene_path, output_dir)
                print(f"\nSaved files:")
                for split, path in saved_files.items():
                    print(f"  {split}: {path}")
        elif args.mode == 'per_room':
            result = generator.generate_room_mst_data(scene_path)
            print_room_mst_summary(result)
        else:
            result = generator.generate_mst_data(scene_path)
            print_mst_summary(result)
        all_results = [result]
    else:
        # Multiple scenes
        scene_dirs = [d for d in scenes_root.iterdir() 
                      if d.is_dir() and (d / 'labels.json').exists()]
        
        print(f"Processing {min(len(scene_dirs), args.max_scenes)} scenes...")
        all_results = []
        
        for scene_path in sorted(scene_dirs)[:args.max_scenes]:
            if args.mode == 'complete':
                result = generator.generate_complete_mst_data(scene_path)
                print_complete_mst_summary(result)
                
                if args.output:
                    output_dir = Path(args.output)
                    generator.save_complete_mst_data(scene_path, output_dir)
            elif args.mode == 'per_room':
                result = generator.generate_room_mst_data(scene_path)
                print_room_mst_summary(result)
            else:
                result = generator.generate_mst_data(scene_path)
                print_mst_summary(result)
            all_results.append(result)
        
        # Aggregate summary
        print("\n" + "=" * 70)
        print("AGGREGATE SUMMARY")
        print("=" * 70)
        
        if args.mode == 'complete':
            total_mst = sum(r['edge_counts']['mst_edges'] for r in all_results if 'edge_counts' in r)
            total_valid = sum(r['edge_counts']['non_mst_valid_edges'] for r in all_results if 'edge_counts' in r)
            total_invalid = sum(r['edge_counts']['non_mst_invalid_edges'] for r in all_results if 'edge_counts' in r)
            
            print(f"Total MST edges (training):          {total_mst}")
            print(f"Total non-MST valid (with image):    {total_valid}")
            print(f"Total non-MST invalid (blind eval):  {total_invalid}")
            
            total = total_mst + total_valid + total_invalid
            if total > 0:
                print(f"\nDistribution:")
                print(f"  Training:        {total_mst/total*100:.1f}%")
                print(f"  Eval with image: {total_valid/total*100:.1f}%")
                print(f"  Blind eval:      {total_invalid/total*100:.1f}%")
        elif args.mode == 'per_room':
            total_mst = sum(r.get('total_mst_edges', 0) for r in all_results)
            total_non_mst = sum(r.get('total_non_mst_edges', 0) for r in all_results)
            print(f"Total MST edges (training): {total_mst}")
            print(f"Total non-MST edges (eval): {total_non_mst}")
        else:
            total_mst = sum(r['mst_stats']['mst_edges'] for r in all_results if 'mst_stats' in r)
            total_non_mst = sum(r['mst_stats']['non_mst_edges'] for r in all_results if 'mst_stats' in r)
            print(f"Total MST edges (training): {total_mst}")
            print(f"Total non-MST edges (eval): {total_non_mst}")
            if total_mst + total_non_mst > 0:
                print(f"Train/Eval ratio: {total_mst/(total_mst+total_non_mst)*100:.1f}% / {total_non_mst/(total_mst+total_non_mst)*100:.1f}%")
    
    # For non-complete modes, save all results to single file
    if args.output and args.mode != 'complete':
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()