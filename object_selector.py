"""
Object Selector Module for InteriorGS Dataset

This module handles filtering and selecting suitable objects and object pairs
from InteriorGS scene's labels.json file.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from itertools import combinations
from collections import Counter, defaultdict
from dataclasses import dataclass

try:
    from .config import ObjectSelectionConfig
except ImportError:
    from config import ObjectSelectionConfig


@dataclass
class SceneObject:
    """Represents a scene object with its properties from InteriorGS labels.json."""
    id: str
    label: str
    bbox_points: List[Dict[str, float]]  # Original 8 corner points
    dims: np.ndarray  # (width, depth, height) - x, y, z dimensions
    center: np.ndarray  # (x, y, z) center point
    aabb_min: np.ndarray  # Axis-aligned bounding box minimum
    aabb_max: np.ndarray  # Axis-aligned bounding box maximum
    room_index: Optional[int] = None
    
    @property
    def max_dim(self) -> float:
        """Maximum dimension of the object."""
        return float(np.max(self.dims))
    
    @property
    def min_dim(self) -> float:
        """Minimum dimension of the object."""
        return float(np.min(self.dims))
    
    @property
    def volume(self) -> float:
        """Volume of the object bounding box."""
        return float(np.prod(self.dims))
    
    @property
    def height(self) -> float:
        """Height of the object (z dimension)."""
        return float(self.dims[2])
    
    @property
    def top_z(self) -> float:
        """Top z coordinate of the object."""
        return float(self.aabb_max[2])
    
    @property
    def bottom_z(self) -> float:
        """Bottom z coordinate of the object."""
        return float(self.aabb_min[2])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'id': self.id,
            'label': self.label,
            'center': self.center.tolist(),
            'dims': self.dims.tolist(),
            'aabb_min': self.aabb_min.tolist(),
            'aabb_max': self.aabb_max.tolist(),
            'room_index': self.room_index,
            'volume': self.volume,
        }
    
    def get_obb_size(self) -> Tuple[float, float, float]:
        """
        Get object size as (length, width, height).
        Length is the longer of the two horizontal dimensions.
        """
        x_dim, y_dim, z_dim = self.dims
        # length = longer horizontal, width = shorter horizontal
        length = max(x_dim, y_dim)
        width = min(x_dim, y_dim)
        height = z_dim
        return float(length), float(width), float(height)


class ObjectSelector:
    """Selects suitable objects and object pairs from InteriorGS scenes."""
    
    def __init__(self, config: ObjectSelectionConfig):
        self.config = config
    
    def load_labels(self, labels_path: Path) -> List[Dict[str, Any]]:
        """Load objects from labels.json."""
        if not labels_path.exists():
            raise FileNotFoundError(f"labels.json not found: {labels_path}")
        
        with open(labels_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def load_room_polys(self, scene_path: Path) -> List[List[List[float]]]:
        """Load room polygons from structure.json if available."""
        structure_path = scene_path / 'structure.json'
        if not structure_path.exists():
            return []
        
        try:
            with open(structure_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            room_polys = []
            if isinstance(data, dict):
                rooms = data.get('rooms', [])
                for room in rooms:
                    poly = None
                    for key in ['profile', 'polygon', 'floor', 'boundary']:
                        if key in room:
                            poly = room[key]
                            break
                    if poly is not None and isinstance(poly, list) and len(poly) >= 3:
                        room_polys.append(poly)
            elif isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        for key in ['profile', 'polygon', 'floor', 'boundary']:
                            if key in item:
                                room_polys.append(item[key])
                                break
            return room_polys
        except Exception as e:
            print(f"Warning: Failed to parse structure.json: {e}")
            return []
    
    def point_in_poly(self, x: float, y: float, poly: List[List[float]]) -> bool:
        """Check if point (x,y) is inside polygon using ray casting."""
        if poly is None or len(poly) < 3:
            return False
        
        px = [p[0] for p in poly]
        py = [p[1] for p in poly]
        inside = False
        n = len(poly)
        j = n - 1
        
        for i in range(n):
            xi, yi = px[i], py[i]
            xj, yj = px[j], py[j]
            intersect = ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi)
            if intersect:
                inside = not inside
            j = i
        
        return inside
    
    def get_room_index_for_point(self, x: float, y: float, room_polys: List[List[List[float]]]) -> Optional[int]:
        """Return index of room containing point, or None."""
        for i, poly in enumerate(room_polys):
            if self.point_in_poly(x, y, poly):
                return i
        return None
    
    def parse_object(self, item: Dict[str, Any], room_polys: List[List[List[float]]]) -> Optional[SceneObject]:
        """Parse a single object from labels.json data."""
        bbox = item.get('bounding_box', [])
        if not bbox or len(bbox) < 8:
            return None
        
        # Get object ID and label
        obj_id = str(item.get('ins_id') or item.get('id') or '')
        label = str(item.get('label', '')).strip().lower()
        
        if not obj_id:
            return None
        
        # Extract coordinates from bounding box points
        xs = [float(p['x']) for p in bbox]
        ys = [float(p['y']) for p in bbox]
        zs = [float(p['z']) for p in bbox]
        
        aabb_min = np.array([min(xs), min(ys), min(zs)], dtype=float)
        aabb_max = np.array([max(xs), max(ys), max(zs)], dtype=float)
        dims = aabb_max - aabb_min
        center = (aabb_min + aabb_max) / 2
        
        # Get room index
        room_idx = self.get_room_index_for_point(float(center[0]), float(center[1]), room_polys)
        
        return SceneObject(
            id=obj_id,
            label=label,
            bbox_points=bbox,
            dims=dims,
            center=center,
            aabb_min=aabb_min,
            aabb_max=aabb_max,
            room_index=room_idx
        )
    
    def filter_single_object(self, obj: SceneObject) -> Tuple[bool, str]:
        """
        Filter a single object based on semantic and geometric constraints.
        
        Returns:
            (is_valid, rejection_reason)
        """
        cfg = self.config
        
        # Semantic filter - check blacklist
        if obj.label in cfg.blacklist:
            return False, 'blacklist'
        
        # Dimension filters (optional)
        if cfg.enable_dim_filter:
            if any(d < cfg.min_dim_component for d in obj.dims):
                return False, 'dim_too_small'
            if any(d > cfg.max_dim_component for d in obj.dims):
                return False, 'dim_too_large'
        
        # Volume filter (optional)
        if cfg.enable_volume_filter:
            if obj.volume < cfg.min_volume:
                return False, 'volume_too_small'
        
        # Aspect ratio filter (optional - avoid very flat objects)
        if cfg.enable_aspect_ratio_filter:
            if obj.min_dim / (obj.max_dim + 1e-9) < cfg.min_aspect_ratio:
                return False, 'too_flat'
        
        return True, 'ok'
    
    def aabb_min_distance(self, obj_a: SceneObject, obj_b: SceneObject) -> float:
        """Calculate minimum distance between two AABBs."""
        a_min, a_max = obj_a.aabb_min, obj_a.aabb_max
        b_min, b_max = obj_b.aabb_min, obj_b.aabb_max
        
        dx = max(0, max(a_min[0] - b_max[0], b_min[0] - a_max[0]))
        dy = max(0, max(a_min[1] - b_max[1], b_min[1] - a_max[1]))
        dz = max(0, max(a_min[2] - b_max[2], b_min[2] - a_max[2]))
        
        return float(math.sqrt(dx*dx + dy*dy + dz*dz))
    
    def center_distance(self, obj_a: SceneObject, obj_b: SceneObject) -> float:
        """Calculate Euclidean distance between object centers."""
        return float(np.linalg.norm(obj_a.center - obj_b.center))
    
    def filter_object_pair(self, obj_a: SceneObject, obj_b: SceneObject) -> Tuple[bool, str]:
        """
        Filter an object pair based on pair constraints.
        
        Returns:
            (is_valid, rejection_reason)
        """
        cfg = self.config
        
        # Semantic filter - check blacklist for BOTH objects
        # Objects in blacklist should not participate in spatial reasoning
        if obj_a.label in cfg.blacklist:
            return False, 'blacklist_obj_a'
        if obj_b.label in cfg.blacklist:
            return False, 'blacklist_obj_b'
        
        # Same room check - only if both objects have known room indices
        if obj_a.room_index is not None and obj_b.room_index is not None:
            if obj_a.room_index != obj_b.room_index:
                return False, 'different_rooms'
        
        # Distance check using AABB
        dist = self.aabb_min_distance(obj_a, obj_b)
        
        # Dynamic thresholds based on object size
        avg_max_dim = (obj_a.max_dim + obj_b.max_dim) / 2
        dyn_min = max(cfg.min_pair_dist, cfg.dyn_min_mult * avg_max_dim)
        dyn_max = min(cfg.max_pair_dist, cfg.dyn_max_mult * avg_max_dim)
        
        if dist < dyn_min:
            return False, 'too_close'
        if dist > dyn_max:
            return False, 'too_far'
        
        # Size difference check
        dim_ratio = max(obj_a.max_dim, obj_b.max_dim) / (min(obj_a.max_dim, obj_b.max_dim) + 1e-9)
        if dim_ratio > cfg.max_pair_dim_ratio:
            return False, 'size_ratio'
        
        dim_diff = abs(obj_a.max_dim - obj_b.max_dim)
        if dim_diff > cfg.max_pair_dim_diff:
            return False, 'size_diff'
        
        return True, 'ok'
    
    def select_single_objects(self, scene_path: Path) -> List[SceneObject]:
        """
        Select all valid single objects from a scene.
        
        Args:
            scene_path: Path to scene folder containing labels.json
        
        Returns:
            List of SceneObject instances that pass all filters
        """
        labels_path = scene_path / 'labels.json'
        data = self.load_labels(labels_path)
        room_polys = self.load_room_polys(scene_path)
        
        # Parse all objects
        all_objects = []
        for item in data:
            obj = self.parse_object(item, room_polys)
            if obj is not None:
                all_objects.append(obj)
        
        # Filter individual objects
        candidates = []
        for obj in all_objects:
            is_valid, reason = self.filter_single_object(obj)
            if is_valid:
                candidates.append(obj)
        
        # Apply label uniqueness within scene/room to avoid ambiguity
        labels_norm = [c.label for c in candidates]
        global_counts = Counter(labels_norm)
        room_counts = defaultdict(Counter)
        for c in candidates:
            room_counts[c.room_index][c.label] += 1
        
        # Keep objects that are unique globally or unique within their room
        filtered = []
        for c in candidates:
            gc = global_counts[c.label]
            rc = room_counts[c.room_index][c.label]
            # Object is unambiguous if it's the only one of its type
            # or if it's unique in its room
            if gc == 1 or rc == 1:
                filtered.append(c)
        
        return filtered
    
    def get_all_parsed_objects(self, scene_path: Path, include_walls: bool = True) -> List[SceneObject]:
        """
        Get all parsed objects from a scene (no filtering).
        
        Args:
            scene_path: Path to scene folder containing labels.json
            include_walls: If True, also include walls from structure.json
            
        Returns:
            List of all SceneObject instances (no filtering applied)
        """
        labels_path = scene_path / 'labels.json'
        data = self.load_labels(labels_path)
        room_polys = self.load_room_polys(scene_path)
        
        all_objects = []
        for item in data:
            obj = self.parse_object(item, room_polys)
            if obj is not None:
                all_objects.append(obj)
        
        # Add walls from structure.json (walls in labels.json don't have geometry)
        if include_walls:
            wall_objects = self.load_walls_from_structure(scene_path)
            all_objects.extend(wall_objects)
        
        return all_objects
    
    def load_walls_from_structure(self, scene_path: Path) -> List[SceneObject]:
        """
        Load wall geometry from structure.json.
        
        Walls in labels.json don't have bounding_box data, so we need to
        get wall geometry from structure.json which contains:
        - thickness: wall thickness
        - height: wall height
        - location: [[x1, y1], [x2, y2]] - two endpoints of the wall
        
        Args:
            scene_path: Path to scene folder containing structure.json
            
        Returns:
            List of SceneObject instances representing walls
        """
        structure_path = scene_path / 'structure.json'
        if not structure_path.exists():
            return []
        
        try:
            with open(structure_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load structure.json for walls: {e}")
            return []
        
        walls_data = data.get('walls', [])
        if not walls_data:
            return []
        
        wall_objects = []
        for i, wall in enumerate(walls_data):
            try:
                thickness = float(wall.get('thickness', 0.1))
                height = float(wall.get('height', 2.8))
                location = wall.get('location', [])
                
                if len(location) < 2:
                    continue
                
                # Get wall endpoints
                x1, y1 = float(location[0][0]), float(location[0][1])
                x2, y2 = float(location[1][0]), float(location[1][1])
                
                # Compute wall direction and perpendicular for thickness
                dx = x2 - x1
                dy = y2 - y1
                length = math.sqrt(dx*dx + dy*dy)
                if length < 1e-6:
                    continue
                
                # Normalize direction
                dx /= length
                dy /= length
                
                # Perpendicular direction for thickness
                px = -dy * (thickness / 2)
                py = dx * (thickness / 2)
                
                # Compute AABB for the wall
                corners_x = [x1 + px, x1 - px, x2 + px, x2 - px]
                corners_y = [y1 + py, y1 - py, y2 + py, y2 - py]
                
                aabb_min = np.array([min(corners_x), min(corners_y), 0.0], dtype=float)
                aabb_max = np.array([max(corners_x), max(corners_y), height], dtype=float)
                
                dims = aabb_max - aabb_min
                center = (aabb_min + aabb_max) / 2
                
                wall_obj = SceneObject(
                    id=f"wall_{i}",
                    label="wall",
                    bbox_points=[],  # No original bbox points for walls
                    dims=dims,
                    center=center,
                    aabb_min=aabb_min,
                    aabb_max=aabb_max,
                    room_index=None
                )
                wall_objects.append(wall_obj)
                
            except Exception as e:
                # Skip malformed wall data
                continue
        
        return wall_objects
    
    def select_object_pairs(self, scene_path: Path, 
                            objects: Optional[List[SceneObject]] = None,
                            max_pairs: int = 100,
                            use_all_objects: bool = True) -> List[Tuple[SceneObject, SceneObject]]:
        """
        Select all valid object pairs from a scene.
        
        Args:
            scene_path: Path to scene folder
            objects: Pre-filtered list of objects (if None, determined by use_all_objects)
            max_pairs: Maximum number of pairs to return
            use_all_objects: If True and objects is None, use all parsed objects;
                           If False and objects is None, use valid single objects only
        
        Returns:
            List of (obj_a, obj_b) tuples
        """
        if objects is None:
            if use_all_objects:
                objects = self.get_all_parsed_objects(scene_path)
            else:
                objects = self.select_single_objects(scene_path)
        
        valid_pairs = []
        for obj_a, obj_b in combinations(objects, 2):
            is_valid, reason = self.filter_object_pair(obj_a, obj_b)
            if is_valid:
                valid_pairs.append((obj_a, obj_b))
                if len(valid_pairs) >= max_pairs:
                    break
        
        return valid_pairs
    
    def select_object_triples(self, scene_path: Path,
                              objects: Optional[List[SceneObject]] = None,
                              max_triples: int = 50,
                              use_all_objects: bool = True) -> List[Tuple[SceneObject, SceneObject, SceneObject]]:
        """
        Select all valid object triples from a scene.
        
        Args:
            scene_path: Path to scene folder
            objects: Pre-filtered list of objects
            max_triples: Maximum number of triples to return
            use_all_objects: If True and objects is None, use all parsed objects;
                           If False and objects is None, use valid single objects only
        
        Returns:
            List of (obj_a, obj_b, obj_c) tuples
        """
        if objects is None:
            if use_all_objects:
                objects = self.get_all_parsed_objects(scene_path)
            else:
                objects = self.select_single_objects(scene_path)
        
        valid_triples = []
        for combo in combinations(objects, 3):
            # Check all pairs in the triple
            all_valid = True
            for i, obj_a in enumerate(combo):
                for obj_b in combo[i+1:]:
                    is_valid, reason = self.filter_object_pair(obj_a, obj_b)
                    if not is_valid:
                        all_valid = False
                        break
                if not all_valid:
                    break
            
            if all_valid:
                valid_triples.append(combo)
                if len(valid_triples) >= max_triples:
                    break
        
        return valid_triples
    
    def get_scene_statistics(self, scene_path: Path, use_all_objects: bool = True) -> Dict[str, Any]:
        """Get statistics about objects in a scene."""
        labels_path = scene_path / 'labels.json'
        data = self.load_labels(labels_path)
        room_polys = self.load_room_polys(scene_path)
        
        all_objects = []
        for item in data:
            obj = self.parse_object(item, room_polys)
            if obj is not None:
                all_objects.append(obj)
        
        valid_objects = self.select_single_objects(scene_path)
        
        # Object pairs are now selected from all objects by default
        valid_pairs = self.select_object_pairs(scene_path, use_all_objects=use_all_objects)
        
        label_counts = Counter(obj.label for obj in all_objects)
        valid_label_counts = Counter(obj.label for obj in valid_objects)
        
        return {
            'total_objects': len(all_objects),
            'valid_objects': len(valid_objects),
            'valid_pairs': len(valid_pairs),
            'pair_selection_source': 'all_objects' if use_all_objects else 'valid_single_objects',
            'label_distribution': dict(label_counts),
            'valid_label_distribution': dict(valid_label_counts),
            'num_rooms': len(room_polys),
        }
