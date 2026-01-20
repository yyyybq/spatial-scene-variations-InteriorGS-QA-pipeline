"""
Camera Sampler Module for InteriorGS Dataset

This module handles sampling valid camera poses around objects for question generation.
Uses camera_utils.py from active_spatial_pipeline for accurate FOV and occlusion checking.
"""

import json
import math
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field

try:
    from .config import CameraSamplingConfig
    from .object_selector import SceneObject
    from .camera_utils import (
        SceneBounds,
        AABB,
        CameraPose,
        is_target_in_fov,
        is_target_occluded,
        is_wall_occluded,
        count_visible_corners,
    )
except ImportError:
    from config import CameraSamplingConfig
    from object_selector import SceneObject
    from camera_utils import (
        SceneBounds,
        AABB,
        CameraPose,
        is_target_in_fov,
        is_target_occluded,
        is_wall_occluded,
        count_visible_corners,
    )


def scene_object_to_aabb(obj: SceneObject) -> AABB:
    """Convert SceneObject to AABB for visibility checking."""
    return AABB(
        id=str(obj.id),
        label=obj.label,
        bmin=obj.aabb_min,
        bmax=obj.aabb_max
    )


def get_visible_objects(
    objects: List[SceneObject], 
    camera_pose: CameraPose,
    intrinsics: np.ndarray,
    image_width: int,
    image_height: int,
    all_scene_aabbs: Optional[List[AABB]] = None,
    check_occlusion: bool = True,
    min_visible_corners: int = 2
) -> List[SceneObject]:
    """
    Filter objects to only those visible from the camera pose.
    
    Uses accurate projection-based FOV checking from camera_utils.
    
    Args:
        objects: List of all objects to check
        camera_pose: Camera pose
        intrinsics: 3x3 camera intrinsic matrix
        image_width: Image width in pixels
        image_height: Image height in pixels
        all_scene_aabbs: All AABBs in scene for occlusion checking
        check_occlusion: Whether to check for occlusion
        min_visible_corners: Minimum corners visible to count as visible
        
    Returns:
        List of visible objects
    """
    visible = []
    
    for obj in objects:
        # Check if object is in FOV using accurate projection
        in_fov, reason = is_target_in_fov(
            K=intrinsics,
            cam_pos=camera_pose.position,
            cam_target=camera_pose.target,
            target_bmin=obj.aabb_min,
            target_bmax=obj.aabb_max,
            width=image_width,
            height=image_height,
            require_center=True,
            border=5
        )
        
        if not in_fov:
            continue
        
        # Optional: Check occlusion
        if check_occlusion and all_scene_aabbs:
            # Count visible corners
            visible_corners = count_visible_corners(
                cam_pos=camera_pose.position,
                cam_target=camera_pose.target,
                target_bmin=obj.aabb_min,
                target_bmax=obj.aabb_max,
                occluders=all_scene_aabbs,
                target_id=str(obj.id),
                K=intrinsics,
                width=image_width,
                height=image_height
            )
            
            if visible_corners < min_visible_corners:
                continue
        
        visible.append(obj)
    
    return visible


def get_visible_object_pairs(
    object_pairs: List[Tuple[SceneObject, SceneObject]],
    visible_object_ids: set
) -> List[Tuple[SceneObject, SceneObject]]:
    """
    Filter object pairs to only those where BOTH objects are visible.
    """
    visible_pairs = []
    for obj1, obj2 in object_pairs:
        if obj1.id in visible_object_ids and obj2.id in visible_object_ids:
            visible_pairs.append((obj1, obj2))
    return visible_pairs


class CameraSampler:
    """Samples valid camera poses around objects for question generation."""
    
    def __init__(self, config: CameraSamplingConfig):
        self.config = config
        self._intrinsics = None
    
    @property
    def intrinsics(self) -> np.ndarray:
        """Get camera intrinsics matrix K."""
        if self._intrinsics is None:
            fov_rad = np.radians(self.config.fov_deg)
            fx = fy = self.config.image_width / (2 * np.tan(fov_rad / 2))
            cx = self.config.image_width / 2
            cy = self.config.image_height / 2
            self._intrinsics = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=float)
        return self._intrinsics
    
    def load_scene_bounds(self, scene_path: Path) -> Optional[SceneBounds]:
        """Load scene bounds from occupancy.json."""
        occupancy_path = scene_path / 'occupancy.json'
        if not occupancy_path.exists():
            return None
        
        try:
            with open(occupancy_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return SceneBounds.from_occupancy(data)
        except Exception as e:
            print(f"Warning: Failed to load occupancy.json: {e}")
            return None
    
    def load_room_polys(self, scene_path: Path) -> List[List[List[float]]]:
        """Load room polygons from structure.json."""
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
            return room_polys
        except Exception as e:
            return []
    
    def load_wall_aabbs(self, scene_path: Path) -> List[AABB]:
        """
        Load wall AABBs from structure.json for wall collision/occlusion detection.
        
        Each wall is modeled as a vertical rectangular prism based on:
        - location: [[x1,y1], [x2,y2]] - wall endpoints
        - thickness: wall thickness in meters
        - height: wall height in meters
        
        Returns:
            List of AABB objects representing walls
        """
        structure_path = scene_path / 'structure.json'
        walls = []
        
        if not structure_path.exists():
            return walls
        
        try:
            with open(structure_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            wall_items = data.get('walls', [])
            
            for idx, w in enumerate(wall_items):
                try:
                    loc = w.get('location', None)
                    thickness = float(w.get('thickness', 0.2) or 0.2)
                    height = float(w.get('height', 2.8) or 2.8)
                    
                    if not loc or len(loc) != 2:
                        continue
                    
                    x1, y1 = float(loc[0][0]), float(loc[0][1])
                    x2, y2 = float(loc[1][0]), float(loc[1][1])
                    
                    p1 = np.array([x1, y1], dtype=float)
                    p2 = np.array([x2, y2], dtype=float)
                    seg = p2 - p1
                    seg_len = float(np.linalg.norm(seg))
                    
                    half_thick = thickness * 0.5
                    
                    if seg_len < 1e-6:
                        # Degenerate: use a square of thickness around point
                        xs = [x1 - half_thick, x1 + half_thick]
                        ys = [y1 - half_thick, y1 + half_thick]
                    else:
                        # Calculate perpendicular direction in XY plane
                        dir_xy = seg / seg_len
                        n = np.array([-dir_xy[1], dir_xy[0]], dtype=float)
                        
                        # Rectangle corners in XY
                        q1 = p1 + n * half_thick
                        q2 = p1 - n * half_thick
                        q3 = p2 + n * half_thick
                        q4 = p2 - n * half_thick
                        
                        xs = [q1[0], q2[0], q3[0], q4[0]]
                        ys = [q1[1], q2[1], q3[1], q4[1]]
                    
                    bmin = np.array([min(xs), min(ys), 0.0], dtype=float)
                    bmax = np.array([max(xs), max(ys), height], dtype=float)
                    
                    walls.append(AABB(id=f"wall_{idx}", label='wall', bmin=bmin, bmax=bmax))
                except Exception:
                    continue
                    
        except Exception as e:
            print(f"[CameraSampler] Warning: Failed to load wall AABBs: {e}")
        
        return walls
    
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
    
    def is_position_valid(self, position: np.ndarray, 
                          scene_bounds: Optional[SceneBounds],
                          room_polys: List[List[List[float]]],
                          objects: List[SceneObject]) -> bool:
        """Check if a camera position is valid."""
        # Check height constraints
        if position[2] < self.config.min_camera_height or position[2] > self.config.max_camera_height:
            return False
        
        # Check scene bounds
        if scene_bounds is not None:
            if not scene_bounds.contains_point_2d(position[0], position[1], margin=0.1):
                return False
        
        # Check if inside any room polygon
        if room_polys:
            in_room = any(self.point_in_poly(position[0], position[1], poly) for poly in room_polys)
            if not in_room:
                return False
        
        # Check collision with objects
        for obj in objects:
            margin = 0.1
            if (obj.aabb_min[0] - margin <= position[0] <= obj.aabb_max[0] + margin and
                obj.aabb_min[1] - margin <= position[1] <= obj.aabb_max[1] + margin and
                obj.aabb_min[2] - margin <= position[2] <= obj.aabb_max[2] + margin):
                return False
        
        return True
    
    def compute_camera_height(self, objects: List[SceneObject]) -> float:
        """Compute appropriate camera height based on object heights."""
        if not objects:
            return self.config.max_camera_height
        
        max_obj_top = max(obj.top_z for obj in objects)
        camera_height = min(max_obj_top + self.config.camera_height_offset, 
                           self.config.max_camera_height)
        camera_height = max(camera_height, self.config.min_camera_height)
        
        return camera_height
    
    def sample_camera_pose(self, objects: List[SceneObject],
                           scene_path: Path,
                           all_scene_objects: Optional[List[SceneObject]] = None) -> Optional[CameraPose]:
        """Sample a single valid camera pose looking at the given objects.
        
        Supports two move patterns:
        - 'around': Sample on a horizontal circle at fixed height (original behavior)
        - 'spherical': Sample on the entire sphere surface around the target
        """
        if not objects:
            return None
        
        # Compute look-at target (center of all objects)
        if len(objects) == 1:
            target = objects[0].center.copy()
        else:
            target = np.mean([obj.center for obj in objects], axis=0)
        
        # Load scene constraints
        scene_bounds = self.load_scene_bounds(scene_path)
        room_polys = self.load_room_polys(scene_path)
        wall_aabbs = self.load_wall_aabbs(scene_path)  # Load walls from structure.json
        collision_objects = all_scene_objects if all_scene_objects else objects
        
        # Sampling parameters
        min_dist = self.config.min_distance
        max_dist = self.config.max_distance
        
        move_pattern = self.config.move_pattern
        
        if move_pattern == 'around':
            # Original behavior: sample on horizontal circle at fixed height
            return self._sample_camera_pose_around(
                target, scene_bounds, room_polys, collision_objects, objects,
                min_dist, max_dist, wall_aabbs
            )
        elif move_pattern == 'spherical':
            # New behavior: sample on entire sphere surface
            return self._sample_camera_pose_spherical(
                target, scene_bounds, room_polys, collision_objects, objects,
                min_dist, max_dist, wall_aabbs
            )
        elif move_pattern == 'linear':
            # Linear movement: move in straight line while keeping object in view
            return self._sample_camera_pose_linear(
                target, scene_bounds, room_polys, collision_objects, objects,
                min_dist, max_dist, wall_aabbs
            )
        else:
            raise ValueError(f"Unknown move_pattern: {move_pattern}. Supported: 'around', 'spherical', 'linear'")
    
    def _sample_camera_pose_around(self, target: np.ndarray,
                                    scene_bounds: Optional[SceneBounds],
                                    room_polys: List[List[List[float]]],
                                    collision_objects: List[SceneObject],
                                    objects: List[SceneObject],
                                    min_dist: float,
                                    max_dist: float,
                                    wall_aabbs: Optional[List[AABB]] = None) -> Optional[CameraPose]:
        """Sample camera pose on a horizontal circle around the target (original 'around' pattern).
        
        Includes visibility validation:
        - All target objects must be in FOV with center in image
        - Minimum visible corners requirement (occlusion check)
        - Wall occlusion check using real walls from structure.json
        """
        # Compute camera height
        camera_height = self.compute_camera_height(objects)
        
        # Prepare AABBs for occlusion checking - include wall AABBs
        all_aabbs = [scene_object_to_aabb(obj) for obj in collision_objects]
        if wall_aabbs:
            all_aabbs.extend(wall_aabbs)  # Add walls from structure.json
        target_ids = {str(obj.id) for obj in objects}
        
        # Try random sampling
        for _ in range(self.config.max_tries):
            yaw = np.random.uniform(0, 360)
            radius = np.random.uniform(min_dist, max_dist)
            
            yaw_rad = np.radians(yaw)
            cam_x = target[0] + radius * np.cos(yaw_rad)
            cam_y = target[1] + radius * np.sin(yaw_rad)
            cam_z = camera_height
            
            position = np.array([cam_x, cam_y, cam_z])
            
            if not self.is_position_valid(position, scene_bounds, room_polys, collision_objects):
                continue
            
            dz = target[2] - position[2]
            horizontal_dist = np.sqrt((target[0] - position[0])**2 + (target[1] - position[1])**2)
            pitch = np.degrees(np.arctan2(-dz, horizontal_dist))
            
            # Create candidate pose for visibility checking
            # Build target object names from the label attribute
            target_object_names = [obj.label for obj in objects]
            
            candidate_pose = CameraPose(
                position=position,
                target=target,
                yaw=yaw,
                pitch=pitch,
                radius=radius,
                target_objects=target_object_names
            )
            
            # Validate visibility for all target objects
            if not self._validate_visibility(candidate_pose, objects, all_aabbs, target_ids):
                continue
            
            return candidate_pose
        
        return None
    
    def _sample_camera_pose_spherical(self, target: np.ndarray,
                                       scene_bounds: Optional[SceneBounds],
                                       room_polys: List[List[List[float]]],
                                       collision_objects: List[SceneObject],
                                       objects: List[SceneObject],
                                       min_dist: float,
                                       max_dist: float,
                                       wall_aabbs: Optional[List[AABB]] = None) -> Optional[CameraPose]:
        """Sample camera pose on the entire sphere surface around the target ('spherical' pattern).
        
        Uses uniform spherical sampling (using theta and phi angles) to sample points
        on a sphere centered at the target object.
        
        Includes visibility validation:
        - All target objects must be in FOV with center in image
        - Minimum visible corners requirement (occlusion check)
        - Wall occlusion check using real walls from structure.json
        """
        # Prepare AABBs for occlusion checking - include wall AABBs
        all_aabbs = [scene_object_to_aabb(obj) for obj in collision_objects]
        if wall_aabbs:
            all_aabbs.extend(wall_aabbs)  # Add walls from structure.json
        target_ids = {str(obj.id) for obj in objects}
        
        # Try random sampling on sphere
        for _ in range(self.config.max_tries):
            radius = np.random.uniform(min_dist, max_dist)
            
            # Uniform spherical sampling
            # theta: azimuthal angle [0, 2*pi] (around z-axis, equivalent to yaw)
            # phi: polar angle [0, pi] (from z-axis down)
            theta = np.random.uniform(0, 2 * np.pi)
            # Use arccos of uniform distribution for uniform area sampling on sphere
            phi = np.arccos(np.random.uniform(-1, 1))
            
            # Convert spherical coordinates to Cartesian
            # x = r * sin(phi) * cos(theta)
            # y = r * sin(phi) * sin(theta)
            # z = r * cos(phi)
            cam_x = target[0] + radius * np.sin(phi) * np.cos(theta)
            cam_y = target[1] + radius * np.sin(phi) * np.sin(theta)
            cam_z = target[2] + radius * np.cos(phi)
            
            position = np.array([cam_x, cam_y, cam_z])
            
            # Check height constraints
            if position[2] < self.config.min_camera_height or position[2] > self.config.max_camera_height:
                continue
            
            # Check scene bounds
            if scene_bounds is not None:
                if not scene_bounds.contains_point_2d(position[0], position[1], margin=0.1):
                    continue
            
            # Check if inside any room polygon
            if room_polys:
                in_room = any(self.point_in_poly(position[0], position[1], poly) for poly in room_polys)
                if not in_room:
                    continue
            
            # Check collision with objects
            collision = False
            for obj in collision_objects:
                margin = 0.1
                if (obj.aabb_min[0] - margin <= position[0] <= obj.aabb_max[0] + margin and
                    obj.aabb_min[1] - margin <= position[1] <= obj.aabb_max[1] + margin and
                    obj.aabb_min[2] - margin <= position[2] <= obj.aabb_max[2] + margin):
                    collision = True
                    break
            if collision:
                continue
            
            # Compute yaw and pitch for the camera pose
            # yaw: angle in xy-plane
            yaw = np.degrees(theta)
            # pitch: angle from horizontal plane
            dz = target[2] - position[2]
            horizontal_dist = np.sqrt((target[0] - position[0])**2 + (target[1] - position[1])**2)
            pitch = np.degrees(np.arctan2(-dz, horizontal_dist + 1e-8))
            
            # Create candidate pose for visibility checking
            # Build target object names from the label attribute
            target_object_names = [obj.label for obj in objects]
            
            candidate_pose = CameraPose(
                position=position,
                target=target,
                yaw=yaw,
                pitch=pitch,
                radius=radius,
                target_objects=target_object_names
            )
            
            # Validate visibility for all target objects
            if not self._validate_visibility(candidate_pose, objects, all_aabbs, target_ids):
                continue
            
            return candidate_pose
        
        return None
    
    def _validate_visibility(self, pose: CameraPose, 
                             objects: List[SceneObject],
                             all_aabbs: List[AABB],
                             target_ids: set) -> bool:
        """
        Validate that all target objects are properly visible from the camera pose.
        
        Checks:
        1. Each object's center must be in the image (FOV check with require_center=True)
        2. Each object must have at least min_visible_corners visible (occlusion check)
           - This check is skipped if skip_occlusion_check=True
        3. Wall occlusion check (fast, only checks walls) - enabled when:
           - skip_occlusion_check=True AND check_wall_occlusion=True
        
        Args:
            pose: Camera pose to validate
            objects: Target objects that must be visible
            all_aabbs: All AABBs in scene for occlusion checking
            target_ids: IDs of target objects (to exclude from self-occlusion)
            
        Returns:
            True if all objects are properly visible, False otherwise
        """
        K = self.intrinsics
        width = self.config.image_width
        height = self.config.image_height
        min_corners = self.config.min_visible_corners
        skip_occlusion = getattr(self.config, 'skip_occlusion_check', False)
        check_wall = getattr(self.config, 'check_wall_occlusion', True)
        
        for obj in objects:
            # Check 1: Object must be in FOV with center in image
            in_fov, reason = is_target_in_fov(
                K=K,
                cam_pos=pose.position,
                cam_target=pose.target,
                target_bmin=obj.aabb_min,
                target_bmax=obj.aabb_max,
                width=width,
                height=height,
                require_center=True,  # Ensure object center is in image
                border=5  # 5 pixel margin from edge
            )
            
            if not in_fov:
                return False
            
            # Check 2: Full occlusion check (visible corners) - SKIP if configured
            if not skip_occlusion:
                visible_corners = count_visible_corners(
                    K=K,
                    cam_pos=pose.position,
                    cam_target=pose.target,
                    target_bmin=obj.aabb_min,
                    target_bmax=obj.aabb_max,
                    width=width,
                    height=height,
                    border=2,
                    check_occlusion=True,
                    occluders=all_aabbs,
                    target_id=str(obj.id)
                )
                
                if visible_corners < min_corners:
                    return False
            elif check_wall:
                # Check 3: Wall-only occlusion check (fast version)
                # Even when skipping full occlusion check, still reject if wall blocks view
                is_occluded, _ = is_wall_occluded(
                    cam_pos=pose.position,
                    target_bmin=obj.aabb_min,
                    target_bmax=obj.aabb_max,
                    occluders=all_aabbs,
                    target_id=str(obj.id),
                    sample_corners=True,
                    occlusion_threshold=0.5  # Reject if 50%+ of points occluded by wall
                )
                
                if is_occluded:
                    return False
        
        return True
    
    def _validate_visibility_linear(self, pose: CameraPose, 
                                    objects: List[SceneObject],
                                    all_aabbs: List[AABB],
                                    target_ids: set) -> bool:
        """
        Validate visibility for linear pattern (relaxed version).
        
        For linear pattern, camera orientation is FIXED, so object may not be at image center.
        Only check that object is within FOV (anywhere in frame), not necessarily centered.
        Also skip full occlusion check to improve success rate.
        
        Args:
            pose: Camera pose to validate
            objects: Target objects that must be visible
            all_aabbs: All AABBs in scene for occlusion checking
            target_ids: IDs of target objects
            
        Returns:
            True if objects are visible somewhere in frame
        """
        K = self.intrinsics
        width = self.config.image_width
        height = self.config.image_height
        
        for obj in objects:
            # Relaxed check: Object just needs to be in FOV (don't require center in image)
            in_fov, reason = is_target_in_fov(
                K=K,
                cam_pos=pose.position,
                cam_target=pose.target,
                target_bmin=obj.aabb_min,
                target_bmax=obj.aabb_max,
                width=width,
                height=height,
                require_center=False,  # Don't require center - just any part visible
                border=0  # No margin requirement
            )
            
            if not in_fov:
                return False
            
            # Optional: Quick wall occlusion check
            check_wall = getattr(self.config, 'check_wall_occlusion', True)
            if check_wall:
                is_occluded, _ = is_wall_occluded(
                    cam_pos=pose.position,
                    target_bmin=obj.aabb_min,
                    target_bmax=obj.aabb_max,
                    occluders=all_aabbs,
                    target_id=str(obj.id),
                    sample_corners=True,
                    occlusion_threshold=0.7  # More lenient - only reject if 70%+ blocked
                )
                
                if is_occluded:
                    return False
        
        return True
    
    def sample_cameras(self, scene_path: Path, 
                       objects: List[SceneObject],
                       num_samples: int = 5,
                       all_scene_objects: Optional[List[SceneObject]] = None) -> List[CameraPose]:
        """Sample multiple valid camera poses for given objects.
        
        For linear pattern: Returns a single trajectory with multiple poses.
        For other patterns: Returns independent camera poses.
        """
        # For linear pattern, use generate_linear_poses to get a proper trajectory
        if self.config.move_pattern == 'linear':
            return self.generate_linear_poses(scene_path, objects, all_scene_objects)
        
        # For other patterns, sample independent poses
        poses = []
        attempts = 0
        max_total_attempts = num_samples * 10
        
        while len(poses) < num_samples and attempts < max_total_attempts:
            pose = self.sample_camera_pose(objects, scene_path, all_scene_objects)
            if pose is not None:
                poses.append(pose)
            attempts += 1
        
        return poses
    
    def get_intrinsics_dict(self) -> Dict[str, Any]:
        """Get camera intrinsics as dictionary."""
        K = self.intrinsics
        return {
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
            'width': self.config.image_width,
            'height': self.config.image_height,
            'fov_deg': self.config.fov_deg,
        }
    
    def build_camera_matrix(self, pose: CameraPose) -> np.ndarray:
        """Build 4x4 camera-to-world transformation matrix."""
        position = pose.position
        target = pose.target
        
        forward = target - position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-8:
            forward = np.array([0.0, 1.0, 0.0])
        else:
            forward = forward / forward_norm
        
        up = np.array([0.0, 0.0, 1.0])
        
        right = np.cross(forward, up)
        right_norm = np.linalg.norm(right)
        if right_norm < 1e-8:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / right_norm
        
        down = np.cross(forward, right)
        down = down / np.linalg.norm(down)
        
        R = np.column_stack([right, down, forward])
        
        c2w = np.eye(4, dtype=float)
        c2w[:3, :3] = R
        c2w[:3, 3] = position
        
        return c2w
    
    # =========================================================================
    # ROTATION PATTERN METHODS
    # =========================================================================
    
    def compute_room_centers(self, scene_path: Path) -> List[Dict[str, Any]]:
        """
        Compute the center of each room from structure.json.
        
        Returns:
            List of dictionaries with room info:
            [{'room_idx': 0, 'center': np.array([x, y]), 'polygon': [...]}]
        """
        room_polys = self.load_room_polys(scene_path)
        room_centers = []
        
        for room_idx, poly in enumerate(room_polys):
            if not poly or len(poly) < 3:
                continue
            
            # Compute centroid of polygon
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            center_x = sum(xs) / len(xs)
            center_y = sum(ys) / len(ys)
            
            room_centers.append({
                'room_idx': room_idx,
                'center': np.array([center_x, center_y], dtype=float),
                'polygon': poly,
            })
        
        return room_centers
    
    def generate_rotation_poses(self, scene_path: Path, 
                                 all_scene_objects: Optional[List[SceneObject]] = None
                                 ) -> List[Tuple[CameraPose, int]]:
        """
        Generate camera poses for rotation pattern.
        
        For each room, stand at the center and rotate 360 degrees.
        
        Args:
            scene_path: Path to scene folder
            all_scene_objects: All objects in the scene for collision checking
            
        Returns:
            List of (CameraPose, room_idx) tuples
        """
        room_centers = self.compute_room_centers(scene_path)
        if not room_centers:
            print("  [Warning] No room polygons found, cannot generate rotation poses")
            return []
        
        interval = self.config.rotation_interval
        camera_height = self.config.rotation_camera_height
        num_angles = int(360 / interval)
        
        collision_objects = all_scene_objects if all_scene_objects else []
        
        all_poses = []
        
        for room_info in room_centers:
            room_idx = room_info['room_idx']
            center_2d = room_info['center']
            
            # Camera position: room center at fixed height
            cam_x, cam_y = center_2d[0], center_2d[1]
            cam_z = camera_height
            position = np.array([cam_x, cam_y, cam_z])
            
            # Check if camera position collides with any object
            collision = False
            for obj in collision_objects:
                margin = 0.1
                if (obj.aabb_min[0] - margin <= position[0] <= obj.aabb_max[0] + margin and
                    obj.aabb_min[1] - margin <= position[1] <= obj.aabb_max[1] + margin and
                    obj.aabb_min[2] - margin <= position[2] <= obj.aabb_max[2] + margin):
                    collision = True
                    break
            
            if collision:
                print(f"  [Warning] Room {room_idx} center collides with object, skipping")
                continue
            
            # Generate poses for each angle
            for angle_idx in range(num_angles):
                yaw = angle_idx * interval
                yaw_rad = np.radians(yaw)
                
                # Look direction: 1 meter in front of camera at same height
                look_distance = 1.0
                target_x = cam_x + look_distance * np.cos(yaw_rad)
                target_y = cam_y + look_distance * np.sin(yaw_rad)
                target_z = cam_z  # Look horizontally
                target = np.array([target_x, target_y, target_z])
                
                pose = CameraPose(
                    position=position.copy(),
                    target=target,
                    yaw=yaw,
                    pitch=0.0,  # Looking horizontally
                    radius=0.0,  # Not applicable for rotation
                    target_objects=[],  # Will be filled later based on visibility
                )
                
                all_poses.append((pose, room_idx))
        
        return all_poses

    # =========================================================================
    # LINEAR PATTERN METHODS
    # =========================================================================
    
    def _sample_camera_pose_linear(self, target: np.ndarray,
                                    scene_bounds: Optional[SceneBounds],
                                    room_polys: List[List[List[float]]],
                                    collision_objects: List[SceneObject],
                                    objects: List[SceneObject],
                                    min_dist: float,
                                    max_dist: float,
                                    wall_aabbs: Optional[List[AABB]] = None) -> Optional[CameraPose]:
        """Sample a single camera pose for linear movement pattern.
        
        Linear pattern: Camera moves in a straight line while keeping the target object in view.
        
        Two sub-patterns:
        - 'center_forward': Object at center of view, camera walks toward object (head tracks object)
        - 'side_sweep': Object starts at right edge, camera walks straight (no head turn), object sweeps right-to-left
        
        This method samples a single pose from the linear trajectory.
        Use generate_linear_poses() to get the full trajectory.
        """
        # Compute camera height
        camera_height = self.compute_camera_height(objects)
        
        # Prepare AABBs for occlusion checking - include wall AABBs
        all_aabbs = [scene_object_to_aabb(obj) for obj in collision_objects]
        if wall_aabbs:
            all_aabbs.extend(wall_aabbs)
        target_ids = {str(obj.id) for obj in objects}
        
        sub_pattern = getattr(self.config, 'linear_sub_pattern', 'center_forward')
        
        # Try random initial directions
        for _ in range(self.config.max_tries):
            # Random initial angle (direction from target to camera)
            initial_yaw = np.random.uniform(0, 360)
            initial_yaw_rad = np.radians(initial_yaw)
            
            # Random distance from target
            radius = np.random.uniform(min_dist, max_dist)
            
            # Camera position
            cam_x = target[0] + radius * np.cos(initial_yaw_rad)
            cam_y = target[1] + radius * np.sin(initial_yaw_rad)
            cam_z = camera_height
            position = np.array([cam_x, cam_y, cam_z])
            
            # Validate position
            if not self.is_position_valid(position, scene_bounds, room_polys, collision_objects):
                continue
            
            # For single pose sampling, camera looks at the target
            dz = target[2] - position[2]
            horizontal_dist = np.sqrt((target[0] - position[0])**2 + (target[1] - position[1])**2)
            pitch = np.degrees(np.arctan2(-dz, horizontal_dist))
            
            target_object_names = [obj.label for obj in objects]
            
            candidate_pose = CameraPose(
                position=position,
                target=target,
                yaw=initial_yaw,
                pitch=pitch,
                radius=radius,
                target_objects=target_object_names
            )
            
            # Validate visibility
            if not self._validate_visibility(candidate_pose, objects, all_aabbs, target_ids):
                continue
            
            return candidate_pose
        
        return None
    
    def generate_linear_poses(self, scene_path: Path,
                              objects: List[SceneObject],
                              all_scene_objects: Optional[List[SceneObject]] = None,
                              sub_pattern: Optional[str] = None) -> List[CameraPose]:
        """
        Generate a sequence of camera poses for linear movement pattern.
        
        For each trajectory, generates multiple poses along a STRAIGHT LINE path.
        Camera orientation (yaw/pitch) remains FIXED throughout the trajectory.
        Only camera POSITION changes along a linear path.
        
        Two sub-patterns:
        - 'approach': Camera moves toward object along a straight line (forward walk)
        - 'pass_by': Camera moves along a line that passes by the object (side walk)
        
        Args:
            scene_path: Path to scene folder
            objects: Target objects to focus on
            all_scene_objects: All objects in scene for collision checking
            sub_pattern: Override sub-pattern (default: use config value)
            
        Returns:
            List of CameraPose objects along the linear trajectory
        """
        if not objects:
            return []
        
        # Compute target center
        if len(objects) == 1:
            target = objects[0].center.copy()
        else:
            target = np.mean([obj.center for obj in objects], axis=0)
        
        # Load scene constraints
        scene_bounds = self.load_scene_bounds(scene_path)
        room_polys = self.load_room_polys(scene_path)
        wall_aabbs = self.load_wall_aabbs(scene_path)
        collision_objects = all_scene_objects if all_scene_objects else objects
        
        # Parameters
        min_dist = self.config.min_distance
        max_dist = self.config.max_distance
        camera_height = self.compute_camera_height(objects)
        num_steps = getattr(self.config, 'linear_num_steps', 5)
        fov_margin = getattr(self.config, 'linear_fov_margin', 0.1)
        move_distance = getattr(self.config, 'linear_move_distance', 0.3)  # Total movement distance
        sub_pat = sub_pattern or getattr(self.config, 'linear_sub_pattern', 'approach')
        
        # FOV calculations
        fov_rad = np.radians(self.config.fov_deg)
        half_fov_rad = fov_rad / 2
        
        # Prepare AABBs for occlusion checking
        all_aabbs = [scene_object_to_aabb(obj) for obj in collision_objects]
        if wall_aabbs:
            all_aabbs.extend(wall_aabbs)
        target_ids = {str(obj.id) for obj in objects}
        
        poses = []
        
        # Try to find a valid trajectory
        for _ in range(self.config.max_tries):
            # Random initial direction (from target to camera start position)
            initial_yaw = np.random.uniform(0, 360)
            initial_yaw_rad = np.radians(initial_yaw)
            
            # Direction vector from target to initial camera position (in XY plane)
            dir_from_target = np.array([np.cos(initial_yaw_rad), np.sin(initial_yaw_rad), 0.0])
            
            trajectory_poses = []
            trajectory_valid = True
            
            if sub_pat == 'approach':
                # =============================================================
                # APPROACH: Camera walks in a straight line toward the object
                # =============================================================
                # Camera moves a fixed distance (linear_move_distance) toward object
                # Camera orientation is FIXED (always looking toward object direction)
                # Position changes, but yaw/pitch stay constant
                
                # Random starting distance from object (between min and max)
                start_radius = np.random.uniform(min_dist + move_distance, max_dist)
                
                # Start position: at start_radius from object
                start_pos = np.array([
                    target[0] + start_radius * dir_from_target[0],
                    target[1] + start_radius * dir_from_target[1],
                    camera_height
                ])
                
                # End position: move_distance closer to object
                end_radius = start_radius - move_distance
                end_pos = np.array([
                    target[0] + end_radius * dir_from_target[0],
                    target[1] + end_radius * dir_from_target[1],
                    camera_height
                ])
                
                # Movement direction (normalized): from start toward end (toward object)
                move_dir = end_pos - start_pos
                move_len = np.linalg.norm(move_dir)
                if move_len > 0:
                    move_dir = move_dir / move_len
                
                # FIXED camera orientation: looking toward the target direction
                # (from camera toward target, so opposite of dir_from_target)
                look_dir = -dir_from_target
                fixed_yaw = np.degrees(np.arctan2(look_dir[1], look_dir[0]))
                
                # Compute fixed pitch (looking slightly down toward target)
                mid_radius = (start_radius + end_radius) / 2
                dz = target[2] - camera_height
                fixed_pitch = np.degrees(np.arctan2(-dz, mid_radius))
                
                # FIXED look target (a point far ahead in the look direction)
                look_distance = 10.0
                fixed_look_target = np.array([
                    start_pos[0] + look_distance * look_dir[0],
                    start_pos[1] + look_distance * look_dir[1],
                    target[2]  # Look at object height
                ])
                
                for step in range(num_steps):
                    t = step / max(1, num_steps - 1)
                    
                    # Linear interpolation: position moves along straight line
                    position = start_pos + t * (end_pos - start_pos)
                    current_dist = np.linalg.norm(position[:2] - target[:2])
                    
                    if not self.is_position_valid(position, scene_bounds, room_polys, collision_objects):
                        trajectory_valid = False
                        break
                    
                    target_object_names = [obj.label for obj in objects]
                    
                    pose = CameraPose(
                        position=position,
                        target=fixed_look_target,  # FIXED look target
                        yaw=fixed_yaw,             # FIXED yaw
                        pitch=fixed_pitch,         # FIXED pitch
                        radius=current_dist,
                        target_objects=target_object_names
                    )
                    
                    # Use relaxed visibility check for linear pattern
                    if not self._validate_visibility_linear(pose, objects, all_aabbs, target_ids):
                        trajectory_valid = False
                        break
                    
                    trajectory_poses.append(pose)
                    
            elif sub_pat == 'pass_by':
                # =============================================================
                # PASS_BY: Camera walks in a straight line past the object
                # =============================================================
                # Camera moves in a straight line that passes by the object
                # At the middle of the trajectory, object is roughly at center of view
                # Camera orientation is FIXED (looks in the direction of initial setup)
                
                # Random distance from object
                base_radius = np.random.uniform(min_dist, max_dist)
                
                # Direction perpendicular to dir_from_target (in XY plane)
                perp_dir = np.array([-dir_from_target[1], dir_from_target[0], 0.0])
                
                # Use move_distance for sweep distance (same as approach)
                sweep_distance = move_distance
                
                # Base position: at base_radius from object
                base_pos = np.array([
                    target[0] + base_radius * dir_from_target[0],
                    target[1] + base_radius * dir_from_target[1],
                    camera_height
                ])
                
                # Start and end positions (along perpendicular line, centered at base_pos)
                start_pos = base_pos - (sweep_distance / 2) * perp_dir
                end_pos = base_pos + (sweep_distance / 2) * perp_dir
                
                # FIXED camera orientation: looking toward the target direction
                look_dir = -dir_from_target
                fixed_yaw = np.degrees(np.arctan2(look_dir[1], look_dir[0]))
                
                # Compute fixed pitch (looking slightly down toward target)
                dz = target[2] - camera_height
                fixed_pitch = np.degrees(np.arctan2(-dz, base_radius))
                
                # FIXED look target
                look_distance = 10.0
                fixed_look_target = np.array([
                    base_pos[0] + look_distance * look_dir[0],
                    base_pos[1] + look_distance * look_dir[1],
                    target[2]
                ])
                
                for step in range(num_steps):
                    t = step / max(1, num_steps - 1)
                    
                    # Linear interpolation: position moves along straight line
                    position = start_pos + t * (end_pos - start_pos)
                    current_dist = np.linalg.norm(position[:2] - target[:2])
                    
                    if not self.is_position_valid(position, scene_bounds, room_polys, collision_objects):
                        trajectory_valid = False
                        break
                    
                    target_object_names = [obj.label for obj in objects]
                    
                    pose = CameraPose(
                        position=position,
                        target=fixed_look_target,  # FIXED look target
                        yaw=fixed_yaw,             # FIXED yaw
                        pitch=fixed_pitch,         # FIXED pitch
                        radius=current_dist,
                        target_objects=target_object_names
                    )
                    
                    # Use relaxed visibility check for linear pattern
                    if not self._validate_visibility_linear(pose, objects, all_aabbs, target_ids):
                        trajectory_valid = False
                        break
                    
                    trajectory_poses.append(pose)
            
            if trajectory_valid and len(trajectory_poses) == num_steps:
                return trajectory_poses
        
        return []