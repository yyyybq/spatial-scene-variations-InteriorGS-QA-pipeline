"""
Main Pipeline Module for InteriorGS Question Generation

This module orchestrates the complete question generation process:
1. Load scene and filter/select objects
2. For each selected object/pair, sample camera poses
3. For each camera pose, render images and generate questions
4. Save the dataset in JSONL format with image paths
"""

import json
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import asdict

try:
    from .config import PipelineConfig
    from .object_selector import ObjectSelector
    from .camera_sampler import CameraSampler, get_visible_objects, get_visible_object_pairs, scene_object_to_aabb
    from .question_generator import QuestionGenerator
    from .render_utils import SceneRenderer, RenderConfig, camera_pose_to_matrices, compute_intrinsics
except ImportError:
    from config import PipelineConfig
    from object_selector import ObjectSelector
    from camera_sampler import CameraSampler, get_visible_objects, get_visible_object_pairs, scene_object_to_aabb
    from question_generator import QuestionGenerator
    from render_utils import SceneRenderer, RenderConfig, camera_pose_to_matrices, compute_intrinsics


def _run_async(coro):
    """Helper to run async coroutine in sync context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    
    if loop is not None and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


class InteriorGSQuestionPipeline:
    """
    Main pipeline for generating spatial reasoning questions from InteriorGS dataset.
    
    Pipeline Flow:
        1. Load scene and filter/select objects based on semantic and geometric constraints
        2. For each selected object/pair, sample valid camera poses
        3. For each camera pose, render images and generate questions
        4. Save the dataset in JSONL format with image paths
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        config.validate()
        
        self.object_selector = ObjectSelector(config.object_selection)
        self.camera_sampler = CameraSampler(config.camera_sampling)
        self.question_generator = QuestionGenerator(config.question_config)
        
        # Rendering setup
        self.renderer = None
        self.render_enabled = config.render_config.enable_rendering
        
        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def get_scene_list(self) -> List[str]:
        """Get list of scenes to process."""
        if self.config.scene_list:
            return self.config.scene_list
        
        # Auto-discover scenes
        scenes_root = Path(self.config.scenes_root)
        scenes = []
        
        for item in scenes_root.iterdir():
            if item.is_dir():
                labels_path = item / 'labels.json'
                if labels_path.exists():
                    scenes.append(item.name)
        
        return sorted(scenes)
    
    def process_scene(self, scene_name: str, scene_output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Process a single scene and generate questions with rendered images.
        
        Dispatches to different processing methods based on move_pattern:
        - 'around' / 'spherical': Object-centric approach (sample cameras around objects)
        - 'rotation': Room-centric approach (stand at room center and rotate)
        
        Args:
            scene_name: Name of the scene folder
            scene_output_dir: Output directory for this scene (for images)
        
        Returns:
            List of question dictionaries with image paths
        """
        move_pattern = self.config.camera_sampling.move_pattern
        
        if move_pattern == 'rotation':
            return self._process_scene_rotation(scene_name, scene_output_dir)
        else:
            return self._process_scene_object_centric(scene_name, scene_output_dir)
    
    def _process_scene_object_centric(self, scene_name: str, scene_output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Process a scene using object-centric approach (around/spherical patterns).
        
        Logic:
        1. Select objects/pairs first
        2. For each object/pair, sample camera poses around it
        3. For each (object, camera) combination, generate questions
        
        Args:
            scene_name: Name of the scene folder
            scene_output_dir: Output directory for this scene (for images)
        
        Returns:
            List of question dictionaries with image paths
        """
        scene_path = Path(self.config.scenes_root) / scene_name
        
        if not scene_path.exists():
            print(f"Scene not found: {scene_path}")
            return []
        
        print(f"Processing scene: {scene_name}")
        
        # Step 1: Select valid objects (for single-object questions)
        try:
            single_objects = self.object_selector.select_single_objects(scene_path)
            print(f"  Found {len(single_objects)} valid single objects")
        except Exception as e:
            print(f"  Error selecting objects: {e}")
            return []
        
        if not single_objects:
            print(f"  No valid objects found, skipping scene")
            return []
        
        # Step 2: Select object pairs from ALL objects (not just valid single objects)
        # This allows more pairs to be found, as pair constraints are different from single object constraints
        object_pairs = self.object_selector.select_object_pairs(scene_path, objects=None, use_all_objects=True)
        print(f"  Found {len(object_pairs)} valid object pairs (from all objects)")
        
        all_questions = []
        all_camera_poses = []  # Track all unique camera poses for rendering
        camera_pose_to_idx = {}  # Map camera pose to index
        
        # Get ALL parsed objects for collision checking (including those not in single_objects)
        all_scene_objects = self.object_selector.get_all_parsed_objects(scene_path)
        
        # Prepare for visibility checking
        fov_deg = self.config.camera_sampling.fov_deg
        image_width = self.config.render_config.image_width
        image_height = self.config.render_config.image_height
        intrinsics = compute_intrinsics(image_width, image_height, fov_deg)
        all_scene_aabbs = [scene_object_to_aabb(obj) for obj in all_scene_objects]
        
        num_cameras_per_item = self.config.camera_sampling.num_cameras_per_item
        
        # ========== Process single-object questions ==========
        # For each object, sample cameras around it, then generate questions
        print(f"  Processing single-object questions...")
        single_obj_stats = {'objects': 0, 'cameras': 0, 'questions': 0}
        
        for obj in single_objects:
            # Sample camera poses specifically for this object
            camera_poses = self.camera_sampler.sample_cameras(
                scene_path, [obj],
                num_samples=num_cameras_per_item,
                all_scene_objects=all_scene_objects
            )
            
            if not camera_poses:
                continue
            
            single_obj_stats['objects'] += 1
            single_obj_stats['cameras'] += len(camera_poses)
            
            for camera_pose in camera_poses:
                # Get or create pose index
                pose_key = (tuple(camera_pose.position.tolist()), tuple(camera_pose.target.tolist()))
                if pose_key not in camera_pose_to_idx:
                    pose_idx = len(all_camera_poses)
                    camera_pose_to_idx[pose_key] = pose_idx
                    all_camera_poses.append(camera_pose)
                else:
                    pose_idx = camera_pose_to_idx[pose_key]
                
                # Generate single-object questions for this (object, camera) pair
                questions = self.question_generator.generate_single_object_questions(
                    obj, camera_pose
                )
                for q in questions:
                    q['scene_id'] = scene_name
                    q['camera_pose_idx'] = pose_idx
                all_questions.extend(questions)
                single_obj_stats['questions'] += len(questions)
        
        print(f"    Single-object: {single_obj_stats['objects']} objects × {num_cameras_per_item} cameras → {single_obj_stats['questions']} questions")
        
        # ========== Process pair-object questions ==========
        # For each object pair, sample cameras that see both, then generate questions
        print(f"  Processing pair-object questions...")
        pair_stats = {'pairs': 0, 'cameras': 0, 'questions': 0}
        
        for obj1, obj2 in object_pairs:
            # Sample camera poses that can see both objects
            camera_poses = self.camera_sampler.sample_cameras(
                scene_path, [obj1, obj2],
                num_samples=num_cameras_per_item,
                all_scene_objects=all_scene_objects
            )
            
            if not camera_poses:
                continue
            
            pair_stats['pairs'] += 1
            pair_stats['cameras'] += len(camera_poses)
            
            for camera_pose in camera_poses:
                # Get or create pose index
                pose_key = (tuple(camera_pose.position.tolist()), tuple(camera_pose.target.tolist()))
                if pose_key not in camera_pose_to_idx:
                    pose_idx = len(all_camera_poses)
                    camera_pose_to_idx[pose_key] = pose_idx
                    all_camera_poses.append(camera_pose)
                else:
                    pose_idx = camera_pose_to_idx[pose_key]
                
                # Generate pair-object questions
                questions = self.question_generator.generate_pair_object_questions(
                    obj1, obj2, camera_pose
                )
                for q in questions:
                    q['scene_id'] = scene_name
                    q['camera_pose_idx'] = pose_idx
                all_questions.extend(questions)
                pair_stats['questions'] += len(questions)
        
        print(f"    Pair-object: {pair_stats['pairs']} pairs × {num_cameras_per_item} cameras → {pair_stats['questions']} questions")
        
        # ========== Process multi-object questions ==========
        # For multi-object questions, we need at least 3 objects visible from the same camera
        # Reuse camera poses from pair-object processing
        print(f"  Processing multi-object questions...")
        multi_obj_stats = {'questions': 0}
        
        if len(single_objects) >= 3:
            # Use a subset of camera poses for multi-object questions
            multi_obj_camera_poses = list(all_camera_poses)[:min(len(all_camera_poses), num_cameras_per_item * 3)]
            
            for camera_pose in multi_obj_camera_poses:
                pose_key = (tuple(camera_pose.position.tolist()), tuple(camera_pose.target.tolist()))
                pose_idx = camera_pose_to_idx.get(pose_key, len(all_camera_poses) - 1)
                
                # Generate multi-object questions using all valid single objects
                questions = self.question_generator.generate_multi_object_questions(
                    single_objects, camera_pose,
                    max_questions_per_type=self.config.question_config.max_questions_per_type
                )
                for q in questions:
                    q['scene_id'] = scene_name
                    q['camera_pose_idx'] = pose_idx
                all_questions.extend(questions)
                multi_obj_stats['questions'] += len(questions)
        
        print(f"    Multi-object: {multi_obj_stats['questions']} questions")
        
        # ========== Render images for all unique camera poses ==========
        print(f"  Total unique camera poses: {len(all_camera_poses)}")
        
        rendered_images = {}
        if self.render_enabled and scene_output_dir:
            rendered_images = self._render_camera_poses(
                scene_name, all_camera_poses, scene_output_dir
            )
            print(f"  Rendered {len(rendered_images)} images")
            
            # Update questions with image paths (new format: {experiment}/{scene_id}/{move_pattern}/{object}_{view}.png)
            experiment_name = self.config.experiment_name
            move_pattern = self.config.camera_sampling.move_pattern
            for q in all_questions:
                pose_idx = q.get('camera_pose_idx')
                if pose_idx is not None and pose_idx in rendered_images:
                    # Path relative to output_dir: {experiment}/{scene_id}/{move_pattern}/{filename}
                    image_path = rendered_images[pose_idx]
                    relative_path = f"{experiment_name}/{scene_name}/{move_pattern}/{image_path.name}"
                    q['image'] = relative_path
        
        # Check max questions per scene
        if len(all_questions) > self.config.max_questions_per_scene:
            all_questions = all_questions[:self.config.max_questions_per_scene]
        
        print(f"  Generated {len(all_questions)} total questions")
        return all_questions
    
    def _process_scene_rotation(self, scene_name: str, scene_output_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
        """
        Process a scene using rotation mode (room-centric approach).
        
        Logic:
        1. Find room centers and generate rotation camera poses (72 per room at 5° intervals)
        2. For each camera pose, detect which objects/pairs are visible
        3. Generate questions for visible objects (same object may appear in multiple images)
        
        This is the inverse of object-centric approach: cameras are determined first,
        then we check what's visible from each camera position.
        
        Args:
            scene_name: Name of the scene folder
            scene_output_dir: Output directory for this scene (for images)
        
        Returns:
            List of question dictionaries with image paths
        """
        print(f"Processing scene: {scene_name} (rotation mode)")
        
        # Get scene path
        scene_path = Path(self.config.scenes_root) / scene_name
        
        # Validate scene exists
        if not scene_path.exists():
            print(f"  [Error] Scene path not found: {scene_path}")
            return []
        
        # Get all scene objects first (needed for visibility checking)
        all_scene_objects = self.object_selector.load_scene_objects(scene_path)
        if not all_scene_objects:
            print(f"  [Warning] No objects found in scene")
            return []
        
        print(f"  Loaded {len(all_scene_objects)} objects from scene")
        
        # Build AABBs for visibility checking (reuse existing helper)
        all_scene_aabbs = [scene_object_to_aabb(obj) for obj in all_scene_objects]
        
        # ========== Generate rotation cameras for all rooms ==========
        print(f"  Generating rotation cameras...")
        
        # Get rotation parameters from config
        rotation_interval = self.config.camera_sampling.rotation_interval
        camera_height = self.config.camera_sampling.rotation_camera_height
        
        # Generate rotation cameras (72 per room for 360° at 5° intervals)
        all_room_cameras = self.camera_sampler.generate_rotation_cameras(
            scene_path,
            rotation_interval_deg=rotation_interval,
            camera_height=camera_height
        )
        
        if not all_room_cameras:
            print(f"  [Warning] No rotation cameras generated")
            return []
        
        total_cameras = sum(len(poses) for poses in all_room_cameras.values())
        print(f"  Generated {total_cameras} cameras across {len(all_room_cameras)} rooms")
        
        # ========== For each camera pose, detect visible objects and generate questions ==========
        all_questions = []
        all_camera_poses = []
        camera_pose_to_idx = {}
        
        single_obj_stats = {'cameras': 0, 'objects_detected': 0, 'questions': 0}
        pair_stats = {'cameras': 0, 'pairs_detected': 0, 'questions': 0}
        multi_obj_stats = {'cameras': 0, 'questions': 0}
        
        # Get camera intrinsics for visibility checking
        intrinsics = self.camera_sampler.intrinsics
        image_width = self.config.camera_sampling.image_width
        image_height = self.config.camera_sampling.image_height
        
        for room_name, room_camera_poses in all_room_cameras.items():
            print(f"  Processing room: {room_name} ({len(room_camera_poses)} camera poses)")
            
            for camera_pose in room_camera_poses:
                # Register this camera pose
                pose_key = (tuple(camera_pose.position.tolist()), tuple(camera_pose.target.tolist()))
                if pose_key not in camera_pose_to_idx:
                    pose_idx = len(all_camera_poses)
                    camera_pose_to_idx[pose_key] = pose_idx
                    all_camera_poses.append(camera_pose)
                else:
                    pose_idx = camera_pose_to_idx[pose_key]
                
                # ========== Detect visible single objects for this pose ==========
                # Use existing module-level function get_visible_objects
                visible_objects = get_visible_objects(
                    objects=all_scene_objects,
                    camera_pose=camera_pose,
                    intrinsics=intrinsics,
                    image_width=image_width,
                    image_height=image_height,
                    all_scene_aabbs=all_scene_aabbs,
                    check_occlusion=True,
                    min_visible_corners=self.config.camera_sampling.min_visible_corners
                )
                
                if visible_objects:
                    single_obj_stats['cameras'] += 1
                    single_obj_stats['objects_detected'] += len(visible_objects)
                    
                    for obj in visible_objects:
                        # Generate single-object questions for this (object, camera) pair
                        questions = self.question_generator.generate_single_object_questions(
                            obj, camera_pose
                        )
                        for q in questions:
                            q['scene_id'] = scene_name
                            q['camera_pose_idx'] = pose_idx
                            q['room'] = room_name
                            q['camera_yaw'] = camera_pose.yaw
                        all_questions.extend(questions)
                        single_obj_stats['questions'] += len(questions)
                
                # ========== Detect visible object pairs for this pose ==========
                # Use existing module-level function get_visible_object_pairs
                visible_ids = {obj.id for obj in visible_objects}
                # Generate all pairs from visible objects
                visible_pairs = []
                visible_list = list(visible_objects)
                for i in range(len(visible_list)):
                    for j in range(i + 1, len(visible_list)):
                        visible_pairs.append((visible_list[i], visible_list[j]))
                
                if visible_pairs:
                    pair_stats['cameras'] += 1
                    pair_stats['pairs_detected'] += len(visible_pairs)
                    
                    for obj1, obj2 in visible_pairs:
                        # Generate pair-object questions
                        questions = self.question_generator.generate_pair_object_questions(
                            obj1, obj2, camera_pose
                        )
                        for q in questions:
                            q['scene_id'] = scene_name
                            q['camera_pose_idx'] = pose_idx
                            q['room'] = room_name
                            q['camera_yaw'] = camera_pose.yaw
                        all_questions.extend(questions)
                        pair_stats['questions'] += len(questions)
                
                # ========== Generate multi-object questions if enough objects visible ==========
                if len(visible_objects) >= 3:
                    multi_obj_stats['cameras'] += 1
                    questions = self.question_generator.generate_multi_object_questions(
                        visible_objects, camera_pose,
                        max_questions_per_type=self.config.question_config.max_questions_per_type
                    )
                    for q in questions:
                        q['scene_id'] = scene_name
                        q['camera_pose_idx'] = pose_idx
                        q['room'] = room_name
                        q['camera_yaw'] = camera_pose.yaw
                    all_questions.extend(questions)
                    multi_obj_stats['questions'] += len(questions)
        
        print(f"    Single-object: {single_obj_stats['cameras']} cameras, {single_obj_stats['objects_detected']} visible objects → {single_obj_stats['questions']} questions")
        print(f"    Pair-object: {pair_stats['cameras']} cameras, {pair_stats['pairs_detected']} visible pairs → {pair_stats['questions']} questions")
        print(f"    Multi-object: {multi_obj_stats['cameras']} cameras → {multi_obj_stats['questions']} questions")
        
        # ========== Render images for all camera poses ==========
        print(f"  Total camera poses: {len(all_camera_poses)}")
        
        rendered_images = {}
        if self.render_enabled and scene_output_dir:
            rendered_images = self._render_rotation_poses(
                scene_name, all_camera_poses, all_room_cameras, scene_output_dir
            )
            print(f"  Rendered {len(rendered_images)} images")
            
            # Update questions with image paths
            experiment_name = self.config.experiment_name
            move_pattern = self.config.camera_sampling.move_pattern
            for q in all_questions:
                pose_idx = q.get('camera_pose_idx')
                if pose_idx is not None and pose_idx in rendered_images:
                    image_path = rendered_images[pose_idx]
                    relative_path = f"{experiment_name}/{scene_name}/{move_pattern}/{image_path.name}"
                    q['image'] = relative_path
        
        # Check max questions per scene
        if len(all_questions) > self.config.max_questions_per_scene:
            all_questions = all_questions[:self.config.max_questions_per_scene]
        
        print(f"  Generated {len(all_questions)} total questions")
        return all_questions
    
    def _render_rotation_poses(self, scene_id: str, camera_poses: List,
                                room_cameras: Dict[str, List], images_dir: Path) -> Dict[int, Path]:
        """
        Render images for rotation mode camera poses.
        
        Path format: {experiment_name}/{scene_id}/rotation/{room_name}_{yaw_deg}.png
        
        Args:
            scene_id: Scene identifier
            camera_poses: List of all CameraPose objects (used for indexing)
            room_cameras: Dictionary mapping room_name -> list of camera poses
            images_dir: Base directory for images
            
        Returns:
            Dictionary mapping pose_idx to image file path
        """
        rendered = {}
        
        # Get experiment name and move pattern from config
        experiment_name = self.config.experiment_name
        move_pattern = self.config.camera_sampling.move_pattern
        
        # Create output directory: {output_dir}/{experiment_name}/{scene_id}/rotation/
        exp_scene_pattern_dir = self.output_dir / experiment_name / scene_id / move_pattern
        exp_scene_pattern_dir.mkdir(parents=True, exist_ok=True)
        
        # Build a mapping from pose to its room and yaw
        pose_to_info = {}
        for room_name, poses in room_cameras.items():
            for pose in poses:
                pose_key = (tuple(pose.position.tolist()), tuple(pose.target.tolist()))
                pose_to_info[pose_key] = {
                    'room_name': room_name,
                    'yaw': pose.yaw
                }
        
        try:
            # Create render config from pipeline config
            render_cfg = RenderConfig(
                scenes_root=self.config.scenes_root,
                render_backend=self.config.render_config.render_backend,
                client_url=self.config.render_config.client_url,
                image_width=self.config.render_config.image_width,
                image_height=self.config.render_config.image_height,
                fov_deg=self.config.render_config.fov_deg,
                gpu_device=self.config.render_config.gpu_device,
            )
            
            # Run rendering asynchronously
            rendered = _run_async(self._render_rotation_poses_async(
                scene_id, camera_poses, exp_scene_pattern_dir, render_cfg, pose_to_info
            ))
            
        except Exception as e:
            print(f"  [Warning] Rendering failed: {e}")
            print(f"  [Warning] Questions will be generated without images")
        
        return rendered
    
    async def _render_rotation_poses_async(self, scene_id: str, camera_poses: List,
                                            output_dir: Path, render_cfg: RenderConfig,
                                            pose_to_info: Dict) -> Dict[int, Path]:
        """Async implementation of rotation pose rendering."""
        rendered = {}
        
        async with SceneRenderer(render_cfg) as renderer:
            await renderer.set_scene(scene_id)
            
            for pose_idx, pose in enumerate(camera_poses):
                try:
                    # Convert pose to camera matrices
                    pose_dict = pose.to_dict() if hasattr(pose, 'to_dict') else pose
                    intrinsics, extrinsics_c2w = camera_pose_to_matrices(
                        pose_dict,
                        render_cfg.image_width,
                        render_cfg.image_height,
                        render_cfg.fov_deg
                    )
                    
                    # Render image
                    image = await renderer.render_image(intrinsics, extrinsics_c2w)
                    
                    if image is not None:
                        # Get room name and yaw from pose info
                        pose_key = (tuple(pose.position.tolist()), tuple(pose.target.tolist()))
                        info = pose_to_info.get(pose_key, {})
                        room_name = self._sanitize_name(info.get('room_name', f'room_{pose_idx}'))
                        yaw_deg = info.get('yaw', 0.0)
                        
                        # Build filename: {room_name}_{yaw_deg}.png
                        # Round yaw to integer for cleaner filenames
                        yaw_int = int(round(yaw_deg)) % 360
                        image_filename = f"{room_name}_{yaw_int:03d}deg.png"
                        image_path = output_dir / image_filename
                        image.save(image_path)
                        rendered[pose_idx] = image_path
                    else:
                        print(f"    [Warning] Failed to render pose {pose_idx}")
                        
                except Exception as e:
                    print(f"    [Warning] Error rendering pose {pose_idx}: {e}")
        
        return rendered
    
    def _render_camera_poses(self, scene_id: str, camera_poses: List, 
                              images_dir: Path) -> Dict[int, Path]:
        """
        Render images for all camera poses with new path format.
        
        New path format: {experiment_name}/{scene_id}/{move_pattern}/{object_name}_{view_idx}.png
        - For single object poses: {object_label}_{view_idx}.png
        - For object pairs: {object1_label}_{object2_label}_{view_idx}.png
        
        Args:
            scene_id: Scene identifier
            camera_poses: List of CameraPose objects
            images_dir: Base directory (will create subdirectories for experiment/scene/pattern)
            
        Returns:
            Dictionary mapping pose_idx to image file path
        """
        rendered = {}
        
        # Get experiment name and move pattern from config
        experiment_name = self.config.experiment_name
        move_pattern = self.config.camera_sampling.move_pattern
        
        # Create output directory: {output_dir}/{experiment_name}/{scene_id}/{move_pattern}/
        exp_scene_pattern_dir = self.output_dir / experiment_name / scene_id / move_pattern
        exp_scene_pattern_dir.mkdir(parents=True, exist_ok=True)
        
        # Track view indices per object/pair
        view_counters = {}  # key: object_key (e.g., "chair" or "chair_table"), value: next view index
        
        try:
            # Create render config from pipeline config
            render_cfg = RenderConfig(
                scenes_root=self.config.scenes_root,
                render_backend=self.config.render_config.render_backend,
                client_url=self.config.render_config.client_url,
                image_width=self.config.render_config.image_width,
                image_height=self.config.render_config.image_height,
                fov_deg=self.config.render_config.fov_deg,
                gpu_device=self.config.render_config.gpu_device,
            )
            
            # Run rendering asynchronously
            rendered = _run_async(self._render_poses_async(
                scene_id, camera_poses, exp_scene_pattern_dir, render_cfg, view_counters
            ))
            
        except Exception as e:
            print(f"  [Warning] Rendering failed: {e}")
            print(f"  [Warning] Questions will be generated without images")
        
        return rendered
    
    async def _render_poses_async(self, scene_id: str, camera_poses: List,
                                   output_dir: Path, render_cfg: RenderConfig,
                                   view_counters: Dict[str, int]) -> Dict[int, Path]:
        """Async implementation of pose rendering with new path format."""
        rendered = {}
        
        async with SceneRenderer(render_cfg) as renderer:
            await renderer.set_scene(scene_id)
            
            for pose_idx, pose in enumerate(camera_poses):
                try:
                    # Convert pose to camera matrices
                    pose_dict = pose.to_dict() if hasattr(pose, 'to_dict') else pose
                    intrinsics, extrinsics_c2w = camera_pose_to_matrices(
                        pose_dict,
                        render_cfg.image_width,
                        render_cfg.image_height,
                        render_cfg.fov_deg
                    )
                    
                    # Render image
                    image = await renderer.render_image(intrinsics, extrinsics_c2w)
                    
                    if image is not None:
                        # Generate object key from target_objects
                        target_objects = pose_dict.get('target_objects', [])
                        if target_objects:
                            # Sanitize object names (remove spaces, special chars)
                            sanitized_names = [self._sanitize_name(name) for name in target_objects]
                            object_key = '_'.join(sorted(sanitized_names))
                        else:
                            object_key = f"unknown_{pose_idx}"
                        
                        # Get and increment view counter for this object/pair
                        if object_key not in view_counters:
                            view_counters[object_key] = 0
                        view_idx = view_counters[object_key]
                        view_counters[object_key] += 1
                        
                        # Build filename: {object_name}_{view_idx}.png
                        image_filename = f"{object_key}_{view_idx}.png"
                        image_path = output_dir / image_filename
                        image.save(image_path)
                        rendered[pose_idx] = image_path
                    else:
                        print(f"    [Warning] Failed to render pose {pose_idx}")
                        
                except Exception as e:
                    print(f"    [Warning] Error rendering pose {pose_idx}: {e}")
        
        return rendered
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize object name for use in filename."""
        # Replace spaces and special characters with underscores
        import re
        sanitized = re.sub(r'[^\w\-]', '_', name.strip().lower())
        # Remove multiple underscores
        sanitized = re.sub(r'_+', '_', sanitized)
        # Remove leading/trailing underscores
        sanitized = sanitized.strip('_')
        return sanitized if sanitized else 'unknown'
    
    def run(self, verbose: bool = True, scene_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Run the complete pipeline.
        
        Args:
            verbose: Whether to print progress
            scene_id: If specified, only process this single scene
        
        Returns:
            List of all generated question dictionaries
        """
        if scene_id:
            return self.run_single_scene(scene_id, verbose=verbose)
        
        scenes = self.get_scene_list()
        
        if verbose:
            print(f"Found {len(scenes)} scenes to process")
            print(f"Output directory: {self.output_dir}")
            print()
        
        all_questions = []
        scene_stats = {}
        
        for scene_idx, scene_name in enumerate(scenes):
            if verbose:
                print(f"[{scene_idx+1}/{len(scenes)}] ", end='')
            
            # Create scene output directory for rendering
            scene_output = self.output_dir / scene_name
            scene_output.mkdir(parents=True, exist_ok=True)
            
            questions = self.process_scene(scene_name, scene_output_dir=scene_output)
            scene_stats[scene_name] = len(questions)
            
            # Save scene questions
            if questions:
                with open(scene_output / 'questions.jsonl', 'w', encoding='utf-8') as f:
                    for q in questions:
                        f.write(json.dumps(q, ensure_ascii=False) + '\n')
            
            all_questions.extend(questions)
        
        # Save final dataset as JSONL
        dataset_path = self.output_dir / 'questions.jsonl'
        with open(dataset_path, 'w', encoding='utf-8') as f:
            for q in all_questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        
        # Also save as JSON for compatibility
        with open(self.output_dir / 'questions.json', 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        question_stats = self._get_question_statistics(all_questions)
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_questions': len(all_questions),
            'num_scenes': len(scenes),
            'scene_stats': scene_stats,
            'question_type_stats': question_stats,
            'config': self.config.to_dict()
        }
        
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print()
            print("=" * 60)
            print(f"Pipeline completed!")
            print(f"Total questions: {len(all_questions)}")
            print(f"Question types: {question_stats}")
            print(f"Dataset saved to: {dataset_path}")
            print("=" * 60)
        
        return all_questions
    
    def run_single_scene(self, scene_name: str, verbose: bool = True) -> List[Dict[str, Any]]:
        """
        Run pipeline for a single scene.
        
        Args:
            scene_name: Name/ID of the scene to process
            verbose: Whether to print progress
        
        Returns:
            List of question dictionaries
        """
        if verbose:
            print(f"Processing single scene: {scene_name}")
            print(f"Output directory: {self.output_dir}")
            if self.render_enabled:
                print(f"Rendering enabled: {self.config.scenes_root}")
            print()
        
        # Create scene output directory first (needed for rendering)
        scene_output = self.output_dir / scene_name
        scene_output.mkdir(parents=True, exist_ok=True)
        
        # Process scene with rendering
        questions = self.process_scene(scene_name, scene_output_dir=scene_output)
        
        if not questions:
            if verbose:
                print(f"No questions generated for scene {scene_name}")
            return []
        
        # Save questions as JSONL
        questions_path = scene_output / 'questions.jsonl'
        with open(questions_path, 'w', encoding='utf-8') as f:
            for q in questions:
                f.write(json.dumps(q, ensure_ascii=False) + '\n')
        
        # Save metadata
        question_stats = self._get_question_statistics(questions)
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'total_questions': len(questions),
            'scene_id': scene_name,
            'question_type_stats': question_stats,
            'config': self.config.to_dict()
        }
        
        metadata_path = scene_output / 'metadata.json'
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        if verbose:
            print()
            print("=" * 60)
            print(f"Pipeline completed for scene: {scene_name}")
            print(f"Total questions: {len(questions)}")
            print(f"Question types: {question_stats}")
            print(f"Output directory: {scene_output}")
            print("=" * 60)
        
        return questions
    
    def _get_question_statistics(self, questions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get statistics about generated questions by type."""
        stats = {}
        for q in questions:
            q_type = q.get('question_type', 'unknown')
            stats[q_type] = stats.get(q_type, 0) + 1
        return stats


def run_pipeline(scenes_root: str, output_dir: str, 
                 scene_id: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
    """
    Convenience function to run the pipeline.
    
    Args:
        scenes_root: Root directory containing InteriorGS scene folders
        output_dir: Output directory for generated questions
        scene_id: If specified, only process this single scene
        **kwargs: Additional configuration options
    
    Returns:
        List of all generated questions
    
    Examples:
        # Process all scenes
        run_pipeline('/path/to/InteriorGS', '/path/to/output')
        
        # Process only one specific scene
        run_pipeline('/path/to/InteriorGS', '/path/to/output', scene_id='0267_840790')
    """
    config = PipelineConfig(
        scenes_root=scenes_root,
        output_dir=output_dir,
        **kwargs
    )
    
    pipeline = InteriorGSQuestionPipeline(config)
    return pipeline.run(scene_id=scene_id)
