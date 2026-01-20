#!/usr/bin/env python3
"""
MST-based VQA Data Generator

This module generates VQA (Visual Question Answering) data for spatial reasoning
based on MST (Minimum Spanning Tree) splits.

It performs:
1. Read MST edge data (train/eval_with_image splits)
2. For each object pair edge, sample valid camera poses
3. Render images at those camera poses
4. Generate object_pair_distance_vector questions
5. Output JSONL format training/evaluation data

Usage:
    python mst_vqa_generator.py \
        --scenes_root /path/to/InteriorGS \
        --mst_splits_dir /path/to/mst_splits \
        --output_dir /path/to/output \
        --scene_id 0267_840790
"""

import json
import os
import argparse
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

try:
    from config import PipelineConfig, CameraSamplingConfig, RenderConfig as PipelineRenderConfig, ObjectSelectionConfig
    from object_selector import ObjectSelector, SceneObject
    from camera_sampler import CameraSampler, CameraPose
    from render_utils import SceneRenderer, RenderConfig, camera_pose_to_matrices, compute_intrinsics
    from question_utils import construct_object_pair_distance_vector_qa
except ImportError:
    from .config import PipelineConfig, CameraSamplingConfig, RenderConfig as PipelineRenderConfig, ObjectSelectionConfig
    from .object_selector import ObjectSelector, SceneObject
    from .camera_sampler import CameraSampler, CameraPose
    from .render_utils import SceneRenderer, RenderConfig, camera_pose_to_matrices, compute_intrinsics
    from .question_utils import construct_object_pair_distance_vector_qa


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


@dataclass
class MSTVQAConfig:
    """Configuration for MST VQA generation."""
    scenes_root: str
    mst_splits_dir: str
    output_dir: str
    
    # Camera sampling
    num_cameras_per_pair: int = 3
    fov_deg: float = 60.0
    move_pattern: str = 'around'
    skip_occlusion_check: bool = False  # Skip slow occlusion check for speed
    check_wall_occlusion: bool = True   # Even with skip_occlusion_check, still check wall occlusion
    
    # Rendering
    enable_rendering: bool = True
    image_width: int = 512
    image_height: int = 512
    render_backend: str = 'local'  # "local" or "client"
    gpu_device: int = 0
    
    # Experiment naming
    experiment_name: str = 'mst_vqa'


class MSTVQAGenerator:
    """
    Generates VQA data from MST splits.
    
    Pipeline:
    1. Load MST edge data (object pairs)
    2. For each pair, sample camera poses that can see both objects
    3. Render images at each camera pose
    4. Generate object_pair_distance_vector questions
    5. Save as JSONL with image paths
    """
    
    def __init__(self, config: MSTVQAConfig):
        self.config = config
        self.scenes_root = Path(config.scenes_root)
        self.mst_splits_dir = Path(config.mst_splits_dir)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize camera sampler with RELAXED settings for faster processing
        camera_config = CameraSamplingConfig(
            num_cameras_per_item=config.num_cameras_per_pair,
            fov_deg=config.fov_deg,
            move_pattern=config.move_pattern,
            # Relaxed settings to increase success rate
            max_tries=100,  # Reduce from 300 to speed up
            min_visible_corners=1,  # Reduce from 2 - only need 1 corner visible
            max_occlusion_ratio=0.8,  # Increase from 0.6 - allow more occlusion
            min_visibility_ratio=0.02,  # Reduce from 0.05 - allow smaller objects
            # Skip slow occlusion check if configured
            skip_occlusion_check=config.skip_occlusion_check,
            # Wall occlusion check (fast, even when skip_occlusion_check=True)
            check_wall_occlusion=config.check_wall_occlusion,
        )
        self.camera_sampler = CameraSampler(camera_config)
        
        # Object selector for loading scene objects
        object_config = ObjectSelectionConfig()
        self.object_selector = ObjectSelector(object_config)
    
    def load_mst_edges(self, scene_id: str, split: str = 'train') -> List[Dict[str, Any]]:
        """
        Load MST edge data for a scene.
        
        Args:
            scene_id: Scene identifier
            split: 'train' for MST edges, 'eval_with_image' for non-MST valid edges,
                   'eval_blind' for blind evaluation edges
            
        Returns:
            List of edge dictionaries
        """
        if split == 'train':
            filename = f"{scene_id}_train_mst.json"
        elif split == 'eval_with_image':
            filename = f"{scene_id}_eval_with_image.json"
        elif split == 'eval_blind':
            filename = f"{scene_id}_eval_blind.json"
        else:
            raise ValueError(f"Unknown split: {split}")
        
        filepath = self.mst_splits_dir / filename
        if not filepath.exists():
            print(f"MST file not found: {filepath}")
            return []
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data.get('edges', [])
    
    def get_scene_objects_by_id(self, scene_path: Path) -> Dict[str, SceneObject]:
        """
        Load all scene objects and create a lookup by ID.
        
        Returns:
            Dictionary mapping object ID to SceneObject
        """
        all_objects = self.object_selector.get_all_parsed_objects(scene_path)
        return {obj.id: obj for obj in all_objects}
    
    def process_scene(self, scene_id: str, split: str = 'train') -> List[Dict[str, Any]]:
        """
        Process a scene and generate VQA data for MST edges.
        
        Args:
            scene_id: Scene identifier
            split: 'train', 'eval_with_image', or 'eval_blind'
            
        Returns:
            List of VQA question dictionaries
        """
        scene_path = self.scenes_root / scene_id
        if not scene_path.exists():
            print(f"Scene not found: {scene_path}")
            return []
        
        print(f"\n{'='*60}")
        print(f"Processing scene: {scene_id} (split: {split})")
        print(f"{'='*60}")
        
        # Load MST edges
        edges = self.load_mst_edges(scene_id, split)
        if not edges:
            print(f"  No edges found for {split} split")
            return []
        print(f"  Loaded {len(edges)} edges from {split} split")
        
        # Load scene objects
        objects_by_id = self.get_scene_objects_by_id(scene_path)
        print(f"  Loaded {len(objects_by_id)} scene objects")
        
        # ========== EVAL_BLIND: Text-only QA without images ==========
        if split == 'eval_blind':
            return self._process_scene_blind(scene_id, edges, objects_by_id)
        
        # ========== TRAIN / EVAL_WITH_IMAGE: With camera sampling and rendering ==========
        # Get all scene objects for collision checking
        all_scene_objects = list(objects_by_id.values())
        
        # Process each edge
        all_questions = []
        all_camera_poses = []
        camera_to_objects = {}  # Map pose_idx to (obj_a, obj_b)
        
        stats = {
            'total_edges': len(edges),
            'edges_with_valid_cameras': 0,
            'total_cameras': 0,
            'total_questions': 0,
            'skipped_missing_objects': 0,
            'skipped_no_valid_camera': 0,
        }
        
        print(f"  Processing {len(edges)} edges...")
        for edge_idx, edge in enumerate(edges):
            # Progress indicator every 10 edges
            if edge_idx % 10 == 0:
                print(f"    [{edge_idx}/{len(edges)}] Processing edge {edge_idx}...", flush=True)
            
            obj_a_id = edge['obj_a_id']
            obj_b_id = edge['obj_b_id']
            
            # Get SceneObject instances
            obj_a = objects_by_id.get(obj_a_id)
            obj_b = objects_by_id.get(obj_b_id)
            
            if obj_a is None or obj_b is None:
                stats['skipped_missing_objects'] += 1
                continue
            
            # Sample camera poses for this pair
            camera_poses = self.camera_sampler.sample_cameras(
                scene_path, [obj_a, obj_b],
                num_samples=self.config.num_cameras_per_pair,
                all_scene_objects=all_scene_objects
            )
            
            if not camera_poses:
                stats['skipped_no_valid_camera'] += 1
                continue
            
            stats['edges_with_valid_cameras'] += 1
            stats['total_cameras'] += len(camera_poses)
            
            # Generate questions for each camera pose
            for camera_pose in camera_poses:
                pose_idx = len(all_camera_poses)
                all_camera_poses.append(camera_pose)
                camera_to_objects[pose_idx] = (obj_a, obj_b)
                
                # Generate object_pair_distance_vector question
                qa = construct_object_pair_distance_vector_qa(obj_a, obj_b, camera_pose)
                if qa:
                    qa['scene_id'] = scene_id
                    qa['split'] = split
                    qa['edge_idx'] = edge_idx
                    qa['camera_pose_idx'] = pose_idx
                    # Store world-space vector for reference
                    qa['world_vector_a_to_b'] = edge.get('vector_a_to_b', None)
                    all_questions.append(qa)
                    stats['total_questions'] += 1
        
        print(f"\n📊 Camera Sampling Statistics:")
        print(f"   Total edges: {stats['total_edges']}")
        print(f"   Edges with valid cameras: {stats['edges_with_valid_cameras']}")
        print(f"   Skipped (missing objects): {stats['skipped_missing_objects']}")
        print(f"   Skipped (no valid camera): {stats['skipped_no_valid_camera']}")
        print(f"   Total camera poses: {stats['total_cameras']}")
        print(f"   Total questions: {stats['total_questions']}")
        
        # Render images if enabled
        if self.config.enable_rendering and all_camera_poses:
            print(f"\n🎨 Rendering {len(all_camera_poses)} images...")
            rendered_images = self._render_camera_poses(
                scene_id, all_camera_poses, camera_to_objects, split
            )
            print(f"   Rendered {len(rendered_images)} images")
            
            # Update questions with image paths
            for q in all_questions:
                pose_idx = q.get('camera_pose_idx')
                if pose_idx is not None and pose_idx in rendered_images:
                    q['image'] = rendered_images[pose_idx]
        
        return all_questions
    
    def _process_scene_blind(self, scene_id: str, edges: List[Dict], 
                              objects_by_id: Dict[str, SceneObject]) -> List[Dict[str, Any]]:
        """
        Process scene for eval_blind split - text-only QA without images.
        
        For blind evaluation, we generate questions without camera poses or images.
        This tests whether the model actually needs visual input to answer.
        
        Args:
            scene_id: Scene identifier
            edges: List of edge dictionaries
            objects_by_id: Dictionary mapping object ID to SceneObject
            
        Returns:
            List of VQA question dictionaries (no images)
        """
        all_questions = []
        stats = {
            'total_edges': len(edges),
            'valid_edges': 0,
            'skipped_missing_objects': 0,
        }
        
        print(f"  [BLIND] Processing {len(edges)} edges (text-only, no images)...")
        for edge_idx, edge in enumerate(edges):
            # Progress indicator every 1000 edges (since there are many)
            if edge_idx % 1000 == 0:
                print(f"    [{edge_idx}/{len(edges)}] Processing edge {edge_idx}...", flush=True)
            
            obj_a_id = edge['obj_a_id']
            obj_b_id = edge['obj_b_id']
            
            # Get SceneObject instances
            obj_a = objects_by_id.get(obj_a_id)
            obj_b = objects_by_id.get(obj_b_id)
            
            if obj_a is None or obj_b is None:
                stats['skipped_missing_objects'] += 1
                continue
            
            stats['valid_edges'] += 1
            
            # Generate text-only question (no camera pose, no image)
            # For blind eval, we ask about spatial relationship without providing visual context
            qa = self._construct_blind_qa(obj_a, obj_b, edge)
            if qa:
                qa['scene_id'] = scene_id
                qa['split'] = 'eval_blind'
                qa['edge_idx'] = edge_idx
                qa['image'] = None  # Explicitly no image
                all_questions.append(qa)
        
        print(f"\n📊 Blind QA Statistics:")
        print(f"   Total edges: {stats['total_edges']}")
        print(f"   Valid edges: {stats['valid_edges']}")
        print(f"   Skipped (missing objects): {stats['skipped_missing_objects']}")
        print(f"   Total questions: {len(all_questions)}")
        
        return all_questions
    
    def _construct_blind_qa(self, obj_a: SceneObject, obj_b: SceneObject, 
                            edge: Dict) -> Optional[Dict[str, Any]]:
        """
        Construct a blind QA question without camera pose or image.
        
        The question asks about spatial relationship using only object names.
        The answer is the world-space vector (ground truth).
        """
        import numpy as np
        
        # Get world-space vector from edge data
        vector_a_to_b = edge.get('vector_a_to_b')
        if vector_a_to_b is None:
            # Compute from object centers
            center_a = np.array(obj_a.center)
            center_b = np.array(obj_b.center)
            vector_a_to_b = (center_b - center_a).tolist()
        
        # Distance
        distance = np.linalg.norm(vector_a_to_b)
        
        # Construct question and answer
        question = (
            f"Consider two objects in a room: '{obj_a.label}' and '{obj_b.label}'. "
            f"What is the displacement vector from the center of '{obj_a.label}' "
            f"to the center of '{obj_b.label}' in world coordinates (x, y, z)?"
        )
        
        answer = f"[{vector_a_to_b[0]:.3f}, {vector_a_to_b[1]:.3f}, {vector_a_to_b[2]:.3f}]"
        
        return {
            'question_type': 'object_pair_distance_vector_blind',
            'question': question,
            'answer': answer,
            'obj_a_id': obj_a.id,
            'obj_b_id': obj_b.id,
            'obj_a_label': obj_a.label,
            'obj_b_label': obj_b.label,
            'world_vector_a_to_b': vector_a_to_b,
            'distance': float(distance),
        }

    def _render_camera_poses(self, scene_id: str, camera_poses: List[CameraPose],
                              camera_to_objects: Dict[int, Tuple[SceneObject, SceneObject]],
                              split: str) -> Dict[int, str]:
        """
        Render images for all camera poses.
        
        Args:
            scene_id: Scene identifier
            camera_poses: List of camera poses
            camera_to_objects: Map pose_idx to (obj_a, obj_b)
            split: 'train' or 'eval_with_image'
            
        Returns:
            Dictionary mapping pose_idx to relative image path
        """
        # Create output directory: {output_dir}/{experiment_name}/{split}/{scene_id}/
        exp_name = self.config.experiment_name
        images_dir = self.output_dir / exp_name / split / scene_id
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Track view indices per object pair
        view_counters = defaultdict(int)
        
        try:
            render_cfg = RenderConfig(
                scenes_root=str(self.scenes_root),
                render_backend=self.config.render_backend,
                image_width=self.config.image_width,
                image_height=self.config.image_height,
                fov_deg=self.config.fov_deg,
                gpu_device=self.config.gpu_device,
            )
            
            rendered = _run_async(self._render_poses_async(
                scene_id, camera_poses, camera_to_objects, images_dir, 
                render_cfg, view_counters, split
            ))
            return rendered
            
        except Exception as e:
            print(f"  [Warning] Rendering failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    async def _render_poses_async(self, scene_id: str, camera_poses: List[CameraPose],
                                   camera_to_objects: Dict[int, Tuple[SceneObject, SceneObject]],
                                   output_dir: Path, render_cfg: RenderConfig,
                                   view_counters: Dict[str, int], split: str) -> Dict[int, str]:
        """Async implementation of pose rendering."""
        rendered = {}
        exp_name = self.config.experiment_name
        
        async with SceneRenderer(render_cfg) as renderer:
            await renderer.set_scene(scene_id)
            
            for pose_idx, pose in enumerate(camera_poses):
                try:
                    pose_dict = pose.to_dict() if hasattr(pose, 'to_dict') else pose
                    intrinsics, extrinsics_c2w = camera_pose_to_matrices(
                        pose_dict,
                        render_cfg.image_width,
                        render_cfg.image_height,
                        render_cfg.fov_deg
                    )
                    
                    image = await renderer.render_image(intrinsics, extrinsics_c2w)
                    
                    if image is not None:
                        # Get object pair for this pose
                        obj_a, obj_b = camera_to_objects[pose_idx]
                        
                        # Create filename: {obj_a_label}_{obj_b_label}_{view_idx}.png
                        obj_key = f"{self._sanitize_name(obj_a.label)}_{self._sanitize_name(obj_b.label)}"
                        view_idx = view_counters[obj_key]
                        view_counters[obj_key] += 1
                        
                        filename = f"{obj_key}_{view_idx}.png"
                        image_path = output_dir / filename
                        image.save(image_path)
                        
                        # Store relative path: {experiment}/{split}/{scene_id}/{filename}
                        relative_path = f"{exp_name}/{split}/{scene_id}/{filename}"
                        rendered[pose_idx] = relative_path
                    else:
                        print(f"    [Warning] Failed to render pose {pose_idx}")
                        
                except Exception as e:
                    print(f"    [Warning] Error rendering pose {pose_idx}: {e}")
        
        return rendered
    
    def _sanitize_name(self, name: str) -> str:
        """Sanitize object name for use in filename."""
        import re
        sanitized = re.sub(r'[^\w\-]', '_', name.strip().lower())
        sanitized = re.sub(r'_+', '_', sanitized)
        sanitized = sanitized.strip('_')
        return sanitized if sanitized else 'unknown'
    
    def run(self, scene_id: Optional[str] = None, 
            splits: List[str] = ['train', 'eval_with_image']) -> Dict[str, List[Dict]]:
        """
        Run the complete VQA generation pipeline.
        
        Args:
            scene_id: If specified, only process this scene
            splits: List of splits to process
            
        Returns:
            Dictionary mapping split name to list of questions
        """
        results = {}
        
        for split in splits:
            print(f"\n{'#'*60}")
            print(f"# Processing split: {split}")
            print(f"{'#'*60}")
            
            if scene_id:
                questions = self.process_scene(scene_id, split)
            else:
                # Process all scenes that have MST files
                questions = []
                mst_files = list(self.mst_splits_dir.glob(f"*_{split.replace('eval_with_image', 'eval_with_image')}.json"))
                if split == 'train':
                    mst_files = list(self.mst_splits_dir.glob("*_train_mst.json"))
                
                for mst_file in mst_files:
                    # Extract scene_id from filename
                    if split == 'train':
                        sid = mst_file.stem.replace('_train_mst', '')
                    else:
                        sid = mst_file.stem.replace('_eval_with_image', '')
                    
                    scene_questions = self.process_scene(sid, split)
                    questions.extend(scene_questions)
            
            results[split] = questions
            
            # Save split-specific JSONL
            if questions:
                output_path = self.output_dir / f"{split}.jsonl"
                with open(output_path, 'w', encoding='utf-8') as f:
                    for q in questions:
                        f.write(json.dumps(q, ensure_ascii=False) + '\n')
                print(f"\n✅ Saved {len(questions)} questions to {output_path}")
        
        # Save combined metadata
        metadata = {
            'generated_at': datetime.now().isoformat(),
            'config': {
                'scenes_root': str(self.scenes_root),
                'mst_splits_dir': str(self.mst_splits_dir),
                'output_dir': str(self.output_dir),
                'num_cameras_per_pair': self.config.num_cameras_per_pair,
                'experiment_name': self.config.experiment_name,
                'enable_rendering': self.config.enable_rendering,
            },
            'statistics': {
                split: {
                    'num_questions': len(questions),
                    'num_unique_scenes': len(set(q['scene_id'] for q in questions)) if questions else 0,
                }
                for split, questions in results.items()
            }
        }
        
        with open(self.output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        # Print summary
        print(f"\n{'='*60}")
        print("GENERATION COMPLETE")
        print(f"{'='*60}")
        for split, questions in results.items():
            print(f"  {split}: {len(questions)} questions")
        print(f"  Output directory: {self.output_dir}")
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Generate VQA data from MST splits',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument('--scenes_root', type=str, required=True,
                        help='Root directory containing InteriorGS scenes')
    parser.add_argument('--mst_splits_dir', type=str, required=True,
                        help='Directory containing MST split files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for VQA data')
    parser.add_argument('--scene_id', type=str, default=None,
                        help='Process only this scene')
    parser.add_argument('--splits', type=str, nargs='+', 
                        default=['train', 'eval_with_image'],
                        help='Splits to process (default: train eval_with_image)')
    
    # Camera sampling options
    parser.add_argument('--num_cameras', type=int, default=3,
                        help='Number of camera poses per object pair')
    parser.add_argument('--fov_deg', type=float, default=60.0,
                        help='Camera field of view in degrees')
    parser.add_argument('--move_pattern', type=str, default='around',
                        choices=['around', 'spherical'],
                        help='Camera movement pattern')
    parser.add_argument('--skip_occlusion_check', action='store_true',
                        help='Skip slow occlusion check for faster processing')
    parser.add_argument('--no_wall_occlusion_check', action='store_true',
                        help='Disable wall occlusion check (by default wall check is ON even when skip_occlusion_check)')
    
    # Rendering options
    parser.add_argument('--enable_rendering', action='store_true', default=True,
                        help='Enable image rendering (default: True)')
    parser.add_argument('--no_rendering', action='store_true',
                        help='Disable image rendering')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image width and height')
    parser.add_argument('--render_backend', type=str, default='local',
                        choices=['local', 'client'],
                        help='Rendering backend: local or client')
    parser.add_argument('--gpu_device', type=int, default=0,
                        help='GPU device ID')
    
    # Experiment naming
    parser.add_argument('--experiment_name', type=str, default='mst_vqa',
                        help='Experiment name for output path structure')
    
    args = parser.parse_args()
    
    # Build config
    config = MSTVQAConfig(
        scenes_root=args.scenes_root,
        mst_splits_dir=args.mst_splits_dir,
        output_dir=args.output_dir,
        num_cameras_per_pair=args.num_cameras,
        fov_deg=args.fov_deg,
        move_pattern=args.move_pattern,
        skip_occlusion_check=args.skip_occlusion_check,
        check_wall_occlusion=not args.no_wall_occlusion_check,  # Default ON
        enable_rendering=not args.no_rendering,
        image_width=args.image_size,
        image_height=args.image_size,
        render_backend=args.render_backend,
        gpu_device=args.gpu_device,
        experiment_name=args.experiment_name,
    )
    
    generator = MSTVQAGenerator(config)
    generator.run(scene_id=args.scene_id, splits=args.splits)


if __name__ == '__main__':
    main()
