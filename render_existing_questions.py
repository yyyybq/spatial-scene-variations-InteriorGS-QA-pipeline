#!/usr/bin/env python3
"""
Render Images for Existing VQA Questions
=========================================

This script renders images for already generated VQA questions data.
It reads the questions.jsonl file, extracts unique camera poses, 
renders images, and updates the questions with image paths.

Usage:
    python render_existing_questions.py \
        --data_dir /path/to/output/0267_840790 \
        --scenes_root /path/to/InteriorGS

    # Or process all scenes in data directory:
    python render_existing_questions.py \
        --data_root /path/to/output \
        --scenes_root /path/to/InteriorGS

    # With custom rendering options:
    python render_existing_questions.py \
        --data_dir /path/to/output/0267_840790 \
        --scenes_root /path/to/InteriorGS \
        --width 640 --height 480 --fov 60 \
        --gpu_device 0

Output Structure:
    data_dir/
        questions.jsonl          # Updated with image paths
        questions_backup.jsonl   # Backup of original questions
        images/
            pose_000.png
            pose_001.png
            ...
        render_log.json         # Rendering statistics
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import shutil

# Optional tqdm for progress bar
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get('desc', '')
        if desc:
            print(f"{desc}...")
        return iterable

# Add current directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from render_utils import RenderConfig, SceneRenderer, camera_pose_to_matrices


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


def load_questions(jsonl_path: Path) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    questions = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def save_questions(questions: List[Dict[str, Any]], jsonl_path: Path):
    """Save questions to JSONL file."""
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


def extract_unique_camera_poses(questions: List[Dict[str, Any]]) -> Dict[int, Dict]:
    """
    Extract unique camera poses from questions.
    
    Args:
        questions: List of question dictionaries
        
    Returns:
        Dictionary mapping camera_pose_idx to camera_pose dict
    """
    unique_poses = {}
    
    for q in questions:
        pose_idx = q.get('camera_pose_idx')
        if pose_idx is not None and pose_idx not in unique_poses:
            camera_pose = q.get('camera_pose')
            if camera_pose is not None:
                unique_poses[pose_idx] = camera_pose
    
    return unique_poses


async def render_poses(
    scene_id: str,
    camera_poses: Dict[int, Dict],
    images_dir: Path,
    render_config: RenderConfig,
    skip_existing: bool = True
) -> Dict[int, Path]:
    """
    Render images for all camera poses.
    
    Args:
        scene_id: Scene identifier
        camera_poses: Dictionary mapping pose_idx to camera_pose dict
        images_dir: Directory to save rendered images
        render_config: Render configuration
        skip_existing: Whether to skip already rendered images
        
    Returns:
        Dictionary mapping pose_idx to image file path
    """
    rendered = {}
    images_dir.mkdir(parents=True, exist_ok=True)
    
    async with SceneRenderer(render_config) as renderer:
        await renderer.set_scene(scene_id)
        
        for pose_idx, pose in tqdm(sorted(camera_poses.items()), desc="Rendering"):
            image_path = images_dir / f"pose_{pose_idx:03d}.png"
            
            # Skip if already exists
            if skip_existing and image_path.exists():
                print(f"  Skipping pose {pose_idx} (already exists)")
                rendered[pose_idx] = image_path
                continue
            
            try:
                # Convert pose to camera matrices
                intrinsics, extrinsics_c2w = camera_pose_to_matrices(
                    pose,
                    render_config.image_width,
                    render_config.image_height,
                    render_config.fov_deg
                )
                
                # Render image
                image = await renderer.render_image(intrinsics, extrinsics_c2w)
                
                if image is not None:
                    image.save(image_path)
                    rendered[pose_idx] = image_path
                    print(f"  Rendered pose {pose_idx}: {image_path.name}")
                else:
                    print(f"  [Warning] Failed to render pose {pose_idx}")
                    
            except Exception as e:
                print(f"  [Error] Error rendering pose {pose_idx}: {e}")
    
    return rendered


def update_questions_with_images(
    questions: List[Dict[str, Any]], 
    rendered_images: Dict[int, Path],
    images_dir: Path
) -> List[Dict[str, Any]]:
    """
    Update questions with image paths.
    
    Args:
        questions: List of question dictionaries
        rendered_images: Dictionary mapping pose_idx to image path
        images_dir: Base images directory
        
    Returns:
        Updated questions list
    """
    updated_questions = []
    
    for q in questions:
        pose_idx = q.get('camera_pose_idx')
        if pose_idx is not None and pose_idx in rendered_images:
            q = q.copy()  # Don't modify original
            image_path = rendered_images[pose_idx]
            # Store relative path: images/pose_xxx.png
            q['image'] = f"images/{image_path.name}"
        updated_questions.append(q)
    
    return updated_questions


def process_scene_data(
    data_dir: Path,
    scenes_root: str,
    render_config: RenderConfig,
    skip_existing: bool = True,
    backup: bool = True
) -> Dict[str, Any]:
    """
    Process a single scene's data directory and render images.
    
    Args:
        data_dir: Path to scene data directory (containing questions.jsonl)
        scenes_root: Root directory for Gaussian Splatting scenes
        render_config: Render configuration
        skip_existing: Whether to skip already rendered images
        backup: Whether to backup original questions.jsonl
        
    Returns:
        Processing statistics
    """
    questions_path = data_dir / 'questions.jsonl'
    
    if not questions_path.exists():
        print(f"[Error] questions.jsonl not found in {data_dir}")
        return {'success': False, 'error': 'questions.jsonl not found'}
    
    # Load questions
    print(f"Loading questions from {questions_path}")
    questions = load_questions(questions_path)
    print(f"  Loaded {len(questions)} questions")
    
    if not questions:
        return {'success': False, 'error': 'No questions loaded'}
    
    # Get scene_id from first question
    scene_id = questions[0].get('scene_id')
    if not scene_id:
        # Try to infer from directory name
        scene_id = data_dir.name
    print(f"  Scene ID: {scene_id}")
    
    # Extract unique camera poses
    camera_poses = extract_unique_camera_poses(questions)
    print(f"  Found {len(camera_poses)} unique camera poses")
    
    if not camera_poses:
        return {'success': False, 'error': 'No camera poses found in questions'}
    
    # Create images directory
    images_dir = data_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Update render config with scenes_root
    render_config.scenes_root = scenes_root
    
    # Render images
    print(f"Rendering images to {images_dir}")
    rendered_images = _run_async(render_poses(
        scene_id, camera_poses, images_dir, render_config, skip_existing
    ))
    print(f"  Rendered {len(rendered_images)} images")
    
    # Backup original questions
    if backup:
        backup_path = data_dir / 'questions_backup.jsonl'
        if not backup_path.exists():
            shutil.copy(questions_path, backup_path)
            print(f"  Created backup: {backup_path.name}")
    
    # Update questions with image paths
    updated_questions = update_questions_with_images(questions, rendered_images, images_dir)
    
    # Save updated questions
    save_questions(updated_questions, questions_path)
    print(f"  Updated questions.jsonl with image paths")
    
    # Save render log
    log = {
        'rendered_at': datetime.now().isoformat(),
        'scene_id': scene_id,
        'total_questions': len(questions),
        'unique_poses': len(camera_poses),
        'rendered_images': len(rendered_images),
        'render_config': {
            'width': render_config.image_width,
            'height': render_config.image_height,
            'fov_deg': render_config.fov_deg,
            'render_backend': render_config.render_backend,
        }
    }
    
    log_path = data_dir / 'render_log.json'
    with open(log_path, 'w') as f:
        json.dump(log, f, indent=2)
    print(f"  Saved render log: {log_path.name}")
    
    return {
        'success': True,
        'scene_id': scene_id,
        'total_questions': len(questions),
        'unique_poses': len(camera_poses),
        'rendered_images': len(rendered_images)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Render images for existing VQA questions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Data paths - mutually exclusive
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data_dir', type=str,
                       help='Path to a single scene data directory containing questions.jsonl')
    group.add_argument('--data_root', type=str,
                       help='Root directory containing multiple scene data directories')
    
    # Scenes root (required)
    parser.add_argument('--scenes_root', type=str, required=True,
                        help='Root directory for Gaussian Splatting scenes (InteriorGS)')
    
    # Rendering options
    parser.add_argument('--width', type=int, default=640,
                        help='Rendered image width (default: 640)')
    parser.add_argument('--height', type=int, default=480,
                        help='Rendered image height (default: 480)')
    parser.add_argument('--fov', type=float, default=60.0,
                        help='Field of view in degrees (default: 60)')
    parser.add_argument('--render_backend', type=str, default='local',
                        choices=['local', 'client'],
                        help='Rendering backend (default: local)')
    parser.add_argument('--gpu_device', type=int, default=None,
                        help='GPU device ID (default: auto)')
    
    # Processing options
    parser.add_argument('--skip_existing', action='store_true', default=True,
                        help='Skip already rendered images (default: True)')
    parser.add_argument('--no_skip_existing', dest='skip_existing', action='store_false',
                        help='Re-render all images')
    parser.add_argument('--no_backup', action='store_true', default=False,
                        help='Do not create backup of original questions.jsonl')
    
    args = parser.parse_args()
    
    # Create render config
    render_config = RenderConfig(
        scenes_root=args.scenes_root,
        render_backend=args.render_backend,
        image_width=args.width,
        image_height=args.height,
        fov_deg=args.fov,
        gpu_device=args.gpu_device
    )
    
    # Process single scene or multiple scenes
    if args.data_dir:
        data_dir = Path(args.data_dir)
        result = process_scene_data(
            data_dir, args.scenes_root, render_config,
            skip_existing=args.skip_existing,
            backup=not args.no_backup
        )
        
        if result['success']:
            print(f"\nSuccess! Rendered {result['rendered_images']} images")
        else:
            print(f"\nFailed: {result.get('error', 'Unknown error')}")
            sys.exit(1)
    
    else:
        # Process all scenes in data_root
        data_root = Path(args.data_root)
        
        # Find all scene directories
        scene_dirs = [d for d in data_root.iterdir() 
                      if d.is_dir() and (d / 'questions.jsonl').exists()]
        
        print(f"Found {len(scene_dirs)} scene directories to process\n")
        
        results = []
        for scene_dir in scene_dirs:
            print(f"\n{'='*60}")
            print(f"Processing: {scene_dir.name}")
            print('='*60)
            
            result = process_scene_data(
                scene_dir, args.scenes_root, render_config,
                skip_existing=args.skip_existing,
                backup=not args.no_backup
            )
            results.append(result)
        
        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print('='*60)
        
        success_count = sum(1 for r in results if r['success'])
        total_images = sum(r.get('rendered_images', 0) for r in results if r['success'])
        
        print(f"Scenes processed: {success_count}/{len(scene_dirs)}")
        print(f"Total images rendered: {total_images}")


if __name__ == '__main__':
    main()
