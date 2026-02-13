#!/usr/bin/env python3
"""
Batch generation script for InteriorQA dataset.

This script generates all move patterns (around, spherical, linear) for selected scenes
and creates demo visualizations for each pattern.

Usage:
    python generate_interiorqa_batch.py
"""

import argparse
import json
import os
import subprocess
import sys
import random
from pathlib import Path

# Configuration
SCENES_ROOT = "/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_BASE = "/scratch/by2593/project/sceneshift/data/InteriorQA"
SCRIPT_DIR = Path(__file__).parent

# Selected scenes (scene_267 + 4 random scenes)
SELECTED_SCENES = [
    "0267_840790",  # Required scene
    "0414_840040",  # Random
    "0400_840162",  # Random
    "0048_839893",  # Random
    "0363_840304",  # Random
]

# Move patterns to generate
MOVE_PATTERNS = ["around", "spherical", "linear"]

# Linear sub-patterns
LINEAR_SUB_PATTERNS = ["approach", "pass_by"]


def run_pipeline(scene_id: str, move_pattern: str, output_dir: str, 
                 linear_sub_pattern: str = None, enable_rendering: bool = True):
    """Run the question generation pipeline for a scene with a specific move pattern."""
    
    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "run_pipeline.py"),
        "--scenes_root", SCENES_ROOT,
        "--output_dir", output_dir,
        "--scene_id", scene_id,
        "--move_pattern", move_pattern,
        "--num_cameras", "5",
        "--max_questions_per_type", "10",
    ]
    
    if enable_rendering:
        cmd.append("--enable_rendering")
    
    if linear_sub_pattern:
        cmd.extend(["--linear_sub_pattern", linear_sub_pattern])
        cmd.extend(["--linear_num_steps", "5"])
    
    print(f"\n{'='*60}")
    print(f"Running: Scene={scene_id}, Pattern={move_pattern}")
    if linear_sub_pattern:
        print(f"         Linear sub-pattern={linear_sub_pattern}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    return result.returncode == 0


def create_demo(questions_file: str, output_root: str, pattern: str, demo_output_dir: str):
    """Create demo visualization for a generated dataset."""
    
    if not os.path.exists(questions_file):
        print(f"  Warning: Questions file not found: {questions_file}")
        return False
    
    # Load questions to find available objects
    objects = []
    with open(questions_file, 'r') as f:
        for line in f:
            if line.strip():
                q = json.loads(line)
                if 'objects' in q and len(q['objects']) > 0:
                    label = q['objects'][0].get('label', '')
                    if label and label not in objects:
                        objects.append(label)
    
    if not objects:
        print(f"  Warning: No objects found in {questions_file}")
        return False
    
    # Create demos for up to 3 random objects
    demo_objects = random.sample(objects, min(3, len(objects)))
    
    os.makedirs(demo_output_dir, exist_ok=True)
    
    for obj_label in demo_objects:
        safe_label = obj_label.replace(' ', '_').replace('/', '_')
        demo_file = os.path.join(demo_output_dir, f"demo_{pattern}_{safe_label}.png")
        
        cmd = [
            sys.executable,
            str(SCRIPT_DIR / "create_demo.py"),
            "--questions_file", questions_file,
            "--output_file", demo_file,
            "--output_root", output_root,
            "--pattern", pattern,
            "--object_label", obj_label,
            "--num_views", "5",
        ]
        
        print(f"  Creating demo for object '{obj_label}'...")
        result = subprocess.run(cmd, cwd=str(SCRIPT_DIR), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    Warning: Demo creation failed")
            if result.stderr:
                print(f"    Error: {result.stderr[:200]}")
        else:
            print(f"    Saved: {demo_file}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Batch generate InteriorQA data')
    parser.add_argument('--skip-generation', action='store_true', 
                        help='Skip data generation, only create demos')
    parser.add_argument('--skip-demos', action='store_true',
                        help='Skip demo creation')
    parser.add_argument('--no-render', action='store_true',
                        help='Disable rendering (faster, no images)')
    args = parser.parse_args()
    
    # Create output base directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    # Track success/failure
    results = []
    
    for scene_id in SELECTED_SCENES:
        scene_output_base = os.path.join(OUTPUT_BASE, scene_id)
        
        for move_pattern in MOVE_PATTERNS:
            if move_pattern == "linear":
                # Generate both linear sub-patterns
                for sub_pattern in LINEAR_SUB_PATTERNS:
                    pattern_name = f"linear_{sub_pattern}"
                    pattern_output_dir = os.path.join(scene_output_base, pattern_name)
                    
                    if not args.skip_generation:
                        success = run_pipeline(
                            scene_id=scene_id,
                            move_pattern=move_pattern,
                            output_dir=pattern_output_dir,
                            linear_sub_pattern=sub_pattern,
                            enable_rendering=not args.no_render
                        )
                        results.append((scene_id, pattern_name, success))
                    
                    # Create demos
                    if not args.skip_demos:
                        questions_file = os.path.join(pattern_output_dir, "default", "questions.jsonl")
                        demo_dir = os.path.join(pattern_output_dir, "demos")
                        create_demo(questions_file, pattern_output_dir, move_pattern, demo_dir)
            else:
                pattern_output_dir = os.path.join(scene_output_base, move_pattern)
                
                if not args.skip_generation:
                    success = run_pipeline(
                        scene_id=scene_id,
                        move_pattern=move_pattern,
                        output_dir=pattern_output_dir,
                        enable_rendering=not args.no_render
                    )
                    results.append((scene_id, move_pattern, success))
                
                # Create demos
                if not args.skip_demos:
                    questions_file = os.path.join(pattern_output_dir, "default", "questions.jsonl")
                    demo_dir = os.path.join(pattern_output_dir, "demos")
                    create_demo(questions_file, pattern_output_dir, move_pattern, demo_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    if results:
        success_count = sum(1 for _, _, s in results if s)
        total_count = len(results)
        print(f"Generation: {success_count}/{total_count} successful")
        
        for scene_id, pattern, success in results:
            status = "✓" if success else "✗"
            print(f"  {status} {scene_id} / {pattern}")
    
    print(f"\nOutput directory: {OUTPUT_BASE}")
    print("\nDirectory structure:")
    print("  {scene_id}/")
    print("    around/")
    print("      default/images/ + questions.jsonl")
    print("      demos/")
    print("    spherical/")
    print("      default/images/ + questions.jsonl")
    print("      demos/")
    print("    linear_approach/")
    print("      default/images/ + questions.jsonl")
    print("      demos/")
    print("    linear_pass_by/")
    print("      default/images/ + questions.jsonl")
    print("      demos/")


if __name__ == '__main__':
    main()
