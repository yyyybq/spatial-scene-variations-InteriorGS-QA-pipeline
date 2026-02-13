#!/usr/bin/env python3
"""
Add Multi-Object Questions to Existing Dataset

This script adds missing multi-object questions (object_comparison_absolute_distance 
and object_comparison_relative_distance) to existing questions.jsonl files.

The multi-object questions were not generated in the original run due to the 
max_questions_per_scene limit truncating them.

Usage:
    python add_multi_object_questions.py \
        --scenes_root /path/to/InteriorGS \
        --dataset_root /path/to/generated_dataset \
        [--max_questions_per_type 5]
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict
import sys

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import ObjectSelectionConfig, QuestionConfig
from object_selector import ObjectSelector
from question_generator import QuestionGenerator
from camera_utils import CameraPose


def load_questions(questions_file: Path):
    """Load questions from a JSONL file."""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def save_questions(questions, output_file: Path):
    """Save questions to a JSONL file."""
    with open(output_file, 'w') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


def get_question_types_count(questions):
    """Count questions by type."""
    counts = defaultdict(int)
    for q in questions:
        counts[q.get('question_type', 'unknown')] += 1
    return dict(counts)


def add_multi_object_questions(
    scenes_root: Path,
    dataset_root: Path,
    max_questions_per_type: int = 5,
    verbose: bool = True
):
    """
    Add multi-object questions to all questions.jsonl files in the dataset.
    
    Args:
        scenes_root: Root directory containing InteriorGS scene folders
        dataset_root: Root directory of the generated dataset
        max_questions_per_type: Maximum multi-object questions per type per camera pose
        verbose: Print verbose output
    """
    # Setup components
    obj_selector = ObjectSelector(ObjectSelectionConfig())
    question_config = QuestionConfig(
        max_questions_per_type=max_questions_per_type,
        enabled_question_types=[
            'object_comparison_absolute_distance',
            'object_comparison_relative_distance',
        ]
    )
    question_generator = QuestionGenerator(question_config)
    
    patterns = ['around', 'spherical', 'linear_approach', 'linear_passby', 'rotation']
    
    total_added = 0
    results = []
    
    for pattern in patterns:
        pattern_dir = dataset_root / pattern
        if not pattern_dir.exists():
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing pattern: {pattern}")
            print(f"{'='*60}")
        
        # Find all scene directories with questions.jsonl
        for scene_dir in pattern_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            
            # Handle nested structure: {pattern}/{scene}/{scene}/questions.jsonl
            questions_file = None
            for candidate in [
                scene_dir / 'questions.jsonl',
                scene_dir / scene_dir.name / 'questions.jsonl',
            ]:
                if candidate.exists():
                    questions_file = candidate
                    break
            
            if not questions_file:
                continue
            
            scene_id = scene_dir.name
            scene_path = scenes_root / scene_id
            
            if not scene_path.exists():
                if verbose:
                    print(f"  Scene not found: {scene_id}")
                continue
            
            if verbose:
                print(f"\n  Processing scene: {scene_id}")
            
            # Load existing questions
            existing_questions = load_questions(questions_file)
            existing_types = get_question_types_count(existing_questions)
            
            if verbose:
                print(f"    Existing questions: {len(existing_questions)}")
                print(f"    Existing types: {existing_types}")
            
            # Check if multi-object questions already exist
            has_multi_abs = 'object_comparison_absolute_distance' in existing_types
            has_multi_rel = 'object_comparison_relative_distance' in existing_types
            
            if has_multi_abs and has_multi_rel:
                if verbose:
                    print(f"    Multi-object questions already exist, skipping.")
                continue
            
            # Get single objects for multi-object question generation
            try:
                single_objects = obj_selector.select_single_objects(scene_path)
            except Exception as e:
                if verbose:
                    print(f"    Error loading objects: {e}")
                continue
            
            if len(single_objects) < 3:
                if verbose:
                    print(f"    Not enough objects ({len(single_objects)}) for multi-object questions")
                continue
            
            if verbose:
                print(f"    Found {len(single_objects)} single objects")
            
            # Extract unique camera poses from existing questions
            camera_poses_dict = {}
            for q in existing_questions:
                pose_idx = q.get('camera_pose_idx')
                pose_data = q.get('camera_pose')
                if pose_idx is not None and pose_data:
                    if pose_idx not in camera_poses_dict:
                        try:
                            camera_poses_dict[pose_idx] = CameraPose.from_dict(pose_data)
                        except Exception as e:
                            pass
            
            if verbose:
                print(f"    Found {len(camera_poses_dict)} unique camera poses")
            
            if not camera_poses_dict:
                if verbose:
                    print(f"    No valid camera poses found, skipping")
                continue
            
            # Generate multi-object questions for each camera pose
            multi_questions = []
            scene_id_from_q = existing_questions[0].get('scene_id', scene_id) if existing_questions else scene_id
            
            for pose_idx, camera_pose in sorted(camera_poses_dict.items()):
                questions = question_generator.generate_multi_object_questions(
                    single_objects, camera_pose,
                    max_questions_per_type=max_questions_per_type
                )
                
                for q in questions:
                    q['scene_id'] = scene_id_from_q
                    q['camera_pose_idx'] = pose_idx
                    # Copy image path from an existing question with same pose_idx
                    for existing_q in existing_questions:
                        if existing_q.get('camera_pose_idx') == pose_idx:
                            if 'image' in existing_q:
                                q['image'] = existing_q['image']
                            if 'room' in existing_q:
                                q['room'] = existing_q['room']
                            break
                
                multi_questions.extend(questions)
            
            if verbose:
                multi_types = get_question_types_count(multi_questions)
                print(f"    Generated {len(multi_questions)} multi-object questions: {multi_types}")
            
            # Append multi-object questions to existing file
            if multi_questions:
                all_questions = existing_questions + multi_questions
                save_questions(all_questions, questions_file)
                
                total_added += len(multi_questions)
                results.append({
                    'pattern': pattern,
                    'scene_id': scene_id,
                    'questions_added': len(multi_questions),
                    'types_added': get_question_types_count(multi_questions),
                })
                
                if verbose:
                    print(f"    ✓ Added {len(multi_questions)} questions to {questions_file}")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Total multi-object questions added: {total_added}")
    print(f"Files updated: {len(results)}")
    
    if results:
        print("\nDetails:")
        for r in results:
            print(f"  {r['pattern']}/{r['scene_id']}: +{r['questions_added']} ({r['types_added']})")
    
    return total_added, results


def main():
    parser = argparse.ArgumentParser(
        description='Add multi-object questions to existing dataset'
    )
    parser.add_argument('--scenes_root', required=True, type=str,
                        help='Root directory containing InteriorGS scene folders')
    parser.add_argument('--dataset_root', required=True, type=str,
                        help='Root directory of the generated dataset')
    parser.add_argument('--max_questions_per_type', type=int, default=5,
                        help='Max multi-object questions per type per camera pose')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    
    args = parser.parse_args()
    
    scenes_root = Path(args.scenes_root)
    dataset_root = Path(args.dataset_root)
    
    if not scenes_root.exists():
        print(f"Error: scenes_root not found: {scenes_root}")
        sys.exit(1)
    
    if not dataset_root.exists():
        print(f"Error: dataset_root not found: {dataset_root}")
        sys.exit(1)
    
    add_multi_object_questions(
        scenes_root=scenes_root,
        dataset_root=dataset_root,
        max_questions_per_type=args.max_questions_per_type,
        verbose=args.verbose
    )


if __name__ == '__main__':
    main()
