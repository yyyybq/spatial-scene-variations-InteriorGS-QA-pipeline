#!/usr/bin/env python3
"""
Convert single-view questions.jsonl to multi-view multi_view_questions.jsonl

原始格式：同一个question_id出现N次，每次对应一个视角（一张图片）
转换后格式：同一个question_id出现N次，但每条数据包含递增数量的视角：
  - 第1条：1张图片（随机选择1个视角）
  - 第2条：2张图片（随机选择2个视角）
  - ...
  - 第N条：N张图片（所有视角）

这样可以评估多视角信息对模型回答准确率的影响。
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict
import argparse
from copy import deepcopy


def load_questions(questions_file):
    """Load questions from a JSONL file."""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def save_questions(questions, output_file):
    """Save questions to a JSONL file."""
    with open(output_file, 'w') as f:
        for q in questions:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')


def group_by_question_id(questions):
    """Group questions by question_id."""
    grouped = defaultdict(list)
    for q in questions:
        qid = q.get("question_id", "")
        grouped[qid].append(q)
    return grouped


def convert_to_multiview(questions, seed=42):
    """
    Convert single-view questions to multi-view format.
    
    For each unique question_id with N views:
    - Generate N new entries
    - Entry i contains i randomly selected views (i = 1, 2, ..., N)
    - All entries share the same question and answer
    
    Returns:
        list: Converted questions in multi-view format
    """
    random.seed(seed)
    
    # Group questions by question_id
    grouped = group_by_question_id(questions)
    
    multiview_questions = []
    
    for question_id, views in grouped.items():
        num_views = len(views)
        
        # Sort views by camera_pose_idx for consistency
        views_sorted = sorted(views, key=lambda x: x.get("camera_pose_idx", 0))
        
        # Shuffle for random selection
        views_shuffled = views_sorted.copy()
        random.shuffle(views_shuffled)
        
        # Generate N entries with 1, 2, ..., N views
        for num_images in range(1, num_views + 1):
            # Select first num_images views from shuffled list
            selected_views = views_shuffled[:num_images]
            
            # Sort selected views by camera_pose_idx for consistent ordering
            selected_views_sorted = sorted(selected_views, key=lambda x: x.get("camera_pose_idx", 0))
            
            # Create new multi-view entry based on first view
            base_entry = deepcopy(selected_views_sorted[0])
            
            # Collect all images and camera poses
            images = [v.get("image", "") for v in selected_views_sorted]
            camera_poses = [v.get("camera_pose", {}) for v in selected_views_sorted]
            camera_pose_indices = [v.get("camera_pose_idx", 0) for v in selected_views_sorted]
            
            # Update entry with multi-view information
            base_entry["images"] = images  # List of image paths
            base_entry["image"] = images[0]  # Keep first image for backward compatibility
            base_entry["camera_poses"] = camera_poses  # List of camera poses
            base_entry["camera_pose_indices"] = camera_pose_indices  # List of camera indices
            base_entry["num_views"] = num_images  # Number of views in this entry
            base_entry["total_available_views"] = num_views  # Total views available for this question
            base_entry["multiview_entry_idx"] = num_images - 1  # 0-indexed entry number
            
            multiview_questions.append(base_entry)
    
    # Sort by question_id and num_views for organized output
    multiview_questions.sort(key=lambda x: (x.get("question_id", ""), x.get("num_views", 0)))
    
    return multiview_questions


def convert_single_file(input_file, output_file, seed=42):
    """Convert a single questions.jsonl file."""
    print(f"Processing: {input_file}")
    
    questions = load_questions(input_file)
    print(f"  Loaded {len(questions)} questions")
    
    multiview_questions = convert_to_multiview(questions, seed=seed)
    print(f"  Generated {len(multiview_questions)} multi-view entries")
    
    save_questions(multiview_questions, output_file)
    print(f"  Saved to: {output_file}")
    
    # Statistics
    grouped = group_by_question_id(questions)
    num_unique = len(grouped)
    views_counts = [len(v) for v in grouped.values()]
    print(f"  Unique question IDs: {num_unique}")
    print(f"  Views per question: min={min(views_counts)}, max={max(views_counts)}, avg={sum(views_counts)/len(views_counts):.2f}")
    
    return len(multiview_questions)


def convert_dataset(dataset_root, seed=42, output_filename="multi_view_questions.jsonl"):
    """Convert the entire dataset."""
    dataset_root = Path(dataset_root)
    
    patterns = ["around", "spherical", "linear_approach", "linear_passby", "rotation"]
    
    total_converted = 0
    conversion_stats = []
    
    for pattern in patterns:
        pattern_dir = dataset_root / pattern
        if not pattern_dir.exists():
            continue
        
        print(f"\n{'='*60}")
        print(f"Pattern: {pattern}")
        print(f"{'='*60}")
        
        for scene_dir in sorted(pattern_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            scene_id = scene_dir.name
            
            # Check nested structure
            nested_dir = scene_dir / scene_id
            if nested_dir.exists() and (nested_dir / "questions.jsonl").exists():
                actual_dir = nested_dir
            elif (scene_dir / "questions.jsonl").exists():
                actual_dir = scene_dir
            else:
                continue
            
            input_file = actual_dir / "questions.jsonl"
            output_file = actual_dir / output_filename
            
            count = convert_single_file(input_file, output_file, seed=seed)
            total_converted += count
            conversion_stats.append({
                "pattern": pattern,
                "scene_id": scene_id,
                "output_file": str(output_file),
                "num_entries": count
            })
    
    print(f"\n{'='*60}")
    print("CONVERSION COMPLETE")
    print(f"{'='*60}")
    print(f"Total multi-view entries created: {total_converted}")
    print(f"Files converted: {len(conversion_stats)}")
    
    # Save conversion log
    log_file = dataset_root / "multiview_conversion_log.json"
    with open(log_file, 'w') as f:
        json.dump({
            "seed": seed,
            "output_filename": output_filename,
            "total_entries": total_converted,
            "conversions": conversion_stats
        }, f, indent=2)
    print(f"Conversion log saved to: {log_file}")
    
    return conversion_stats


def demo_conversion():
    """Show a demo of how the conversion works."""
    print("\n" + "="*80)
    print("DEMO: How Multi-View Conversion Works")
    print("="*80)
    
    # Create sample data
    sample_questions = [
        {
            "question": "What is the size of the table?",
            "answer": "[1.0, 0.5, 0.8]",
            "question_type": "object_size",
            "question_id": "object_size_1",
            "camera_pose_idx": i,
            "image": f"images/table_{i}.png",
            "camera_pose": {"position": [i, 0, 1], "yaw": i*36}
        }
        for i in range(5)
    ]
    
    print("\nOriginal format (5 separate entries for same question):")
    print("-" * 60)
    for q in sample_questions[:3]:
        print(f"  question_id: {q['question_id']}, camera_pose_idx: {q['camera_pose_idx']}, image: {q['image']}")
    print("  ...")
    
    # Convert
    multiview = convert_to_multiview(sample_questions, seed=42)
    
    print("\nConverted multi-view format (5 entries with 1,2,3,4,5 views):")
    print("-" * 60)
    for q in multiview:
        print(f"  num_views: {q['num_views']}, images: {q['images']}")
    
    print("\nThis allows evaluating model performance across different numbers of input views!")


def main():
    parser = argparse.ArgumentParser(description="Convert single-view to multi-view questions")
    parser.add_argument("--dataset_root", type=str, 
                        default="/scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_all_type",
                        help="Root directory of the dataset")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for view selection")
    parser.add_argument("--output_filename", type=str, default="multi_view_questions.jsonl",
                        help="Output filename for converted questions")
    parser.add_argument("--demo", action="store_true",
                        help="Show a demo of the conversion")
    parser.add_argument("--single_file", type=str, default=None,
                        help="Convert a single file instead of the whole dataset")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_conversion()
        return
    
    if args.single_file:
        input_path = Path(args.single_file)
        output_path = input_path.parent / args.output_filename
        convert_single_file(input_path, output_path, seed=args.seed)
    else:
        convert_dataset(args.dataset_root, seed=args.seed, output_filename=args.output_filename)


if __name__ == "__main__":
    main()
