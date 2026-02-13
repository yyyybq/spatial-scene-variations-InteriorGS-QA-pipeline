#!/usr/bin/env python3
"""
Dataset Statistics Analysis for InteriorGS_5scenes_all_type
Analyzes the VQA dataset structure, question types, objects, and image-question relationships.



 python analyze_dataset_statistics.py --dataset_root /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix_v2

 python convert_to_multiview.py --dataset_root /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix_v2
 
 python create_benchmark.py --dataset_root /scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix_v2
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import argparse


def load_questions(questions_file):
    """Load questions from a JSONL file."""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_metadata(metadata_file):
    """Load metadata from a JSON file."""
    with open(metadata_file, 'r') as f:
        return json.load(f)


def analyze_single_scene(scene_dir):
    """Analyze a single scene directory."""
    questions_file = scene_dir / "questions.jsonl"
    metadata_file = scene_dir / "metadata.json"
    images_dir = scene_dir / "images"
    
    if not questions_file.exists():
        return None
    
    questions = load_questions(questions_file)
    metadata = load_metadata(metadata_file) if metadata_file.exists() else {}
    
    # Count images
    num_images = len(list(images_dir.glob("*.png"))) if images_dir.exists() else 0
    
    # Analyze questions
    question_types = Counter()
    objects_per_question = []
    unique_objects = set()
    unique_question_ids = set()
    image_to_questions = defaultdict(list)  # Track which questions use which images
    question_id_to_cameras = defaultdict(set)  # Track camera poses per question_id
    
    for q in questions:
        question_types[q.get("question_type", "unknown")] += 1
        objects_per_question.append(len(q.get("objects", [])))
        
        for obj in q.get("objects", []):
            unique_objects.add(obj.get("label", "unknown"))
        
        # Track image-question relationship
        image_path = q.get("image", "")
        image_to_questions[image_path].append(q)
        
        # Track unique question_ids and their camera poses
        question_id = q.get("question_id", "")
        camera_pose_idx = q.get("camera_pose_idx", -1)
        unique_question_ids.add(question_id)
        question_id_to_cameras[question_id].add(camera_pose_idx)
    
    # Calculate views per unique question
    views_per_question = [len(cameras) for cameras in question_id_to_cameras.values()]
    
    return {
        "num_questions": len(questions),
        "num_images": num_images,
        "num_unique_question_ids": len(unique_question_ids),
        "question_types": dict(question_types),
        "avg_objects_per_question": sum(objects_per_question) / len(objects_per_question) if objects_per_question else 0,
        "unique_objects": list(unique_objects),
        "num_unique_objects": len(unique_objects),
        "image_to_question_count": {k: len(v) for k, v in image_to_questions.items()},
        "questions_per_image": Counter(len(v) for v in image_to_questions.values()),
        "views_per_question": Counter(views_per_question),
        "avg_views_per_question": sum(views_per_question) / len(views_per_question) if views_per_question else 0,
        "question_id_to_cameras": {k: list(v) for k, v in question_id_to_cameras.items()},
    }


def analyze_dataset(dataset_root):
    """Analyze the entire dataset."""
    dataset_root = Path(dataset_root)
    
    patterns = ["around", "spherical", "linear_approach", "linear_passby", "rotation"]
    
    all_stats = {}
    pattern_summaries = {}
    
    for pattern in patterns:
        pattern_dir = dataset_root / pattern
        if not pattern_dir.exists():
            continue
        
        pattern_stats = {}
        total_questions = 0
        total_images = 0
        total_unique_questions = 0
        all_question_types = Counter()
        all_objects = set()
        all_views_per_q = []
        
        for scene_dir in pattern_dir.iterdir():
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
            
            stats = analyze_single_scene(actual_dir)
            if stats:
                pattern_stats[scene_id] = stats
                total_questions += stats["num_questions"]
                total_images += stats["num_images"]
                total_unique_questions += stats["num_unique_question_ids"]
                all_question_types.update(stats["question_types"])
                all_objects.update(stats["unique_objects"])
                all_views_per_q.extend(stats["views_per_question"].elements())
        
        all_stats[pattern] = pattern_stats
        pattern_summaries[pattern] = {
            "num_scenes": len(pattern_stats),
            "total_questions": total_questions,
            "total_images": total_images,
            "total_unique_question_ids": total_unique_questions,
            "question_types": dict(all_question_types),
            "num_unique_objects": len(all_objects),
            "unique_objects": sorted(list(all_objects)),
            "avg_views_per_question": sum(all_views_per_q) / len(all_views_per_q) if all_views_per_q else 0,
            "views_distribution": dict(Counter(all_views_per_q)),
        }
    
    return all_stats, pattern_summaries


def print_report(all_stats, pattern_summaries, output_file=None):
    """Print a comprehensive report."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("DATASET STATISTICS ANALYSIS")
    lines.append("InteriorGS_5scenes_all_type")
    lines.append("=" * 80)
    
    # Overall summary
    lines.append("\n" + "=" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("=" * 80)
    
    total_q = sum(s["total_questions"] for s in pattern_summaries.values())
    total_img = sum(s["total_images"] for s in pattern_summaries.values())
    total_unique_q = sum(s["total_unique_question_ids"] for s in pattern_summaries.values())
    
    lines.append(f"Total Questions (all patterns): {total_q}")
    lines.append(f"Total Images (all patterns): {total_img}")
    lines.append(f"Total Unique Question IDs: {total_unique_q}")
    lines.append(f"Patterns: {', '.join(pattern_summaries.keys())}")
    
    # Per-pattern summary
    for pattern, summary in pattern_summaries.items():
        lines.append("\n" + "-" * 60)
        lines.append(f"Pattern: {pattern.upper()}")
        lines.append("-" * 60)
        lines.append(f"  Scenes: {summary['num_scenes']}")
        lines.append(f"  Total Questions (rows in JSONL): {summary['total_questions']}")
        lines.append(f"  Total Unique Question IDs: {summary['total_unique_question_ids']}")
        lines.append(f"  Total Images: {summary['total_images']}")
        lines.append(f"  Average Views per Unique Question: {summary['avg_views_per_question']:.2f}")
        lines.append(f"\n  Question Type Distribution:")
        for qt, count in sorted(summary["question_types"].items(), key=lambda x: -x[1]):
            lines.append(f"    - {qt}: {count}")
        lines.append(f"\n  Views per Question Distribution:")
        for views, count in sorted(summary["views_distribution"].items()):
            lines.append(f"    - {views} view(s): {count} questions")
    
    # QA-Image Relationship Explanation
    lines.append("\n" + "=" * 80)
    lines.append("QA-IMAGE RELATIONSHIP ANALYSIS")
    lines.append("=" * 80)
    
    lines.append("""
Structure Overview:
-------------------
Each question entry in questions.jsonl contains:
- question: The question text
- answer: Ground truth answer
- question_type: Type of spatial question
- question_id: Unique identifier for the question (e.g., "object_size_66")
- primary_object: ID of the main object being queried
- objects: List of objects with 3D properties (center, dims, aabb)
- camera_pose: Camera position, target, yaw, pitch, radius
- camera_pose_idx: Index of the camera view (0, 1, 2, ... for different views)
- image: Path to the corresponding rendered image
- scene_id: Scene identifier

QA-Image Relationship:
----------------------
1. Each row in questions.jsonl = 1 question + 1 image (1:1 mapping per row)

2. The same question_id appears MULTIPLE times with DIFFERENT camera_pose_idx values.
   This means the SAME question is asked from DIFFERENT viewpoints.

3. For example, "object_size_66" might appear 10 times, each with:
   - camera_pose_idx: 0, 1, 2, ..., 9
   - Different camera positions/angles
   - Different images: "images/side_table_0.png", "images/side_table_1.png", etc.

4. The number of views per question depends on the movement pattern:
""")
    
    for pattern, summary in pattern_summaries.items():
        avg_views = summary["avg_views_per_question"]
        views_dist = summary["views_distribution"]
        lines.append(f"   - {pattern}: {avg_views:.1f} avg views, distribution: {dict(sorted(views_dist.items()))}")
    
    lines.append("""
Movement Patterns:
------------------
- spherical: Camera moves on a sphere around the target object
- around: Camera orbits around the target object at varying angles
- linear_approach: Camera moves in a straight line towards the object
- linear_passby: Camera moves past the object in a straight line

Image Naming Convention:
------------------------
- Single object questions: {object_label}_{camera_idx}.png
  Example: side_table_0.png, side_table_1.png, ...
  
- Object pair questions: {object1_label}_{object2_label}_{camera_idx}.png
  Example: chair_table_0.png, bed_wardrobe_0.png, ...
""")
    
    # Per-scene detailed stats
    lines.append("\n" + "=" * 80)
    lines.append("PER-SCENE DETAILED STATISTICS")
    lines.append("=" * 80)
    
    for pattern, scenes in all_stats.items():
        lines.append(f"\n{'='*60}")
        lines.append(f"Pattern: {pattern}")
        lines.append(f"{'='*60}")
        
        for scene_id, stats in scenes.items():
            lines.append(f"\n  Scene: {scene_id}")
            lines.append(f"  {'-'*40}")
            lines.append(f"    Questions (JSONL rows): {stats['num_questions']}")
            lines.append(f"    Unique Question IDs: {stats['num_unique_question_ids']}")
            lines.append(f"    Images: {stats['num_images']}")
            lines.append(f"    Avg Views/Question: {stats['avg_views_per_question']:.2f}")
            lines.append(f"    Unique Objects: {stats['num_unique_objects']}")
            lines.append(f"    Objects: {', '.join(sorted(stats['unique_objects']))}")
            
            lines.append(f"\n    Question Types:")
            for qt, count in sorted(stats["question_types"].items(), key=lambda x: -x[1]):
                lines.append(f"      - {qt}: {count}")
            
            lines.append(f"\n    Views per Question Distribution:")
            for views, count in sorted(stats["views_per_question"].items()):
                lines.append(f"      - {views} view(s): {count} unique questions")
    
    # Sample question structure
    lines.append("\n" + "=" * 80)
    lines.append("SAMPLE QUESTION STRUCTURE")
    lines.append("=" * 80)
    
    # Get a sample question from first available scene
    sample_q = None
    for pattern, scenes in all_stats.items():
        for scene_id, stats in scenes.items():
            if "question_id_to_cameras" in stats and stats["question_id_to_cameras"]:
                # Get first question_id
                sample_qid = list(stats["question_id_to_cameras"].keys())[0]
                sample_cameras = stats["question_id_to_cameras"][sample_qid]
                lines.append(f"\nExample: question_id = '{sample_qid}'")
                lines.append(f"  This question appears with camera_pose_idx values: {sorted(sample_cameras)}")
                lines.append(f"  Total views for this question: {len(sample_cameras)}")
                break
        if sample_qid:
            break
    
    report = "\n".join(lines)
    print(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_file}")
    
    return report


def export_json_stats(all_stats, pattern_summaries, output_file):
    """Export statistics to JSON format."""
    export_data = {
        "pattern_summaries": pattern_summaries,
        "per_scene_stats": {}
    }
    
    for pattern, scenes in all_stats.items():
        export_data["per_scene_stats"][pattern] = {}
        for scene_id, stats in scenes.items():
            # Remove large dictionaries for cleaner export
            clean_stats = {k: v for k, v in stats.items() 
                          if k not in ["image_to_question_count", "question_id_to_cameras"]}
            clean_stats["questions_per_image"] = dict(stats["questions_per_image"])
            clean_stats["views_per_question"] = dict(stats["views_per_question"])
            export_data["per_scene_stats"][pattern][scene_id] = clean_stats
    
    with open(output_file, 'w') as f:
        json.dump(export_data, f, indent=2)
    print(f"JSON statistics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze VQA dataset statistics")
    parser.add_argument("--dataset_root", type=str, 
                        default="/scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_all_type",
                        help="Root directory of the dataset")
    parser.add_argument("--output_txt", type=str, default=None,
                        help="Output text file for the report")
    parser.add_argument("--output_json", type=str, default=None,
                        help="Output JSON file for the statistics")
    
    args = parser.parse_args()
    
    print(f"Analyzing dataset at: {args.dataset_root}")
    all_stats, pattern_summaries = analyze_dataset(args.dataset_root)
    
    # Default output files
    if args.output_txt is None:
        args.output_txt = os.path.join(args.dataset_root, "dataset_statistics_report.txt")
    if args.output_json is None:
        args.output_json = os.path.join(args.dataset_root, "dataset_statistics.json")
    
    print_report(all_stats, pattern_summaries, args.output_txt)
    export_json_stats(all_stats, pattern_summaries, args.output_json)


if __name__ == "__main__":
    main()
