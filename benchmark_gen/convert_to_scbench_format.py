#!/usr/bin/env python3
"""
Convert InteriorGS benchmark to Spatial-Consistency-Bench compatible format

This script converts the generated InteriorGS VQA benchmark to a format
compatible with the Spatial-Consistency-Bench inference pipeline.

Usage:
    python convert_to_scbench_format.py \
        --input /path/to/benchmark.jsonl \
        --output /path/to/scbench_format.jsonl \
        --dataset_root /path/to/InteriorGS_dataset
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List


def convert_sample(sample: Dict[str, Any], dataset_root: Path) -> Dict[str, Any]:
    """
    Convert a single sample to Spatial-Consistency-Bench format.
    
    Original format:
    {
        "question": "...",
        "answer": "...",
        "question_type": "...",
        "image": "images/xxx.png",
        "image_full_path": "around/scene/scene/images/xxx.png",
        "images": ["path1.png", "path2.png", ...],  # for multi-view
        ...
    }
    
    Target format:
    {
        "id": "unique_id",
        "question": "... <image_start>[image_1]<image_end> ...",
        "images": {"image_1": "full/path/to/image.png", ...},
        "gt_answer": "...",
        ...
    }
    """
    question_id = sample.get('question_id', 'unknown')
    scene_id = sample.get('scene_id', '')
    pattern = sample.get('pattern', '')
    num_views = sample.get('num_views', 1)
    
    # Generate unique ID
    unique_id = f"{pattern}_{scene_id}_{question_id}_v{num_views}"
    
    # Get image paths
    images_list = sample.get('images', [])
    if not images_list and sample.get('image'):
        images_list = [sample['image']]
    
    # Build image dict with placeholders
    images_dict = {}
    image_placeholders = []
    
    for i, img_path in enumerate(images_list):
        key = f"image_{i+1}"
        # Construct full path
        if sample.get('image_full_path') and i == 0:
            full_path = str(dataset_root / sample['image_full_path'])
        else:
            # For multi-view, construct path from pattern
            full_path = str(dataset_root / pattern / scene_id / scene_id / img_path)
        
        images_dict[key] = full_path
        image_placeholders.append(f"<image_start>[{key}]<image_end>")
    
    # Build question with image placeholders
    original_question = sample.get('question', '')
    question_type = sample.get('question_type', '')
    
    # Add image context to question
    if image_placeholders:
        if num_views > 1:
            image_context = f"Given {num_views} views of the scene:\n" + "\n".join(image_placeholders) + "\n\n"
        else:
            image_context = f"Given this view of the scene:\n{image_placeholders[0]}\n\n"
        question_with_images = image_context + original_question
    else:
        question_with_images = original_question
    
    # Get ground truth answer
    gt_answer = sample.get('answer', '')
    
    # Build converted sample
    converted = {
        'id': unique_id,
        'question': question_with_images,
        'images': images_dict,
        'gt_answer': gt_answer,
        # Preserve metadata for evaluation
        'question_type': question_type,
        'scene_id': scene_id,
        'pattern': pattern,
        'num_views': num_views,
        'original_question_id': question_id,
        # Additional metadata
        'objects': sample.get('objects', []),
        'camera_pose': sample.get('camera_pose'),
        'camera_poses': sample.get('camera_poses'),
    }
    
    return converted


def convert_benchmark(
    input_file: Path,
    output_file: Path,
    dataset_root: Path,
    max_samples: int = None,
    filter_question_types: List[str] = None,
    filter_num_views: int = None
) -> int:
    """
    Convert entire benchmark file.
    
    Args:
        input_file: Path to input benchmark.jsonl
        output_file: Path to output file
        dataset_root: Root directory of the dataset
        max_samples: Maximum number of samples to convert
        filter_question_types: Only include these question types
        filter_num_views: Only include samples with this many views
        
    Returns:
        Number of samples converted
    """
    converted_samples = []
    
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            
            sample = json.loads(line.strip())
            
            # Apply filters
            if filter_question_types:
                if sample.get('question_type') not in filter_question_types:
                    continue
            
            if filter_num_views is not None:
                if sample.get('num_views', 1) != filter_num_views:
                    continue
            
            converted = convert_sample(sample, dataset_root)
            converted_samples.append(converted)
    
    # Save converted samples
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        for sample in converted_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Converted {len(converted_samples)} samples")
    print(f"Output saved to: {output_file}")
    
    # Print statistics
    from collections import Counter
    type_counts = Counter(s['question_type'] for s in converted_samples)
    print("\nQuestion type distribution:")
    for qt, count in sorted(type_counts.items()):
        print(f"  {qt}: {count}")
    
    view_counts = Counter(s['num_views'] for s in converted_samples)
    print("\nViews distribution:")
    for nv, count in sorted(view_counts.items()):
        print(f"  {nv} views: {count}")
    
    return len(converted_samples)


def main():
    parser = argparse.ArgumentParser(
        description='Convert InteriorGS benchmark to SC-Bench format'
    )
    parser.add_argument('--input', '-i', required=True, type=str,
                        help='Input benchmark.jsonl file')
    parser.add_argument('--output', '-o', required=True, type=str,
                        help='Output file path')
    parser.add_argument('--dataset_root', '-d', required=True, type=str,
                        help='Root directory of the dataset')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to convert')
    parser.add_argument('--question_types', nargs='+', default=None,
                        help='Filter to specific question types')
    parser.add_argument('--num_views', type=int, default=None,
                        help='Filter to samples with specific number of views')
    
    args = parser.parse_args()
    
    convert_benchmark(
        input_file=Path(args.input),
        output_file=Path(args.output),
        dataset_root=Path(args.dataset_root),
        max_samples=args.max_samples,
        filter_question_types=args.question_types,
        filter_num_views=args.num_views
    )


if __name__ == '__main__':
    main()
