#!/usr/bin/env python3
"""
Benchmark Selection Script for InteriorGS VQA Dataset

Creates a balanced benchmark dataset with:
1. Equal number of questions across scenes (scene-balanced)
2. Balanced question types within each scene (type-balanced)
3. Only uses linear_approach, linear_passby, rotation patterns (no around/spherical)
4. Allows single-view questions for multi-object question types

Output:
- benchmark.jsonl: The selected benchmark questions
- benchmark_analysis.md: Analysis of the benchmark composition
"""

import json
import os
import random
from pathlib import Path
from collections import defaultdict, Counter
import argparse
from typing import List, Dict, Any, Tuple

# Multi-object question types that only have 1 view
MULTI_OBJECT_TYPES = [
    'object_comparison_absolute_distance',
    'object_comparison_relative_distance',
]

# All 9 question types
ALL_QUESTION_TYPES = [
    'object_size',
    'object_distance_to_camera',
    'object_size_comparison_relative',
    'object_size_comparison_absolute',
    'object_pair_distance_center',
    'object_pair_distance_center_w_size',
    'object_pair_distance_vector',
    'object_comparison_absolute_distance',
    'object_comparison_relative_distance',
]


def load_questions(questions_file: Path) -> List[Dict]:
    """Load questions from a JSONL file."""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def group_questions_by_id(questions: List[Dict]) -> Dict[str, List[Dict]]:
    """Group questions by question_id (same question, different views)."""
    grouped = defaultdict(list)
    for q in questions:
        grouped[q.get('question_id', '')].append(q)
    return grouped


def select_balanced_questions_v2(
    all_questions: Dict[str, Dict[str, List[Dict]]],  # pattern -> scene -> questions
    questions_per_type_per_scene: int = 10,
    min_views_per_question: int = 1,
    seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Select balanced questions across scenes and question types.
    
    New strategy:
    1. Pool all questions from all patterns (linear_approach, linear_passby, rotation)
    2. For each scene, select equal number of questions per question type
    3. For multi-object types (min_views=1), allow single-view questions
    4. For other types, prefer questions with more views
    
    Args:
        all_questions: Nested dict of pattern -> scene -> list of questions
        questions_per_type_per_scene: Target number of questions per type per scene
        min_views_per_question: Minimum views for non-multi-object questions
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (selected_questions, selection_stats)
    """
    random.seed(seed)
    
    # Step 1: Pool all questions by scene (merge patterns)
    # Structure: scene_id -> question_type -> list of (qid, questions, pattern)
    scene_pool = defaultdict(lambda: defaultdict(list))
    
    for pattern, scenes in all_questions.items():
        for scene_id, questions in scenes.items():
            grouped = group_questions_by_id(questions)
            
            for qid, qs in grouped.items():
                q_type = qs[0].get('question_type', 'unknown')
                num_views = len(qs)
                
                # Check min_views requirement
                # For multi-object types, allow any number of views
                # For other types, require min_views_per_question
                if q_type in MULTI_OBJECT_TYPES:
                    is_valid = True
                else:
                    is_valid = (num_views >= min_views_per_question)
                
                if is_valid:
                    scene_pool[scene_id][q_type].append({
                        'qid': qid,
                        'questions': qs,
                        'pattern': pattern,
                        'num_views': num_views,
                    })
    
    # Step 2: For each scene, select balanced questions per type
    selected = []
    stats = {
        'per_pattern': defaultdict(lambda: {
            'scenes': {},
            'total_unique_qids': 0,
            'total_entries': 0,
            'question_types': Counter(),
        }),
        'per_scene': {},
        'per_question_type': Counter(),
        'total_unique_question_ids': 0,
        'total_entries': 0,
        'views_distribution': Counter(),
    }
    
    print(f"\nSelecting {questions_per_type_per_scene} questions per type per scene...")
    
    for scene_id in sorted(scene_pool.keys()):
        type_pool = scene_pool[scene_id]
        scene_selected = []
        scene_question_types = Counter()
        
        print(f"\n  Scene: {scene_id}")
        
        for q_type in ALL_QUESTION_TYPES:
            candidates = type_pool.get(q_type, [])
            
            if not candidates:
                print(f"    {q_type}: 0 available (skipped)")
                continue
            
            # Sort by num_views descending, then shuffle within same view count
            candidates_sorted = sorted(candidates, key=lambda x: -x['num_views'])
            
            # Select up to questions_per_type_per_scene
            num_to_select = min(questions_per_type_per_scene, len(candidates_sorted))
            selected_candidates = candidates_sorted[:num_to_select]
            
            # Shuffle for variety
            random.shuffle(selected_candidates)
            
            print(f"    {q_type}: {len(candidates)} available, selected {num_to_select}")
            
            for cand in selected_candidates:
                qid = cand['qid']
                pattern = cand['pattern']
                num_views = cand['num_views']
                
                for q in cand['questions']:
                    q_copy = q.copy()
                    q_copy['pattern'] = pattern
                    q_copy['scene_id'] = scene_id
                    scene_selected.append(q_copy)
                
                scene_question_types[q_type] += 1
                stats['per_question_type'][q_type] += 1
                stats['views_distribution'][num_views] += num_views  # count all view entries
                
                # Update pattern stats
                stats['per_pattern'][pattern]['total_unique_qids'] += 1
                stats['per_pattern'][pattern]['total_entries'] += num_views
                stats['per_pattern'][pattern]['question_types'][q_type] += 1
        
        selected.extend(scene_selected)
        
        # Scene stats
        unique_qids = sum(scene_question_types.values())
        total_entries = len(scene_selected)
        
        stats['per_scene'][scene_id] = {
            'unique_question_ids': unique_qids,
            'total_entries': total_entries,
            'question_types': dict(scene_question_types),
            'avg_views': total_entries / unique_qids if unique_qids > 0 else 0,
        }
        
        stats['total_unique_question_ids'] += unique_qids
        stats['total_entries'] += total_entries
        
        print(f"    Total: {unique_qids} unique QIDs, {total_entries} entries")
    
    # Convert defaultdict to regular dict
    stats['per_pattern'] = {
        pattern: {
            'total_unique_qids': p_stats['total_unique_qids'],
            'total_entries': p_stats['total_entries'],
            'question_types': dict(p_stats['question_types']),
            'scenes': {},  # We'll fill this differently
        }
        for pattern, p_stats in stats['per_pattern'].items()
    }
    stats['per_question_type'] = dict(stats['per_question_type'])
    stats['views_distribution'] = dict(stats['views_distribution'])
    
    return selected, stats


# Keep the old function for backward compatibility
def select_balanced_questions(
    all_questions: Dict[str, Dict[str, List[Dict]]],  # pattern -> scene -> questions
    questions_per_scene: int = 100,
    min_views_per_question: int = 3,
    seed: int = 42
) -> Tuple[List[Dict], Dict]:
    """
    Select balanced questions across scenes and question types (legacy version).
    """
    random.seed(seed)
    
    selected = []
    stats = {
        'per_pattern': {},
        'per_scene': {},
        'per_question_type': Counter(),
        'total_unique_question_ids': 0,
        'total_entries': 0,
        'views_distribution': Counter(),
    }
    
    for pattern, scenes in all_questions.items():
        pattern_stats = {
            'scenes': {},
            'total_unique_qids': 0,
            'total_entries': 0,
            'question_types': Counter(),
        }
        
        for scene_id, questions in scenes.items():
            # Group by question_id
            grouped = group_questions_by_id(questions)
            
            # Filter by minimum views (with exception for multi-object types)
            valid_qids = {}
            for qid, qs in grouped.items():
                q_type = qs[0].get('question_type', 'unknown')
                if q_type in MULTI_OBJECT_TYPES:
                    valid_qids[qid] = qs  # Allow any number of views
                elif len(qs) >= min_views_per_question:
                    valid_qids[qid] = qs
            
            if not valid_qids:
                print(f"  Warning: No valid questions in {pattern}/{scene_id}")
                continue
            
            # Group valid questions by type
            type_to_qids = defaultdict(list)
            for qid, qs in valid_qids.items():
                q_type = qs[0].get('question_type', 'unknown')
                type_to_qids[q_type].append(qid)
            
            # Calculate how many questions per type to select
            num_types = len(type_to_qids)
            base_per_type = questions_per_scene // num_types
            remainder = questions_per_scene % num_types
            
            scene_selected_qids = []
            scene_question_types = Counter()
            
            # Sort types by name for reproducibility
            sorted_types = sorted(type_to_qids.keys())
            
            for i, q_type in enumerate(sorted_types):
                qids = type_to_qids[q_type]
                # Distribute remainder to first few types
                num_to_select = base_per_type + (1 if i < remainder else 0)
                num_to_select = min(num_to_select, len(qids))
                
                # Prioritize questions with more views
                qids_with_views = [(qid, len(valid_qids[qid])) for qid in qids]
                qids_with_views.sort(key=lambda x: -x[1])  # Sort by views descending
                
                # Take top ones, but shuffle within same view count for variety
                selected_qids = [qid for qid, _ in qids_with_views[:num_to_select]]
                random.shuffle(selected_qids)
                
                scene_selected_qids.extend(selected_qids)
                scene_question_types[q_type] += len(selected_qids)
            
            # Add all views of selected questions
            scene_entries = []
            for qid in scene_selected_qids:
                for q in valid_qids[qid]:
                    # Add pattern and scene info
                    q_copy = q.copy()
                    q_copy['pattern'] = pattern
                    q_copy['scene_id'] = scene_id
                    scene_entries.append(q_copy)
                    stats['views_distribution'][len(valid_qids[qid])] += 1
            
            selected.extend(scene_entries)
            
            # Update stats
            scene_stats = {
                'unique_question_ids': len(scene_selected_qids),
                'total_entries': len(scene_entries),
                'question_types': dict(scene_question_types),
                'avg_views': len(scene_entries) / len(scene_selected_qids) if scene_selected_qids else 0,
            }
            
            pattern_stats['scenes'][scene_id] = scene_stats
            pattern_stats['total_unique_qids'] += len(scene_selected_qids)
            pattern_stats['total_entries'] += len(scene_entries)
            pattern_stats['question_types'].update(scene_question_types)
            
            stats['per_question_type'].update(scene_question_types)
            stats['total_unique_question_ids'] += len(scene_selected_qids)
            stats['total_entries'] += len(scene_entries)
        
        stats['per_pattern'][pattern] = {
            'total_unique_qids': pattern_stats['total_unique_qids'],
            'total_entries': pattern_stats['total_entries'],
            'question_types': dict(pattern_stats['question_types']),
            'scenes': pattern_stats['scenes'],
        }
    
    stats['per_question_type'] = dict(stats['per_question_type'])
    stats['views_distribution'] = dict(stats['views_distribution'])
    
    return selected, stats


def load_all_questions(dataset_root: Path, patterns: List[str]) -> Dict[str, Dict[str, List[Dict]]]:
    """Load all questions from dataset."""
    all_questions = {}
    
    for pattern in patterns:
        pattern_dir = dataset_root / pattern
        if not pattern_dir.exists():
            continue
        
        all_questions[pattern] = {}
        
        for scene_dir in sorted(pattern_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            
            scene_id = scene_dir.name
            
            # Check nested structure
            nested_dir = scene_dir / scene_id
            if nested_dir.exists() and (nested_dir / "questions.jsonl").exists():
                questions_file = nested_dir / "questions.jsonl"
            elif (scene_dir / "questions.jsonl").exists():
                questions_file = scene_dir / "questions.jsonl"
            else:
                continue
            
            questions = load_questions(questions_file)
            if questions:
                # Add image path prefix for correct referencing
                for q in questions:
                    if 'image' in q:
                        # Update image path to be relative from dataset root
                        q['image_full_path'] = f"{pattern}/{scene_id}/{scene_id}/{q['image']}"
                
                all_questions[pattern][scene_id] = questions
                print(f"  Loaded {len(questions)} questions from {pattern}/{scene_id}")
    
    return all_questions


def generate_analysis_report(stats: Dict, output_file: Path):
    """Generate markdown analysis report."""
    lines = []
    
    lines.append("# Benchmark Dataset Analysis")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    lines.append(f"- **Total Unique Question IDs**: {stats['total_unique_question_ids']}")
    lines.append(f"- **Total Entries (with all views)**: {stats['total_entries']}")
    
    # Handle both new (per_scene) and old (per_pattern) formats
    if 'per_pattern' in stats and stats['per_pattern']:
        lines.append(f"- **Patterns**: {', '.join(stats['per_pattern'].keys())}")
    
    if stats['total_unique_question_ids'] > 0:
        lines.append(f"- **Average Views per Question**: {stats['total_entries'] / stats['total_unique_question_ids']:.2f}")
    lines.append("")
    
    # Question type distribution
    lines.append("## Question Type Distribution")
    lines.append("")
    lines.append("| Question Type | Count | Percentage |")
    lines.append("|--------------|-------|------------|")
    total_types = sum(stats['per_question_type'].values())
    for q_type, count in sorted(stats['per_question_type'].items(), key=lambda x: -x[1]):
        pct = count / total_types * 100 if total_types > 0 else 0
        lines.append(f"| {q_type} | {count} | {pct:.1f}% |")
    lines.append("")
    
    # Views distribution
    if stats.get('views_distribution'):
        lines.append("## Views Distribution")
        lines.append("")
        lines.append("| Number of Views | Entry Count |")
        lines.append("|----------------|-------------|")
        for views, count in sorted(stats['views_distribution'].items()):
            lines.append(f"| {views} | {count} |")
        lines.append("")
    
    # Per-scene breakdown (new format)
    if 'per_scene' in stats and stats['per_scene']:
        lines.append("## Per-Scene Breakdown")
        lines.append("")
        lines.append("| Scene | Unique QIDs | Entries | Avg Views |")
        lines.append("|-------|-------------|---------|-----------|")
        for scene_id, s_stats in sorted(stats['per_scene'].items()):
            avg_views = s_stats.get('avg_views', 0)
            lines.append(f"| {scene_id} | {s_stats['unique_question_ids']} | {s_stats['total_entries']} | {avg_views:.1f} |")
        lines.append("")
        
        # Detailed per-scene question types
        lines.append("## Per-Scene Question Type Distribution")
        lines.append("")
        for scene_id, s_stats in sorted(stats['per_scene'].items()):
            lines.append(f"### {scene_id}")
            lines.append("")
            lines.append(f"- Unique Question IDs: {s_stats['unique_question_ids']}")
            lines.append(f"- Total Entries: {s_stats['total_entries']}")
            lines.append(f"- Average Views: {s_stats.get('avg_views', 0):.1f}")
            lines.append("")
            lines.append("| Question Type | Count |")
            lines.append("|--------------|-------|")
            for q_type, count in sorted(s_stats['question_types'].items(), key=lambda x: -x[1]):
                lines.append(f"| {q_type} | {count} |")
            lines.append("")
    
    # Per-pattern breakdown (old format, for backward compatibility)
    if 'per_pattern' in stats and stats['per_pattern']:
        has_scenes = any(p_stats.get('scenes') for p_stats in stats['per_pattern'].values())
        if has_scenes:
            lines.append("## Per-Pattern Breakdown")
            lines.append("")
            
            for pattern, p_stats in stats['per_pattern'].items():
                lines.append(f"### {pattern.upper()}")
                lines.append("")
                lines.append(f"- Unique Question IDs: {p_stats['total_unique_qids']}")
                lines.append(f"- Total Entries: {p_stats['total_entries']}")
                lines.append("")
                
                if p_stats.get('scenes'):
                    lines.append("| Scene | Unique QIDs | Entries | Avg Views |")
                    lines.append("|-------|-------------|---------|-----------|")
                    for scene_id, s_stats in sorted(p_stats['scenes'].items()):
                        lines.append(f"| {scene_id} | {s_stats['unique_question_ids']} | {s_stats['total_entries']} | {s_stats['avg_views']:.1f} |")
                    lines.append("")
                
                lines.append("**Question Types:**")
                lines.append("")
                for q_type, count in sorted(p_stats['question_types'].items(), key=lambda x: -x[1]):
                    lines.append(f"- {q_type}: {count}")
                lines.append("")
    
    # Selection criteria
    lines.append("## Selection Criteria")
    lines.append("")
    lines.append("The benchmark was created with the following goals:")
    lines.append("")
    lines.append("1. **Balanced across scenes**: Each scene contributes equal number of questions per type")
    lines.append("2. **Balanced question types**: All 9 question types are equally represented")
    lines.append("3. **Patterns**: Only uses linear_approach, linear_passby, rotation (excludes around/spherical)")
    lines.append("4. **Multi-object support**: Allows single-view questions for multi-object comparison types")
    lines.append("")
    
    report = "\n".join(lines)
    
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Analysis report saved to: {output_file}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Create balanced benchmark from VQA dataset")
    parser.add_argument("--dataset_root", type=str,
                        default="/scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_all_type",
                        help="Root directory of the dataset")
    parser.add_argument("--output_jsonl", type=str, default=None,
                        help="Output JSONL file for benchmark")
    parser.add_argument("--output_analysis", type=str, default=None,
                        help="Output markdown file for analysis")
    parser.add_argument("--questions_per_type", type=int, default=10,
                        help="Target number of questions per type per scene (for balanced mode)")
    parser.add_argument("--questions_per_scene", type=int, default=50,
                        help="Target number of unique question IDs per scene per pattern (legacy mode)")
    parser.add_argument("--min_views", type=int, default=1,
                        help="Minimum number of views required per question (except multi-object types)")
    parser.add_argument("--patterns", type=str, nargs='+',
                        default=["linear_approach", "linear_passby", "rotation"],
                        help="Patterns to include (default: only linear and rotation)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--balanced", action="store_true", default=True,
                        help="Use balanced sampling (equal per type per scene)")
    parser.add_argument("--legacy", action="store_true", default=False,
                        help="Use legacy sampling mode")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset_root)
    
    # Default output files
    if args.output_jsonl is None:
        args.output_jsonl = dataset_root / "benchmark.jsonl"
    else:
        args.output_jsonl = Path(args.output_jsonl)
    
    if args.output_analysis is None:
        args.output_analysis = dataset_root / "benchmark_analysis.md"
    else:
        args.output_analysis = Path(args.output_analysis)
    
    print("=" * 60)
    print("Creating Balanced Benchmark Dataset")
    print("=" * 60)
    print(f"Dataset root: {dataset_root}")
    print(f"Patterns: {args.patterns}")
    print(f"Questions per type per scene: {args.questions_per_type}")
    print(f"Minimum views: {args.min_views}")
    print(f"Random seed: {args.seed}")
    print(f"Balanced mode: {not args.legacy}")
    print()
    
    # Load all questions
    print("Loading questions...")
    all_questions = load_all_questions(dataset_root, args.patterns)
    print()
    
    # Select balanced questions
    print("Selecting balanced benchmark...")
    if args.legacy:
        selected, stats = select_balanced_questions(
            all_questions,
            questions_per_scene=args.questions_per_scene,
            min_views_per_question=args.min_views,
            seed=args.seed
        )
    else:
        selected, stats = select_balanced_questions_v2(
            all_questions,
            questions_per_type_per_scene=args.questions_per_type,
            min_views_per_question=args.min_views,
            seed=args.seed
        )
    print()
    
    # Save benchmark
    print(f"Saving benchmark to {args.output_jsonl}...")
    with open(args.output_jsonl, 'w') as f:
        for q in selected:
            f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    # Generate analysis report
    print(f"Generating analysis report...")
    generate_analysis_report(stats, args.output_analysis)
    
    # Print summary
    print()
    print("=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Total unique question IDs: {stats['total_unique_question_ids']}")
    print(f"Total entries (with all views): {stats['total_entries']}")
    if stats['total_unique_question_ids'] > 0:
        print(f"Average views per question: {stats['total_entries'] / stats['total_unique_question_ids']:.2f}")
    print()
    print("Per-scene breakdown:")
    for scene_id, s_stats in sorted(stats.get('per_scene', {}).items()):
        print(f"  {scene_id}: {s_stats['unique_question_ids']} unique QIDs, {s_stats['total_entries']} entries")
    print()
    print("Question type distribution:")
    for q_type, count in sorted(stats['per_question_type'].items(), key=lambda x: -x[1]):
        print(f"  {q_type}: {count}")
    print()
    print(f"Benchmark saved to: {args.output_jsonl}")
    print(f"Analysis saved to: {args.output_analysis}")


if __name__ == "__main__":
    main()
