#!/usr/bin/env python3
"""
InteriorGS Question Generation Script

This script runs the complete pipeline to generate spatial reasoning questions
from InteriorGS dataset.

Usage:
    python run_pipeline.py \\
        --scenes_root /path/to/InteriorGS \\
        --output_dir /path/to/output

Example with specific scenes:
    python run_pipeline.py \\
        --scenes_root /path/to/InteriorGS \\
        --output_dir /path/to/output \\
        --scenes 0267_840790 0002_839955

Example with single scene:
    python run_pipeline.py \\
        --scenes_root /path/to/InteriorGS \\
        --output_dir /path/to/output \\
        --scene_id 0267_840790
"""

import argparse
import json
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from config import PipelineConfig, ObjectSelectionConfig, CameraSamplingConfig, QuestionConfig, RenderConfig
from pipeline import InteriorGSQuestionPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Spatial Reasoning Questions from InteriorGS Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Required arguments
    parser.add_argument('--scenes_root', required=True, type=str,
                        help='Root directory containing InteriorGS scene folders with labels.json')
    parser.add_argument('--output_dir', required=True, type=str,
                        help='Output directory for generated questions')
    
    # Scene selection
    parser.add_argument('--scenes', nargs='+', type=str, default=None,
                        help='Specific scene names to process (default: all scenes)')
    parser.add_argument('--scene_id', type=str, default=None,
                        help='Process only this single scene by ID (e.g., 0267_840790)')
    
    # Camera sampling
    parser.add_argument('--num_cameras', type=int, default=3,
                        help='Number of camera poses to sample per scene')
    parser.add_argument('--move_pattern', type=str, default='around', choices=['around', 'spherical', 'linear'],
                        help='Camera sampling pattern: around (horizontal circle), spherical (full sphere), or linear (straight line movement)')
    parser.add_argument('--linear_sub_pattern', type=str, default='approach',
                        choices=['approach', 'pass_by'],
                        help='Sub-pattern for linear movement: approach (walk toward object, fixed orientation), '
                             'pass_by (walk past object, fixed orientation)')
    parser.add_argument('--linear_num_steps', type=int, default=5,
                        help='Number of poses along the linear trajectory')
    parser.add_argument('--linear_move_distance', type=float, default=0.3,
                        help='Total movement distance along trajectory in meters (default: 0.3m)')
    parser.add_argument('--experiment_name', type=str, default='default',
                        help='Experiment name for output directory structure. '
                             'Images saved as: {output_dir}/{experiment_name}/{move_pattern}/{object_name}_{view_idx}.png')
    
    # Question generation
    parser.add_argument('--question_types', nargs='+', type=str, default=None,
                        help='Question types to generate (default: all). Options: '
                             'object_size, object_distance_to_camera, '
                             'object_size_comparison_relative, object_size_comparison_absolute, '
                             'object_pair_distance_center, object_pair_distance_center_w_size, '
                             'object_pair_distance_vector, '
                             'object_comparison_absolute_distance, object_comparison_relative_distance')
    
    # Limits
    parser.add_argument('--max_questions_per_scene', type=int, default=1000,
                        help='Maximum questions to generate per scene')
    parser.add_argument('--max_questions_per_type', type=int, default=5,
                        help='Maximum questions per type per camera pose')
    parser.add_argument('--max_tries', type=int, default=300,
                        help='Maximum camera sampling attempts per object/pair (default: 300)')
    parser.add_argument('--min_views_required', type=int, default=0,
                        help='Minimum number of views required per question. '
                             'Questions with fewer views will be filtered out. '
                             '0 = no filtering (default: 0)')

    # Processing options
    parser.add_argument('--no_intermediate', action='store_true',
                        help='Do not save intermediate per-scene results')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Print verbose output')
    parser.add_argument('--flat_output', action='store_true', default=False,
                        help='Output directly to output_dir without creating scene subdirectory')
    
    # Rendering options
    parser.add_argument('--enable_rendering', action='store_true', default=False,
                        help='Enable image rendering for each camera pose')
    parser.add_argument('--render_backend', type=str, default='local', choices=['local', 'client'],
                        help='Rendering backend: local (GPU) or client (WebSocket)')
    parser.add_argument('--render_width', type=int, default=640,
                        help='Rendered image width')
    parser.add_argument('--render_height', type=int, default=480,
                        help='Rendered image height')
    parser.add_argument('--render_fov', type=float, default=60.0,
                        help='Field of view for rendering (degrees)')
    parser.add_argument('--gpu_device', type=int, default=None,
                        help='GPU device ID for rendering (default: auto)')
    
    # Config file
    parser.add_argument('--config', type=str, default=None,
                        help='Load configuration from JSON file')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Handle scene_id as a shortcut for --scenes with single scene
    scene_id = args.scene_id
    scenes_list = args.scenes
    if scene_id:
        scenes_list = [scene_id]
    
    # Load config from file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = PipelineConfig.from_dict(config_dict)
    else:
        # Build config from arguments
        object_config = ObjectSelectionConfig()
        
        camera_config = CameraSamplingConfig(
            num_cameras_per_item=args.num_cameras,
            move_pattern=args.move_pattern,
            linear_sub_pattern=args.linear_sub_pattern,
            linear_num_steps=args.linear_num_steps,
            linear_move_distance=args.linear_move_distance,
            max_tries=args.max_tries
        )
        
        question_config = QuestionConfig(
            max_questions_per_type=args.max_questions_per_type
        )
        
        if args.question_types:
            question_config.enabled_question_types = args.question_types
        
        # Render config
        render_config = RenderConfig(
            enable_rendering=args.enable_rendering,
            render_backend=args.render_backend,
            image_width=args.render_width,
            image_height=args.render_height,
            fov_deg=args.render_fov,
            gpu_device=args.gpu_device,
        )
        
        config = PipelineConfig(
            scenes_root=args.scenes_root,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            scene_list=scenes_list,
            object_selection=object_config,
            camera_sampling=camera_config,
            question_config=question_config,
            render_config=render_config,
            save_intermediate=not args.no_intermediate,
            max_questions_per_scene=args.max_questions_per_scene,
            min_views_required=args.min_views_required,
        )
    
    # Override paths from command line if provided
    if args.scenes_root:
        config.scenes_root = args.scenes_root
    if args.output_dir:
        config.output_dir = args.output_dir
    if scenes_list:
        config.scene_list = scenes_list
    
    # Create and run pipeline
    pipeline = InteriorGSQuestionPipeline(config)
    
    if scene_id:
        results = pipeline.run_single_scene(scene_id, verbose=args.verbose, flat_output=args.flat_output)
    else:
        results = pipeline.run(verbose=args.verbose)
    
    return results


if __name__ == '__main__':
    main()
