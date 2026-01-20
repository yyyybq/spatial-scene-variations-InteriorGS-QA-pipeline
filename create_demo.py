#!/usr/bin/env python3
"""
Move Pattern Demo Generator

A universal demo visualization script for all 4 camera movement patterns:
- around: Horizontal circle around object
- spherical: Spherical sampling around object  
- rotation: Room center 360° rotation
- linear: Linear trajectory toward/past object

This script creates a demo visualization showing:
- Multiple sequential images from different camera poses
- Corresponding QA pairs for each image
- Pattern-specific information (distance, angle, etc.)
- All combined into a single visualization image

Usage:
    # For object-centric patterns (around, spherical, linear)
    python create_pattern_demo.py \\
        --questions_file /path/to/questions.jsonl \\
        --output_file /path/to/demo.png \\
        --pattern linear \\
        --object_label "sofa" \\
        --num_views 5

    # For room-centric pattern (rotation)
    python create_pattern_demo.py \\
        --questions_file /path/to/questions.jsonl \\
        --output_file /path/to/demo.png \\
        --pattern rotation \\
        --room_name "bedroom" \\
        --num_views 8
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Please install required packages: pip install numpy pillow")
    sys.exit(1)


# =============================================================================
# Pattern Configuration
# =============================================================================

PATTERN_CONFIG = {
    'around': {
        'title': 'Horizontal Circle Pattern',
        'description': 'Camera circles around object on horizontal plane',
        'icon': '🔄',
        'view_label': 'Angle',
        'metric': 'yaw',  # Use yaw angle as the varying metric
        'metric_format': '{:.0f}°',
    },
    'spherical': {
        'title': 'Spherical Sampling Pattern',
        'description': 'Camera samples on sphere surface around object',
        'icon': '🌐',
        'view_label': 'View',
        'metric': 'elevation',  # Varies both yaw and pitch
        'metric_format': 'θ={:.0f}°',
    },
    'rotation': {
        'title': 'Room Rotation Pattern',
        'description': 'Camera at room center, rotating 360°',
        'icon': '🏠',
        'view_label': 'Yaw',
        'metric': 'yaw',
        'metric_format': '{:.0f}°',
    },
    'linear': {
        'title': 'Linear Trajectory Pattern',
        'description': 'Camera moves in straight line toward object',
        'icon': '➡️',
        'view_label': 'Step',
        'metric': 'radius',  # Distance to object
        'metric_format': 'd={:.1f}m',
    },
}


# =============================================================================
# Data Loading and Filtering
# =============================================================================

def load_questions(questions_file: str) -> List[Dict[str, Any]]:
    """Load questions from JSONL file."""
    questions = []
    with open(questions_file, 'r') as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))
    return questions


def load_metadata(questions_file: str) -> Dict[str, Any]:
    """Load metadata.json from the same directory as questions file."""
    metadata_path = Path(questions_file).parent / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return {}


def get_available_objects(questions: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Get list of available objects with their question counts."""
    labels = []
    for q in questions:
        if 'objects' in q and len(q['objects']) > 0:
            labels.append(q['objects'][0].get('label', ''))
    
    return Counter(labels).most_common()


def get_available_rooms(questions: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Get list of available rooms (for rotation pattern)."""
    rooms = []
    for q in questions:
        camera_pose = q.get('camera_pose', {})
        room = camera_pose.get('room_name') or camera_pose.get('target_room')
        if room:
            rooms.append(room)
    
    return Counter(rooms).most_common()


def filter_questions_for_object(
    questions: List[Dict[str, Any]], 
    object_label: str,
    question_type: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Filter questions for a specific object and optionally by question type."""
    filtered = []
    for q in questions:
        primary_label = None
        if 'objects' in q and len(q['objects']) > 0:
            primary_label = q['objects'][0].get('label', '').lower()
        
        if object_label.lower() in (primary_label or ''):
            if question_type is None or q.get('question_type') == question_type:
                filtered.append(q)
    
    return filtered


def filter_questions_for_room(
    questions: List[Dict[str, Any]],
    room_name: str
) -> List[Dict[str, Any]]:
    """Filter questions for a specific room (for rotation pattern)."""
    filtered = []
    for q in questions:
        camera_pose = q.get('camera_pose', {})
        room = camera_pose.get('room_name') or camera_pose.get('target_room', '')
        if room_name.lower() in room.lower():
            filtered.append(q)
    
    return filtered


def group_questions_by_camera_pose(questions: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    """Group questions by camera pose index."""
    grouped = {}
    for q in questions:
        pose_idx = q.get('camera_pose_idx', 0)
        if pose_idx not in grouped:
            grouped[pose_idx] = []
        grouped[pose_idx].append(q)
    return grouped


def get_metric_value(question: Dict[str, Any], metric: str) -> float:
    """Extract metric value from question's camera pose."""
    camera_pose = question.get('camera_pose', {})
    
    if metric == 'yaw':
        return camera_pose.get('yaw', 0)
    elif metric == 'pitch':
        return camera_pose.get('pitch', 0)
    elif metric == 'radius':
        return camera_pose.get('radius', 0)
    elif metric == 'elevation':
        # Combine yaw and pitch for spherical
        return camera_pose.get('pitch', 0)
    else:
        return 0


def select_views_for_pattern(
    questions: List[Dict[str, Any]],
    pattern: str,
    num_views: int = 5
) -> List[Dict[str, Any]]:
    """
    Select representative views based on pattern type.
    
    For each pattern, we want to show the progression:
    - around: Different yaw angles around the object
    - spherical: Different elevation/azimuth combinations
    - rotation: Different yaw angles (0°, 45°, 90°, ...)
    - linear: Different distances (far to near)
    """
    if not questions:
        return []
    
    # Group by camera pose
    grouped = group_questions_by_camera_pose(questions)
    pose_indices = sorted(grouped.keys())
    
    # Select evenly spaced poses
    if len(pose_indices) <= num_views:
        selected_indices = pose_indices
    else:
        step = len(pose_indices) / num_views
        selected_indices = [pose_indices[int(i * step)] for i in range(num_views)]
    
    # Get one question per selected pose
    selected = []
    for idx in selected_indices:
        if idx in grouped and grouped[idx]:
            selected.append(grouped[idx][0])
    
    # Sort by metric for better visualization
    config = PATTERN_CONFIG.get(pattern, PATTERN_CONFIG['around'])
    metric = config['metric']
    
    if pattern == 'linear':
        # Sort by distance (far to near)
        selected.sort(key=lambda q: get_metric_value(q, metric), reverse=True)
    elif pattern == 'rotation':
        # Sort by yaw angle (0° to 360°)
        selected.sort(key=lambda q: get_metric_value(q, metric))
    else:
        # Sort by pose index
        selected.sort(key=lambda q: q.get('camera_pose_idx', 0))
    
    return selected


# =============================================================================
# Image Handling
# =============================================================================

def get_image_path_from_question(question: Dict[str, Any], output_root: str) -> Optional[str]:
    """Extract full image path from question."""
    rel_path = question.get('image')
    if rel_path:
        full_path = os.path.join(output_root, rel_path)
        if os.path.exists(full_path):
            return full_path
    return None


def find_images_in_directory(
    images_dir: str,
    object_label: Optional[str] = None,
    room_name: Optional[str] = None
) -> Dict[int, str]:
    """Find images in directory, optionally filtered by object or room."""
    image_paths = {}
    
    if not os.path.exists(images_dir):
        return image_paths
    
    for f in os.listdir(images_dir):
        if not f.endswith('.png'):
            continue
        
        # Match based on object label
        if object_label:
            obj_prefix = object_label.lower().replace(' ', '_')
            if f.lower().startswith(obj_prefix):
                try:
                    # Format: {object}_{idx}.png
                    parts = f.replace('.png', '').split('_')
                    idx = int(parts[-1])
                    image_paths[idx] = os.path.join(images_dir, f)
                except:
                    pass
        
        # Match based on room name (for rotation pattern)
        elif room_name:
            room_prefix = room_name.lower().replace(' ', '_')
            if f.lower().startswith(room_prefix):
                try:
                    # Format: {room}_{yaw}deg.png
                    if 'deg' in f:
                        yaw = int(f.split('_')[-1].replace('deg.png', ''))
                        image_paths[yaw] = os.path.join(images_dir, f)
                except:
                    pass
        else:
            # No filter, try to extract index
            try:
                parts = f.replace('.png', '').split('_')
                idx = int(parts[-1])
                image_paths[idx] = os.path.join(images_dir, f)
            except:
                pass
    
    return image_paths


def create_placeholder_image(width: int, height: int, text: str = "No Image") -> Image.Image:
    """Create a placeholder image with text."""
    img = Image.new('RGB', (width, height), color=(220, 220, 220))
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    # Draw text centered
    lines = text.split('\n')
    total_height = len(lines) * 20
    y = (height - total_height) // 2
    
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        text_width = bbox[2] - bbox[0]
        x = (width - text_width) // 2
        draw.text((x, y), line, fill=(120, 120, 120), font=font)
        y += 20
    
    return img


# =============================================================================
# Visualization
# =============================================================================

def create_pattern_demo(
    questions: List[Dict[str, Any]],
    pattern: str,
    output_file: str,
    output_root: str = None,
    image_paths: Dict[int, str] = None,
    title: Optional[str] = None,
    image_size: Tuple[int, int] = (320, 240),
    max_columns: int = 5,
    show_qa_types: int = 3
):
    """
    Create a demo visualization for any move pattern.
    
    Args:
        questions: List of questions (already filtered for object/room)
        pattern: Move pattern type ('around', 'spherical', 'rotation', 'linear')
        output_file: Path to save the demo image
        output_root: Root directory for resolving image paths
        image_paths: Optional pre-computed image paths
        title: Custom title (auto-generated if None)
        image_size: Size of each view image
        max_columns: Maximum columns in the grid
        show_qa_types: Number of QA types to show per view
    """
    if not questions:
        print("No questions to visualize!")
        return
    
    config = PATTERN_CONFIG.get(pattern, PATTERN_CONFIG['around'])
    
    # Group questions by pose
    pose_to_questions = {}
    for q in questions:
        pose_idx = q.get('camera_pose_idx', 0)
        if pose_idx not in pose_to_questions:
            pose_to_questions[pose_idx] = []
        pose_to_questions[pose_idx].append(q)
    
    # Get unique poses in order
    pose_indices = sorted(pose_to_questions.keys())
    num_views = min(len(pose_indices), max_columns * 2)  # Max 2 rows
    pose_indices = pose_indices[:num_views]
    
    # Get all question types
    all_q_types = set()
    for qs in pose_to_questions.values():
        for q in qs:
            all_q_types.add(q.get('question_type', 'unknown'))
    q_types = sorted(list(all_q_types))[:show_qa_types]
    
    # Layout calculations
    img_w, img_h = image_size
    padding = 15
    row_height = 22
    
    columns = min(num_views, max_columns)
    rows = (num_views + columns - 1) // columns
    
    cell_width = img_w + padding * 2
    qa_section_height = len(q_types) * row_height + 10
    cell_height = img_h + qa_section_height + 40  # Extra for header
    
    title_height = 90
    legend_height = 35
    total_width = columns * cell_width + padding * 2
    total_height = title_height + rows * cell_height + legend_height + padding
    
    # Create canvas
    canvas = Image.new('RGB', (total_width, total_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    
    # Load fonts
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
        header_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 10)
    except:
        title_font = header_font = subtitle_font = text_font = small_font = ImageFont.load_default()
    
    # Draw title
    if title is None:
        # Get object/room info
        if questions and 'objects' in questions[0] and questions[0]['objects']:
            target = questions[0]['objects'][0].get('label', 'Unknown')
        else:
            target = 'Scene'
        title = f"{config['icon']} {config['title']}"
    
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_x = (total_width - (title_bbox[2] - title_bbox[0])) // 2
    draw.text((title_x, 12), title, fill=(30, 30, 30), font=title_font)
    
    # Draw subtitle with pattern description
    subtitle = config['description']
    if questions and 'objects' in questions[0] and questions[0]['objects']:
        target = questions[0]['objects'][0].get('label', 'Unknown')
        subtitle = f"Target: {target} | {subtitle}"
    
    sub_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
    sub_x = (total_width - (sub_bbox[2] - sub_bbox[0])) // 2
    draw.text((sub_x, 42), subtitle, fill=(80, 80, 80), font=subtitle_font)
    
    # Draw view count and pattern info
    info_text = f"Pattern: {pattern} | Views: {num_views} | Question Types: {len(all_q_types)}"
    info_bbox = draw.textbbox((0, 0), info_text, font=small_font)
    info_x = (total_width - (info_bbox[2] - info_bbox[0])) // 2
    draw.text((info_x, 62), info_text, fill=(120, 120, 120), font=small_font)
    
    # Prepare image paths if not provided
    if image_paths is None:
        image_paths = {}
    
    # Draw each view
    for view_idx, pose_idx in enumerate(pose_indices):
        row = view_idx // columns
        col = view_idx % columns
        
        cell_x = padding + col * cell_width
        cell_y = title_height + row * cell_height
        
        qs_for_pose = pose_to_questions.get(pose_idx, [])
        
        # Draw view header
        metric_value = get_metric_value(qs_for_pose[0], config['metric']) if qs_for_pose else 0
        metric_str = config['metric_format'].format(metric_value)
        header_text = f"{config['view_label']} {view_idx + 1}: {metric_str}"
        draw.text((cell_x + padding, cell_y), header_text, fill=(0, 80, 160), font=header_font)
        
        # Draw image
        img_y = cell_y + 20
        img = None
        
        # Try to get image from question
        if qs_for_pose and output_root:
            img_path = get_image_path_from_question(qs_for_pose[0], output_root)
            if img_path and os.path.exists(img_path):
                img = Image.open(img_path)
        
        # Try using view_idx in image_paths
        if img is None and view_idx in image_paths and os.path.exists(str(image_paths.get(view_idx, ''))):
            img = Image.open(image_paths[view_idx])
        
        # Try using pose_idx in image_paths
        if img is None and pose_idx in image_paths and os.path.exists(str(image_paths.get(pose_idx, ''))):
            img = Image.open(image_paths[pose_idx])
        
        # Fallback to placeholder
        if img is None:
            placeholder_text = f"{config['view_label']} {view_idx + 1}\n{metric_str}"
            img = create_placeholder_image(img_w, img_h, placeholder_text)
        else:
            img = img.resize(image_size, Image.Resampling.LANCZOS)
        
        canvas.paste(img, (cell_x + padding, img_y))
        
        # Draw image border
        draw.rectangle(
            [cell_x + padding - 1, img_y - 1,
             cell_x + padding + img_w, img_y + img_h],
            outline=(200, 200, 200), width=1
        )
        
        # Draw QA section
        qa_y = img_y + img_h + 5
        
        for q_idx, q_type in enumerate(q_types):
            answer = "N/A"
            for q in qs_for_pose:
                if q.get('question_type') == q_type:
                    answer = str(q.get('answer', 'N/A'))
                    # Truncate long answers
                    if len(answer) > 25:
                        answer = answer[:22] + "..."
                    break
            
            # Shorten question type for display
            short_type = q_type.replace('object_', '').replace('_', ' ').title()
            if len(short_type) > 15:
                short_type = short_type[:12] + "..."
            
            text = f"{short_type}: {answer}"
            y_pos = qa_y + q_idx * row_height
            draw.text((cell_x + padding, y_pos), text, fill=(60, 60, 60), font=text_font)
    
    # Draw legend at bottom
    legend_y = total_height - legend_height
    
    # Pattern legend
    legend_items = [
        f"Pattern: {pattern.upper()}",
        f"Metric: {config['metric']}",
    ]
    
    # Question types legend
    q_type_str = " | ".join([t.replace('object_', '').replace('_', ' ').title()[:15] for t in q_types])
    legend_items.append(f"QA Types: {q_type_str}")
    
    legend_text = " | ".join(legend_items)
    draw.text((padding, legend_y + 5), legend_text, fill=(100, 100, 100), font=small_font)
    
    # Save
    canvas.save(output_file, quality=95)
    print(f"\nDemo saved to: {output_file}")
    print(f"  Pattern: {pattern}")
    print(f"  Views: {num_views}")
    print(f"  QA Types: {len(all_q_types)}")
    
    return canvas


def create_pattern_comparison_demo(
    questions: List[Dict[str, Any]],
    object_label: str,
    output_file: str,
    output_root: str = None,
    patterns: List[str] = None
):
    """
    Create a comparison demo showing the same object across different patterns.
    (Placeholder for future implementation)
    """
    # TODO: Implement multi-pattern comparison
    pass


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Create Demo Visualization for Any Move Pattern',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Linear pattern demo for 'bed' object
  python create_pattern_demo.py \\
      --questions_file output/0267_840790/questions.jsonl \\
      --output_file demo_linear_bed.png \\
      --pattern linear \\
      --object_label bed

  # Around pattern demo for 'sofa' object
  python create_pattern_demo.py \\
      --questions_file output/0267_840790/questions.jsonl \\
      --output_file demo_around_sofa.png \\
      --pattern around \\
      --object_label sofa

  # Rotation pattern demo for 'bedroom'
  python create_pattern_demo.py \\
      --questions_file output/0267_840790/questions.jsonl \\
      --output_file demo_rotation_bedroom.png \\
      --pattern rotation \\
      --room_name bedroom

  # List available objects
  python create_pattern_demo.py \\
      --questions_file output/0267_840790/questions.jsonl \\
      --list_objects
        """
    )
    
    parser.add_argument('--questions_file', type=str, required=True,
                        help='Path to questions.jsonl file')
    parser.add_argument('--output_file', type=str, default='pattern_demo.png',
                        help='Output demo image path')
    parser.add_argument('--output_root', type=str, default=None,
                        help='Root directory for resolving image paths')
    parser.add_argument('--images_dir', type=str, default=None,
                        help='Directory containing rendered images')
    
    # Pattern selection
    parser.add_argument('--pattern', type=str, default='linear',
                        choices=['around', 'spherical', 'rotation', 'linear'],
                        help='Move pattern type')
    
    # Object/room selection
    parser.add_argument('--object_label', type=str, default=None,
                        help='Filter by object label (for around/spherical/linear)')
    parser.add_argument('--room_name', type=str, default=None,
                        help='Filter by room name (for rotation pattern)')
    
    # Visualization options
    parser.add_argument('--num_views', type=int, default=5,
                        help='Number of views to show in demo')
    parser.add_argument('--image_width', type=int, default=320,
                        help='Width of each view image')
    parser.add_argument('--image_height', type=int, default=240,
                        help='Height of each view image')
    parser.add_argument('--max_columns', type=int, default=5,
                        help='Maximum columns in the grid')
    parser.add_argument('--qa_types', type=int, default=3,
                        help='Number of QA types to show per view')
    
    # Info options
    parser.add_argument('--list_objects', action='store_true',
                        help='List available objects and exit')
    parser.add_argument('--list_rooms', action='store_true',
                        help='List available rooms and exit')
    
    args = parser.parse_args()
    
    # Load questions
    print(f"Loading questions from: {args.questions_file}")
    questions = load_questions(args.questions_file)
    print(f"  Loaded {len(questions)} questions")
    
    # Handle list options
    if args.list_objects:
        objects = get_available_objects(questions)
        print("\nAvailable objects:")
        for label, count in objects[:30]:
            print(f"  {label}: {count} questions")
        return
    
    if args.list_rooms:
        rooms = get_available_rooms(questions)
        if rooms:
            print("\nAvailable rooms:")
            for room, count in rooms[:20]:
                print(f"  {room}: {count} questions")
        else:
            print("\nNo room information found in questions.")
            print("Room filtering is only available for rotation pattern data.")
        return
    
    # Determine output root
    if args.output_root:
        output_root = args.output_root
        # Ensure output directory exists
        os.makedirs(output_root, exist_ok=True)
    else:
        output_root = str(Path(args.questions_file).parent.parent)
    
    # Filter questions based on pattern
    if args.pattern == 'rotation' and args.room_name:
        filtered = filter_questions_for_room(questions, args.room_name)
        filter_desc = f"room '{args.room_name}'"
    elif args.object_label:
        filtered = filter_questions_for_object(questions, args.object_label)
        filter_desc = f"object '{args.object_label}'"
    else:
        # Auto-select the most common object
        objects = get_available_objects(questions)
        if objects:
            args.object_label = objects[0][0]
            filtered = filter_questions_for_object(questions, args.object_label)
            filter_desc = f"object '{args.object_label}' (auto-selected)"
        else:
            filtered = questions
            filter_desc = "all questions"
    
    print(f"  Filtered to {len(filtered)} questions for {filter_desc}")
    
    if not filtered:
        print("\nNo matching questions found!")
        print("\nAvailable objects:")
        for label, count in get_available_objects(questions)[:15]:
            print(f"  {label}: {count}")
        return
    
    # If output file is default, build descriptive filename (after filtering so we know object/room)
    if args.output_file == 'pattern_demo.png':
        name_parts = [args.pattern]
        if args.object_label:
            safe_label = args.object_label.replace(' ', '_').replace('/', '_')
            name_parts.append(safe_label)
        elif args.room_name:
            safe_room = args.room_name.replace(' ', '_').replace('/', '_')
            name_parts.append(safe_room)
        
        demo_filename = f"demo_{'_'.join(name_parts)}.png"
        if args.output_root:
            args.output_file = os.path.join(args.output_root, demo_filename)
        else:
            args.output_file = demo_filename
    
    # Select views for the pattern
    selected = select_views_for_pattern(filtered, args.pattern, args.num_views)
    print(f"  Selected {len(selected)} views for demo")
    
    # Find images
    image_paths = {}
    if args.images_dir and os.path.exists(args.images_dir):
        image_paths = find_images_in_directory(
            args.images_dir,
            object_label=args.object_label,
            room_name=args.room_name
        )
        print(f"  Found {len(image_paths)} images in directory")
    
    # Create demo
    create_pattern_demo(
        filtered,
        args.pattern,
        args.output_file,
        output_root=output_root,
        image_paths=image_paths,
        image_size=(args.image_width, args.image_height),
        max_columns=args.max_columns,
        show_qa_types=args.qa_types
    )


if __name__ == '__main__':
    main()
