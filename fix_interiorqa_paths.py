#!/usr/bin/env python3
"""
Fix InteriorQA image paths.

This script fixes the redundant path structure in existing InteriorQA data:
- Old structure: InteriorQA/{scene_id}/{pattern}/default/{scene_id}/{pattern}/*.png
- New structure: InteriorQA/{scene_id}/{pattern}/images/*.png

It also updates the questions.jsonl files to use the new relative paths.

Usage:
    python fix_interiorqa_paths.py --data_dir /path/to/InteriorQA
    
    # Dry run (show what would be done without making changes)
    python fix_interiorqa_paths.py --data_dir /path/to/InteriorQA --dry_run
"""

import argparse
import json
import os
import shutil
from pathlib import Path


def find_pattern_dirs(data_dir: Path):
    """Find all pattern directories (around, spherical, linear_approach, linear_pass_by)."""
    patterns = []
    
    for scene_dir in data_dir.iterdir():
        if not scene_dir.is_dir():
            continue
        
        for pattern_dir in scene_dir.iterdir():
            if not pattern_dir.is_dir():
                continue
            
            # Check if this is a pattern directory (has questions.jsonl)
            if (pattern_dir / "questions.jsonl").exists():
                patterns.append(pattern_dir)
    
    return patterns


def get_old_image_dir(pattern_dir: Path):
    """Get the old (redundant) image directory path."""
    # The old structure has: pattern_dir/default/{scene_id}/{move_pattern}/
    default_dir = pattern_dir / "default"
    
    if not default_dir.exists():
        return None
    
    # Find the nested structure
    for scene_subdir in default_dir.iterdir():
        if scene_subdir.is_dir():
            for pattern_subdir in scene_subdir.iterdir():
                if pattern_subdir.is_dir():
                    # Check if it contains images
                    images = list(pattern_subdir.glob("*.png"))
                    if images:
                        return pattern_subdir
    
    return None


def fix_pattern_dir(pattern_dir: Path, dry_run: bool = False):
    """Fix the image paths for a single pattern directory."""
    print(f"\nProcessing: {pattern_dir}")
    
    # Find old image directory
    old_image_dir = get_old_image_dir(pattern_dir)
    
    if old_image_dir is None:
        print(f"  No images found in old structure, skipping")
        return False
    
    print(f"  Old image dir: {old_image_dir}")
    
    # Create new images directory
    new_image_dir = pattern_dir / "images"
    
    # Count images
    images = list(old_image_dir.glob("*.png"))
    print(f"  Found {len(images)} images to move")
    
    if not dry_run:
        new_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Move images
        for img in images:
            new_path = new_image_dir / img.name
            if not new_path.exists():
                shutil.move(str(img), str(new_path))
            else:
                # If already exists, just remove the old one
                img.unlink()
        
        # Remove empty directories
        try:
            # Remove the nested structure
            default_dir = pattern_dir / "default"
            if default_dir.exists():
                shutil.rmtree(str(default_dir))
        except Exception as e:
            print(f"  Warning: Could not remove old directories: {e}")
    else:
        print(f"  Would create: {new_image_dir}")
        print(f"  Would move {len(images)} images")
        print(f"  Would remove: {pattern_dir / 'default'}")
    
    # Update questions.jsonl
    questions_file = pattern_dir / "questions.jsonl"
    if questions_file.exists():
        print(f"  Updating questions.jsonl...")
        
        updated_questions = []
        update_count = 0
        
        with open(questions_file, 'r') as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    
                    if 'image' in q:
                        old_path = q['image']
                        # Extract just the filename
                        filename = Path(old_path).name
                        new_path = f"images/{filename}"
                        
                        if old_path != new_path:
                            q['image'] = new_path
                            update_count += 1
                    
                    updated_questions.append(q)
        
        print(f"  Updated {update_count} image paths in questions")
        
        if not dry_run:
            with open(questions_file, 'w') as f:
                for q in updated_questions:
                    f.write(json.dumps(q, ensure_ascii=False) + '\n')
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Fix InteriorQA image paths')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to InteriorQA data directory')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be done without making changes')
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return 1
    
    print(f"InteriorQA Path Fixer")
    print(f"Data directory: {data_dir}")
    if args.dry_run:
        print("Mode: DRY RUN (no changes will be made)")
    print()
    
    # Find all pattern directories
    pattern_dirs = find_pattern_dirs(data_dir)
    print(f"Found {len(pattern_dirs)} pattern directories to process")
    
    # Process each
    success_count = 0
    for pattern_dir in pattern_dirs:
        if fix_pattern_dir(pattern_dir, dry_run=args.dry_run):
            success_count += 1
    
    print()
    print("=" * 60)
    print(f"Processed: {success_count}/{len(pattern_dirs)} directories")
    if args.dry_run:
        print("This was a dry run. Run without --dry_run to apply changes.")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    exit(main())
