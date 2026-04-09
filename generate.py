#!/usr/bin/env python3
"""
SceneShift-Bench QA Generation

Generates spatial reasoning VQA data for the SceneShift benchmark.
Uses 50 curated InteriorGS scenes, all 13 question types, and 4 camera patterns.

Usage:
    python generate.py                                    # default: 50 scenes, all types
    python generate.py --output /path/to/output           # custom output dir
    python generate.py --scenes 0267_840790 0003_839989   # specific scenes
    python generate.py --max-objects 4                    # max 4 objects per scene
"""
import sys
import os
import json
import time
import random
import argparse
from pathlib import Path
from itertools import combinations

sys.path.insert(0, str(Path(__file__).parent))

from config import ObjectSelectionConfig, CameraSamplingConfig, QuestionConfig
from object_selector import ObjectSelector
from scenes import DEFAULT_SCENES, QUESTION_TYPES, PATTERNS

random.seed(42)

SCENES_ROOT = "/scratch/by2593/project/Active_Spatial/InteriorGS"
DEFAULT_OUTPUT = "/scratch/by2593/project/sceneshift/data/sceneshift_bench_50_v4"

MAX_OBJECTS_PER_SCENE = 5
MIN_OBJECTS_PER_SCENE = 3


def select_focus_objects(scene_path: Path, max_objects: int = MAX_OBJECTS_PER_SCENE) -> list:
    """Select 3-5 diverse objects from a scene.
    
    Deduplicates by label (keeps largest instance), then takes top by volume.
    """
    obj_selector = ObjectSelector(ObjectSelectionConfig())
    all_valid = obj_selector.select_single_objects(scene_path)

    if len(all_valid) <= max_objects:
        return all_valid

    by_label = {}
    for obj in all_valid:
        if obj.label not in by_label or obj.volume > by_label[obj.label].volume:
            by_label[obj.label] = obj

    unique_objs = sorted(by_label.values(), key=lambda o: o.volume, reverse=True)
    return unique_objs[:max_objects]


def select_focus_pairs(objects: list) -> list:
    """Select ALL C(n,2) pairs from focus objects with random order swap for Yes/No balance."""
    pairs = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if random.random() < 0.5:
                pairs.append((objects[i], objects[j]))
            else:
                pairs.append((objects[j], objects[i]))
    return pairs


def run_scene_pattern(scene_id, focus_objects, focus_pairs, pattern, num_cameras, linear_sub):
    """Generate questions for one (scene, pattern) combination."""
    from camera_sampler import CameraSampler
    from question_generator import QuestionGenerator

    cam_cfg = CameraSamplingConfig(
        num_cameras_per_item=num_cameras,
        move_pattern=pattern,
        max_tries=100,
        skip_occlusion_check=True,
        per_angle=18,
        rotation_interval=30.0,
    )
    if linear_sub:
        cam_cfg.linear_sub_pattern = linear_sub
        cam_cfg.linear_num_steps = 5
        cam_cfg.linear_move_distance = 0.3

    q_cfg = QuestionConfig(
        enabled_question_types=QUESTION_TYPES,
        max_questions_per_type=10,
    )

    camera_sampler = CameraSampler(cam_cfg)
    question_gen = QuestionGenerator(q_cfg)

    scene_path = Path(SCENES_ROOT) / scene_id
    obj_selector = ObjectSelector(ObjectSelectionConfig())
    all_scene_objects = obj_selector.get_all_parsed_objects(scene_path)

    all_questions = []
    all_camera_poses = []
    camera_pose_to_idx = {}

    def _register_pose(camera_pose):
        pose_key = (tuple(camera_pose.position.tolist()), tuple(camera_pose.target.tolist()))
        if pose_key not in camera_pose_to_idx:
            idx = len(all_camera_poses)
            camera_pose_to_idx[pose_key] = idx
            all_camera_poses.append(camera_pose)
            return idx
        return camera_pose_to_idx[pose_key]

    def _tag_questions(questions, scene_id, pose_idx):
        for q in questions:
            q['scene_id'] = scene_id
            q['camera_pose_idx'] = pose_idx

    if pattern == 'rotation':
        # Rotation: room-centric 360° cameras, then questions for all focus objects
        try:
            rotation_results = camera_sampler.generate_rotation_poses(
                scene_path, all_scene_objects=all_scene_objects
            )
        except Exception as e:
            print(f"    Rotation camera generation failed: {e}")
            return []

        if not rotation_results:
            return []

        all_rot_poses = [pose for pose, _ in rotation_results]
        MAX_ROTATION_POSES = 6
        if len(all_rot_poses) > MAX_ROTATION_POSES:
            step = max(1, len(all_rot_poses) // MAX_ROTATION_POSES)
            all_rot_poses = all_rot_poses[::step][:MAX_ROTATION_POSES]

        for camera_pose in all_rot_poses:
            pose_idx = _register_pose(camera_pose)

            for obj in focus_objects:
                qs = question_gen.generate_single_object_questions(obj, camera_pose)
                _tag_questions(qs, scene_id, pose_idx)
                all_questions.extend(qs)

            for obj1, obj2 in focus_pairs:
                qs = question_gen.generate_pair_object_questions(obj1, obj2, camera_pose)
                _tag_questions(qs, scene_id, pose_idx)
                all_questions.extend(qs)

            if len(focus_objects) >= 3:
                qs = question_gen.generate_multi_object_questions(
                    focus_objects, camera_pose, max_questions_per_type=q_cfg.max_questions_per_type
                )
                _tag_questions(qs, scene_id, pose_idx)
                all_questions.extend(qs)

    else:
        # Object-centric patterns: sample cameras around individual objects/pairs
        for obj in focus_objects:
            camera_poses = camera_sampler.sample_cameras(
                scene_path, [obj], num_samples=num_cameras, all_scene_objects=all_scene_objects
            )
            for camera_pose in (camera_poses or []):
                pose_idx = _register_pose(camera_pose)
                qs = question_gen.generate_single_object_questions(obj, camera_pose)
                _tag_questions(qs, scene_id, pose_idx)
                all_questions.extend(qs)

        for obj1, obj2 in focus_pairs:
            camera_poses = camera_sampler.sample_cameras(
                scene_path, [obj1, obj2], num_samples=num_cameras, all_scene_objects=all_scene_objects
            )
            for camera_pose in (camera_poses or []):
                pose_idx = _register_pose(camera_pose)
                qs = question_gen.generate_pair_object_questions(obj1, obj2, camera_pose)
                _tag_questions(qs, scene_id, pose_idx)
                all_questions.extend(qs)

        if len(focus_objects) >= 3:
            multi_poses = list(all_camera_poses)[:min(len(all_camera_poses), num_cameras * 3)]
            for camera_pose in multi_poses:
                pose_idx = _register_pose(camera_pose)
                qs = question_gen.generate_multi_object_questions(
                    focus_objects, camera_pose, max_questions_per_type=q_cfg.max_questions_per_type
                )
                _tag_questions(qs, scene_id, pose_idx)
                all_questions.extend(qs)

    return all_questions


def main():
    parser = argparse.ArgumentParser(description="SceneShift-Bench QA Generation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--scenes", nargs="+", default=None, help="Scene IDs (default: 50 curated scenes)")
    parser.add_argument("--max-objects", type=int, default=MAX_OBJECTS_PER_SCENE)
    parser.add_argument("--min-objects", type=int, default=MIN_OBJECTS_PER_SCENE)
    args = parser.parse_args()

    scenes = args.scenes or DEFAULT_SCENES
    output_base = args.output
    os.makedirs(output_base, exist_ok=True)

    total_questions = 0
    failed = []
    skipped = []
    t0 = time.time()
    total_runs = len(scenes) * len(PATTERNS)
    done = 0

    obj_selector = ObjectSelector(ObjectSelectionConfig())

    for si, scene_id in enumerate(scenes):
        scene_path = Path(SCENES_ROOT) / scene_id
        focus_objects = select_focus_objects(scene_path, args.max_objects)

        if len(focus_objects) < args.min_objects:
            print(f"\n[SKIP] {scene_id}: only {len(focus_objects)} objects (need {args.min_objects})")
            skipped.append(scene_id)
            done += len(PATTERNS)
            continue

        focus_pairs = select_focus_pairs(focus_objects)
        focus_labels = [o.label for o in focus_objects]
        print(f"\n{'='*60}")
        print(f"Scene {si+1}/{len(scenes)}: {scene_id} — {len(focus_objects)} objects: {focus_labels}")
        print(f"  {len(focus_pairs)} pairs")

        for pattern, num_cameras, linear_sub in PATTERNS:
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total_runs - done) if done > 0 else 0

            out_dir = Path(output_base) / pattern / scene_id
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"  [{done}/{total_runs}] {pattern} (ETA: {eta/60:.0f}min)", end=" ", flush=True)

            try:
                questions = run_scene_pattern(
                    scene_id, focus_objects, focus_pairs, pattern, num_cameras, linear_sub
                )

                qfile = out_dir / "questions.jsonl"
                with open(qfile, "w") as f:
                    for q in questions:
                        f.write(json.dumps(q, default=str) + "\n")

                meta = {
                    "scene_id": scene_id,
                    "pattern": pattern,
                    "num_cameras": num_cameras,
                    "num_questions": len(questions),
                    "focus_objects": focus_labels,
                    "num_focus_objects": len(focus_objects),
                    "num_focus_pairs": len(focus_pairs),
                    "question_types": list(set(q["question_type"] for q in questions)) if questions else [],
                }
                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

                n = len(questions)
                total_questions += n
                print(f"=> {n} Qs (total: {total_questions})", flush=True)

            except Exception as e:
                print(f"FAIL: {e}", flush=True)
                import traceback
                traceback.print_exc()
                failed.append(f"{scene_id}/{pattern}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"Total questions: {total_questions}")
    print(f"Skipped scenes: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    for f in failed:
        print(f"  - {f}")

    summary = {
        "total_questions": total_questions,
        "total_scenes": len(scenes),
        "scenes_processed": len(scenes) - len(skipped),
        "scenes_skipped": skipped,
        "patterns": [p[0] for p in PATTERNS],
        "question_types": QUESTION_TYPES,
        "max_objects_per_scene": args.max_objects,
        "elapsed_minutes": round(elapsed / 60, 1),
        "failed": failed,
    }
    with open(f"{output_base}/generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {output_base}/generation_summary.json")


if __name__ == "__main__":
    main()
