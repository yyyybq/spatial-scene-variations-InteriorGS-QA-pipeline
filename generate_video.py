#!/usr/bin/env python3
"""
SceneShift-Bench Video QA Generation

Generates spatial reasoning VQA data with video trajectories.
For each (QA item, pattern), generates 10 distinct video trajectories.

Usage:
    python generate_video.py                                      # default: scene 0267
    python generate_video.py --scenes 0267_840790 0003_839989     # specific scenes
    python generate_video.py --num-trajectories 5 --num-frames 15 # custom params
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

from config import (
    ObjectSelectionConfig, CameraSamplingConfig,
    QuestionConfig, VideoTrajectoryConfig,
)
from object_selector import ObjectSelector, SceneObject
from camera_sampler import CameraPose
from question_generator import QuestionGenerator
from video_trajectory_sampler import VideoTrajectorySampler
from scenes import DEFAULT_SCENES, QUESTION_TYPES

random.seed(42)

SCENES_ROOT = "/scratch/by2593/project/Active_Spatial/InteriorGS"
DEFAULT_OUTPUT = "/scratch/by2593/project/sceneshift/data/sceneshift_video_v1"

MAX_OBJECTS_PER_SCENE = 5
MIN_OBJECTS_PER_SCENE = 3

# Video-mode patterns: (pattern_name, linear_sub_pattern)
VIDEO_PATTERNS = [
    ("around",    None),
    ("spherical", None),
    ("linear",    "approach"),
    ("rotation",  None),
]


def select_focus_objects(scene_path: Path, max_objects: int = MAX_OBJECTS_PER_SCENE) -> list:
    """Select 3-5 diverse objects from a scene (deduplicate by label, top by volume)."""
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
    """Select ALL C(n,2) pairs with random order swap."""
    pairs = []
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if random.random() < 0.5:
                pairs.append((objects[i], objects[j]))
            else:
                pairs.append((objects[j], objects[i]))
    return pairs


def get_reference_pose(trajectory: list, reference: str = "middle") -> CameraPose:
    """Get the reference camera pose from a trajectory for answer computation."""
    if reference == "first":
        return trajectory[0]
    elif reference == "last":
        return trajectory[-1]
    else:  # middle
        return trajectory[len(trajectory) // 2]


def run_scene_pattern_video(
    scene_id: str,
    focus_objects: list,
    focus_pairs: list,
    pattern: str,
    linear_sub: str,
    video_config: VideoTrajectoryConfig,
):
    """Generate video QA data for one (scene, pattern) combination.
    
    For each object/pair, generates num_trajectories video trajectories,
    and for each trajectory generates the full set of questions.
    """
    cam_cfg = CameraSamplingConfig(
        move_pattern=pattern,
        max_tries=100,
        skip_occlusion_check=True,
        per_angle=18,
        rotation_interval=30.0,
        image_width=video_config.frame_width,
        image_height=video_config.frame_height,
        fov_deg=video_config.fov_deg,
    )
    if linear_sub:
        cam_cfg.linear_sub_pattern = linear_sub
        cam_cfg.linear_move_distance = video_config.linear_total_distance

    q_cfg = QuestionConfig(
        enabled_question_types=QUESTION_TYPES,
        max_questions_per_type=10,
    )

    traj_sampler = VideoTrajectorySampler(cam_cfg, video_config)
    question_gen = QuestionGenerator(q_cfg)

    scene_path = Path(SCENES_ROOT) / scene_id
    obj_selector = ObjectSelector(ObjectSelectionConfig())
    all_scene_objects = obj_selector.get_all_parsed_objects(scene_path)

    all_questions = []
    all_trajectories = {}  # traj_key -> list of CameraPose

    def _make_traj_key(obj_key: str, traj_idx: int) -> str:
        return f"{obj_key}_traj{traj_idx:02d}"

    # ---- Single objects ----
    for obj in focus_objects:
        obj_key = f"{obj.label}_{obj.id}"
        print(f"    Object: {obj_key}", end=" ", flush=True)

        trajectories = traj_sampler.generate_trajectories(
            scene_path, [obj], all_scene_objects, pattern=pattern,
        )
        print(f"=> {len(trajectories)} trajectories", flush=True)

        for traj_idx, traj in enumerate(trajectories):
            traj_key = _make_traj_key(obj_key, traj_idx)
            all_trajectories[traj_key] = traj

            ref_pose = get_reference_pose(traj, video_config.reference_frame)
            qs = question_gen.generate_single_object_questions(obj, ref_pose)

            for q in qs:
                q["scene_id"] = scene_id
                q["pattern"] = pattern
                q["trajectory_key"] = traj_key
                q["trajectory_idx"] = traj_idx
                q["num_frames"] = len(traj)
                q["reference_frame_idx"] = len(traj) // 2
                q["camera_poses"] = [p.to_dict() for p in traj]

            all_questions.extend(qs)

    # ---- Pair objects ----
    for obj1, obj2 in focus_pairs:
        pair_key = f"{obj1.label}_{obj1.id}__{obj2.label}_{obj2.id}"
        print(f"    Pair: {pair_key}", end=" ", flush=True)

        trajectories = traj_sampler.generate_trajectories(
            scene_path, [obj1, obj2], all_scene_objects, pattern=pattern,
        )
        print(f"=> {len(trajectories)} trajectories", flush=True)

        for traj_idx, traj in enumerate(trajectories):
            traj_key = _make_traj_key(pair_key, traj_idx)
            all_trajectories[traj_key] = traj

            ref_pose = get_reference_pose(traj, video_config.reference_frame)
            qs = question_gen.generate_pair_object_questions(obj1, obj2, ref_pose)

            for q in qs:
                q["scene_id"] = scene_id
                q["pattern"] = pattern
                q["trajectory_key"] = traj_key
                q["trajectory_idx"] = traj_idx
                q["num_frames"] = len(traj)
                q["reference_frame_idx"] = len(traj) // 2
                q["camera_poses"] = [p.to_dict() for p in traj]

            all_questions.extend(qs)

    # ---- Multi-object questions (reuse single-object trajectories) ----
    if len(focus_objects) >= 3:
        # Reuse trajectories from the first object
        first_obj = focus_objects[0]
        first_key = f"{first_obj.label}_{first_obj.id}"
        for traj_idx in range(video_config.num_trajectories):
            traj_key = _make_traj_key(first_key, traj_idx)
            if traj_key not in all_trajectories:
                continue
            traj = all_trajectories[traj_key]
            ref_pose = get_reference_pose(traj, video_config.reference_frame)
            qs = question_gen.generate_multi_object_questions(
                focus_objects, ref_pose, max_questions_per_type=q_cfg.max_questions_per_type,
            )
            for q in qs:
                q["scene_id"] = scene_id
                q["pattern"] = pattern
                q["trajectory_key"] = traj_key
                q["trajectory_idx"] = traj_idx
                q["num_frames"] = len(traj)
                q["reference_frame_idx"] = len(traj) // 2
                q["camera_poses"] = [p.to_dict() for p in traj]

            all_questions.extend(qs)

    return all_questions, all_trajectories


def main():
    parser = argparse.ArgumentParser(description="SceneShift-Bench Video QA Generation")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output directory")
    parser.add_argument("--scenes", nargs="+", default=["0267_840790"], help="Scene IDs")
    parser.add_argument("--max-objects", type=int, default=MAX_OBJECTS_PER_SCENE)
    parser.add_argument("--min-objects", type=int, default=MIN_OBJECTS_PER_SCENE)
    parser.add_argument("--num-trajectories", type=int, default=10)
    parser.add_argument("--num-frames", type=int, default=30)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--output-format", choices=["mp4", "frames"], default="frames",
                        help="Output format: mp4 (video files) or frames (PNG directories)")
    parser.add_argument("--patterns", nargs="+", default=None,
                        help="Patterns to generate (default: all 4)")
    args = parser.parse_args()

    video_cfg = VideoTrajectoryConfig(
        num_trajectories=args.num_trajectories,
        num_frames=args.num_frames,
        fps=args.fps,
        output_format=args.output_format,
    )

    scenes = args.scenes
    output_base = args.output
    os.makedirs(output_base, exist_ok=True)

    patterns = VIDEO_PATTERNS
    if args.patterns:
        patterns = [(p, sub) for p, sub in VIDEO_PATTERNS if p in args.patterns]

    total_questions = 0
    total_trajectories = 0
    failed = []
    skipped = []
    t0 = time.time()
    total_runs = len(scenes) * len(patterns)
    done = 0

    for si, scene_id in enumerate(scenes):
        scene_path = Path(SCENES_ROOT) / scene_id
        focus_objects = select_focus_objects(scene_path, args.max_objects)

        if len(focus_objects) < args.min_objects:
            print(f"\n[SKIP] {scene_id}: only {len(focus_objects)} objects (need {args.min_objects})")
            skipped.append(scene_id)
            done += len(patterns)
            continue

        focus_pairs = select_focus_pairs(focus_objects)
        focus_labels = [o.label for o in focus_objects]
        print(f"\n{'='*60}")
        print(f"Scene {si+1}/{len(scenes)}: {scene_id} — {len(focus_objects)} objects: {focus_labels}")
        print(f"  {len(focus_pairs)} pairs")

        for pattern, linear_sub in patterns:
            done += 1
            elapsed = time.time() - t0
            eta = (elapsed / done) * (total_runs - done) if done > 0 else 0

            out_dir = Path(output_base) / pattern / scene_id
            out_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n  [{done}/{total_runs}] {pattern} (ETA: {eta/60:.0f}min)", flush=True)

            try:
                questions, trajectories = run_scene_pattern_video(
                    scene_id, focus_objects, focus_pairs,
                    pattern, linear_sub, video_cfg,
                )

                # Save questions.jsonl (without full camera_poses to save space)
                qfile = out_dir / "questions.jsonl"
                with open(qfile, "w") as f:
                    for q in questions:
                        # Store camera_poses separately in trajectory files
                        q_save = {k: v for k, v in q.items() if k != "camera_poses"}
                        f.write(json.dumps(q_save, default=str) + "\n")

                # Save trajectory files
                traj_dir = out_dir / "trajectories"
                traj_dir.mkdir(exist_ok=True)
                for traj_key, traj in trajectories.items():
                    traj_file = traj_dir / f"{traj_key}.json"
                    traj_data = {
                        "trajectory_key": traj_key,
                        "num_frames": len(traj),
                        "pattern": pattern,
                        "scene_id": scene_id,
                        "camera_poses": [p.to_dict() for p in traj],
                    }
                    with open(traj_file, "w") as f:
                        json.dump(traj_data, f, indent=2)

                # Save metadata
                meta = {
                    "scene_id": scene_id,
                    "pattern": pattern,
                    "num_trajectories": len(trajectories),
                    "num_questions": len(questions),
                    "num_frames_per_trajectory": video_cfg.num_frames,
                    "fps": video_cfg.fps,
                    "focus_objects": focus_labels,
                    "num_focus_objects": len(focus_objects),
                    "num_focus_pairs": len(focus_pairs),
                    "question_types": list(set(q["question_type"] for q in questions)) if questions else [],
                    "trajectory_keys": list(trajectories.keys()),
                }
                with open(out_dir / "metadata.json", "w") as f:
                    json.dump(meta, f, indent=2)

                nq = len(questions)
                nt = len(trajectories)
                total_questions += nq
                total_trajectories += nt
                print(f"  => {nt} trajectories, {nq} Qs (total: {total_questions} Qs, {total_trajectories} trajs)")

            except Exception as e:
                print(f"  FAIL: {e}", flush=True)
                import traceback
                traceback.print_exc()
                failed.append(f"{scene_id}/{pattern}")

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} minutes")
    print(f"Total questions: {total_questions}")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Skipped: {len(skipped)}, Failed: {len(failed)}")
    for f in failed:
        print(f"  - {f}")

    summary = {
        "total_questions": total_questions,
        "total_trajectories": total_trajectories,
        "total_scenes": len(scenes),
        "scenes_processed": len(scenes) - len(skipped),
        "scenes_skipped": skipped,
        "patterns": [p for p, _ in patterns],
        "num_trajectories_per_pattern": video_cfg.num_trajectories,
        "num_frames_per_trajectory": video_cfg.num_frames,
        "fps": video_cfg.fps,
        "elapsed_minutes": round(elapsed / 60, 1),
        "failed": failed,
    }
    with open(Path(output_base) / "generation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
