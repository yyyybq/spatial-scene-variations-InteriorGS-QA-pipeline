#!/usr/bin/env python3
"""
Render videos for SceneShift-Bench video QA data.

Reads trajectory JSON files, renders each frame via Gaussian Splatting,
and outputs either MP4 videos or PNG frame directories.

Usage:
    python render_video.py --gpu 0                              # render all
    python render_video.py --gpu 0 --pattern around             # only around
    python render_video.py --gpu 0 --output-format mp4          # render as mp4
    python render_video.py --gpu 0 --output-format frames       # render as PNGs
"""
import json
import sys
import os
import asyncio
import time
import argparse
from pathlib import Path
from collections import defaultdict

VAGEN_PATH = "/scratch/by2593/project/Active_Spatial/VAGEN"
sys.path.insert(0, VAGEN_PATH)
sys.path.insert(0, str(Path(__file__).parent))

from render_utils import RenderConfig, SceneRenderer, camera_pose_to_matrices

SCENES_ROOT = "/scratch/by2593/project/Active_Spatial/InteriorGS"
DEFAULT_DATA_ROOT = "/scratch/by2593/project/sceneshift/data/sceneshift_video_v1"
PATTERNS = ["around", "linear", "spherical", "rotation"]


def collect_render_tasks(data_root: Path, patterns: list = None, scenes: list = None):
    """Collect all trajectory files that need rendering.
    
    Returns list of (pattern, scene_id, traj_key, traj_data, output_dir).
    """
    patterns = patterns or PATTERNS
    scene_filter = set(scenes) if scenes else None
    tasks = []

    for pattern in patterns:
        pattern_dir = data_root / pattern
        if not pattern_dir.exists():
            continue
        for scene_dir in sorted(pattern_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            if scene_filter and scene_dir.name not in scene_filter:
                continue
            traj_dir = scene_dir / "trajectories"
            if not traj_dir.exists():
                continue

            scene_id = scene_dir.name

            for traj_file in sorted(traj_dir.glob("*.json")):
                traj_key = traj_file.stem

                # Check if already rendered
                video_out = scene_dir / "videos" / f"{traj_key}.mp4"
                frames_out = scene_dir / "frames" / traj_key
                if video_out.exists() or (frames_out.exists() and any(frames_out.iterdir())):
                    continue

                with open(traj_file) as f:
                    traj_data = json.load(f)

                tasks.append((pattern, scene_id, traj_key, traj_data, scene_dir))

    return tasks


async def render_trajectory_frames(
    renderer, traj_data: dict, config: RenderConfig
) -> list:
    """Render all frames of a trajectory, return list of PIL images."""
    import numpy as np

    frames = []
    for pose_dict in traj_data["camera_poses"]:
        intrinsics, extrinsics_c2w = camera_pose_to_matrices(
            pose_dict, config.image_width, config.image_height, config.fov_deg
        )
        image = await renderer.render_image(intrinsics, extrinsics_c2w)
        if image is not None:
            frames.append(image)
        else:
            frames.append(None)
    return frames


def save_as_frames(frames: list, output_dir: Path):
    """Save frames as individual PNG files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(frames):
        if img is not None:
            img.save(output_dir / f"frame_{i:04d}.png")


def save_as_mp4(frames: list, output_path: Path, fps: int = 10):
    """Save frames as MP4 video using imageio."""
    import numpy as np
    try:
        import imageio.v3 as iio
        use_v3 = True
    except ImportError:
        import imageio
        use_v3 = False

    valid_frames = []
    for img in frames:
        if img is not None:
            valid_frames.append(np.array(img))

    if not valid_frames:
        print(f"  [WARN] No valid frames to save for {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if use_v3:
        iio.imwrite(str(output_path), np.stack(valid_frames), fps=fps, codec="libx264")
    else:
        imageio.mimwrite(str(output_path), valid_frames, fps=fps, codec="libx264")


async def render_all(data_root: Path, gpu_id: int = 0,
                     output_format: str = "frames", fps: int = 10,
                     patterns: list = None, scenes: list = None):
    tasks = collect_render_tasks(data_root, patterns, scenes)
    if not tasks:
        print("All trajectories already rendered!")
        return

    by_scene = defaultdict(list)
    for pattern, scene_id, traj_key, traj_data, scene_dir in tasks:
        by_scene[scene_id].append((pattern, traj_key, traj_data, scene_dir))

    total = len(tasks)
    total_scenes = len(by_scene)
    total_frames = sum(len(t[3]["camera_poses"]) for t in tasks)
    print(f"=== Rendering {total} trajectories ({total_frames} frames) "
          f"across {total_scenes} scenes on GPU {gpu_id} ===")
    print(f"    Output format: {output_format}, FPS: {fps}")

    config = RenderConfig(
        scenes_root=SCENES_ROOT,
        render_backend="local",
        image_width=640,
        image_height=480,
        fov_deg=60.0,
        gpu_device=gpu_id,
    )

    rendered = 0
    failed = 0
    t0 = time.time()

    async with SceneRenderer(config) as renderer:
        for si, (scene_id, scene_tasks) in enumerate(sorted(by_scene.items())):
            elapsed = time.time() - t0
            rate = rendered / elapsed if elapsed > 0 else 0
            remaining = (total - rendered) / rate if rate > 0 else 0
            print(f"\n[{si+1}/{total_scenes}] Scene {scene_id} ({len(scene_tasks)} trajectories) | "
                  f"Done: {rendered}/{total} | Rate: {rate:.1f}/s | ETA: {remaining/60:.0f}min")

            try:
                await renderer.set_scene(scene_id)
            except Exception as e:
                print(f"  [ERROR] Cannot load scene {scene_id}: {e}")
                failed += len(scene_tasks)
                continue

            for pattern, traj_key, traj_data, scene_dir in scene_tasks:
                n_frames = len(traj_data["camera_poses"])
                print(f"  {pattern}/{traj_key} ({n_frames} frames)", end=" ", flush=True)

                try:
                    frames = await render_trajectory_frames(renderer, traj_data, config)
                    valid_count = sum(1 for f in frames if f is not None)

                    if valid_count == 0:
                        print(f"=> FAILED (0/{n_frames} frames)")
                        failed += 1
                        continue

                    if output_format == "mp4":
                        out_path = scene_dir / "videos" / f"{traj_key}.mp4"
                        save_as_mp4(frames, out_path, fps=fps)
                        print(f"=> {out_path.name} ({valid_count}/{n_frames} frames)")
                    else:
                        out_dir = scene_dir / "frames" / traj_key
                        save_as_frames(frames, out_dir)
                        print(f"=> frames/{traj_key}/ ({valid_count}/{n_frames} frames)")

                    rendered += 1

                except Exception as e:
                    print(f"=> ERROR: {e}")
                    failed += 1

    elapsed = time.time() - t0
    print(f"\n=== DONE: {rendered} rendered, {failed} failed in {elapsed/60:.1f}min ===")


def update_questions_with_video_paths(data_root: Path, output_format: str = "frames"):
    """Add video_path field to all questions.jsonl entries."""
    for pattern in PATTERNS:
        pattern_dir = data_root / pattern
        if not pattern_dir.exists():
            continue
        for scene_dir in sorted(pattern_dir.iterdir()):
            if not scene_dir.is_dir():
                continue
            qfile = scene_dir / "questions.jsonl"
            if not qfile.exists():
                continue

            updated = []
            with open(qfile) as f:
                for line in f:
                    q = json.loads(line)
                    traj_key = q.get("trajectory_key", "")
                    if output_format == "mp4":
                        q["video_path"] = f"{pattern}/{scene_dir.name}/videos/{traj_key}.mp4"
                    else:
                        q["video_path"] = f"{pattern}/{scene_dir.name}/frames/{traj_key}/"
                    updated.append(q)

            with open(qfile, "w") as f:
                for q in updated:
                    f.write(json.dumps(q, default=str) + "\n")

            print(f"  Updated {len(updated)} questions in {qfile.relative_to(data_root)}")


def main():
    parser = argparse.ArgumentParser(description="Render videos for SceneShift-Bench")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--data", default=DEFAULT_DATA_ROOT, help="Data root directory")
    parser.add_argument("--output-format", choices=["mp4", "frames"], default="frames")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--pattern", nargs="+", default=None, help="Only render these patterns")
    parser.add_argument("--scenes", nargs="+", default=None, help="Only render these scene IDs")
    parser.add_argument("--update-only", action="store_true", help="Only update questions.jsonl with video paths")
    args = parser.parse_args()

    data_root = Path(args.data)

    if args.update_only:
        update_questions_with_video_paths(data_root, args.output_format)
        return

    asyncio.run(render_all(
        data_root, args.gpu, args.output_format, args.fps, args.pattern,
        args.scenes
    ))

    # After rendering, update questions.jsonl with video paths
    update_questions_with_video_paths(data_root, args.output_format)


if __name__ == "__main__":
    main()
