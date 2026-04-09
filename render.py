#!/usr/bin/env python3
"""
Render images for SceneShift-Bench questions.

Reads questions.jsonl from each (pattern, scene) directory, renders the
camera poses via Gaussian Splatting, and saves PNG images.

Usage:
    python render.py --gpu 0                       # render all missing images
    python render.py --gpu 5 --data /path/to/data  # custom data dir
    python render.py --update-only                  # only add image_path to questions.jsonl
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
DEFAULT_DATA_ROOT = "/scratch/by2593/project/sceneshift/data/sceneshift_bench_50_v4"
PATTERNS = ["around", "linear", "spherical", "rotation"]


def collect_render_tasks(data_root: Path):
    """Collect all unique (scene, pattern, pose_idx, camera_pose) needing rendering."""
    tasks = []
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

            scene_id = scene_dir.name
            img_dir = scene_dir / "images"

            seen_poses = {}
            with open(qfile) as f:
                for line in f:
                    q = json.loads(line)
                    pidx = q["camera_pose_idx"]
                    if pidx not in seen_poses:
                        seen_poses[pidx] = q["camera_pose"]

            for pidx in sorted(seen_poses.keys()):
                out_path = img_dir / f"pose_{pidx:04d}.png"
                if not out_path.exists():
                    tasks.append((pattern, scene_id, pidx, seen_poses[pidx], img_dir))

    return tasks


async def render_all(data_root: Path, gpu_id: int = 0):
    tasks = collect_render_tasks(data_root)
    if not tasks:
        print("All images already rendered!")
        return

    by_scene = defaultdict(list)
    for pattern, scene_id, pidx, camera_pose, img_dir in tasks:
        by_scene[scene_id].append((pattern, pidx, camera_pose, img_dir))

    total = len(tasks)
    total_scenes = len(by_scene)
    print(f"=== Rendering {total} images across {total_scenes} scenes on GPU {gpu_id} ===")

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
            print(f"\n[{si+1}/{total_scenes}] Scene {scene_id} ({len(scene_tasks)} poses) | "
                  f"Done: {rendered}/{total} | Rate: {rate:.1f}/s | ETA: {remaining/60:.0f}min")

            try:
                await renderer.set_scene(scene_id)
            except Exception as e:
                print(f"  [ERROR] Cannot load scene {scene_id}: {e}")
                failed += len(scene_tasks)
                continue

            for pattern, pidx, camera_pose, img_dir in scene_tasks:
                img_dir.mkdir(parents=True, exist_ok=True)
                out_path = img_dir / f"pose_{pidx:04d}.png"

                try:
                    intrinsics, extrinsics_c2w = camera_pose_to_matrices(
                        camera_pose, config.image_width, config.image_height, config.fov_deg
                    )
                    image = await renderer.render_image(intrinsics, extrinsics_c2w)

                    if image is not None:
                        image.save(out_path)
                        rendered += 1
                    else:
                        print(f"  [FAIL] None: {pattern}/{scene_id}/pose_{pidx:04d}")
                        failed += 1
                except Exception as e:
                    print(f"  [ERROR] {pattern}/{scene_id}/pose_{pidx:04d}: {e}")
                    failed += 1

    elapsed = time.time() - t0
    print(f"\n=== DONE: {rendered} rendered, {failed} failed in {elapsed/60:.1f}min ===")


def update_questions_jsonl(data_root: Path):
    """Add image_path field to all questions.jsonl files."""
    updated_files = 0
    updated_qs = 0
    missing_imgs = 0

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

            questions = []
            changed = False
            with open(qfile) as f:
                for line in f:
                    q = json.loads(line)
                    pidx = q["camera_pose_idx"]
                    img_rel = f"images/pose_{pidx:04d}.png"
                    img_abs = scene_dir / img_rel

                    if img_abs.exists():
                        if q.get("image_path") != img_rel:
                            q["image_path"] = img_rel
                            changed = True
                            updated_qs += 1
                    else:
                        missing_imgs += 1

                    questions.append(q)

            if changed:
                with open(qfile, "w") as f:
                    for q in questions:
                        f.write(json.dumps(q, ensure_ascii=False) + "\n")
                updated_files += 1

    print(f"Updated {updated_files} files, {updated_qs} questions with image_path")
    if missing_imgs:
        print(f"WARNING: {missing_imgs} questions missing rendered images")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render SceneShift-Bench images")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--data", default=DEFAULT_DATA_ROOT, help="Data root directory")
    parser.add_argument("--update-only", action="store_true", help="Only update questions.jsonl")
    args = parser.parse_args()

    data_root = Path(args.data)
    if args.update_only:
        update_questions_jsonl(data_root)
    else:
        asyncio.run(render_all(data_root, gpu_id=args.gpu))
        print("\nUpdating questions.jsonl with image paths...")
        update_questions_jsonl(data_root)
