#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


DATASET_ROOT = Path("/scratch/by2593/project/sceneshift/data/sceneshift_bench_50_v4")
WEBSITE_ROOT = DATASET_ROOT / "website"
DATA_DIR = WEBSITE_ROOT / "data"
CHUNKS_DIR = DATA_DIR / "chunks"
ANALYSIS_IMAGE = DATASET_ROOT / "analysis" / "data_distribution.png"


def read_json(path: Path):
    with open(path) as handle:
        return json.load(handle)


def read_jsonl(path: Path):
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def compact_question(question: dict, pattern: str, scene_id: str, chunk_file: str) -> dict:
    camera_pose = question.get("camera_pose") or {}
    objects = question.get("objects") or []
    return {
        "id": question.get("question_id"),
        "scene_id": scene_id,
        "pattern": pattern,
        "question_type": question.get("question_type"),
        "question": question.get("question"),
        "answer": question.get("answer"),
        "answer_text": question.get("answer_text"),
        "answer_value": question.get("answer_value"),
        "primary_object": question.get("primary_object"),
        "object_labels": [obj.get("label") for obj in objects],
        "camera_pose_idx": question.get("camera_pose_idx"),
        "camera_target_objects": camera_pose.get("target_objects", []),
        "choices": question.get("choices"),
        "mc_source_type": question.get("mc_source_type"),
        "chunk_file": chunk_file,
    }


def summarize_numeric(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "mean": round(mean(values), 4),
    }


def main() -> None:
    WEBSITE_ROOT.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    generation_summary = read_json(DATASET_ROOT / "generation_summary.json")

    question_index = []
    scene_summaries = []
    chunk_manifest = []
    label_counter = Counter()
    pattern_counts = Counter()
    question_type_counts = Counter()
    pattern_type_counts = defaultdict(Counter)
    yes_no_counts = defaultdict(Counter)
    mc_answer_counts = Counter()
    mc_source_counts = Counter()
    numeric_answers = defaultdict(list)
    scene_question_counts = Counter()
    scene_pattern_counts = defaultdict(Counter)
    scene_type_counts = defaultdict(Counter)
    total_camera_pose_refs = Counter()

    question_files = sorted(DATASET_ROOT.glob("*/**/questions.jsonl"))
    for questions_path in question_files:
        pattern = questions_path.parent.parent.name
        scene_id = questions_path.parent.name
        metadata_path = questions_path.parent / "metadata.json"
        metadata = read_json(metadata_path) if metadata_path.exists() else {}

        records = []
        for question in read_jsonl(questions_path):
            chunk_file = f"{pattern}__{scene_id}.json"
            compact = compact_question(question, pattern, scene_id, chunk_file)
            records.append(question)
            question_index.append(compact)
            pattern_counts[pattern] += 1
            question_type_counts[question["question_type"]] += 1
            pattern_type_counts[pattern][question["question_type"]] += 1
            scene_question_counts[scene_id] += 1
            scene_pattern_counts[scene_id][pattern] += 1
            scene_type_counts[scene_id][question["question_type"]] += 1
            total_camera_pose_refs[(scene_id, pattern, question.get("camera_pose_idx"))] += 1

            for obj in question.get("objects", []):
                label = obj.get("label")
                if label:
                    label_counter[label] += 1

            if question["question_type"] in {
                "relative_size",
                "relative_distance",
                "relative_distance_to_camera",
            }:
                # Relative questions now use A/B choices; resolve to Yes/No via answer_text
                answer_text = question.get("answer_text") or question.get("answer")
                yes_no_counts[question["question_type"]][answer_text] += 1

            if question["question_type"] == "mc":
                mc_answer_counts[question.get("answer")] += 1
                mc_source_counts[question.get("mc_source_type", "unknown")] += 1

            if question["question_type"] in {
                "object_size",
                "object_distance_to_camera",
                "object_pair_distance_center",
            }:
                try:
                    numeric_answers[question["question_type"]].append(float(question.get("answer")))
                except (TypeError, ValueError):
                    pass

        chunk_payload = {
            "scene_id": scene_id,
            "pattern": pattern,
            "metadata": metadata,
            "questions": records,
        }
        chunk_path = CHUNKS_DIR / f"{pattern}__{scene_id}.json"
        with open(chunk_path, "w") as handle:
            json.dump(chunk_payload, handle, ensure_ascii=False)

        chunk_manifest.append(
            {
                "scene_id": scene_id,
                "pattern": pattern,
                "question_count": len(records),
                "metadata_path": str(metadata_path.relative_to(DATASET_ROOT)) if metadata_path.exists() else None,
                "chunk_file": chunk_path.name,
            }
        )

        scene_summaries.append(
            {
                "scene_id": scene_id,
                "pattern": pattern,
                "question_count": len(records),
                "focus_objects": metadata.get("focus_objects", []),
                "num_focus_objects": metadata.get("num_focus_objects", 0),
                "num_focus_pairs": metadata.get("num_focus_pairs", 0),
                "question_types": metadata.get("question_types", []),
                "num_cameras": metadata.get("num_cameras"),
            }
        )

    scene_overview = []
    for scene_id in sorted(scene_question_counts):
        scene_overview.append(
            {
                "scene_id": scene_id,
                "total_questions": scene_question_counts[scene_id],
                "by_pattern": dict(scene_pattern_counts[scene_id]),
                "by_type": dict(scene_type_counts[scene_id]),
                "camera_pose_refs": sum(
                    1
                    for key in total_camera_pose_refs
                    if key[0] == scene_id and key[2] is not None
                ),
            }
        )

    # ── Derived fields for the website ──
    all_patterns = sorted(pattern_counts.keys())
    all_question_types = sorted(question_type_counts.keys())
    all_scenes = sorted(scene_question_counts.keys())

    # Histograms for numeric answer types
    import numpy as np
    numeric_histograms = {}
    for key, values in numeric_answers.items():
        if not values:
            continue
        arr = np.array(values)
        counts, bin_edges = np.histogram(arr, bins=20)
        numeric_histograms[key] = {
            "bin_edges": [round(float(e), 4) for e in bin_edges],
            "counts": [int(c) for c in counts],
            "stats": {
                "count": len(values),
                "min": round(float(arr.min()), 4),
                "max": round(float(arr.max()), 4),
                "mean": round(float(arr.mean()), 4),
                "median": round(float(np.median(arr)), 4),
            },
        }

    # Scene heatmaps
    scene_pattern_heatmap = []
    scene_type_heatmap = []
    for sid in all_scenes:
        scene_pattern_heatmap.append([scene_pattern_counts[sid].get(p, 0) for p in all_patterns])
        scene_type_heatmap.append([scene_type_counts[sid].get(qt, 0) for qt in all_question_types])

    # Per-scene info for scene table
    scene_overview_enriched = []
    for item in scene_overview:
        sid = item["scene_id"]
        # Find focus objects from metadata
        focus_objs = set()
        for cm in chunk_manifest:
            if cm["scene_id"] == sid and cm.get("metadata_path"):
                mpath = DATASET_ROOT / cm["metadata_path"]
                if mpath.exists():
                    meta = read_json(mpath)
                    focus_objs.update(meta.get("focus_objects", []))
        item["object_labels"] = sorted(focus_objs)
        item["num_objects"] = len(focus_objs)
        item["has_image"] = (DATASET_ROOT / "website" / "data" / "scene_images" / f"{sid}.png").exists()
        scene_overview_enriched.append(item)

    dashboard = {
        "generation_summary": generation_summary,
        "total_questions": len(question_index),
        "total_scenes": len(all_scenes),
        "all_patterns": all_patterns,
        "all_question_types": all_question_types,
        "all_scenes": all_scenes,
        "pattern_counts": dict(pattern_counts),
        "question_type_counts": dict(question_type_counts),
        "pattern_type_cross": {pattern: dict(counter) for pattern, counter in pattern_type_counts.items()},
        "yes_no_counts": {key: dict(counter) for key, counter in yes_no_counts.items()},
        "mc_answer_dist": dict(mc_answer_counts),
        "mc_source_dist": dict(mc_source_counts),
        "numeric_histograms": numeric_histograms,
        "numeric_answer_stats": {key: summarize_numeric(values) for key, values in numeric_answers.items()},
        "top_object_labels": [
            {"label": label, "count": count} for label, count in label_counter.most_common(100)
        ],
        "scene_overview": scene_overview_enriched,
        "scene_pattern_heatmap": scene_pattern_heatmap,
        "scene_type_heatmap": scene_type_heatmap,
        "chunk_manifest": chunk_manifest,
    }

    with open(DATA_DIR / "dashboard.json", "w") as handle:
        json.dump(dashboard, handle, ensure_ascii=False)

    with open(DATA_DIR / "question_index.json", "w") as handle:
        json.dump(question_index, handle, ensure_ascii=False)

    with open(DATA_DIR / "scene_summaries.json", "w") as handle:
        json.dump(scene_summaries, handle, ensure_ascii=False)

    if ANALYSIS_IMAGE.exists():
        shutil.copy2(ANALYSIS_IMAGE, WEBSITE_ROOT / "assets" / "analysis.png")

    print(f"Built website data at: {WEBSITE_ROOT}")
    print(f"Question index records: {len(question_index)}")
    print(f"Chunks: {len(chunk_manifest)}")


if __name__ == "__main__":
    main()