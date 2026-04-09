# SceneShift-Bench Question Generation Pipeline (InteriorGS)

Generates spatial reasoning VQA from InteriorGS 3D indoor scenes.

- **50 curated scenes**, **13 question types**, **4 camera trajectory modes**
- Output: `questions.jsonl` + rendered PNG images per scene/pattern

## Quick Start

```bash
# Generate QA for all 50 default scenes
python generate.py

# Generate for specific scenes only
python generate.py --scenes 0267_840790 0003_839989

# Custom output directory
python generate.py --output /path/to/output

# Render images (requires GPU + gsplat)
python render.py --gpu 0

# Only update questions.jsonl with image_path (no re-render)
python render.py --update-only

# Build website dashboard
python build_benchmark_website.py
```

## Pipeline Flow

```
InteriorGS Scene (labels.json + structure.json + occupancy.json)
  │
  ├─[1] Object Selection ─── object_selector.py
  │     Semantic blacklist → geometric filter → 3-5 objects + all pairs
  │
  ├─[2] Camera Sampling ──── camera_sampler.py + camera_utils.py
  │     4 trajectory modes → collision/visibility validation → N poses
  │
  ├─[3] Question Generation ─ question_generator.py + question_utils.py
  │     13 question types → numerical + MC + relative
  │
  ├─[4] Rendering (optional) ─ render.py + render_utils.py
  │     Gaussian Splatting → PNG images
  │
  └─ Output: {pattern}/{scene_id}/questions.jsonl + images/
```

## Question Types (13 total)

### Numerical (9 types)

| Type | Objects | Answer |
|------|---------|--------|
| `object_size` | 1 | number (2dp) |
| `object_distance_to_camera` | 1 | number (2dp) |
| `object_size_comparison_relative` | 2 | ratio (2dp) |
| `object_size_comparison_absolute` | 2 | number (2dp) |
| `object_pair_distance_center` | 2 | number (2dp) |
| `object_pair_distance_center_w_size` | 2 | number (2dp) |
| `object_pair_distance_vector` | 2 | [x, y, z] (2dp) |
| `object_comparison_absolute_distance` | 3-4 | number (2dp) |
| `object_comparison_relative_distance` | 3-4 | ratio (2dp) |

### Relative MC — A/B (3 types)

| Type | Objects | Answer |
|------|---------|--------|
| `relative_size` | 2 | A or B |
| `relative_distance` | 3 (pivot) | A or B |
| `relative_distance_to_camera` | 2 | A or B |

### Numerical MC — A/B/C/D (1 type, 3 sub-types)

| `mc` sub-type | Source | Answer |
|---------------|--------|--------|
| `mc_object_size` | 1 obj | A-D |
| `mc_object_distance_to_camera` | 1 obj | A-D |
| `mc_object_pair_distance_center` | 2 obj | A-D |

## Camera Patterns

| Mode | Behavior |
|------|----------|
| `around` | Horizontal circle around object |
| `spherical` | Sphere surface sampling (varied height) |
| `linear` | Straight walk toward object (approach) |
| `rotation` | Stand at room center, rotate 360° |

## Scene Selection

50 scenes are selected from 76 user-curated candidates (seed=42).
Scene IDs and question types are defined in [`scenes.py`](scenes.py).

## Project Structure

```
question_gen_InteriorGS/
│
│  Core Modules
├── config.py                 # Configuration dataclasses
├── object_selector.py        # Object filtering & pair selection
├── camera_sampler.py         # Camera pose sampling (4 modes)
├── camera_utils.py           # Camera math (projection, FOV, occlusion)
├── question_templates.py     # Question text templates + post prompts
├── question_utils.py         # QA constructors (13 question types)
├── question_generator.py     # Question generation orchestrator
├── render_utils.py           # Gaussian Splatting rendering wrapper
├── pipeline.py               # General-purpose pipeline orchestrator
│
│  Entry Points
├── generate.py               # Generate QA data (main script)
├── render.py                 # Render images from camera poses
├── build_benchmark_website.py # Build website dashboard
│
│  Config & Data
├── scenes.py                 # 50 scene IDs, question types, patterns
├── requirements.txt          # Python dependencies
├── __init__.py               # Package exports
└── README.md
```

## Output Format

```
output_dir/
├── around/
│   └── {scene_id}/
│       ├── questions.jsonl     # QA data
│       ├── metadata.json       # Scene/pattern metadata
│       └── images/
│           ├── pose_0000.png
│           └── ...
├── spherical/
├── linear/
├── rotation/
└── generation_summary.json
```

## Input Requirements

Each InteriorGS scene folder must contain:

| File | Required | Content |
|------|----------|---------|
| `labels.json` | **Yes** | Object annotations with 3D bounding boxes |
| `structure.json` | Optional | Room polygons for spatial validation |
| `occupancy.json` | Optional | Scene bounds for camera position validation |
