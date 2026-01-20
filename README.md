# Spatial Reasoning Question Generation Pipeline for InteriorGS

A modular pipeline for generating spatial reasoning questions and answers from 3D indoor scenes in the InteriorGS dataset. This pipeline automatically selects objects, samples valid camera viewpoints, renders images, and generates diverse question-answer pairs for training and evaluating vision-language models on spatial understanding tasks.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Architecture](#pipeline-architecture)
- [Question Types](#question-types)
- [Usage](#usage)
- [Output Format](#output-format)
- [Configuration](#configuration)
- [Move Patterns](#move-patterns-camera-trajectory-types)
- [Rendering Images](#rendering-images)
- [Project Structure](#project-structure)
- [Data Requirements](#data-requirements)

## Overview

This pipeline generates spatial reasoning questions from InteriorGS 3D indoor scenes by:

1. **Object Selection** - Filtering meaningful objects from scene annotations
2. **Pair Selection** - Finding valid object pairs for relational questions
3. **Camera Sampling** - Generating realistic viewpoints that observe target objects
4. **Question Generation** - Creating diverse question-answer pairs about spatial relationships
5. **Image Rendering** - Rendering images from sampled camera poses (optional)

## Features

- **9 question types** covering size estimation, distance measurement, and spatial relationships
- **Automatic object filtering** based on semantic categories and geometric constraints
- **Smart camera placement** with collision detection and visibility validation
- **Gaussian Splatting rendering** support for photorealistic images
- **Flexible configuration** via command-line arguments or JSON config files
- **Batch processing** for multiple scenes with progress tracking

## Installation

```bash
cd /path/to/question_gen_InteriorGS
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```bash
# Minimal example: Generate questions for a single scene (no images)
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790
```

### Full Example with All Key Parameters

```bash
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --experiment_name my_experiment \
    --move_pattern around \
    --num_cameras 5 \
    --enable_rendering
```

### Key Parameters Explained

| Parameter | Description | Options |
|-----------|-------------|---------|
| `--scenes_root` | Path to InteriorGS dataset folder | Required |
| `--output_dir` | Where to save generated data | Required |
| `--scene_id` | Scene folder name to process | e.g., `0267_840790` |
| `--experiment_name` | Subfolder name for outputs (default: `default`) | Any string |
| `--move_pattern` | Camera trajectory type | `around`, `spherical`, `rotation`, `linear` |
| `--num_cameras` | Number of camera poses per object (default: 3) | Integer |
| `--enable_rendering` | Render images from camera poses | Flag (no value needed) |

### Move Pattern Quick Reference

| Pattern | Description | Use Case |
|---------|-------------|----------|
| `around` | Circle around object horizontally | Default, multi-view of single object |
| `spherical` | Sample on sphere surface | Varied viewing angles including up/down |
| `rotation` | Stand at room center, rotate 360° | Room panorama, seeing all objects |
| `linear` | Walk toward or past object | Simulating approach/passing motion |

#### Linear Pattern Sub-types

The `linear` pattern has two sub-patterns controlled by `--linear_sub_pattern`:

| Sub-pattern | Description | Visual Effect |
|-------------|-------------|---------------|
| `approach` | Walk **toward** the object | Object gets larger, stays centered |
| `pass_by` | Walk **past** the object (sideways) | Object moves across the field of view |

### Example: Different Move Patterns

```bash
# 1. Default "around" pattern - circle around objects
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern around \
    --num_cameras 5 \
    --experiment_name around_demo

# 2. "spherical" pattern - varied heights and angles
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern spherical \
    --num_cameras 8 \
    --experiment_name spherical_demo

# 3a. "linear" pattern with "approach" - walk toward object
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern linear \
    --linear_sub_pattern approach \
    --linear_num_steps 5 \
    --experiment_name linear_approach_demo

# 3b. "linear" pattern with "pass_by" - walk past object
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern linear \
    --linear_sub_pattern pass_by \
    --linear_num_steps 5 \
    --experiment_name linear_passby_demo

# 4. "rotation" pattern - 360° room scan
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern rotation \
    --experiment_name rotation_demo
```

### Output Structure

After running, you'll find:
```
output_dir/
├── {experiment_name}/           # e.g., "my_experiment"
│   └── {scene_id}/              # e.g., "0267_840790"
│       └── {move_pattern}/      # e.g., "around"
│           └── *.png            # Rendered images (if --enable_rendering)
└── {scene_id}/
    ├── questions.jsonl          # Generated QA pairs
    └── metadata.json            # Statistics
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           RAW DATA (InteriorGS Scene)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│  scene_folder/                                                                  │
│  ├── labels.json        # Object annotations with 3D bounding boxes            │
│  ├── structure.json     # Room polygons (optional, for room validation)        │
│  └── occupancy.json     # Scene bounds (optional, for position validation)     │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 1: OBJECT SELECTION                                 │
│                        (object_selector.py)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Input: labels.json (all objects in scene)                                      │
│                                                                                 │
│  Filters Applied:                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Semantic Filter (Blacklist) [Always enabled]                         │   │
│  │    - Exclude: wall, floor, ceiling, door, window, lamp, carpet...       │   │
│  │    - Keep: sofa, table, chair, cabinet, bed, tv...                      │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ 2. Geometric Filters [OPTIONAL - can be toggled on/off]                 │   │
│  │    - enable_dim_filter=True:                                            │   │
│  │        min_dim_component: 0.1m (no tiny objects)                        │   │
│  │        max_dim_component: 3.0m (no huge objects)                        │   │
│  │    - enable_volume_filter=True:                                         │   │
│  │        min_volume: 0.01m^3 (no flat/thin objects)                       │   │
│  │    - enable_aspect_ratio_filter=True:                                   │   │
│  │        min_aspect_ratio: 0.05 (no extremely elongated objects)          │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ 3. Uniqueness Filter                                                    │   │
│  │    - Keep only objects with unique labels (avoid ambiguity)             │   │
│  │    - Or unique within their room                                        │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Output: List[SceneObject] - Valid single objects                               │
│          Example: 36 valid objects from 459 total                               │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 2: OBJECT PAIR SELECTION                            │
│                        (object_selector.py)                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Input: List of valid single objects                                            │
│                                                                                 │
│  Pair Constraints:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ - Same room (if room info available)                                    │   │
│  │ - Distance: 0.3m < dist < 5.0m                                          │   │
│  │ - Size ratio: max_dim_A / max_dim_B < 3.0                               │   │
│  │ - Size difference: |max_dim_A - max_dim_B| < 2.5m                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Output: List[(ObjectA, ObjectB)] - Valid object pairs                          │
│          Example: 46 valid pairs                                                │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 3: CAMERA POSE SAMPLING                             │
│                        (camera_sampler.py)                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  Goal: Find valid camera positions that can observe the objects                 │
│                                                                                 │
│  Sampling Process:                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ 1. Compute camera height based on object heights                        │   │
│  │    height = min(max_object_top + 0.2m, 1.8m)                            │   │
│  │    height = max(height, 0.8m)                                           │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ 2. Random sampling (up to 300 attempts)                                 │   │
│  │    - Random yaw angle: 0 to 360 degrees                                 │   │
│  │    - Random distance: 0.5m to 4.0m from target                          │   │
│  │    - Camera position: (x, y) around object center                       │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ 3. Validity Checks                                                      │   │
│  │    [x] Within scene bounds (x, y only)                                  │   │
│  │    [x] Inside room polygon (if available)                               │   │
│  │    [x] Not colliding with any object                                    │   │
│  │    [x] Height within 0.8m to 1.8m                                       │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  Output: List[CameraPose] with position, target, yaw, pitch, radius             │
│          Example: 3 valid camera poses                                          │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        STEP 4: QUESTION GENERATION                              │
│                        (question_generator.py + question_utils.py)              │
├─────────────────────────────────────────────────────────────────────────────────┤
│  For each camera pose, generate questions based on visible objects:             │
│                                                                                 │
│  SINGLE OBJECT QUESTIONS (per object)                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ object_size:                                                            │   │
│  │   Q: "What is the length, width, height of the {sofa}?"                 │   │
│  │   A: "[1.8, 0.9, 0.8]" (from OBB dimensions)                            │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ object_distance_to_camera:                                              │   │
│  │   Q: "What is the distance of {sofa} from the camera?"                  │   │
│  │   A: "2.5" (Euclidean distance: camera_pos to object_center)            │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  PAIR OBJECT QUESTIONS (per object pair x 3 dimensions)                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ object_size_comparison_relative:                                        │   │
│  │   Q: "What is the ratio of the {length} of {sofa} to {chair}?"          │   │
│  │   A: "1.5" (sofa_length / chair_length)                                 │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ object_size_comparison_absolute:                                        │   │
│  │   Q: "If {chair} has length 1.2m, what is length of {sofa}?"            │   │
│  │   A: "1.8" (direct measurement)                                         │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ object_pair_distance_center:                                            │   │
│  │   Q: "What is distance between {sofa} and {chair}?"                     │   │
│  │   A: "2.1" (Euclidean: center_A to center_B)                            │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ object_pair_distance_vector:                                            │   │
│  │   Q: "What is vector from {sofa} to {chair} in camera coords?"          │   │
│  │   A: "[0.5, -0.3, 1.2]" (transformed to camera local frame)             │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
│  MULTI-OBJECT QUESTIONS (4 objects: A, B, X, Y)                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │ object_comparison_absolute_distance:                                    │   │
│  │   Q: "If distance X to Y is 2.0m, what is distance A to B?"             │   │
│  │   A: "3.5" (compute dist(A,B))                                          │   │
│  ├─────────────────────────────────────────────────────────────────────────┤   │
│  │ object_comparison_relative_distance:                                    │   │
│  │   Q: "What is ratio of dist(A,B) to dist(X,Y)?"                         │   │
│  │   A: "1.75" (dist(A,B) / dist(X,Y))                                     │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           STEP 5: OUTPUT                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│  output_dir/                                                                    │
│  └── {experiment_name}/                                                         │
│      └── {scene_id}/                                                            │
│          └── {move_pattern}/   # 'around', 'spherical', 'rotation', or 'linear' │
│              ├── {object_name}_{view_idx}.png  # For around/spherical/linear    │
│              ├── {room_name}_{yaw}deg.png      # For rotation mode              │
│              └── ...                                                            │
│  └── {scene_id}/               # Scene data folder                              │
│      ├── questions.jsonl       # All questions (one JSON per line)              │
│      └── metadata.json         # Statistics and config                          │
└─────────────────────────────────────────────────────────────────────────────────┘
```

## Question Types

### Single Object Questions

| Type | Description | Answer Format |
|------|-------------|---------------|
| `object_size` | Estimate length, width, height of an object | `[L, W, H]` in meters |
| `object_distance_to_camera` | Distance from object center to camera | Single number in meters |

### Pair Object Questions

| Type | Description | Answer Format |
|------|-------------|---------------|
| `object_size_comparison_relative` | Ratio of dimensions between two objects | Single ratio number |
| `object_size_comparison_absolute` | Object size given reference object dimension | Single number in meters |
| `object_pair_distance_center` | Distance between two object centers | Single number in meters |
| `object_pair_distance_center_w_size` | Distance with object size context | Single number in meters |
| `object_pair_distance_vector` | Vector from object A to B in camera coordinates | `[x, y, z]` in meters |

### Multi-Object Questions

| Type | Description | Answer Format |
|------|-------------|---------------|
| `object_comparison_absolute_distance` | Given distance A-B, find distance X-Y | Single number in meters |
| `object_comparison_relative_distance` | Ratio of distances between object pairs | Single ratio number |

## Usage

### Command Line Interface

```bash
# Process single scene (questions only, no rendering)
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790

# Process single scene WITH image rendering
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --enable_rendering

# Process all scenes with rendering
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --enable_rendering

# With custom options
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --num_cameras 5 \
    --max_questions_per_scene 500 \
    --question_types object_size object_pair_distance_center \
    --enable_rendering \
    --render_fov 60.0

# Example: Linear pattern with fixed camera orientation
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern linear \
    --linear_sub_pattern approach \
    --linear_num_steps 5 \
    --num_cameras 3 \
    --enable_rendering \
    --experiment_name linear_demo
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--scenes_root` | required | Root directory containing InteriorGS scene folders |
| `--output_dir` | required | Output directory for generated questions |
| `--scene_id` | None | Process only this single scene |
| `--scenes` | None | List of specific scenes to process |
| `--num_cameras` | 3 | Number of camera poses per object/pair |
| `--max_questions_per_scene` | 1000 | Maximum questions per scene |
| `--question_types` | all | Specific question types to generate |

### Rendering Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--enable_rendering` | False | Enable image rendering |
| `--render_backend` | local | Rendering backend: `local` or `client` |
| `--render_width` | 640 | Image width in pixels |
| `--render_height` | 480 | Image height in pixels |
| `--render_fov` | 60.0 | Field of view in degrees |
| `--gpu_device` | auto | GPU device ID for rendering |

### Python API

```python
from pipeline import InteriorGSQuestionPipeline
from config import PipelineConfig

# Create config
config = PipelineConfig(
    scenes_root='/path/to/InteriorGS',
    output_dir='/path/to/output',
)

# Run pipeline
pipeline = InteriorGSQuestionPipeline(config)
questions = pipeline.run()

# Or run for single scene
questions = pipeline.run_single_scene('0267_840790')
```

## Output Format

### Directory Structure

```
output_dir/
├── {experiment_name}/              # Experiment folder (e.g., "default")
│   └── {scene_id}/                 # Scene folder
│       └── {move_pattern}/         # Move pattern: "around", "spherical", "rotation", or "linear"
│           ├── {object_name}_{view_idx}.png  # For around/spherical/linear
│           ├── {room_name}_{yaw}deg.png      # For rotation mode
│           └── ...
├── {scene_id}/                     # Scene data folder
│   ├── questions.jsonl             # All questions (one JSON per line)
│   └── metadata.json               # Statistics and config
└── questions.jsonl                 # Combined questions from all scenes
```

### Question JSON Format

Each line in `questions.jsonl` contains:

```json
{
    "question": "What is the estimated length, width, and height of the sofa in this scene in meters?...",
    "answer": "[1.8, 0.9, 0.8]",
    "question_type": "object_size",
    "question_id": "object_size_sofa_123",
    "primary_object": "sofa_123",
    "objects": [
        {
            "id": "sofa_123",
            "label": "sofa",
            "center": [1.0, 2.0, 0.4],
            "dims": [1.8, 0.9, 0.8],
            "aabb_min": [...],
            "aabb_max": [...]
        }
    ],
    "camera_pose": {
        "position": [0.0, -2.0, 1.5],
        "target": [1.0, 2.0, 0.4],
        "yaw": 45.0,
        "pitch": -10.0,
        "radius": 2.5
    },
    "image": "default/0267_840790/around/sofa_0.png",
    "scene_id": "0267_840790"
}
```

### Metadata JSON Format

The `metadata.json` file contains:
- Generation timestamp
- Total question count
- Per-scene statistics
- Question type distribution
- Pipeline configuration

## Configuration

### Object Selection Config

```python
ObjectSelectionConfig(
    # Semantic blacklist - objects to exclude
    blacklist={'wall', 'floor', 'ceiling', 'door', 'window', 'lamp', ...},
    
    # Geometric filter toggles (set False to disable)
    enable_dim_filter=False,           # Dimension filtering
    enable_volume_filter=False,        # Volume filtering
    enable_aspect_ratio_filter=False,  # Aspect ratio filtering
    
    # Geometric constraints (when filters enabled)
    min_dim_component=0.1,   # Min dimension per axis (m)
    max_dim_component=3.0,   # Max dimension per axis (m)
    min_volume=0.01,         # Min volume (m^3)
    min_aspect_ratio=0.05,   # Min shortest/longest edge ratio
    
    # Pair constraints
    min_pair_dist=0.3,       # Min distance between pairs (m)
    max_pair_dist=5.0,       # Max distance between pairs (m)
    max_pair_dim_ratio=3.0,  # Max ratio of longest edges
    max_pair_dim_diff=2.5,   # Max absolute difference in longest edge (m)
)
```

### Camera Sampling Config

```python
CameraSamplingConfig(
    num_cameras_per_item=3,    # Camera poses per object/pair
    max_tries=300,             # Max sampling attempts
    
    # Height constraints
    min_camera_height=0.8,     # Min camera height (m)
    max_camera_height=1.8,     # Max camera height (m)
    camera_height_offset=0.2,  # Height offset above object top
    
    # Distance constraints
    min_distance=0.5,          # Min distance to objects (m)
    max_distance=4.0,          # Max distance to objects (m)
    
    # Camera intrinsics
    fov_deg=60.0,              # Field of view (degrees)
    image_width=640,
    image_height=480,
    
    # Move pattern: 'around', 'spherical', 'rotation', or 'linear'
    # - 'around': Horizontal circle around object (default)
    # - 'spherical': Sample on sphere surface around object
    # - 'rotation': Stand at room center, rotate 360°
    # - 'linear': Walk toward or past object in straight line
    move_pattern='around',
    
    # Rotation mode parameters (only used when move_pattern='rotation')
    rotation_interval=5.0,       # Degrees between each camera pose (360/5 = 72 images)
    rotation_camera_height=1.5,  # Fixed camera height (m)
    
    # Linear mode parameters (only used when move_pattern='linear')
    # Camera orientation stays FIXED, only position changes along straight line
    linear_sub_pattern='approach',  # 'approach' (walk toward) or 'pass_by' (walk past)
    linear_num_steps=5,             # Number of poses along trajectory
)
```

See [Move Patterns](#move-patterns-camera-trajectory-types) section below for detailed move pattern documentation.
```

## Move Patterns (Camera Trajectory Types)

Move patterns determine how camera poses are sampled to observe target objects. Different patterns simulate various ways humans observe objects in indoor environments, generating diverse training data.

### Why 4 Move Patterns?

| Simulated Behavior | Move Pattern | Description |
|-------------------|--------------|-------------|
| 🚶 **Walk around object** | `around` | Circle around an object, always facing it |
| 🌐 **Multi-angle observation** | `spherical` | Observe from different heights and angles (crouch, stand, look down) |
| 🔄 **Stand and look around** | `rotation` | Stand at room center, rotate 360° to look around |
| 🚶‍♂️ **Approach/pass by object** | `linear` | Walk toward object or pass by it |

### Move Pattern Details

---

#### 1. `around` - Horizontal Circle Mode (Default)

**Scenario**: Person circles around an object (e.g., sofa, table), always facing it.

**Sampling Logic**:
- Object-centric
- Sample camera positions uniformly on horizontal plane (fixed height)
- Camera always points toward object center

**Diagram**:
```
        [cam1]
           ↘
    [cam4] → [object] ← [cam2]
           ↗
        [cam3]
```

**Command Example**:
```bash
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern around \
    --num_cameras 5
```

**Output Naming**: `{object_name}_{view_idx}.png`
- Example: `sofa_0.png`, `sofa_1.png`, `sofa_2.png`

---

#### 2. `spherical` - Spherical Sampling Mode

**Scenario**: Observe object from various heights and angles (crouch down, stand up, look from above).

**Sampling Logic**:
- Object-centric
- Sample camera positions uniformly on sphere surface
- Camera can be at different heights (above, level with, below object)
- Camera always points toward object center

**Diagram**:
```
       [cam_top]
          ↓
    [cam] → [object] ← [cam]
          ↑
      [cam_low]
```

**Command Example**:
```bash
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern spherical \
    --num_cameras 8
```

**Output Naming**: `{object_name}_{view_idx}.png`
- Example: `table_0.png`, `table_1.png`, ...

---

#### 3. `rotation` - Room Center Rotation Mode

**Scenario**: Stand at room center, rotate 360° in place to look around at all objects.

**Sampling Logic**:
- Room-centric (not object-centric)
- Camera fixed at room center point
- Rotate at fixed angle intervals (e.g., every 5°), generating 72 images
- Detect visible objects in each image and generate questions for them

**Diagram**:
```
    ┌─────────────────────┐
    │  [sofa]     [TV]    │
    │                     │
    │      [camera]←→360° │
    │         ↑           │
    │  [table]   [chair]  │
    └─────────────────────┘
```

**Special Parameters**:
- `rotation_interval`: Rotation interval in degrees (default 5°, 360/5=72 images)
- `rotation_camera_height`: Fixed camera height (default 1.5m)

**Command Example**:
```bash
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern rotation \
    --rotation_interval 10.0 \
    --rotation_camera_height 1.6
```

**Output Naming**: `{room_name}_{yaw_deg}deg.png`
- Example: `bedroom_000deg.png`, `bedroom_005deg.png`, ..., `bedroom_355deg.png`

---

#### 4. `linear` - Linear Trajectory Mode

**Scenario**: Simulate walking toward an object or passing by it.

**Key Feature**: Camera orientation (yaw/pitch) remains **FIXED** throughout the trajectory. Only camera **POSITION** changes along a straight line.

**Sampling Logic**:
- Sample multiple camera positions along a straight line path
- Camera always looks in the same direction (no head turning)
- Two sub-patterns simulate different walking behaviors

**Sub-patterns (linear_sub_pattern)**:

| Sub-pattern | Description | Visual Effect |
|-------------|-------------|---------------|
| `approach` | Walk toward object in a straight line | Object gets larger, stays roughly centered |
| `pass_by` | Walk along a line that passes by object | Object moves across field of view |

**approach Diagram** (camera walks toward object):
```
                      Fixed look direction
                            ↓
    [cam1]----[cam2]----[cam3]----[cam4]---→ [object]
       ├─────────────────────────────────────→
       Position moves, orientation stays fixed
```

**pass_by Diagram** (camera walks past object):
```
                 Fixed look direction
                        ↓
    [cam1]----[cam2]----[cam3]----[cam4]
       │         │         │         │
       ↓         ↓         ↓         ↓
    (all cameras look in same direction)
              [object on side] ←─ appears to move across FOV
```

**Special Parameters**:
- `linear_sub_pattern`: Sub-pattern, `approach` or `pass_by`
- `linear_num_steps`: Number of camera poses along trajectory (default 5)

**Command Example**:
```bash
# Walk toward object (approach)
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern linear \
    --linear_sub_pattern approach \
    --linear_num_steps 5

# Walk past object (pass_by)
python run_pipeline.py \
    --scenes_root /path/to/InteriorGS \
    --output_dir /path/to/output \
    --scene_id 0267_840790 \
    --move_pattern linear \
    --linear_sub_pattern pass_by \
    --linear_num_steps 7
```

**Output Naming**: `{object_name}_{step_idx}.png`
- Example: `chair_0.png`, `chair_1.png`, ..., `chair_4.png`

---

### Move Pattern Comparison Summary

| Property | `around` | `spherical` | `rotation` | `linear` |
|----------|----------|-------------|------------|----------|
| **Center** | Object | Object | Room | Object |
| **Camera Movement** | Horizontal circle | Sphere surface | Fixed + rotate | Straight line |
| **Camera Direction** | Object center | Object center | Rotation direction | Fixed (no head turn) |
| **Height Variation** | Fixed | Varies | Fixed | Fixed |
| **Use Case** | Multi-angle single object | Full 3D observation | Room panorama | Approach/pass by |
| **Output Count** | N per object | N per object | 72 per room | N per object |

### Configuration Example

```python
CameraSamplingConfig(
    # Move pattern: 'around', 'spherical', 'rotation', 'linear'
    move_pattern='around',
    
    # Common parameters
    num_cameras_per_item=3,    # Cameras per object (around/spherical/linear)
    min_distance=0.5,          # Min distance (m)
    max_distance=4.0,          # Max distance (m)
    min_camera_height=0.8,     # Min camera height (m)
    max_camera_height=1.8,     # Max camera height (m)
    
    # Rotation mode parameters
    rotation_interval=5.0,       # Rotation interval degrees (360/5 = 72 images)
    rotation_camera_height=1.5,  # Fixed camera height (m)
    
    # Linear mode parameters
    linear_sub_pattern='center_forward',  # 'center_forward' or 'side_sweep'
    linear_num_steps=5,                   # Number of poses along trajectory
)

### Question Config

```python
QuestionConfig(
    enabled_question_types=[
        'object_size',
        'object_distance_to_camera',
        'object_size_comparison_relative',
        'object_size_comparison_absolute',
        'object_pair_distance_center',
        'object_pair_distance_center_w_size',
        'object_pair_distance_vector',
        'object_comparison_absolute_distance',
        'object_comparison_relative_distance',
    ],
    dimensions=['length', 'width', 'height'],
    max_questions_per_type=5,
)
```

## Rendering Images

### Render Images for Existing Questions

If you generated questions without images, you can render them separately:

```bash
# Using the shell script
./render_images.sh 0267_840790       # Single scene
./render_images.sh --all             # All scenes

# Using Python directly
python render_existing_questions.py \
    --data_dir /path/to/output/0267_840790 \
    --scenes_root /path/to/InteriorGS \
    --width 640 --height 480 --fov 60

# Render all scenes
python render_existing_questions.py \
    --data_root /path/to/output \
    --scenes_root /path/to/InteriorGS
```

### Rendering Options

| Option | Default | Description |
|--------|---------|-------------|
| `--data_dir` | - | Path to a single scene data directory |
| `--data_root` | - | Root directory with multiple scene directories |
| `--scenes_root` | required | InteriorGS root directory |
| `--width` | 640 | Image width in pixels |
| `--height` | 480 | Image height in pixels |
| `--fov` | 60 | Field of view in degrees |
| `--gpu_device` | auto | GPU device ID |
| `--skip_existing` | True | Skip already rendered images |
| `--no_backup` | False | Don't backup original questions.jsonl |

### Rendering Process

1. Reads `questions.jsonl` from the data directory
2. Extracts unique camera poses (by `camera_pose_idx`)
3. Renders an image for each camera pose
4. Updates `questions.jsonl` with `"image": "images/pose_xxx.png"` paths
5. Saves backup as `questions_backup.jsonl`
6. Creates `render_log.json` with statistics

## Project Structure

```
question_gen_InteriorGS/
├── __init__.py                    # Package exports
├── config.py                      # Configuration dataclasses
├── object_selector.py             # Object filtering and selection
├── camera_sampler.py              # Camera pose sampling
├── camera_utils.py                # Camera utilities (FOV, occlusion)
├── question_templates.py          # Question text templates
├── question_utils.py              # Question construction utilities
├── question_generator.py          # Question generation logic
├── pipeline.py                    # Main pipeline orchestration
├── run_pipeline.py                # CLI entry point
├── render_utils.py                # Rendering utilities (SceneRenderer)
├── render_existing_questions.py   # Render images for existing questions
├── render_images.sh               # Quick render script
├── example_config.json            # Example configuration file
└── README.md                      # This file
```

## Data Requirements

The pipeline expects InteriorGS scenes with the following structure:

### Required Files

**labels.json** - Object annotations with 3D bounding boxes:
```json
[
    {
        "ins_id": "chair_1",
        "label": "chair",
        "bounding_box": [
            {"x": 0.0, "y": 0.0, "z": 0.0},
            {"x": 0.5, "y": 0.0, "z": 0.0},
            {"x": 0.5, "y": 0.5, "z": 0.0},
            {"x": 0.0, "y": 0.5, "z": 0.0},
            {"x": 0.0, "y": 0.0, "z": 0.8},
            {"x": 0.5, "y": 0.0, "z": 0.8},
            {"x": 0.5, "y": 0.5, "z": 0.8},
            {"x": 0.0, "y": 0.5, "z": 0.8}
        ]
    }
]
```

### Optional Files

- **structure.json** - Room polygons for spatial validation
- **occupancy.json** - Scene bounds for camera position validation

## Example Statistics

For a typical scene (e.g., `0267_840790`):

```
Raw Input:  459 objects in labels.json

Step 1 (Object Selection):
  - After semantic filter (blacklist)
  - After geometric filter (size, volume, aspect ratio)
  - After uniqueness filter (avoid ambiguous labels)
  => 36 valid single objects

Step 2 (Pair Selection):
  - Pair constraints (distance, size ratio, same room)
  => 46 valid object pairs

Step 3 (Camera Sampling):
  - Position validation, collision check
  => 3 valid camera poses per item

Step 4 (Question Generation):
  => 585 questions total
     ├── object_size: 30
     ├── object_distance_to_camera: 30
     ├── object_size_comparison_relative: 135
     ├── object_size_comparison_absolute: 135
     ├── object_pair_distance_center: 45
     ├── object_pair_distance_center_w_size: 135
     ├── object_pair_distance_vector: 45
     ├── object_comparison_absolute_distance: 15
     └── object_comparison_relative_distance: 15
```
