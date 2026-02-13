#!/bin/bash
# Batch generation script for 4 scenes with all patterns
# Scenes: 0015_840888, 0042_839881, 0100_839971, 0200_840150

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENES_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_BASE="/scratch/by2593/project/sceneshift/vqa_demo_all"

# Scenes to process (excluding 0267_840790)
SCENES=("0015_840888" "0042_839881" "0100_839971" "0200_840150","0267_840790")

# Patterns to generate
PATTERNS=("around" "spherical" "linear_approach" "linear_passby")

echo "========================================="
echo "Batch VQA Generation for 4 Scenes"
echo "========================================="
echo "Scenes: ${SCENES[@]}"
echo "Patterns: ${PATTERNS[@]}"
echo "Output: ${OUTPUT_BASE}"
echo ""

cd "$SCRIPT_DIR"

for scene in "${SCENES[@]}"; do
    echo "========================================="
    echo "Processing scene: ${scene}"
    echo "========================================="
    
    for pattern in "${PATTERNS[@]}"; do
        echo ""
        echo "--- Pattern: ${pattern} ---"
        
        # Determine pattern-specific output directory
        OUTPUT_DIR="${OUTPUT_BASE}/${pattern}/${scene}"
        
        # Set up pattern-specific arguments
        if [[ "$pattern" == "around" ]]; then
            MOVE_PATTERN="around"
            EXTRA_ARGS=""
        elif [[ "$pattern" == "spherical" ]]; then
            MOVE_PATTERN="spherical"
            EXTRA_ARGS=""
        elif [[ "$pattern" == "linear_approach" ]]; then
            MOVE_PATTERN="linear"
            EXTRA_ARGS="--linear_sub_pattern approach --linear_num_steps 5 --linear_move_distance 0.3"
        elif [[ "$pattern" == "linear_passby" ]]; then
            MOVE_PATTERN="linear"
            EXTRA_ARGS="--linear_sub_pattern pass_by --linear_num_steps 5 --linear_move_distance 0.3"
        fi
        
        echo "Running pipeline for ${scene} with ${pattern}..."
        
        python run_pipeline.py \
            --scenes_root "${SCENES_ROOT}" \
            --output_dir "${OUTPUT_DIR}" \
            --scene_id "${scene}" \
            --move_pattern "${MOVE_PATTERN}" \
            ${EXTRA_ARGS} \
            --num_cameras 10 \
            --enable_rendering \
            --experiment_name "${pattern}" \
            --verbose
        
        echo "Completed ${scene} / ${pattern}"
    done
done

echo ""
echo "========================================="
echo "All scene/pattern combinations completed!"
echo "========================================="
