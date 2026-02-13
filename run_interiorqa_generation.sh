#!/bin/bash
# =============================================================================
# InteriorQA Dataset Generation Script
# 
# This script generates all move patterns for 5 selected scenes and creates demos.
# Usage: bash run_interiorqa_generation.sh
# =============================================================================

set -e  # Exit on error

# Activate conda environment
source /scratch/by2593/miniconda3/bin/activate vagen

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCENES_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_BASE="/scratch/by2593/project/sceneshift/data/InteriorQA"

# Selected scenes (scene_267 + 4 random scenes)
SCENES=(
    "0267_840790"
    "0414_840040"
    "0400_840162"
    "0048_839893"
    "0363_840304"
)

# Move patterns
PATTERNS=("around" "spherical" "linear")

# Settings
NUM_CAMERAS=5
MAX_QUESTIONS_PER_TYPE=10
ENABLE_RENDERING=true  # Set to true to enable rendering

echo "============================================================"
echo "InteriorQA Dataset Generation"
echo "============================================================"
echo "Scenes: ${SCENES[*]}"
echo "Patterns: ${PATTERNS[*]}"
echo "Output: ${OUTPUT_BASE}"
echo "Rendering: ${ENABLE_RENDERING}"
echo "============================================================"

# Create output directory
mkdir -p "${OUTPUT_BASE}"

# Function to run pipeline for a scene and pattern
run_pipeline() {
    local scene_id=$1
    local move_pattern=$2
    local output_dir=$3
    local linear_sub_pattern=$4
    
    echo ""
    echo "============================================================"
    echo "Scene: ${scene_id} | Pattern: ${move_pattern} ${linear_sub_pattern}"
    echo "Output: ${output_dir}"
    echo "============================================================"
    
    local cmd="python ${SCRIPT_DIR}/run_pipeline.py \
        --scenes_root ${SCENES_ROOT} \
        --output_dir ${output_dir} \
        --scene_id ${scene_id} \
        --move_pattern ${move_pattern} \
        --num_cameras ${NUM_CAMERAS} \
        --max_questions_per_type ${MAX_QUESTIONS_PER_TYPE} \
        --flat_output"
    
    if [ "${ENABLE_RENDERING}" = true ]; then
        cmd="${cmd} --enable_rendering"
    fi
    
    if [ -n "${linear_sub_pattern}" ]; then
        cmd="${cmd} --linear_sub_pattern ${linear_sub_pattern} --linear_num_steps 5"
    fi
    
    eval ${cmd}
}

# Function to create demos for a pattern
create_demos() {
    local questions_file=$1
    local output_root=$2
    local pattern=$3
    local demo_dir=$4
    
    if [ ! -f "${questions_file}" ]; then
        echo "Warning: Questions file not found: ${questions_file}"
        return
    fi
    
    mkdir -p "${demo_dir}"
    
    # Get available objects from questions file
    local objects=$(python -c "
import json
objects = set()
with open('${questions_file}', 'r') as f:
    for line in f:
        if line.strip():
            q = json.loads(line)
            if 'objects' in q and len(q['objects']) > 0:
                label = q['objects'][0].get('label', '')
                if label:
                    objects.add(label)
import random
obj_list = list(objects)
random.shuffle(obj_list)
for o in obj_list[:3]:
    print(o)
" 2>/dev/null)
    
    if [ -z "${objects}" ]; then
        echo "Warning: No objects found in ${questions_file}"
        return
    fi
    
    echo "Creating demos for pattern: ${pattern}"
    
    # Create demo for each object
    while IFS= read -r obj_label; do
        if [ -n "${obj_label}" ]; then
            safe_label=$(echo "${obj_label}" | tr ' /' '__')
            demo_file="${demo_dir}/demo_${pattern}_${safe_label}.png"
            
            echo "  Creating demo for object: ${obj_label}"
            python "${SCRIPT_DIR}/create_demo.py" \
                --questions_file "${questions_file}" \
                --output_file "${demo_file}" \
                --output_root "${output_root}" \
                --pattern "${pattern}" \
                --object_label "${obj_label}" \
                --num_views 5 \
                2>/dev/null || echo "    Warning: Demo creation failed for ${obj_label}"
        fi
    done <<< "${objects}"
}

# Main generation loop
for scene_id in "${SCENES[@]}"; do
    scene_output_base="${OUTPUT_BASE}/${scene_id}"
    
    for move_pattern in "${PATTERNS[@]}"; do
        if [ "${move_pattern}" = "linear" ]; then
            # Generate both linear sub-patterns
            for sub_pattern in "approach" "pass_by"; do
                pattern_name="linear_${sub_pattern}"
                pattern_output_dir="${scene_output_base}/${pattern_name}"
                
                run_pipeline "${scene_id}" "${move_pattern}" "${pattern_output_dir}" "${sub_pattern}"
                
                # Create demos - questions are now directly in pattern folder
                questions_file="${pattern_output_dir}/questions.jsonl"
                demo_dir="${pattern_output_dir}/demos"
                create_demos "${questions_file}" "${pattern_output_dir}" "${move_pattern}" "${demo_dir}"
            done
        else
            pattern_output_dir="${scene_output_base}/${move_pattern}"
            
            run_pipeline "${scene_id}" "${move_pattern}" "${pattern_output_dir}" ""
            
            # Create demos - questions are now directly in pattern folder
            questions_file="${pattern_output_dir}/questions.jsonl"
            demo_dir="${pattern_output_dir}/demos"
            create_demos "${questions_file}" "${pattern_output_dir}" "${move_pattern}" "${demo_dir}"
        fi
    done
done

echo ""
echo "============================================================"
echo "GENERATION COMPLETE"
echo "============================================================"
echo "Output directory: ${OUTPUT_BASE}"
echo ""
echo "Directory structure:"
echo "  {scene_id}/"
echo "    around/"
echo "      questions.jsonl + metadata.json + images/"
echo "      demos/"
echo "    spherical/"
echo "      questions.jsonl + metadata.json + images/"
echo "      demos/"
echo "    linear_approach/"
echo "      questions.jsonl + metadata.json + images/"
echo "      demos/"
echo "    linear_pass_by/"
echo "      questions.jsonl + metadata.json + images/"
echo "      demos/"
echo ""

# Print summary statistics
echo "Summary statistics:"
for scene_id in "${SCENES[@]}"; do
    echo "  Scene: ${scene_id}"
    for pattern_dir in around spherical linear_approach linear_pass_by; do
        qfile="${OUTPUT_BASE}/${scene_id}/${pattern_dir}/questions.jsonl"
        if [ -f "${qfile}" ]; then
            count=$(wc -l < "${qfile}")
            echo "    ${pattern_dir}: ${count} questions"
        fi
    done
done

echo ""
echo "Done!"
