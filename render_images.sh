#!/bin/bash
# Render images for existing VQA questions
#
# This script renders images for already generated VQA data.
# 
# Usage:
#   ./render_images.sh <scene_id>
#   ./render_images.sh --all
#
# Examples:
#   ./render_images.sh 0267_840790
#   ./render_images.sh --all

set -e

# Configuration - modify these paths as needed
# You can set environment variables SCENES_ROOT and DATA_ROOT to override defaults
SCENES_ROOT="${SCENES_ROOT:-/path/to/InteriorGS}"
DATA_ROOT="${DATA_ROOT:-/path/to/output}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Default rendering settings
WIDTH=640
HEIGHT=480
FOV=60

# Parse arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <scene_id> | --all"
    echo ""
    echo "Options:"
    echo "  <scene_id>  Process a single scene (e.g., 0267_840790)"
    echo "  --all       Process all scenes in data directory"
    echo ""
    echo "Environment variables:"
    echo "  SCENES_ROOT  Path to InteriorGS dataset (current: ${SCENES_ROOT})"
    echo "  DATA_ROOT    Path to output directory (current: ${DATA_ROOT})"
    exit 1
fi

if [ "$1" = "--all" ]; then
    # Process all scenes
    echo "Rendering images for all scenes in ${DATA_ROOT}"
    python "${SCRIPT_DIR}/render_existing_questions.py" \
        --data_root "${DATA_ROOT}" \
        --scenes_root "${SCENES_ROOT}" \
        --width ${WIDTH} \
        --height ${HEIGHT} \
        --fov ${FOV} \
        --skip_existing
else
    # Process single scene
    SCENE_ID=$1
    DATA_DIR="${DATA_ROOT}/${SCENE_ID}"
    
    if [ ! -d "${DATA_DIR}" ]; then
        echo "Error: Scene directory not found: ${DATA_DIR}"
        exit 1
    fi
    
    echo "Rendering images for scene: ${SCENE_ID}"
    python "${SCRIPT_DIR}/render_existing_questions.py" \
        --data_dir "${DATA_DIR}" \
        --scenes_root "${SCENES_ROOT}" \
        --width ${WIDTH} \
        --height ${HEIGHT} \
        --fov ${FOV} \
        --skip_existing
fi

echo ""
echo "Done!"
