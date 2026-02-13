#!/bin/bash
# Batch generation script for 4 scenes with all patterns
# With RESUME functionality - checks existing outputs and continues from where it left off
# Scenes: 0015_840888, 0042_839881, 0100_839971, 0200_840150
# bash generate_all_scenes_resume.sh [--check] [--force] [--scene SCENE_ID] [--pattern PATTERN]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENES_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_BASE="/scratch/by2593/project/sceneshift/datasets/InteriorGS_5scenes_camerafix_v2"

# Scenes to process (excluding 0267_840790)
# Note: 0100_839971 was replaced with 0106_839990 because it had no valid objects
SCENES=("0015_840888" "0042_839881" "0106_839990" "0200_840150" "0267_840790")

# Patterns to generate
PATTERNS=("around" "spherical" "linear_approach" "linear_passby" "rotation")

# Default configurable parameters
NUM_CAMERAS=15                    # Number of camera views per object/pair
MAX_QUESTIONS_PER_SCENE=50000     # Maximum questions per scene (increased to avoid truncating multi-object questions)
MAX_QUESTIONS_PER_TYPE=5          # Maximum questions per type per camera pose
LINEAR_NUM_STEPS=5                # Number of steps for linear trajectory
LINEAR_MOVE_DISTANCE=0.3          # Distance for linear movement (meters)
MAX_TRIES=200                     # Maximum camera sampling attempts
MIN_VIEWS_REQUIRED=10              # Minimum views required (0 = no filtering)
RENDER_WIDTH=512                  # Rendered image width
RENDER_HEIGHT=512                 # Rendered image height
RENDER_FOV=60.0                   # Field of view for rendering

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse command line arguments
CHECK_ONLY=false
FORCE_RERUN=false
SPECIFIC_SCENE=""
SPECIFIC_PATTERN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --check)
            CHECK_ONLY=true
            shift
            ;;
        --force)
            FORCE_RERUN=true
            shift
            ;;
        --scene)
            SPECIFIC_SCENE="$2"
            shift 2
            ;;
        --pattern)
            SPECIFIC_PATTERN="$2"
            shift 2
            ;;
        --num_cameras)
            NUM_CAMERAS="$2"
            shift 2
            ;;
        --max_questions)
            MAX_QUESTIONS_PER_SCENE="$2"
            shift 2
            ;;
        --max_questions_per_type)
            MAX_QUESTIONS_PER_TYPE="$2"
            shift 2
            ;;
        --linear_steps)
            LINEAR_NUM_STEPS="$2"
            shift 2
            ;;
        --linear_distance)
            LINEAR_MOVE_DISTANCE="$2"
            shift 2
            ;;
        --max_tries)
            MAX_TRIES="$2"
            shift 2
            ;;
        --min_views)
            MIN_VIEWS_REQUIRED="$2"
            shift 2
            ;;
        --render_width)
            RENDER_WIDTH="$2"
            shift 2
            ;;
        --render_height)
            RENDER_HEIGHT="$2"
            shift 2
            ;;
        --render_fov)
            RENDER_FOV="$2"
            shift 2
            ;;
        --output_base)
            OUTPUT_BASE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Control Options:"
            echo "  --check                     Only check status, don't run anything"
            echo "  --force                     Force re-run even if already completed"
            echo "  --scene SCENE_ID            Only process specific scene"
            echo "  --pattern PATTERN           Only process specific pattern (around, spherical, linear_approach, linear_passby)"
            echo ""
            echo "Generation Parameters:"
            echo "  --num_cameras N             Number of camera views per object/pair (default: ${NUM_CAMERAS})"
            echo "  --max_questions N           Maximum questions per scene (default: ${MAX_QUESTIONS_PER_SCENE})"
            echo "  --max_questions_per_type N  Maximum questions per type per camera pose (default: ${MAX_QUESTIONS_PER_TYPE})"
            echo "  --max_tries N               Maximum camera sampling attempts (default: ${MAX_TRIES})"
            echo "  --min_views N               Minimum views required per question, filter out if less (default: ${MIN_VIEWS_REQUIRED}, 0=no filter)"
            echo ""
            echo "Linear Pattern Parameters:"
            echo "  --linear_steps N            Number of steps for linear trajectory (default: ${LINEAR_NUM_STEPS})"
            echo "  --linear_distance D         Distance for linear movement in meters (default: ${LINEAR_MOVE_DISTANCE})"
            echo ""
            echo "Rendering Parameters:"
            echo "  --render_width W            Rendered image width (default: ${RENDER_WIDTH})"
            echo "  --render_height H           Rendered image height (default: ${RENDER_HEIGHT})"
            echo "  --render_fov FOV            Field of view for rendering in degrees (default: ${RENDER_FOV})"
            echo ""
            echo "Output Parameters:"
            echo "  --output_base PATH          Base output directory (default: ${OUTPUT_BASE})"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Resume from where it left off"
            echo "  $0 --check                            # Only show status"
            echo "  $0 --scene 0042_839881                # Only run scene 0042_839881"
            echo "  $0 --pattern spherical                # Only run spherical pattern"
            echo "  $0 --force                            # Force re-run everything"
            echo "  $0 --num_cameras 15 --max_questions 2000  # More cameras and questions"
            echo "  $0 --min_views 10                     # Only keep questions with 10+ views"
            echo "  $0 --max_tries 500                    # Try harder to find valid camera poses"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Function to check if a scene/pattern combination is complete
# Returns:
#   0 - Complete (metadata.json exists AND has images)
#   1 - Not started (no output directory)
#   2 - Partial (directory exists but missing files)
check_completion() {
    local scene="$1"
    local pattern="$2"
    local output_dir="${OUTPUT_BASE}/${pattern}/${scene}"
    
    # Check if output directory exists
    if [[ ! -d "$output_dir" ]]; then
        echo "1"  # Not started
        return
    fi
    
    # Find metadata.json - it can be in the root or in a subdirectory
    local metadata_file=""
    if [[ -f "${output_dir}/metadata.json" ]]; then
        metadata_file="${output_dir}/metadata.json"
    elif [[ -f "${output_dir}/${scene}/metadata.json" ]]; then
        metadata_file="${output_dir}/${scene}/metadata.json"
    fi
    
    # Find questions.jsonl
    local questions_file=""
    if [[ -f "${output_dir}/questions.jsonl" ]]; then
        questions_file="${output_dir}/questions.jsonl"
    elif [[ -f "${output_dir}/${scene}/questions.jsonl" ]]; then
        questions_file="${output_dir}/${scene}/questions.jsonl"
    fi
    
    # Check if both metadata and questions exist
    if [[ -n "$metadata_file" && -n "$questions_file" ]]; then
        # Check if questions file has content
        local num_questions=$(wc -l < "$questions_file" 2>/dev/null || echo "0")
        if [[ $num_questions -gt 0 ]]; then
            # Check if images exist - just count all pngs in the output directory
            local num_images=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
            if [[ $num_images -gt 0 ]]; then
                echo "0"  # Complete
                return
            fi
        fi
    fi
    
    echo "2"  # Partial
}

# Function to get detailed status
get_status_details() {
    local scene="$1"
    local pattern="$2"
    local output_dir="${OUTPUT_BASE}/${pattern}/${scene}"
    
    local questions_count=0
    local images_count=0
    local has_metadata="no"
    
    # Find and count questions
    local questions_file=""
    if [[ -f "${output_dir}/questions.jsonl" ]]; then
        questions_file="${output_dir}/questions.jsonl"
    elif [[ -f "${output_dir}/${scene}/questions.jsonl" ]]; then
        questions_file="${output_dir}/${scene}/questions.jsonl"
    fi
    
    if [[ -n "$questions_file" && -f "$questions_file" ]]; then
        questions_count=$(wc -l < "$questions_file" 2>/dev/null || echo "0")
    fi
    
    # Check metadata
    if [[ -f "${output_dir}/metadata.json" ]] || [[ -f "${output_dir}/${scene}/metadata.json" ]]; then
        has_metadata="yes"
    fi
    
    # Count images
    images_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
    
    echo "Q:${questions_count}|I:${images_count}|M:${has_metadata}"
}

echo "========================================="
echo "Batch VQA Generation with Resume Support"
echo "========================================="
echo "Scenes: ${SCENES[@]}"
echo "Patterns: ${PATTERNS[@]}"
echo "Output: ${OUTPUT_BASE}"
echo ""
echo "Generation Parameters:"
echo "  num_cameras: ${NUM_CAMERAS}"
echo "  max_questions_per_scene: ${MAX_QUESTIONS_PER_SCENE}"
echo "  max_questions_per_type: ${MAX_QUESTIONS_PER_TYPE}"
echo "  max_tries: ${MAX_TRIES}"
echo "  min_views_required: ${MIN_VIEWS_REQUIRED}"
echo "  linear_steps: ${LINEAR_NUM_STEPS}, linear_distance: ${LINEAR_MOVE_DISTANCE}"
echo "  render: ${RENDER_WIDTH}x${RENDER_HEIGHT}, fov=${RENDER_FOV}"
echo ""

# Filter scenes and patterns if specified
if [[ -n "$SPECIFIC_SCENE" ]]; then
    SCENES=("$SPECIFIC_SCENE")
    echo "Filtering to scene: $SPECIFIC_SCENE"
fi

if [[ -n "$SPECIFIC_PATTERN" ]]; then
    PATTERNS=("$SPECIFIC_PATTERN")
    echo "Filtering to pattern: $SPECIFIC_PATTERN"
fi

echo ""
echo "========================================="
echo "Checking Completion Status..."
echo "========================================="

# Arrays to track status
declare -a COMPLETED=()
declare -a PENDING=()
declare -a PARTIAL=()

for scene in "${SCENES[@]}"; do
    for pattern in "${PATTERNS[@]}"; do
        status=$(check_completion "$scene" "$pattern")
        details=$(get_status_details "$scene" "$pattern")
        
        case $status in
            0)
                COMPLETED+=("${scene}/${pattern}")
                echo -e "${GREEN}[COMPLETE]${NC} ${scene} / ${pattern} - ${details}"
                ;;
            1)
                PENDING+=("${scene}/${pattern}")
                echo -e "${YELLOW}[PENDING]${NC}  ${scene} / ${pattern} - Not started"
                ;;
            2)
                PARTIAL+=("${scene}/${pattern}")
                echo -e "${BLUE}[PARTIAL]${NC}  ${scene} / ${pattern} - ${details}"
                ;;
        esac
    done
done

echo ""
echo "========================================="
echo "Summary"
echo "========================================="
echo -e "${GREEN}Completed:${NC} ${#COMPLETED[@]} combinations"
echo -e "${YELLOW}Pending:${NC}   ${#PENDING[@]} combinations"
echo -e "${BLUE}Partial:${NC}   ${#PARTIAL[@]} combinations"
echo "Total:     $((${#COMPLETED[@]} + ${#PENDING[@]} + ${#PARTIAL[@]})) combinations"

if [[ $CHECK_ONLY == true ]]; then
    echo ""
    echo "Check-only mode. Exiting."
    exit 0
fi

# Determine what needs to run
declare -a TO_RUN=()
if [[ $FORCE_RERUN == true ]]; then
    for scene in "${SCENES[@]}"; do
        for pattern in "${PATTERNS[@]}"; do
            TO_RUN+=("${scene}|${pattern}")
        done
    done
    echo ""
    echo -e "${RED}Force mode: Will re-run ALL combinations${NC}"
else
    # Add pending and partial to run list
    for item in "${PENDING[@]}" "${PARTIAL[@]}"; do
        scene="${item%/*}"
        pattern="${item#*/}"
        TO_RUN+=("${scene}|${pattern}")
    done
fi

if [[ ${#TO_RUN[@]} -eq 0 ]]; then
    echo ""
    echo -e "${GREEN}All combinations are complete! Nothing to run.${NC}"
    exit 0
fi

echo ""
echo "========================================="
echo "Will process ${#TO_RUN[@]} combinations"
echo "========================================="

cd "$SCRIPT_DIR"

# Process each combination
for item in "${TO_RUN[@]}"; do
    scene="${item%|*}"
    pattern="${item#*|}"
    
    echo ""
    echo "========================================="
    echo "Processing: ${scene} / ${pattern}"
    echo "========================================="
    
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
        EXTRA_ARGS="--linear_sub_pattern approach --linear_num_steps ${LINEAR_NUM_STEPS} --linear_move_distance ${LINEAR_MOVE_DISTANCE}"
    elif [[ "$pattern" == "linear_passby" ]]; then
        MOVE_PATTERN="linear"
        EXTRA_ARGS="--linear_sub_pattern pass_by --linear_num_steps ${LINEAR_NUM_STEPS} --linear_move_distance ${LINEAR_MOVE_DISTANCE}"
    fi
    
    # Build min_views argument if specified
    MIN_VIEWS_ARG=""
    if [[ ${MIN_VIEWS_REQUIRED} -gt 0 ]]; then
        MIN_VIEWS_ARG="--min_views_required ${MIN_VIEWS_REQUIRED}"
    fi
    
    echo "Running pipeline for ${scene} with ${pattern}..."
    echo "Output directory: ${OUTPUT_DIR}"
    
    python run_pipeline.py \
        --scenes_root "${SCENES_ROOT}" \
        --output_dir "${OUTPUT_DIR}" \
        --scene_id "${scene}" \
        --move_pattern "${MOVE_PATTERN}" \
        ${EXTRA_ARGS} \
        --num_cameras ${NUM_CAMERAS} \
        --max_questions_per_scene ${MAX_QUESTIONS_PER_SCENE} \
        --max_questions_per_type ${MAX_QUESTIONS_PER_TYPE} \
        --max_tries ${MAX_TRIES} \
        --render_width ${RENDER_WIDTH} \
        --render_height ${RENDER_HEIGHT} \
        --render_fov ${RENDER_FOV} \
        ${MIN_VIEWS_ARG} \
        --enable_rendering \
        --experiment_name "${pattern}" \
        --verbose
    
    echo -e "${GREEN}Completed: ${scene} / ${pattern}${NC}"
done

echo ""
echo "========================================="
echo "All requested combinations processed!"
echo "========================================="

# Final status check
echo ""
echo "Final Status:"
for scene in "${SCENES[@]}"; do
    for pattern in "${PATTERNS[@]}"; do
        status=$(check_completion "$scene" "$pattern")
        details=$(get_status_details "$scene" "$pattern")
        
        case $status in
            0)
                echo -e "${GREEN}[COMPLETE]${NC} ${scene} / ${pattern} - ${details}"
                ;;
            1)
                echo -e "${YELLOW}[PENDING]${NC}  ${scene} / ${pattern} - Not started"
                ;;
            2)
                echo -e "${BLUE}[PARTIAL]${NC}  ${scene} / ${pattern} - ${details}"
                ;;
        esac
    done
done
