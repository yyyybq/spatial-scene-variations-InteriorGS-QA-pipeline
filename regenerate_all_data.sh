#!/bin/bash
# Regenerate all data with camera_number=15
# Outputs to /scratch/by2593/project/sceneshift/data
# Generates: Images + QA + Demo visualizations

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SCENES_ROOT="/scratch/by2593/project/Active_Spatial/InteriorGS"
OUTPUT_BASE="/scratch/by2593/project/sceneshift/data"

# Get all scene directories from InteriorGS
# Filter to only include valid scene directories (format: XXXX_XXXXXX)
mapfile -t ALL_SCENES < <(ls -d ${SCENES_ROOT}/[0-9][0-9][0-9][0-9]_[0-9]* 2>/dev/null | xargs -n1 basename | sort)

echo "Found ${#ALL_SCENES[@]} scenes in InteriorGS"

# Patterns to generate
PATTERNS=("around" "spherical" "linear_approach" "linear_passby")

# Camera settings
NUM_CAMERAS=15

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Parse command line arguments
FORCE_RERUN=false
SPECIFIC_SCENE=""
SPECIFIC_PATTERN=""
MAX_SCENES=""
SKIP_DEMO=false
CHECK_ONLY=false

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
        --max-scenes)
            MAX_SCENES="$2"
            shift 2
            ;;
        --skip-demo)
            SKIP_DEMO=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --check           Only check status, don't run anything"
            echo "  --force           Force re-run even if already completed"
            echo "  --scene SCENE_ID  Only process specific scene"
            echo "  --pattern PATTERN Only process specific pattern (around, spherical, linear_approach, linear_passby)"
            echo "  --max-scenes N    Only process first N scenes"
            echo "  --skip-demo       Skip demo generation"
            echo "  --help, -h        Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Generate all data"
            echo "  $0 --check                  # Only show status"
            echo "  $0 --scene 0042_839881      # Only run scene 0042_839881"
            echo "  $0 --pattern spherical      # Only run spherical pattern"
            echo "  $0 --max-scenes 10          # Only process first 10 scenes"
            echo "  $0 --force                  # Force re-run everything"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Filter scenes and patterns if specified
if [[ -n "$SPECIFIC_SCENE" ]]; then
    ALL_SCENES=("$SPECIFIC_SCENE")
    echo "Filtering to scene: $SPECIFIC_SCENE"
fi

if [[ -n "$SPECIFIC_PATTERN" ]]; then
    PATTERNS=("$SPECIFIC_PATTERN")
    echo "Filtering to pattern: $SPECIFIC_PATTERN"
fi

if [[ -n "$MAX_SCENES" ]]; then
    ALL_SCENES=("${ALL_SCENES[@]:0:$MAX_SCENES}")
    echo "Limiting to first $MAX_SCENES scenes"
fi

SCENES=("${ALL_SCENES[@]}")

# Function to check if a scene/pattern combination is complete
check_completion() {
    local scene="$1"
    local pattern="$2"
    local output_dir="${OUTPUT_BASE}/${pattern}/${scene}"
    
    # Check if output directory exists
    if [[ ! -d "$output_dir" ]]; then
        echo "1"  # Not started
        return
    fi
    
    # Find metadata.json and questions.jsonl
    local metadata_file="${output_dir}/metadata.json"
    local questions_file="${output_dir}/questions.jsonl"
    
    # Check if both metadata and questions exist
    if [[ -f "$metadata_file" && -f "$questions_file" ]]; then
        # Check if questions file has content
        local num_questions=$(wc -l < "$questions_file" 2>/dev/null || echo "0")
        if [[ $num_questions -gt 0 ]]; then
            # Check if images exist
            local num_images=$(find "$output_dir" -name "*.png" 2>/dev/null | head -100 | wc -l)
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
    
    local questions_file="${output_dir}/questions.jsonl"
    
    if [[ -f "$questions_file" ]]; then
        questions_count=$(wc -l < "$questions_file" 2>/dev/null || echo "0")
    fi
    
    if [[ -f "${output_dir}/metadata.json" ]]; then
        has_metadata="yes"
    fi
    
    images_count=$(find "$output_dir" -name "*.png" 2>/dev/null | wc -l)
    
    echo "Q:${questions_count}|I:${images_count}|M:${has_metadata}"
}

echo "========================================="
echo "Full Data Regeneration Pipeline"
echo "========================================="
echo "Scenes: ${#SCENES[@]} scenes"
echo "Patterns: ${PATTERNS[@]}"
echo "Cameras per item: ${NUM_CAMERAS}"
echo "Output: ${OUTPUT_BASE}"
echo ""

# Create output directory
mkdir -p "${OUTPUT_BASE}"

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
                if [[ $CHECK_ONLY == true ]]; then
                    echo -e "${GREEN}[COMPLETE]${NC} ${scene} / ${pattern} - ${details}"
                fi
                ;;
            1)
                PENDING+=("${scene}/${pattern}")
                if [[ $CHECK_ONLY == true ]]; then
                    echo -e "${YELLOW}[PENDING]${NC}  ${scene} / ${pattern} - Not started"
                fi
                ;;
            2)
                PARTIAL+=("${scene}/${pattern}")
                if [[ $CHECK_ONLY == true ]]; then
                    echo -e "${BLUE}[PARTIAL]${NC}  ${scene} / ${pattern} - ${details}"
                fi
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
else
    echo ""
    echo "========================================="
    echo "Will process ${#TO_RUN[@]} combinations"
    echo "========================================="

    cd "$SCRIPT_DIR"

    # Track successful completions
    SUCCESS_COUNT=0
    FAIL_COUNT=0

    # Process each combination
    for item in "${TO_RUN[@]}"; do
        scene="${item%|*}"
        pattern="${item#*|}"
        
        echo ""
        echo "========================================="
        echo -e "${CYAN}Processing: ${scene} / ${pattern}${NC}"
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
            EXTRA_ARGS="--linear_sub_pattern approach --linear_num_steps 5 --linear_move_distance 0.3"
        elif [[ "$pattern" == "linear_passby" ]]; then
            MOVE_PATTERN="linear"
            EXTRA_ARGS="--linear_sub_pattern pass_by --linear_num_steps 5 --linear_move_distance 0.3"
        fi
        
        echo "Running pipeline for ${scene} with ${pattern}..."
        echo "Output directory: ${OUTPUT_DIR}"
        
        # Run the pipeline
        if python run_pipeline.py \
            --scenes_root "${SCENES_ROOT}" \
            --output_dir "${OUTPUT_DIR}" \
            --scene_id "${scene}" \
            --move_pattern "${MOVE_PATTERN}" \
            ${EXTRA_ARGS} \
            --num_cameras ${NUM_CAMERAS} \
            --enable_rendering \
            --experiment_name "${pattern}" \
            --flat_output \
            --verbose; then
            echo -e "${GREEN}Completed: ${scene} / ${pattern}${NC}"
            ((SUCCESS_COUNT++))
        else
            echo -e "${RED}Failed: ${scene} / ${pattern}${NC}"
            ((FAIL_COUNT++))
        fi
    done

    echo ""
    echo "========================================="
    echo "Generation Summary"
    echo "========================================="
    echo -e "${GREEN}Successful:${NC} ${SUCCESS_COUNT}"
    echo -e "${RED}Failed:${NC}     ${FAIL_COUNT}"
fi

# ========================================
# Generate Demos (if not skipped)
# ========================================
if [[ $SKIP_DEMO == false ]]; then
    echo ""
    echo "========================================="
    echo "Generating Demo Visualizations"
    echo "========================================="
    
    DEMO_OUTPUT="${OUTPUT_BASE}/demos"
    mkdir -p "${DEMO_OUTPUT}"
    
    cd "$SCRIPT_DIR"
    
    for pattern in "${PATTERNS[@]}"; do
        echo ""
        echo "Generating demos for pattern: ${pattern}"
        
        PATTERN_DIR="${OUTPUT_BASE}/${pattern}"
        
        if [[ ! -d "$PATTERN_DIR" ]]; then
            echo "  No data found for pattern ${pattern}, skipping..."
            continue
        fi
        
        # Find scenes with data for this pattern
        for scene_dir in "${PATTERN_DIR}"/*/; do
            if [[ ! -d "$scene_dir" ]]; then
                continue
            fi
            
            scene=$(basename "$scene_dir")
            questions_file="${scene_dir}/questions.jsonl"
            
            if [[ ! -f "$questions_file" ]]; then
                continue
            fi
            
            # Check if questions file has content
            num_questions=$(wc -l < "$questions_file" 2>/dev/null || echo "0")
            if [[ $num_questions -eq 0 ]]; then
                continue
            fi
            
            demo_output="${DEMO_OUTPUT}/${pattern}/${scene}_demo.png"
            mkdir -p "$(dirname "$demo_output")"
            
            echo "  Creating demo for ${scene}/${pattern}..."
            
            # Map pattern name to move pattern for demo script
            if [[ "$pattern" == "linear_approach" || "$pattern" == "linear_passby" ]]; then
                demo_pattern="linear"
            else
                demo_pattern="$pattern"
            fi
            
            python create_demo.py \
                --questions_file "${questions_file}" \
                --output_file "${demo_output}" \
                --pattern "${demo_pattern}" \
                --num_views 8 2>/dev/null || echo "    Demo generation failed for ${scene}/${pattern}"
        done
    done
    
    echo ""
    echo -e "${GREEN}Demo generation complete!${NC}"
fi

echo ""
echo "========================================="
echo "All tasks completed!"
echo "========================================="
echo "Output directory: ${OUTPUT_BASE}"
echo ""
echo "Structure:"
echo "  ${OUTPUT_BASE}/"
echo "    around/           # Horizontal circle pattern"
echo "    spherical/        # Spherical sampling pattern"
echo "    linear_approach/  # Linear approach pattern"
echo "    linear_passby/    # Linear pass-by pattern"
echo "    demos/            # Demo visualizations"
