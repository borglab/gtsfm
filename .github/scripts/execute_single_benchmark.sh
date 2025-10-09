#!/bin/bash
# =============================================================================
# GTSFM Benchmark Execution Script
# =============================================================================
# Executes GTSFM on benchmark datasets with specified configurations.
# Compatible with CI workflows and manual command-line usage.
#
# Author: GTSFM Team
# Usage: ./execute_single_benchmark.sh <dataset> <config> <lookahead> <loader> <resolution> <share_intrinsics>
# =============================================================================

set -euo pipefail

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly BENCHMARK_DIR="benchmarks"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log() {
    local level="$1"
    shift
    case "$level" in
        INFO)  echo "ℹ️  $*" ;;
        WARN)  echo "⚠️  $*" >&2 ;;
        ERROR) echo "❌ $*" >&2 ;;
        SUCCESS) echo "✅ $*" ;;
        *) echo "$*" ;;
    esac
}

usage() {
    cat << EOF
Usage: $SCRIPT_NAME <dataset_name> <config_name> <max_frame_lookahead> <loader_name> <max_resolution> <share_intrinsics>

Execute GTSFM on benchmark datasets.

Arguments:
    dataset_name         Name of the dataset (e.g., skydio-8, notre-dame-20)
    config_name          Configuration name (e.g., sift, lightglue)
    max_frame_lookahead  Maximum frame lookahead (e.g., 15)
    loader_name          Loader type (olsson-loader, colmap-loader, astrovision)
    max_resolution       Maximum image resolution (e.g., 760, 1024)
    share_intrinsics     Share intrinsics across images (true, false)

Examples:
    $SCRIPT_NAME skydio-8 sift 15 colmap-loader 760 true
    $SCRIPT_NAME notre-dame-20 lightglue 15 colmap-loader 760 false

EOF
}

validate_args() {
    local dataset_name="$1"
    local config_name="$2"
    local max_frame_lookahead="$3"
    local loader_name="$4"
    local max_resolution="$5"
    local share_intrinsics="$6"
    
    if [[ -z "$dataset_name" || -z "$config_name" || -z "$max_frame_lookahead" || -z "$loader_name" || -z "$max_resolution" || -z "$share_intrinsics" ]]; then
        log ERROR "Missing required arguments"
        usage
        exit 1
    fi
    
    # Validate dataset name
    case "$dataset_name" in
        door-12|palace-fine-arts-281|2011205_rc3|skydio-8|skydio-32|skydio-501|notre-dame-20|gerrard-hall-100|south-building-128)
            ;;
        *)
            log ERROR "Invalid dataset name '$dataset_name'"
            usage
            exit 1
            ;;
    esac
    
    # Validate loader name
    case "$loader_name" in
        olsson-loader|colmap-loader|astrovision)
            ;;
        *)
            log ERROR "Invalid loader name '$loader_name'"
            usage
            exit 1
            ;;
    esac
    
    # Validate share_intrinsics
    case "$share_intrinsics" in
        true|false)
            ;;
        *)
            log ERROR "Invalid share_intrinsics value '$share_intrinsics' (must be true or false)"
            usage
            exit 1
            ;;
    esac
}

# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

configure_dataset_paths() {
    local dataset_name="$1"
    
    case "$dataset_name" in
        door-12)
            DATASET_ROOT="tests/data/set1_lund_door"
            ;;
        palace-fine-arts-281)
            DATASET_ROOT="${BENCHMARK_DIR}/palace-fine-arts-281"
            ;;
        2011205_rc3)
            DATASET_ROOT="${BENCHMARK_DIR}/2011205_rc3"
            ;;
        skydio-8)
            IMAGES_DIR="${BENCHMARK_DIR}/skydio_crane_mast_8imgs_with_exif/images"
            COLMAP_FILES_DIRPATH="${BENCHMARK_DIR}/skydio_crane_mast_8imgs_with_exif/crane_mast_8imgs_colmap_output"
            ;;
        skydio-32)
            IMAGES_DIR="${BENCHMARK_DIR}/skydio-32/images"
            COLMAP_FILES_DIRPATH="${BENCHMARK_DIR}/skydio-32/colmap_crane_mast_32imgs"
            ;;
        skydio-501)
            IMAGES_DIR="${BENCHMARK_DIR}/skydio-crane-mast-501-images"
            COLMAP_FILES_DIRPATH="${BENCHMARK_DIR}/skydio-501-colmap-pseudo-gt"
            ;;
        notre-dame-20)
            IMAGES_DIR="${BENCHMARK_DIR}/notre-dame-20/images"
            COLMAP_FILES_DIRPATH="${BENCHMARK_DIR}/notre-dame-20/notre-dame-20-colmap"
            ;;
        gerrard-hall-100)
            IMAGES_DIR="${BENCHMARK_DIR}/gerrard-hall/images"
            COLMAP_FILES_DIRPATH="${BENCHMARK_DIR}/gerrard-hall/sparse"
            ;;
        south-building-128)
            IMAGES_DIR="${BENCHMARK_DIR}/south-building/images"
            COLMAP_FILES_DIRPATH="${BENCHMARK_DIR}/south-building/sparse"
            ;;
        *)
            log ERROR "Unknown dataset: $dataset_name"
            return 1
            ;;
    esac
}

# =============================================================================
# EXECUTION FUNCTIONS
# =============================================================================

execute_gtsfm() {
    local dataset_name="$1"
    local config_name="$2"
    local max_frame_lookahead="$3"
    local loader_name="$4"
    local max_resolution="$5"
    local share_intrinsics="$6"
    
    # Setup the intrinsics sharing argument
    local share_intrinsics_arg=""
    if [[ "$share_intrinsics" == "true" ]]; then
        share_intrinsics_arg="--share_intrinsics"
    fi
    
    log INFO "Executing GTSFM with $loader_name..."
    
    case "$loader_name" in
        olsson-loader)
            log INFO "Running with Olsson loader on $DATASET_ROOT"
            python gtsfm/runner/run_scene_optimizer_olssonloader.py \
                --dataset_root "$DATASET_ROOT" \
                --config_name unified \
                --correspondence_generator_config_name "$config_name" \
                --max_frame_lookahead "$max_frame_lookahead" \
                --max_resolution "$max_resolution" \
                $share_intrinsics_arg
            ;;
            
        colmap-loader)
            log INFO "Running with COLMAP loader"
            log INFO "Images: $IMAGES_DIR"
            log INFO "COLMAP files: $COLMAP_FILES_DIRPATH"
            python gtsfm/runner/run_scene_optimizer_colmaploader.py \
                --images_dir "$IMAGES_DIR" \
                --colmap_files_dirpath "$COLMAP_FILES_DIRPATH" \
                --config_name unified \
                --correspondence_generator_config_name "$config_name" \
                --max_frame_lookahead "$max_frame_lookahead" \
                --max_resolution "$max_resolution" \
                $share_intrinsics_arg
            ;;
            
        astrovision)
            log INFO "Running with AstroVision loader on $DATASET_ROOT"
            python gtsfm/runner/run_scene_optimizer_astrovision.py \
                --data_dir "$DATASET_ROOT" \
                --config_name unified \
                --correspondence_generator_config_name "$config_name" \
                --max_frame_lookahead "$max_frame_lookahead" \
                --max_resolution "$max_resolution" \
                $share_intrinsics_arg
            ;;
            
        *)
            log ERROR "Unknown loader: $loader_name"
            return 1
            ;;
    esac
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    local dataset_name="${1:-}"
    local config_name="${2:-}"
    local max_frame_lookahead="${3:-}"
    local loader_name="${4:-}"
    local max_resolution="${5:-}"
    local share_intrinsics="${6:-}"
    
    # Handle help requests
    if [[ "$dataset_name" =~ ^(-h|--help|help)$ ]]; then
        usage
        exit 0
    fi
    
    validate_args "$dataset_name" "$config_name" "$max_frame_lookahead" "$loader_name" "$max_resolution" "$share_intrinsics"
    
    log INFO "GTSFM Benchmark Execution"
    log INFO "Dataset: $dataset_name | Config: $config_name | Loader: $loader_name"
    log INFO "Max Frame Lookahead: $max_frame_lookahead | Max Resolution: $max_resolution"
    log INFO "Share Intrinsics: $share_intrinsics"
    
    # Configure dataset paths
    if configure_dataset_paths "$dataset_name"; then
        log SUCCESS "Dataset paths configured"
    else
        log ERROR "Failed to configure dataset paths"
        exit 1
    fi
    
    # Execute GTSFM
    if execute_gtsfm "$dataset_name" "$config_name" "$max_frame_lookahead" "$loader_name" "$max_resolution" "$share_intrinsics"; then
        log SUCCESS "GTSFM execution completed successfully!"
        exit 0
    else
        log ERROR "GTSFM execution failed"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"
