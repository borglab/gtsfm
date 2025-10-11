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

# Source shared utilities
source "$(dirname "$0")/utils.sh"

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

readonly SCRIPT_NAME="$(basename "$0")"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

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
    show_all_datasets
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
    
    validate_dataset "$dataset_name" || { usage; exit 1; }
    validate_loader "$loader_name" || { usage; exit 1; }
    validate_boolean "$share_intrinsics" "share_intrinsics" || { usage; exit 1; }
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
            python gtsfm/runner.py \
                --loader olsson_loader \
                --dataset_dir "$DATASET_ROOT" \
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
            python gtsfm/runner.py \
                --loader colmap_loader \
                --dataset_dir "$(dirname "$IMAGES_DIR")" \
                --images_dir "$IMAGES_DIR" \
                --config_name unified \
                --correspondence_generator_config_name "$config_name" \
                --max_frame_lookahead "$max_frame_lookahead" \
                --max_resolution "$max_resolution" \
                $share_intrinsics_arg
            ;;
            
        astrovision)
            log INFO "Running with AstroVision loader on $DATASET_ROOT"
            python gtsfm/runner.py \
                --loader astrovision_loader \
                --dataset_dir "$DATASET_ROOT" \
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
    show_help_if_requested "$dataset_name" usage
    
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
