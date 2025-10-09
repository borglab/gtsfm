#!/bin/bash
# =============================================================================
# GTSFM Batch Benchmark Downloader
# =============================================================================
# Downloads all benchmark datasets by calling download_single_benchmark.sh
# for each supported dataset.
#
# Author: GTSFM Team
# Usage: ./download_all_benchmarks.sh <source>
# =============================================================================

set -euo pipefail

# Source shared utilities
source "$(dirname "$0")/utils.sh"

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(get_script_dir)"

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

usage() {
    cat << EOF
Usage: $SCRIPT_NAME <source>

Download all GTSFM benchmark datasets.

Arguments:
    source    Download source (wget|test_data)

Examples:
    $SCRIPT_NAME wget        # Download all datasets from GitHub releases
    $SCRIPT_NAME test_data   # Use test data source

EOF
    show_dataset_summary
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

download_all_datasets() {
    local source="$1"
    local failed_datasets=()
    local successful_datasets=()
    
    log HEADER "Starting batch download of ${#GTSFM_DATASETS[@]} datasets"
    
    for dataset in "${GTSFM_DATASETS[@]}"; do
        log INFO "Processing dataset: $dataset"
        echo ""
        
        if bash "$SCRIPT_DIR/download_single_benchmark.sh" "$dataset" "$source"; then
            successful_datasets+=("$dataset")
            log SUCCESS "Completed: $dataset"
        else
            failed_datasets+=("$dataset")
            log ERROR "Failed: $dataset"
        fi
        
        echo ""
    done
    
    # Report results
    log HEADER "Download Summary"
    log SUCCESS "Successfully downloaded: ${#successful_datasets[@]} datasets"
    
    if [[ ${#failed_datasets[@]} -gt 0 ]]; then
        log ERROR "Failed to download: ${#failed_datasets[@]} datasets"
        for dataset in "${failed_datasets[@]}"; do
            log ERROR "  - $dataset"
        done
        return 1
    fi
    
    log SUCCESS "All ${#GTSFM_DATASETS[@]} benchmark datasets downloaded successfully!"
    return 0
}

main() {
    local source="${1:-}"
    
    # Handle help requests
    show_help_if_requested "$source" usage
    
    validate_source "$source" || { usage; exit 1; }
    
    log INFO "GTSFM Batch Benchmark Downloader"
    log INFO "Source: $source"
    
    if download_all_datasets "$source"; then
        log SUCCESS "Batch download completed successfully!"
        exit 0
    else
        log ERROR "Batch download failed"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"