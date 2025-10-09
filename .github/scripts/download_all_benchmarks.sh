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

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly VALID_SOURCES="wget test_data"

# List of all supported datasets
readonly DATASETS=(
    "skydio-8"
    "skydio-32" 
    "skydio-501"
    "notre-dame-20"
    "palace-fine-arts-281"
    "2011205_rc3"
    "gerrard-hall-100"
    "south-building-128"
)

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
        HEADER) echo "🚀 $*" ;;
        *) echo "$*" ;;
    esac
}

usage() {
    cat << EOF
Usage: $SCRIPT_NAME <source>

Download all GTSFM benchmark datasets.

Arguments:
    source    Download source (wget|test_data)

Examples:
    $SCRIPT_NAME wget        # Download all datasets from GitHub releases
    $SCRIPT_NAME test_data   # Use test data source

Available datasets (8 total):
    Small:  skydio-8 (~39MB), notre-dame-20 (~41MB)
    Medium: skydio-32 (~175MB), palace-fine-arts-281 (~184MB)
    Large:  2011205_rc3 (~205MB total), gerrard-hall-100 (~1.0GB)
            south-building-128 (~472MB), skydio-501 (~4.4GB total)

EOF
}

validate_args() {
    local source="$1"
    
    if [[ -z "$source" ]]; then
        log ERROR "Missing required source argument"
        usage
        exit 1
    fi
    
    if [[ ! " $VALID_SOURCES " =~ " $source " ]]; then
        log ERROR "Invalid source '$source'"
        echo "Valid sources: $VALID_SOURCES"
        exit 1
    fi
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

download_all_datasets() {
    local source="$1"
    local failed_datasets=()
    local successful_datasets=()
    
    log HEADER "Starting batch download of ${#DATASETS[@]} datasets"
    
    for dataset in "${DATASETS[@]}"; do
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
    
    log SUCCESS "All ${#DATASETS[@]} benchmark datasets downloaded successfully!"
    return 0
}

main() {
    local source="${1:-}"
    
    # Handle help requests
    if [[ "$source" =~ ^(-h|--help|help)$ ]]; then
        usage
        exit 0
    fi
    
    validate_args "$source"
    
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