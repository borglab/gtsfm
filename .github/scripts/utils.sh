#!/bin/bash
# =============================================================================
# GTSFM Shared Utilities
# =============================================================================
# Common functions and utilities shared across GTSFM scripts.
# Source this file to access logging and utility functions.
#
# Usage: source "$(dirname "$0")/utils.sh"
# =============================================================================

# =============================================================================
# LOGGING FUNCTIONS
# =============================================================================

log() {
    local level="$1"
    shift
    case "$level" in
        INFO)    echo "ℹ️  $*" ;;
        WARN)    echo "⚠️  $*" >&2 ;;
        ERROR)   echo "❌ $*" >&2 ;;
        SUCCESS) echo "✅ $*" ;;
        HEADER)  echo "🚀 $*" ;;
        DEBUG)   echo "🔍 $*" ;;
        *)       echo "$*" ;;
    esac
}

# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

validate_source() {
    local source="$1"
    local valid_sources="wget test_data"
    
    if [[ -z "$source" ]]; then
        log ERROR "Missing required source argument"
        return 1
    fi
    
    if [[ ! " $valid_sources " =~ " $source " ]]; then
        log ERROR "Invalid source '$source'"
        echo "Valid sources: $valid_sources"
        return 1
    fi
    
    return 0
}

validate_dataset() {
    local dataset_name="$1"
    local valid_datasets="door-12 palace-fine-arts-281 2011205_rc3 skydio-8 skydio-32 skydio-501 notre-dame-20 gerrard-hall-100 south-building-128"
    
    if [[ -z "$dataset_name" ]]; then
        log ERROR "Missing required dataset name"
        return 1
    fi
    
    if [[ ! " $valid_datasets " =~ " $dataset_name " ]]; then
        log ERROR "Invalid dataset name '$dataset_name'"
        return 1
    fi
    
    return 0
}

validate_loader() {
    local loader_name="$1"
    local valid_loaders="olsson-loader colmap-loader astrovision"
    
    if [[ -z "$loader_name" ]]; then
        log ERROR "Missing required loader name"
        return 1
    fi
    
    if [[ ! " $valid_loaders " =~ " $loader_name " ]]; then
        log ERROR "Invalid loader name '$loader_name'"
        return 1
    fi
    
    return 0
}

validate_boolean() {
    local value="$1"
    local param_name="$2"
    
    case "$value" in
        true|false)
            return 0
            ;;
        *)
            log ERROR "Invalid $param_name value '$value' (must be true or false)"
            return 1
            ;;
    esac
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

show_help_if_requested() {
    local arg="$1"
    local usage_func="$2"
    
    if [[ "$arg" =~ ^(-h|--help|help)$ ]]; then
        "$usage_func"
        exit 0
    fi
}

get_script_dir() {
    echo "$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
}

# =============================================================================
# DATASET CONSTANTS AND INFORMATION
# =============================================================================

readonly GTSFM_DATASETS=(
    "door-12"
    "skydio-8"
    "skydio-32" 
    "skydio-501"
    "notre-dame-20"
    "palace-fine-arts-281"
    "2011205_rc3"
    "gerrard-hall-100"
    "south-building-128"
)

readonly BENCHMARK_DIR="benchmarks"
readonly CACHE_DIR="../cache"

# Dataset information with sizes
get_dataset_info() {
    local dataset_name="$1"
    case "$dataset_name" in
        door-12)              echo "Test dataset - Lund door sequence (test_data only)" ;;
        skydio-8)             echo "8 images from Skydio-501 crane (~39MB)" ;;
        skydio-32)            echo "32 images from Skydio-501 crane (~175MB)" ;;
        skydio-501)           echo "501-image Crane Mast collection (~2.9GB + 1.5GB cache)" ;;
        notre-dame-20)        echo "Notre Dame cathedral dataset (~41MB)" ;;
        palace-fine-arts-281) echo "Palace of Fine Arts, San Francisco (~184MB)" ;;
        2011205_rc3)          echo "NASA Dawn mission Asteroid 4 Vesta (~128MB + 77MB cache)" ;;
        gerrard-hall-100)     echo "Gerrard Hall building (~1.0GB)" ;;
        south-building-128)   echo "South Building dataset (~472MB)" ;;
        *)                    echo "Unknown dataset" ;;
    esac
}

# Print all datasets with descriptions
show_all_datasets() {
    echo "Available datasets:"
    for dataset in "${GTSFM_DATASETS[@]}"; do
        printf "    %-20s %s\n" "$dataset" "$(get_dataset_info "$dataset")"
    done
    echo ""
}

# Show categorized dataset summary
show_dataset_summary() {
    echo "Available datasets (${#GTSFM_DATASETS[@]} total):"
    echo "    Test:   door-12 (test_data only)"
    echo "    Small:  skydio-8, notre-dame-20"
    echo "    Medium: skydio-32, palace-fine-arts-281"  
    echo "    Large:  2011205_rc3, gerrard-hall-100, south-building-128, skydio-501"
    echo ""
}