#!/bin/bash
# =============================================================================
# GTSFM Benchmark Dataset Downloader
# =============================================================================
# Downloads and extracts benchmark datasets for GTSFM evaluation.
# Datasets are organized in the benchmarks/ directory with cache files at root.
#
# Author: GTSFM Team
# Usage: ./download_single_benchmark.sh <dataset_name> <source>
# =============================================================================

set -euo pipefail

# =============================================================================
# CONSTANTS AND CONFIGURATION
# =============================================================================

readonly SCRIPT_NAME="$(basename "$0")"
readonly BENCHMARK_DIR="benchmarks"
readonly CACHE_DIR="../cache"
readonly VALID_SOURCES="wget test_data"

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
Usage: $SCRIPT_NAME <dataset_name> <source>

Download and extract GTSFM benchmark datasets.

Arguments:
    dataset_name    Name of the dataset to download
    source          Download source (wget|test_data)

Available datasets:
    skydio-8             8 images from Skydio-501 crane (~39MB)
    skydio-32            32 images from Skydio-501 crane (~175MB)
    skydio-501           501-image Crane Mast collection (~2.9GB + 1.5GB cache)
    notre-dame-20        Notre Dame cathedral dataset (~41MB)
    palace-fine-arts-281 Palace of Fine Arts, San Francisco (~184MB)
    2011205_rc3          NASA Dawn mission Asteroid 4 Vesta (~128MB + 77MB cache)
    gerrard-hall-100     Gerrard Hall building (~1.0GB)
    south-building-128   South Building dataset (~472MB)

EOF
}

validate_args() {
    local dataset_name="$1"
    local source="$2"
    
    if [[ -z "$dataset_name" || -z "$source" ]]; then
        log ERROR "Missing required arguments"
        usage
        exit 1
    fi
    
    # Check if dataset name is valid
    case "$dataset_name" in
        skydio-8|skydio-32|skydio-501|notre-dame-20|palace-fine-arts-281|2011205_rc3|gerrard-hall-100|south-building-128)
            ;;
        *)
            log ERROR "Invalid dataset name '$dataset_name'"
            usage
            exit 1
            ;;
    esac
    
    # Check if source is valid
    if [[ ! " $VALID_SOURCES " =~ " $source " ]]; then
        log ERROR "Invalid source '$source'"
        echo "Valid sources: $VALID_SOURCES"
        exit 1
    fi
}

get_dataset_dir() {
    local dataset_name="$1"
    case "$dataset_name" in
        skydio-8) echo "skydio_crane_mast_8imgs_with_exif" ;;
        skydio-32) echo "skydio-32" ;;
        skydio-501) echo "skydio-crane-mast-501-images" ;;
        notre-dame-20) echo "notre-dame-20" ;;
        palace-fine-arts-281) echo "palace-fine-arts-281" ;;
        2011205_rc3) echo "2011205_rc3" ;;
        gerrard-hall-100) echo "gerrard-hall" ;;
        south-building-128) echo "south-building" ;;
    esac
}

dataset_exists() {
    local dataset_name="$1"
    local dataset_dir
    dataset_dir="$(get_dataset_dir "$dataset_name")"
    [[ -d "$dataset_dir" ]]
}

# =============================================================================
# DOWNLOAD AND EXTRACTION FUNCTIONS
# =============================================================================

retry_with_backoff() {
    local max_attempts="$1"
    shift
    
    local attempt=1
    while (( attempt <= max_attempts )); do
        if "$@"; then
            return 0
        fi
        
        local exit_code=$?
        if (( attempt == max_attempts )); then
            log ERROR "Command failed after $max_attempts attempts: $*"
            return $exit_code
        fi
        
        local delay=$((2 ** (attempt - 1)))
        log WARN "Attempt $attempt/$max_attempts failed (exit: $exit_code), retrying in ${delay}s..."
        sleep "$delay"
        ((attempt++))
    done
}

download_file() {
    local url="$1"
    local filename="${url##*/}"
    
    log INFO "Downloading $filename..."
    retry_with_backoff 5 wget --no-verbose "$url"
}

download_dataset_files() {
    local dataset_name="$1"
    local source="$2"
    
    [[ "$source" == "wget" ]] || return 0
    
    log INFO "Downloading $dataset_name dataset files..."
    
    case "$dataset_name" in
        skydio-8)
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/skydio_crane_mast_8imgs_with_exif.zip"
            ;;
        skydio-32)
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/skydio_crane_mast_32imgs_w_colmap_GT.zip"
            ;;
        notre-dame-20)
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/notre-dame-20.zip"
            ;;
        gerrard-hall-100)
            download_file "https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip"
            ;;
        south-building-128)
            download_file "https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip"
            ;;
        skydio-501)
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-crane-mast-501-images/skydio-crane-mast-501-images1.tar.gz"
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-crane-mast-501-images/skydio-crane-mast-501-images2.tar.gz"
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-501-colmap-pseudo-gt/skydio-501-colmap-pseudo-gt.tar.gz"
            download_file "https://github.com/johnwlambert/gtsfm-cache/releases/download/skydio-501-lookahead50-deep-front-end-cache/skydio-501-lookahead50-deep-front-end-cache.tar.gz"
            ;;
        palace-fine-arts-281)
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/palace-fine-arts-281/fine_arts_palace.zip"
            download_file "https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/palace-fine-arts-281/data.mat"
            ;;
        2011205_rc3)
            download_file "https://www.dropbox.com/s/q02mgq1unbw068t/2011205_rc3.zip"
            download_file "https://github.com/johnwlambert/gtsfm-cache/releases/download/2011205_rc3_deep_front_end_cache/cache_rc3_deep.tar.gz"
            ;;
        *)
            log ERROR "No download configuration found for $dataset_name"
            return 1
            ;;
    esac
}

safe_extract_archive() {
    local archive="$1"
    local extract_args="${2:-}"
    
    log INFO "Extracting $archive..."
    
    case "$archive" in
        *.tar.gz)
            tar -xzf "$archive" $extract_args && rm "$archive"
            ;;
        *.zip)
            unzip -qq "$archive" $extract_args && rm "$archive"
            ;;
        *)
            log ERROR "Unsupported archive format: $archive"
            return 1
            ;;
    esac
    
    log SUCCESS "Extracted and removed $archive"
}

organize_cache_files() {
    local cache_source="$1"
    
    [[ -d "$cache_source" ]] || return 0
    
    log INFO "Organizing cache files from $cache_source..."
    
    mkdir -p "$CACHE_DIR"
    
    local moved_any=false
    for cache_type in detector_descriptor matcher; do
        local source_dir="$cache_source/$cache_type"
        local target_dir="$CACHE_DIR/$cache_type"
        
        if [[ -d "$source_dir" ]]; then
            mkdir -p "$target_dir"
            local file_count
            file_count=$(find "$source_dir" -type f | wc -l)
            
            if (( file_count > 0 )); then
                find "$source_dir" -type f -exec mv {} "$target_dir/" \; 2>/dev/null
                log SUCCESS "Moved $file_count $cache_type cache files"
                moved_any=true
            fi
        fi
    done
    
    rm -rf "$cache_source"
    
    if [[ "$moved_any" == true ]]; then
        log SUCCESS "Cache organization complete"
    else
        log WARN "No cache files found to organize"
    fi
}

# =============================================================================
# DATASET EXTRACTION HANDLERS
# =============================================================================

extract_skydio_8() {
    safe_extract_archive "skydio_crane_mast_8imgs_with_exif.zip"
}

extract_skydio_32() {
    safe_extract_archive "skydio_crane_mast_32imgs_w_colmap_GT.zip" "-d skydio-32"
}

extract_skydio_501() {
    # Extract main archives
    for archive in skydio-crane-mast-501-images{1,2}.tar.gz skydio-501-colmap-pseudo-gt.tar.gz; do
        safe_extract_archive "$archive"
    done
    
    # Consolidate image directories
    local images_dir="skydio-crane-mast-501-images"
    mkdir -p "$images_dir"
    
    for source_dir in skydio-crane-mast-501-images{1,2}; do
        if [[ -d "$source_dir" ]]; then
            find "$source_dir" -type f -exec mv {} "$images_dir/" \; 2>/dev/null || true
            rm -rf "$source_dir"
        fi
    done
    
    # Handle cache files
    local cache_temp="skydio-501-cache"
    mkdir -p "$cache_temp"
    safe_extract_archive "skydio-501-lookahead50-deep-front-end-cache.tar.gz" "--directory $cache_temp"
    organize_cache_files "$cache_temp/cache"
}

extract_notre_dame_20() {
    safe_extract_archive "notre-dame-20.zip"
}

extract_palace_fine_arts_281() {
    mkdir -p palace-fine-arts-281
    safe_extract_archive "fine_arts_palace.zip" "-d palace-fine-arts-281/images"
    
    # Move supplementary data file
    if [[ -f "data.mat" ]]; then
        mv data.mat palace-fine-arts-281/
        log SUCCESS "Moved data.mat to palace-fine-arts-281/"
    fi
}

extract_2011205_rc3() {
    safe_extract_archive "2011205_rc3.zip"
    safe_extract_archive "cache_rc3_deep.tar.gz"
    
    # Organize cache files
    organize_cache_files "cache"
}

extract_gerrard_hall_100() {
    safe_extract_archive "gerrard-hall.zip"
}

extract_south_building_128() {
    safe_extract_archive "south-building.zip"
}

# =============================================================================
# MAIN PROCESSING FUNCTIONS
# =============================================================================

process_dataset() {
    local dataset_name="$1"
    local source="$2"
    
    if dataset_exists "$dataset_name"; then
        log INFO "Dataset $dataset_name already exists, skipping..."
        return 0
    fi
    
    # Get description for the dataset
    local desc
    case "$dataset_name" in
        skydio-8) desc="8 images from Skydio-501 crane" ;;
        skydio-32) desc="32 images from Skydio-501 crane" ;;
        skydio-501) desc="501-image Crane Mast collection (includes cache)" ;;
        notre-dame-20) desc="Notre Dame cathedral dataset" ;;
        palace-fine-arts-281) desc="Palace of Fine Arts, San Francisco" ;;
        2011205_rc3) desc="NASA Dawn mission Asteroid 4 Vesta (includes cache)" ;;
        gerrard-hall-100) desc="Gerrard Hall building" ;;
        south-building-128) desc="South Building dataset" ;;
    esac
    
    log INFO "Processing $dataset_name: $desc"
    
    download_dataset_files "$dataset_name" "$source"
    
    # Call dataset-specific extraction function
    local extract_func="extract_${dataset_name//-/_}"
    if type "$extract_func" >/dev/null 2>&1; then
        "$extract_func"
    else
        log ERROR "No extraction handler for $dataset_name"
        return 1
    fi
    
    log SUCCESS "Dataset $dataset_name processed successfully"
}

# =============================================================================
# MAIN EXECUTION
# =============================================================================

main() {
    local dataset_name="${1:-}"
    local source="${2:-}"
    
    # Handle help requests
    if [[ "$dataset_name" =~ ^(-h|--help|help)$ ]]; then
        usage
        exit 0
    fi
    
    validate_args "$dataset_name" "$source"
    
    log INFO "GTSFM Benchmark Dataset Downloader"
    log INFO "Dataset: $dataset_name | Source: $source"
    
    # Setup workspace
    mkdir -p "$BENCHMARK_DIR"
    cd "$BENCHMARK_DIR"
    log INFO "Working directory: $(pwd)"
    
    # Process the dataset
    if process_dataset "$dataset_name" "$source"; then
        log SUCCESS "All operations completed successfully!"
        exit 0
    else
        log ERROR "Dataset processing failed"
        exit 1
    fi
}

# Execute main function with all arguments
main "$@"
