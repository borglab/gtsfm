#!/bin/bash
# This file defines the workflow for downloading, preparing, and storing datasets required for evaluation in CI.
#
# Datasets processed by this file are downloaded and extracted in the benchmarks/ directory.
# The script takes two arguments:
#   1) DATASET_NAME: name of the dataset to be downloaded. Supported datasets are:
#       - skydio-8
#       - skydio-32
#       - skydio-501
#       - notre-dame-20
#       - palace-fine-arts-281
#       - 2011205_rc3
#       - gerrard-hall-100
#       - south-building-128
#   2) DATASET_SRC: source from which to download the dataset. Supported sources are:
#       - wget
#

set -euo pipefail  # Exit on error, undefined vars, pipe failures

DATASET_NAME=$1
DATASET_SRC=$2

# Validate arguments
if [[ -z "${DATASET_NAME:-}" || -z "${DATASET_SRC:-}" ]]; then
    echo "Error: Missing required arguments"
    echo "Usage: $0 <DATASET_NAME> <DATASET_SRC>"
    exit 1
fi

# Validate dataset name
VALID_DATASETS=("skydio-8" "skydio-32" "skydio-501" "notre-dame-20" "palace-fine-arts-281" "2011205_rc3" "gerrard-hall-100" "south-building-128")
if [[ ! " ${VALID_DATASETS[*]} " =~ " ${DATASET_NAME} " ]]; then
    echo "Error: Invalid dataset name '${DATASET_NAME}'"
    echo "Valid datasets: ${VALID_DATASETS[*]}"
    exit 1
fi

# Validate dataset source
if [[ "${DATASET_SRC}" != "wget" && "${DATASET_SRC}" != "test_data" ]]; then
    echo "Error: Invalid dataset source '${DATASET_SRC}'"
    echo "Valid sources: wget, test_data"
    exit 1
fi

echo "Dataset: ${DATASET_NAME}, Download Source: ${DATASET_SRC}"

# Create benchmark directory if it doesn't exist and change to it
BENCHMARK_DIR="benchmarks"
mkdir -p "${BENCHMARK_DIR}"
cd "${BENCHMARK_DIR}"
echo "Working in directory: $(pwd)"

# Function to safely move cache files to project cache directory
function move_cache_files {
  local cache_dir="$1"
  local target_base="../cache"
  
  echo "Moving cache files from ${cache_dir} to ${target_base}..."
  
  mkdir -p "${target_base}"
  
  if [[ -d "${cache_dir}/detector_descriptor" ]]; then
    mkdir -p "${target_base}/detector_descriptor"
    if find "${cache_dir}/detector_descriptor/" -type f -exec mv {} "${target_base}/detector_descriptor/" \; 2>/dev/null; then
      echo "Moved detector_descriptor cache files"
    else
      echo "Warning: No detector_descriptor files to move or move failed"
    fi
  fi
  
  if [[ -d "${cache_dir}/matcher" ]]; then
    mkdir -p "${target_base}/matcher"
    if find "${cache_dir}/matcher/" -type f -exec mv {} "${target_base}/matcher/" \; 2>/dev/null; then
      echo "Moved matcher cache files"
    else
      echo "Warning: No matcher files to move or move failed"
    fi
  fi
  
  # Clean up temporary cache directory
  if [[ -d "${cache_dir}" ]]; then
    rm -rf "${cache_dir}"
    echo "Cleaned up temporary cache directory ${cache_dir}"
  fi
}

function retry {
  local retries=$1
  shift

  local count=0
  until "$@"; do # actual command execution happening here, and will continue till signal 0 (success).
    exit=$?
    wait=$((2 ** count))
    count=$((count + 1))
    if [ $count -lt $retries ]; then
      echo "Retry $count/$retries exited $exit, retrying in $wait seconds..."
      sleep $wait
    else
      echo "Retry $count/$retries exited $exit, no more retries left."
      return $exit
    fi
  done
  return 0
}

# Function to safely extract archives with error checking
function safe_extract {
  local archive_file="$1"
  local extract_args="${2:-}"
  
  echo "Extracting ${archive_file}..."
  
  if [[ "${archive_file}" == *.tar.gz ]]; then
    if tar -xzf "${archive_file}" ${extract_args} && rm "${archive_file}"; then
      echo "Successfully extracted and removed ${archive_file}"
    else
      echo "Error: Failed to extract ${archive_file}"
      return 1
    fi
  elif [[ "${archive_file}" == *.zip ]]; then
    if unzip -qq "${archive_file}" ${extract_args} && rm "${archive_file}"; then
      echo "Successfully extracted and removed ${archive_file}"
    else
      echo "Error: Failed to extract ${archive_file}"
      return 1
    fi
  else
    echo "Error: Unsupported archive format for ${archive_file}"
    return 1
  fi
}

# The last command executed in this function is `unzip`, which will return a non-zero exit code upon failure
function download_and_unzip_dataset_files {
  # Prepare the download URLs.
  if [ "$DATASET_NAME" == "skydio-8" ]; then
    if [ -d "skydio_crane_mast_8imgs_with_exif" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    # Description: 8 images from Skydio-501 facing a single crane face
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/skydio_crane_mast_8imgs_with_exif.zip
    ZIP_FNAME=skydio-8.zip

  elif [ "$DATASET_NAME" == "skydio-32" ]; then
    if [ -d "skydio-32" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    # Description: 32 images from Skydio-501 facing a single crane face
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/skydio_crane_mast_32imgs_w_colmap_GT.zip
    ZIP_FNAME=skydio-32.zip

  elif [ "$DATASET_NAME" == "skydio-501" ]; then
    if [ -d "skydio-crane-mast-501-images" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    # 501-image Crane Mast collection released by Skydio via Sketchfab
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-crane-mast-501-images/skydio-crane-mast-501-images1.tar.gz
    WGET_URL2=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-crane-mast-501-images/skydio-crane-mast-501-images2.tar.gz
    WGET_URL3=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-501-colmap-pseudo-gt/skydio-501-colmap-pseudo-gt.tar.gz
    WGET_URL4=https://github.com/johnwlambert/gtsfm-cache/releases/download/skydio-501-lookahead50-deep-front-end-cache/skydio-501-lookahead50-deep-front-end-cache.tar.gz

  elif [ "$DATASET_NAME" == "notre-dame-20" ]; then
    if [ -d "notre-dame-20" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    # Description: TODO
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/notre-dame-20.zip
    ZIP_FNAME=notre-dame-20.zip

  elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
    if [ -d "palace-fine-arts-281" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    # Description: 281 images captured at the Palace of Fine Arts in San Francisco, CA. Images and pseudo-ground truth
    # poses from Carl Olsson's page: https://www.maths.lth.se/matematiklth/personal/calle/dataset/dataset.html
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/palace-fine-arts-281/fine_arts_palace.zip
    WGET_URL2=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/palace-fine-arts-281/data.mat
    ZIP_FNAME=fine_arts_palace.zip

  elif [ "$DATASET_NAME" == "2011205_rc3" ]; then
    if [ -d "2011205_rc3" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    # Description: images captured during the Rotation Characterization 3 (RC3) phase of NASA's Dawn mission to Asteroid 4
    #   Vesta.
    WGET_URL1=https://www.dropbox.com/s/q02mgq1unbw068t/2011205_rc3.zip
    WGET_URL2=https://github.com/johnwlambert/gtsfm-cache/releases/download/2011205_rc3_deep_front_end_cache/cache_rc3_deep.tar.gz
    ZIP_FNAME=2011205_rc3.zip

  elif [ "$DATASET_NAME" == "gerrard-hall-100" ]; then
    if [ -d "gerrard-hall" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    WGET_URL1=https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip
    ZIP_FNAME=gerrard-hall.zip

  elif [ "$DATASET_NAME" == "south-building-128" ]; then
    if [ -d "south-building" ]; then
      echo "Dataset $DATASET_NAME already exists, skipping download."
      return 0
    fi
    WGET_URL1=https://github.com/colmap/colmap/releases/download/3.11.1/south-building.zip
    ZIP_FNAME=south-building.zip
  fi

  # Download the data.
  if [[ "${DATASET_SRC}" == "wget" ]]; then
    echo "Downloading ${DATASET_NAME} with WGET"
    retry 10 wget "${WGET_URL1}"

    # Check if additional URLs are set and download them
    for url_var in WGET_URL2 WGET_URL3 WGET_URL4; do
      if [[ -n "${!url_var:-}" ]]; then
        retry 10 wget "${!url_var}"
      fi
    done
  fi

  # Extract the data, configure arguments for runner.
  case "${DATASET_NAME}" in
    "skydio-8")
      safe_extract "skydio_crane_mast_8imgs_with_exif.zip"
      ;;
      
    "skydio-32")
      safe_extract "skydio_crane_mast_32imgs_w_colmap_GT.zip" "-d skydio-32"
      ;;
      
    "skydio-501")
      safe_extract "skydio-crane-mast-501-images1.tar.gz"
      safe_extract "skydio-crane-mast-501-images2.tar.gz"
      safe_extract "skydio-501-colmap-pseudo-gt.tar.gz"
      
      # Consolidate image directories
      IMAGES_DIR="skydio-crane-mast-501-images"
      mkdir -p "${IMAGES_DIR}"
      if [[ -d "skydio-crane-mast-501-images1" ]]; then
        mv skydio-crane-mast-501-images1/* "${IMAGES_DIR}/" 2>/dev/null || echo "Warning: No files to move from images1"
        rm -rf skydio-crane-mast-501-images1
      fi
      if [[ -d "skydio-crane-mast-501-images2" ]]; then
        mv skydio-crane-mast-501-images2/* "${IMAGES_DIR}/" 2>/dev/null || echo "Warning: No files to move from images2"
        rm -rf skydio-crane-mast-501-images2
      fi

      # Extract and organize cache files
      mkdir -p skydio-501-cache
      if tar -xzf skydio-501-lookahead50-deep-front-end-cache.tar.gz --directory skydio-501-cache && rm skydio-501-lookahead50-deep-front-end-cache.tar.gz; then
        move_cache_files "skydio-501-cache/cache"
      else
        echo "Error: Failed to extract cache files for skydio-501"
        return 1
      fi
      ;;
      
    "notre-dame-20")
      safe_extract "notre-dame-20.zip"
      ;;
      
    "palace-fine-arts-281")
      mkdir -p palace-fine-arts-281
      safe_extract "fine_arts_palace.zip" "-d palace-fine-arts-281/images"
      # Move data.mat into the dataset directory
      if [[ -f "data.mat" ]]; then
        mv data.mat palace-fine-arts-281/
        echo "Moved data.mat to palace-fine-arts-281/"
      fi
      ;;
      
    "2011205_rc3")
      safe_extract "2011205_rc3.zip"
      if tar -xf cache_rc3_deep.tar.gz && rm cache_rc3_deep.tar.gz; then
        # Move cache files to main project cache directory (cache_rc3_deep.tar.gz extracts to cache/)
        if [[ -d "cache" ]]; then
          move_cache_files "cache"
        fi
      else
        echo "Error: Failed to extract cache files for 2011205_rc3"
        return 1
      fi
      ;;
      
    "gerrard-hall-100")
      safe_extract "gerrard-hall.zip"
      ;;
      
    "south-building-128")
      safe_extract "south-building.zip"
      ;;
      
    *)
      echo "Error: Unknown dataset extraction for ${DATASET_NAME}"
      return 1
      ;;
  esac
}

# Retry in case of corrupted file ("End-of-central-directory signature not found")
if retry 5 download_and_unzip_dataset_files; then
  echo "✅ Successfully downloaded and extracted dataset: ${DATASET_NAME}"
else
  echo "❌ Failed to download and extract dataset: ${DATASET_NAME}"
  exit 1
fi
