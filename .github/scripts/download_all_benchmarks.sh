#!/bin/bash
# This script downloads all benchmark datasets by calling download_single_benchmark.sh for each dataset.
#
# This script takes one argument:
#   1) DATASET_SRC: source from which to download the datasets. Supported sources are:
#       - wget
#       - test_data

DATASET_SRC=$1

if [ -z "$DATASET_SRC" ]; then
    echo "Usage: $0 <DATASET_SRC>"
    echo "DATASET_SRC options: wget, test_data"
    exit 1
fi

echo "Downloading all benchmarks using source: ${DATASET_SRC}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of all supported datasets
DATASETS=(
    "skydio-8"
    "skydio-32" 
    "skydio-501"
    "notre-dame-20"
    "palace-fine-arts-281"
    "2011205_rc3"
    "gerrard-hall-100"
    "south-building-128"
)

# Download each dataset
for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================="
    echo "Downloading dataset: $dataset"
    echo "========================================="
    
    if ! bash "$SCRIPT_DIR/download_single_benchmark.sh" "$dataset" "$DATASET_SRC"; then
        echo "ERROR: Failed to download $dataset"
        exit 1
    fi
    
    echo "Successfully downloaded: $dataset"
done

echo ""
echo "========================================="
echo "All benchmarks downloaded successfully!"
echo "========================================="