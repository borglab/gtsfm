#!/bin/bash

DATASET_NAME=$1
CONFIG_NAME=$2
MAX_FRAME_LOOKAHEAD=$3
LOADER_NAME=$4
MAX_RESOLUTION=$5
SHARE_INTRINSICS=$6

# Base directory for all benchmark datasets
BENCHMARK_DIR="benchmarks"

# Extract the data, configure arguments for runner.
if [ "$DATASET_NAME" == "door-12" ]; then
  DATASET_ROOT=tests/data/set1_lund_door
elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
  DATASET_ROOT="${BENCHMARK_DIR}/palace-fine-arts-281"
elif [ "$DATASET_NAME" == "2011205_rc3" ]; then
  DATASET_ROOT="${BENCHMARK_DIR}/2011205_rc3"
elif [ "$DATASET_NAME" == "skydio-8" ]; then
  IMAGES_DIR=${BENCHMARK_DIR}/skydio_crane_mast_8imgs_with_exif/images
  COLMAP_FILES_DIRPATH=${BENCHMARK_DIR}/skydio_crane_mast_8imgs_with_exif/crane_mast_8imgs_colmap_output
elif [ "$DATASET_NAME" == "skydio-32" ]; then
  IMAGES_DIR=${BENCHMARK_DIR}/skydio-32/images
  COLMAP_FILES_DIRPATH=${BENCHMARK_DIR}/skydio-32/colmap_crane_mast_32imgs
elif [ "$DATASET_NAME" == "skydio-501" ]; then
  IMAGES_DIR="${BENCHMARK_DIR}/skydio-crane-mast-501-images"
  COLMAP_FILES_DIRPATH="${BENCHMARK_DIR}/skydio-501-colmap-pseudo-gt"
elif [ "$DATASET_NAME" == "notre-dame-20" ]; then
  IMAGES_DIR=${BENCHMARK_DIR}/notre-dame-20/images
  COLMAP_FILES_DIRPATH=${BENCHMARK_DIR}/notre-dame-20/notre-dame-20-colmap
elif [ "$DATASET_NAME" == "gerrard-hall-100" ]; then
  IMAGES_DIR=${BENCHMARK_DIR}/gerrard-hall/images
  COLMAP_FILES_DIRPATH=${BENCHMARK_DIR}/gerrard-hall/sparse
elif [ "$DATASET_NAME" == "south-building-128" ]; then
  IMAGES_DIR=${BENCHMARK_DIR}/south-building/images
  COLMAP_FILES_DIRPATH=${BENCHMARK_DIR}/south-building/sparse
fi

echo "Config: ${CONFIG_NAME}, Loader: ${LOADER_NAME}"
echo "Max. Frame Lookahead: ${MAX_FRAME_LOOKAHEAD}, Max. Resolution: ${MAX_RESOLUTION}"
echo "Share intrinsics for all images? ${SHARE_INTRINSICS}"

# Setup the command line arg if intrinsics are to be shared
if [ "$SHARE_INTRINSICS" == "true" ]; then
  export SHARE_INTRINSICS_ARG="--share_intrinsics"
else
  export SHARE_INTRINSICS_ARG=""
fi

# Run GTSFM on the dataset.
if [ "$LOADER_NAME" == "olsson-loader" ]; then
  python gtsfm/runner/run_scene_optimizer_olssonloader.py \
    --dataset_root $DATASET_ROOT \
    --config_name unified \
    --correspondence_generator_config_name ${CONFIG_NAME} \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}

elif [ "$LOADER_NAME" == "colmap-loader" ]; then
  python gtsfm/runner/run_scene_optimizer_colmaploader.py \
    --images_dir ${IMAGES_DIR} \
    --colmap_files_dirpath $COLMAP_FILES_DIRPATH \
    --config_name unified \
    --correspondence_generator_config_name ${CONFIG_NAME} \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}

elif [ "$LOADER_NAME" == "astrovision" ]; then
  python gtsfm/runner/run_scene_optimizer_astrovision.py \
    --data_dir $DATASET_ROOT \
    --config_name unified \
    --correspondence_generator_config_name ${CONFIG_NAME} \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}
fi
