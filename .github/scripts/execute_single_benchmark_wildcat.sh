#!/bin/bash

# Datasets are stored in ~/sfm_datasets in order to reduce runtimes by not 
# redownloading each (very large) dataset every CI run. Any new datasets must be
# downloaded and stored in ~/sfm_datasets before running this action.
DATASET_PREFIX=/home/tdriver6/sfm_datasets

DATASET_NAME=$1
CONFIG_NAME=$2
MAX_FRAME_LOOKAHEAD=$3
IMAGE_EXTENSION=$4
LOADER_NAME=$5
MAX_RESOLUTION=$6
SHARE_INTRINSICS=$7

# Extract the data, configure arguments for runner.
if [ "$DATASET_NAME" == "skydio-501" ]; then
  IMAGES_DIR="skydio-crane-mast-501-images"
  COLMAP_FILES_DIRPATH="skydio-501-colmap-pseudo-gt"

if [ "$DATASET_NAME" == "gendarmenmarkt" ]; then
  IMAGES_DIR=Gendarmenmarkt/images
fi

echo "Config: ${CONFIG_NAME}, Loader: ${LOADER_NAME}"
echo "Max. Frame Lookahead: ${MAX_FRAME_LOOKAHEAD}, Image Extension: ${IMAGE_EXTENSION}, Max. Resolution: ${MAX_RESOLUTION}"
echo "Share intrinsics for all images? ${SHARE_INTRINSICS}"

# Setup the command line arg if intrinsics are to be shared
if [ "$SHARE_INTRINSICS" == "true" ]; then
  export SHARE_INTRINSICS_ARG="--share_intrinsics"
else
  export SHARE_INTRINSICS_ARG=""
fi

# Run GTSFM on the dataset.
if [ "$LOADER_NAME" == "onedsfm-loader" ]; then
  python gtsfm/runner/run_scene_optimizer_1dsfm.py \
    --dataset_root $DATASET_PREFIX/$IMAGES_DIR \
    --config_name ${CONFIG_NAME}.yaml \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG} \
    --num_workers 1 \
    --mvs_off
 

elif [ "$LOADER_NAME" == "colmap-loader" ]; then
  python gtsfm/runner/run_scene_optimizer_colmaploader.py \
    --images_dir $DATASET_PREFIX/${IMAGES_DIR} \
    --colmap_files_dirpath $DATASET_PREFIX/$COLMAP_FILES_DIRPATH \
    --config_name ${CONFIG_NAME}.yaml \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG} \
    --num_workers 1 \
    --mvs_off
fi
