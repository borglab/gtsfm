#!/bin/bash

# Datasets are stored in /usr/local/gtsfm-data in order to reduce runtimes by not 
# redownloading each (very large) dataset every CI run. Any new datasets must be
# downloaded and stored in /usr/local/gtsfm-data before running this action.
DATASET_PREFIX=/home/akrishnan86/gtsfm/data

DATASET_NAME=$1
CONFIG_NAME=$2
MAX_FRAME_LOOKAHEAD=$3
LOADER_NAME=$4
MAX_RESOLUTION=$5
SHARE_INTRINSICS=$6

# Extract the data, configure arguments for runner.
if [ "$DATASET_NAME" == "skydio-501" ]; then
  IMAGES_DIR="skydio-crane-mast-501-images"
  COLMAP_FILES_DIRPATH="skydio-501-colmap-pseudo-gt"
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
  ./run \
    --loader olsson_loader \
    --dataset_dir $DATASET_PREFIX/$DATASET_ROOT \
    --config_name ${CONFIG_NAME}.yaml \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}

#     --correspondence_generator_config_name loftr.yaml \

elif [ "$LOADER_NAME" == "colmap-loader" ]; then
  ./run \
    --loader colmap_loader \
    --dataset_dir $DATASET_PREFIX/$COLMAP_FILES_DIRPATH \
    --images_dir ${IMAGES_DIR} \
    --use_gt_intrinsics \
    --use_gt_extrinsics \
    --config_name ${CONFIG_NAME}.yaml \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG} \
    --num_workers 1

elif [ "$LOADER_NAME" == "astrovision" ]; then
  ./run \
    --loader astrovision_loader \
    --dataset_dir $DATASET_PREFIX/$DATASET_ROOT \
    --config_name ${CONFIG_NAME}.yaml \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}
fi
