#!/bin/bash

DATASET_NAME=$1
CONFIG_NAME=$2
MAX_FRAME_LOOKAHEAD=$3
IMAGE_EXTENSION=$4
LOADER_NAME=$5
MAX_RESOLUTION=$6
SHARE_INTRINSICS=$7

# Extract the data, configure arguments for runner.
if [ "$DATASET_NAME" == "door-12" ]; then
  DATASET_ROOT=tests/data/set1_lund_door
elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
  DATASET_ROOT="palace-fine-arts-281"
elif [ "$DATASET_NAME" == "2011205_rc3" ]; then
  DATASET_ROOT="2011205_rc3"
elif [ "$DATASET_NAME" == "skydio-8" ]; then
  IMAGES_DIR=skydio_crane_mast_8imgs_with_exif/images
  COLMAP_FILES_DIRPATH=skydio_crane_mast_8imgs_with_exif/crane_mast_8imgs_colmap_output
elif [ "$DATASET_NAME" == "skydio-32" ]; then
  IMAGES_DIR=skydio-32/images
  COLMAP_FILES_DIRPATH=skydio-32/colmap_crane_mast_32imgs
elif [ "$DATASET_NAME" == "skydio-501" ]; then
  IMAGES_DIR="skydio-crane-mast-501-images"
  COLMAP_FILES_DIRPATH="skydio-501-colmap-pseudo-gt"
elif [ "$DATASET_NAME" == "notre-dame-20" ]; then
  IMAGES_DIR=notre-dame-20/images
  COLMAP_FILES_DIRPATH=notre-dame-20/notre-dame-20-colmap
elif [ "$DATASET_NAME" == "lettuce-18" ]; then
  IMAGES_DIR="lettuce-18/images"
  COLMAP_FILES_DIRPATH="lettuce-18/colmap_files"
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

echo "Share intrinsics CLI argument: ${SHARE_INTRINSICS_ARG}"

# Run GTSFM on the dataset.
if [ "$LOADER_NAME" == "olsson-loader" ]; then
  python gtsfm/runner/run_scene_optimizer_olssonloader.py \
    --dataset_root $DATASET_ROOT \
    --image_extension $IMAGE_EXTENSION \
    --config_name ${CONFIG_NAME}.yaml \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}

elif [ "$LOADER_NAME" == "colmap-loader" ]; then
  python gtsfm/runner/run_scene_optimizer_colmaploader.py \
    --images_dir ${IMAGES_DIR} \
    --colmap_files_dirpath $COLMAP_FILES_DIRPATH \
    --config_name ${CONFIG_NAME}.yaml \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}

elif [ "$LOADER_NAME" == "astrovision" ]; then
  python gtsfm/runner/run_scene_optimizer_astrovision.py \
    --data_dir $DATASET_ROOT \
    --config_name ${CONFIG_NAME}.yaml \
    --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
    --max_resolution ${MAX_RESOLUTION} \
    ${SHARE_INTRINSICS_ARG}
fi
