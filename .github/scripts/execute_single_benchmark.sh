#!/bin/bash


CONFIG_NAME=$1
DATASET_NAME=$2
MAX_FRAME_LOOKAHEAD=$3
IMAGE_EXTENSION=$4
DATASET_SRC=$5
LOADER_NAME=$6
MAX_RESOLUTION=$7


echo "Config: ${CONFIG_NAME}, Dataset: ${DATASET_NAME}, Download Source: ${DATASET_SRC}, Loader: ${LOADER_NAME}"

# Prepare the download URLs.
if [ "$DATASET_NAME" == "skydio-8" ]; then
  export GDRIVE_FILEID='1mmM1p_NpL7-pnf3iHWeWVKpsm1pcBoD5'
elif [ "$DATASET_NAME" == "skydio-32" ]; then
  export GDRIVE_FILEID='1BQ6jp0DD3D9yhTnrDoEddzlMYT0RRH68'
elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
  WGET_URL1=http://vision.maths.lth.se/calledataset/fine_arts_palace/fine_arts_palace.zip
  WGET_URL2=http://vision.maths.lth.se/calledataset/fine_arts_palace/data.mat
  echo $WGET_URL1
  echo $WGET_URL2
fi

# Download the data.
if [ "$DATASET_SRC" == "gdrive" ]; then
  echo "Downloading ${DATASET_NAME} from GDRIVE"
  export GDRIVE_URL='https://docs.google.com/uc?export=download&id='$GDRIVE_FILEID
  wget --save-cookies cookies.txt $GDRIVE_URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
  wget --load-cookies cookies.txt -O ${DATASET_NAME}.zip $GDRIVE_URL'&confirm='$(<confirm.txt)

elif [ "$DATASET_SRC" == "wget" ]; then
  echo "Downloading ${DATASET_NAME} with WGET"
  wget $WGET_URL1
  wget $WGET_URL2
fi

# Extract the data, configure arguments for runner.
if [ "$DATASET_NAME" == "door-12" ]; then
  DATASET_ROOT=tests/data/set1_lund_door

elif [ "$DATASET_NAME" == "skydio-8" ]; then
  unzip -qq skydio-8.zip
  IMAGES_DIR=skydio_crane_mast_8imgs_with_exif/images
  COLMAP_FILES_DIRPATH=skydio_crane_mast_8imgs_with_exif/crane_mast_8imgs_colmap_output

elif [ "$DATASET_NAME" == "skydio-32" ]; then
  unzip -qq skydio-32.zip -d skydio-32
  COLMAP_FILES_DIRPATH=skydio-32/colmap_crane_mast_32imgs
  IMAGES_DIR=skydio-32/images

elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then \
  mkdir palace-fine-arts-281
  unzip -qq fine_arts_palace.zip -d palace-fine-arts-281/images
  mv data.mat palace-fine-arts-281/
  DATASET_ROOT="palace-fine-arts-281"
fi


# Run GTSFM on the dataset.
if [ "$LOADER_NAME" == "olsson-loader" ]; then
  python gtsfm/runner/run_scene_optimizer.py \
  --dataset_root $DATASET_ROOT \
  --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
  --config_name ${CONFIG_NAME}.yaml \
  --image_extension $IMAGE_EXTENSION \
  --max_resolution ${MAX_RESOLUTION}

elif [ "$LOADER_NAME" == "colmap-loader" ]; then
  python gtsfm/runner/run_scene_optimizer_colmaploader.py \
  --images_dir ${IMAGES_DIR} \
  --colmap_files_dirpath $COLMAP_FILES_DIRPATH \
  --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
  --config_name ${CONFIG_NAME}.yaml \
  --max_resolution ${MAX_RESOLUTION}
fi
