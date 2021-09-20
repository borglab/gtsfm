#!/bin/bash


CONFIG_NAME=$1
DATASET_NAME=$2
MAX_FRAME_LOOKAHEAD=$3
IMAGE_EXTENSION=$4
DATASET_SRC=$5
LOADER_NAME=$6
MAX_RESOLUTION=$7

function retry {
  local retries=$1
  shift

  local count=0
  until "$@"; do # actual command execution happening here, and will continue till signal 0 (success).
    exit=$?
    wait=$((2 ** $count))
    count=$(($count + 1))
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


echo "Config: ${CONFIG_NAME}, Dataset: ${DATASET_NAME}, Download Source: ${DATASET_SRC}, Loader: ${LOADER_NAME}"

# Prepare the download URLs.
if [ "$DATASET_NAME" == "skydio-8" ]; then
  # Description: TODO
  export GDRIVE_FILEID='1mmM1p_NpL7-pnf3iHWeWVKpsm1pcBoD5'

elif [ "$DATASET_NAME" == "skydio-32" ]; then
  # Description: TODO
  export GDRIVE_FILEID='1BQ6jp0DD3D9yhTnrDoEddzlMYT0RRH68'

elif [ "$DATASET_NAME" == "notre-dame-20" ]; then
  # Description: TODO
  export GDRIVE_FILEID='1t_CptH7ZWdKQVW-yw56bpLS83TntNQiK'

elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
  # Description: TODO
  WGET_URL1=http://vision.maths.lth.se/calledataset/fine_arts_palace/fine_arts_palace.zip
  WGET_URL2=http://vision.maths.lth.se/calledataset/fine_arts_palace/data.mat

elif [ "$DATASET_NAME" == "2011205_rc3" ]; then
  # Description: images captured during the Rotation Characterization 3 (RC3) phase of NASA's Dawn mission to Asteroid 4
  #   Vesta.
  WGET_URL1=https://www.dropbox.com/s/q02mgq1unbw068t/2011205_rc3.zip
fi

# Download the data.
if [ "$DATASET_SRC" == "gdrive" ]; then
  echo "Downloading ${DATASET_NAME} from GDRIVE"
  export GDRIVE_URL='https://docs.google.com/uc?export=download&id='$GDRIVE_FILEID
  retry 3 wget --save-cookies cookies.txt $GDRIVE_URL -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt
  retry 3 wget --load-cookies cookies.txt -O ${DATASET_NAME}.zip $GDRIVE_URL'&confirm='$(<confirm.txt)

elif [ "$DATASET_SRC" == "wget" ]; then
  echo "Downloading ${DATASET_NAME} with WGET"
  retry 3 wget $WGET_URL1

  # Check if $WGET_URL2 has been set.
  if [ ! -z "$WGET_URL2" ]; then
    retry 3 wget $WGET_URL2
  fi
  echo $WGET_URL1
  echo $WGET_URL2
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

<<<<<<< HEAD
elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then 
=======
elif [ "$DATASET_NAME" == "notre-dame-20" ]; then
  unzip -qq notre-dame-20.zip
  COLMAP_FILES_DIRPATH=notre-dame-20/notre-dame-20-colmap
  IMAGES_DIR=notre-dame-20/images

elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then \
>>>>>>> 941761452178fae4c12f3ad12b3dc95b8455ff44
  mkdir palace-fine-arts-281
  unzip -qq fine_arts_palace.zip -d palace-fine-arts-281/images
  mv data.mat palace-fine-arts-281/
  DATASET_ROOT="palace-fine-arts-281"

elif [ "$DATASET_NAME" == "2011205_rc3" ]; then 
  unzip -qq 2011205_rc3.zip
  DATASET_ROOT="2011205_rc3"
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

elif [ "$LOADER_NAME" == "astronet" ]; then
  python gtsfm/runner/run_scene_optimizer_astronet.py \
  --data_dir $DATASET_ROOT \
  --max_frame_lookahead $MAX_FRAME_LOOKAHEAD \
  --config_name ${CONFIG_NAME}.yaml \
  --max_resolution ${MAX_RESOLUTION}
fi
