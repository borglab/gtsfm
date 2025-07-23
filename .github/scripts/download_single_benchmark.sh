#!/bin/bash

DATASET_NAME=$1
DATASET_SRC=$2

echo "Dataset: ${DATASET_NAME}, Download Source: ${DATASET_SRC}"

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

# The last command executed in this function is `unzip`, which will return a non-zero exit code upon failure
function download_and_unzip_dataset_files {
  # Prepare the download URLs.
  if [ "$DATASET_NAME" == "skydio-8" ]; then
    # Description: 8 images from Skydio-501 facing a single crane face
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/skydio_crane_mast_8imgs_with_exif.zip
    ZIP_FNAME=skydio-8.zip

  elif [ "$DATASET_NAME" == "skydio-32" ]; then
    # Description: 32 images from Skydio-501 facing a single crane face
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/skydio_crane_mast_32imgs_w_colmap_GT.zip
    ZIP_FNAME=skydio-32.zip

  elif [ "$DATASET_NAME" == "skydio-501" ]; then
    # 501-image Crane Mast collection released by Skydio via Sketchfab
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-crane-mast-501-images/skydio-crane-mast-501-images1.tar.gz
    WGET_URL2=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-crane-mast-501-images/skydio-crane-mast-501-images2.tar.gz
    WGET_URL3=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/skydio-501-colmap-pseudo-gt/skydio-501-colmap-pseudo-gt.tar.gz

  elif [ "$DATASET_NAME" == "notre-dame-20" ]; then
    # Description: TODO
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/gtsfm-ci-small-datasets/notre-dame-20.zip
    ZIP_FNAME=notre-dame-20.zip

  elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
    # Description: 281 images captured at the Palace of Fine Arts in San Francisco, CA. Images and pseudo-ground truth
    # poses from Carl Olsson's page: https://www.maths.lth.se/matematiklth/personal/calle/dataset/dataset.html
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/palace-fine-arts-281/fine_arts_palace.zip
    WGET_URL2=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/palace-fine-arts-281/data.mat
    ZIP_FNAME=fine_arts_palace.zip

  elif [ "$DATASET_NAME" == "2011205_rc3" ]; then
    # Description: images captured during the Rotation Characterization 3 (RC3) phase of NASA's Dawn mission to Asteroid 4
    #   Vesta.
    WGET_URL1=https://www.dropbox.com/s/q02mgq1unbw068t/2011205_rc3.zip
    WGET_URL2=https://github.com/johnwlambert/gtsfm-cache/releases/download/2011205_rc3_deep_front_end_cache/cache_rc3_deep.tar.gz
    ZIP_FNAME=2011205_rc3.zip

  elif [ "$DATASET_NAME" == "gerrard-hall-100" ]; then
    WGET_URL1=https://github.com/colmap/colmap/releases/download/3.11.1/gerrard-hall.zip
    ZIP_FNAME=gerrard-hall.zip

  elif [ "$DATASET_NAME" == "south-building-128" ]; then
    WGET_URL1=https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/south-building-128/south-building-128.zip
    ZIP_FNAME=south-building-128.zip
  fi

  # Download the data.
  if [ "$DATASET_SRC" == "wget" ]; then
    echo "Downloading ${DATASET_NAME} with WGET"
    retry 10 wget $WGET_URL1

    # Check if $WGET_URL2 has been set.
    if [ ! -z "$WGET_URL2" ]; then
      retry 10 wget $WGET_URL2
    fi
    # Check if $WGET_URL3 has been set.
    if [ ! -z "$WGET_URL3" ]; then
      retry 10 wget $WGET_URL3
    fi
    echo $WGET_URL1
    echo $WGET_URL2
  fi

  # Extract the data, configure arguments for runner.
  if [ "$DATASET_NAME" == "skydio-8" ]; then
    unzip -qq skydio_crane_mast_8imgs_with_exif.zip

  elif [ "$DATASET_NAME" == "skydio-32" ]; then
    unzip -qq skydio_crane_mast_32imgs_w_colmap_GT.zip -d skydio-32

  elif [ "$DATASET_NAME" == "skydio-501" ]; then
    tar -xvzf skydio-crane-mast-501-images1.tar.gz
    tar -xvzf skydio-crane-mast-501-images2.tar.gz
    tar -xvzf skydio-501-colmap-pseudo-gt.tar.gz
    IMAGES_DIR="skydio-crane-mast-501-images"
    mkdir $IMAGES_DIR
    mv skydio-crane-mast-501-images1/* $IMAGES_DIR
    mv skydio-crane-mast-501-images2/* $IMAGES_DIR

    mkdir -p cache/detector_descriptor
    mkdir -p cache/matcher
    wget https://github.com/johnwlambert/gtsfm-cache/releases/download/skydio-501-lookahead50-deep-front-end-cache/skydio-501-lookahead50-deep-front-end-cache.tar.gz
    mkdir skydio-501-cache
    tar -xvzf skydio-501-lookahead50-deep-front-end-cache.tar.gz --directory skydio-501-cache
    cp skydio-501-cache/cache/detector_descriptor/* cache/detector_descriptor/
    cp skydio-501-cache/cache/matcher/* cache/matcher/

  elif [ "$DATASET_NAME" == "notre-dame-20" ]; then
    unzip -qq notre-dame-20.zip

  elif [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
    mkdir -p palace-fine-arts-281
    unzip -qq fine_arts_palace.zip -d palace-fine-arts-281/images

  elif [ "$DATASET_NAME" == "2011205_rc3" ]; then
    unzip -qq 2011205_rc3.zip
    tar -xvf cache_rc3_deep.tar.gz

  elif [ "$DATASET_NAME" == "gerrard-hall-100" ]; then
    unzip gerrard-hall.zip

  elif [ "$DATASET_NAME" == "south-building-128" ]; then
    unzip south-building-128.zip

  fi
}

# Retry in case of corrupted file ("End-of-central-directory signature not found")
retry 5 download_and_unzip_dataset_files

# Set up directories
if [ "$DATASET_NAME" == "palace-fine-arts-281" ]; then
  mv data.mat palace-fine-arts-281/
fi
