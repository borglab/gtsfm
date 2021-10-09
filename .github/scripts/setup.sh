#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previously installed using conda
##########################################################

echo "Running .github/scripts/setup.sh..."
conda init
conda info --envs

wget -O 2021_09_02_gtsam_python38_wheel.zip --no-check-certificate "https://drive.google.com/uc?export=download&id=1GSf3NxabSCsu2nPcBUuEhhVq_m1FvdY5"

unzip 2021_09_02_gtsam_python38_wheel.zip
pip install gtsam-4.1.0-cp38-cp38-manylinux2014_x86_64.whl

##########################################################
# Install GTSFM as a module
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .

##########################################################
# Download pre-trained model weights
##########################################################

cd $GITHUB_WORKSPACE
./download_model_weights.sh

##########################################################
# Download the front-end cache (temporary)
##########################################################
wget https://github.com/ayushbaid/gtsfm-cache/archive/main.zip
unzip main.zip 
mv gtsfm-cache-main cache
