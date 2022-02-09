#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previously installed using conda
##########################################################

echo "Running .github/scripts/setup.sh..."
conda init
conda info --envs

wget https://github.com/johnwlambert/gtsfm-datasets-mirror/releases/download/v4.2a5/gtsam-4.2a5-cp38-cp38-manylinux2014_x86_64.whl.zip

unzip gtsam-4.2a5-cp38-cp38-manylinux2014_x86_64.whl.zip
pip install gtsam-4.2a5-cp38-cp38-manylinux2014_x86_64.whl

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
