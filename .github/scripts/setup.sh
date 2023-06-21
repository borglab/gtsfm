#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previously installed using conda
##########################################################

echo "Running .github/scripts/setup.sh..."
conda init
conda info --envs

##########################################################
# Install GTSFM as a module
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .
wget https://www.dropbox.com/s/ukadst2l367z8qy/pytheia-0.1.22-cp38-cp38-manylinux_2_17_x86_64.whl
pip install pytheia-0.1.22-cp38-cp38-manylinux_2_17_x86_64.whl

##########################################################
# Download pre-trained model weights
##########################################################

cd $GITHUB_WORKSPACE
./download_model_weights.sh
