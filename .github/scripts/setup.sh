#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previously installed using conda
##########################################################

echo "Running .github/scripts/setup.sh..."
conda init
conda info --envs

##########################################################
# Install PyTorch (CPU version)
##########################################################

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

##########################################################
# Install GTSFM as a module
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .
git submodule init
git submodule update

##########################################################
# Download pre-trained model weights
##########################################################

cd $GITHUB_WORKSPACE
./download_model_weights.sh
