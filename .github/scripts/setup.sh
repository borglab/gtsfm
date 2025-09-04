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
git submodule update --init --recursive

##########################################################
# Download pre-trained model weights
##########################################################

cd $GITHUB_WORKSPACE
./download_model_weights.sh
