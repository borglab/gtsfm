#!/bin/bash

##########################################################
# GTSFM Environment Setup Script (uv version)
##########################################################

echo "Running .github/scripts/setup.sh..."

cd $GITHUB_WORKSPACE

##########################################################
# Git submodules
##########################################################
git submodule update --init --recursive

##########################################################
# Download pre-trained model weights
##########################################################
./scripts/download_model_weights.sh