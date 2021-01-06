#!/bin/bash

##########################################################
# Install GTSFM dependencies (including GTSAM) using conda
##########################################################

cd $GITHUB_WORKSPACE

conda env create -f environment.yml
conda activate gtsfm-v1

##########################################################
# Install GTSFM as a module
##########################################################

pip install -e .

##########################################################
# Run GTSFM unit tests
##########################################################

cd $GITHUB_WORKSPACE/tests
python -m unittest discover
