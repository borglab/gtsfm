#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previously installed using conda
##########################################################

echo "Running .github/scripts/python.sh..."
conda init
conda info --envs

wget -O 2021_03_12_gtsam_python38_wheel.zip --no-check-certificate "https://drive.google.com/uc?export=download&id=1JvsLYngMzSqvmD6RwEgEb_HXXHfw_EJ2"

unzip 2021_03_12_gtsam_python38_wheel.zip
pip install gtsam-4.1.1-cp38-cp38-manylinux2014_x86_64.whl

##########################################################
# Install GTSFM as a module
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .

##########################################################
# Run GTSFM unit tests
##########################################################

cd $GITHUB_WORKSPACE
pytest tests

# check that main script executes on toy Door dataset
python gtsfm/runner/run_scene_optimizer.py
