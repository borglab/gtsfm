#!/bin/bash

##########################################################
# Install GTSFM dependencies (including GTSAM) using conda
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .

##########################################################
# Run GTSFM unit tests
##########################################################

cd $GITHUB_WORKSPACE/tests
python -m unittest discover
