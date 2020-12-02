#!/bin/bash

##########################################################
# Build the GTSAM Python wrapper, then run GTSFM unit tests
##########################################################

pip install gtsam

##########################################################
# Install GTSFM dependencies
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .

##########################################################
# Run GTSFM unit tests
##########################################################

cd $GITHUB_WORKSPACE/tests
python -m unittest discover
