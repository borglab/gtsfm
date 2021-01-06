#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previoulsy installed using conda
##########################################################

##########################################################
# Install GTSFM as a module
##########################################################

cd $GITHUB_WORKSPACE
pip install -e .

##########################################################
# Run GTSFM unit tests
##########################################################

cd $GITHUB_WORKSPACE/tests
python -m unittest discover
