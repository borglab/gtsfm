#!/bin/bash

##########################################################
# Build the GTSAM Python wrapper, then run GTSFM unit tests
##########################################################

# Install a system package required by our library
# maybe fix pyparsing error?
sudo apt-get install -y wget libicu-dev python3-pip python3-setuptools

git clone https://github.com/borglab/gtsam.git
cd gtsam
sudo $PYTHON -m pip install -r python/requirements.txt

mkdir build
cd build

echo "Building with Python ${PYTHON_VERSION}"

cmake .. -DGTSAM_BUILD_PYTHON=1 \
    -DGTSAM_PYTHON_VERSION=$PYTHON_VERSION \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc) install

make python-install

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
