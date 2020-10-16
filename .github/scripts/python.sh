  
#!/bin/bash

##########################################################
# Build the GTSAM Python wrapper, then run GTSFM unit tests
##########################################################

git clone https://github.com/borglab/gtsam.git
cd gtsam
sudo $PYTHON -m pip install -r python/requirements.txt

mkdir build
cd build

cmake .. -DGTSAM_BUILD_PYTHON=1 \
    -DGTSAM_PYTHON_VERSION=3.8.0 \
    -DCMAKE_BUILD_TYPE=Release

make -j$(nproc) install

make python-install

cd $GITHUB_WORKSPACE/tests
python -m unittest discover
