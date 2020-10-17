  
#!/bin/bash

##########################################################
# Build the GTSAM Python wrapper, then run GTSFM unit tests
##########################################################

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

cd $GITHUB_WORKSPACE/tests
python -m unittest discover
