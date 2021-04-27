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

# check that main script executes on toy Door dataset
python gtsfm/runner/run_scene_optimizer.py

pytest tests --cov gtsfm
coverage report

##########################################################
# Lint with flake8 and pylint
##########################################################

pip install flake8
flake8 --max-line-length 120 --ignore E201,E202,E203,E231,W291,W293,E303,W391,E402,W503,E731 gtsfm
pylint --indent-string='    ' --generated-members=numpy.* ,torch.* ,cv2.* , cv.*, scipy.*, gtsam.* --max-line-length=120
