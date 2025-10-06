#!/bin/bash

# Upgrade pip to the latest version
pip install --upgrade pip

# Install core Python dependencies
pip install black coverage mypy pylint pytest flake8 isort dask[complete] asyncssh graphviz networkx numpy pandas \
    pillow>=8.0.1 scikit-learn seaborn scipy hydra-core torch torchvision>=0.13.0 kornia pycolmap h5py tabulate \
    simplejson parameterized open3d opencv-python>=4.5.4.58 pydegensac colour trimesh[easy] gtsam==4.2 pydot

# Install Matplotlib separately
pip install matplotlib>=3.8.0

# Install Plotly
pip install plotly

# Install Node.js
apt-get update && apt-get install -y nodejs || echo "Failed to install Node.js"

# Verify installations and check for dependency conflicts
pip list

# Install GTSFM package
pip install -e .

# Verify installation by running help command
python gtsfm/runner/run_scene_optimizer_olssonloader.py -h

# Install necessary visualization libraries
pip install pydeck pycolmap
