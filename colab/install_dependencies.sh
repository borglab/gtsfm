#!/bin/bash
echo "Installing GTSFM dependencies..."

# Upgrade pip to the latest version
echo "Upgrading pip..."
python -m pip install --upgrade pip

# Install core Python dependencies
echo "Installing core dependencies..."
pip install black coverage mypy pylint pytest flake8 isort \
    dask[complete] asyncssh graphviz networkx numpy pandas \
    pillow>=8.0.1 scikit-learn seaborn scipy hydra-core \
    torch torchvision>=0.13.0 kornia pycolmap h5py tabulate \
    simplejson parameterized open3d opencv-python>=4.5.4.58 \
    colour trimesh[easy] pydot \
    matplotlib>=3.8.0 plotly pydeck

# Try to install pydegensac (optional, fails on Python 3.12+)
echo "Attempting to install pydegensac (optional)..."
pip install pydegensac || echo "Warning: pydegensac installation failed (expected on Python 3.12+). Some legacy features may be unavailable."

# Install additional visualization and training dependencies
echo "Installing visualization and training dependencies..."
pip install gsplat torchmetrics

# Install GTSAM separately to avoid dependency conflicts
# Note: Version unpinned to allow compatibility with Python 3.12+
echo "Installing GTSAM..."
pip install gtsam

# Install Node.js for visualization tools
echo "Installing Node.js..."
apt-get update && apt-get install -y nodejs || echo "Warning: Failed to install Node.js"

# Install GTSFM package in editable mode
echo "Installing GTSFM package in editable mode..."
pip install -e .

# Verify installation
echo "Verifying installation..."
python gtsfm/runner.py --help

echo "Installation complete!"

