#!/bin/bash

##########################################################
# GTSFM dependencies (including GTSAM) previously installed using conda
##########################################################

echo "Running .github/scripts/setup.sh..."
conda init
conda info --envs

##########################################################
# Install GTSFM as a module
##########################################################

cd $GITHUB_WORKSPACE

if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "✅ CUDA detected, installing with GPU support..."
    pip install -e ".[cuda]"
else
    echo "⚠️ No CUDA detected, installing CPU version..."
    pip install -e .
fi

git submodule update --init --recursive

##########################################################
# Download pre-trained model weights
##########################################################

cd $GITHUB_WORKSPACE
./scripts/download_model_weights.sh
