# Conda Setup Guide

This guide covers installation of GTSfM using Conda package manager.

## Prerequisites

- [MiniConda](https://www.anaconda.com/docs/getting-started/miniconda/install) installed on your system

## Create a Conda Environment

To run GTSfM, first, we need to create a conda environment with the required dependencies.

### Linux (with CUDA support)

```bash
conda env create -f environment_linux.yml
conda activate gtsfm-v1 # you may need "source activate gtsfm-v1" depending upon your bash and conda set-up
```

Check your cuda version then install `torch_scatter` from pre-built wheels

For example, for CUDA 12.1, use cu121 
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
```

For CUDA 13, use cu128
```bash
pip install torch-scatter --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

### macOS (no CUDA support)

```bash
conda env create -f environment_mac.yml
conda activate gtsfm-v1
```

## Install `gtsfm` as a module

Now, install `gtsfm` as a module:

```bash
pip install -e .
```

## Verify Installation


Make sure that you can run `python -c "import gtsfm; import gtsam; print('hello world')"` in python, and you are good to go!

