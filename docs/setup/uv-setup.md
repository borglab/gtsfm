# UV Setup Guide

This guide covers installation of GTSfM using UV, a fast Python package manager.

## Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh 
```

## Install System Packages

Before setting up the Python environment, install required system packages:

### Linux
```bash
sudo apt-get install nodejs npm graphviz
```

### macOS
```bash
brew install node graphviz
```

## Basic Installation

Navigate to the GTSfM directory:

```bash
cd path/to/gtsfm

# Clean existing environment (if any)
rm -rf .venv

# Install on Linux for CPU only and macOS
uv sync --python 3.12

# Install on Linux with CUDA GPU
uv sync --python 3.12 --extra complete
```
##  Install torch-scatter (platform-specific)
For NVIDIA drivers 550+ (which support CUDA 12.8), 
use cu128 because that's what PyTorch 2.7.0 was compiled with.
PyTorch brings its own CUDA runtime, system CUDA version doesn't matter.

```bash
uv pip install torch-scatter --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html

```

## Multi-GPU Installation (For Distributed Computing)

If you have multiple GPUs on the same machine and want to use Dask for distributed GPU computing:

```bash
# Multiple GPUs per node (e.g., 4x or 8x A100)
uv sync --python 3.12 --extra complete --extra multi-gpu
```

This adds `dask-cuda` for GPU-aware distributed scheduling.

### When do you need `--extra multi-gpu`?
- You have multiple GPUs on the same machine
- You want to use Dask to distribute work across GPUs
- You're running on a GPU cluster node

### When you DON'T need it:
- Single GPU workstation
- Laptop with one GPU
- CPU-only machines
- Multiple machines (handled differently)

## Verify Installation

```bash
uv run python -c "import gtsfm; import pydegensac; import torch; import torch_scatter; print('✅ Success')"
```

## Quick Test

Test the installation with a sample dataset:

```bash
uv run ./run --dataset_dir tests/data/set1_lund_door \
--config_name unified_binary.yaml \
--loader olsson \
--num_workers 2 graph_partitioner.max_depth=1
```

Or run a benchmark:

```bash
uv run .github/scripts/execute_single_benchmark.sh skydio-8 lightglue 15 colmap-loader 760 true
```

## Managing Packages with UV

### Adding a new package:
```bash
uv add <package-name>
# Example: uv add numpy
```

### Adding a development dependency:
```bash
uv add --dev <package-name>
# Example: uv add --dev pytest
```

### Removing a package:
```bash
uv remove <package-name>
# Example: uv remove numpy
```

### Installing a package without adding to dependencies:
```bash
uv pip install <package-name>
```

## When to Use `uv lock`

The `uv lock` command updates the lock file (`uv.lock`) without installing packages. Use it when:

- **After manually editing `pyproject.toml`:** When you directly modify dependencies in the configuration file
- **To update dependencies:** When you want to resolve and lock new versions without installing
- **In CI/CD pipelines:** To ensure reproducible builds by generating a lock file
- **Before committing changes:** To update the lock file for team members

```bash
# Update lock file after editing pyproject.toml
uv lock

# Update lock file and install packages
uv lock && uv sync
```

> **Note:** `uv add` and `uv remove` automatically update the lock file, so you typically don't need to run `uv lock` manually after these commands.

## Running Commands with UV

You have two options for running Python commands with UV:

### Option 1: Activate the Virtual Environment 

Activate the virtual environment once per shell session:

**Linux/macOS:**
```bash
source .venv/bin/activate
```

After activation, you can run commands directly without the `uv run` prefix:

```bash
# Verify installation
python -c "import gtsfm; import gtsam; print('hello world')"

```

To deactivate the environment:
```bash
deactivate
```

### Option 2: Use `uv run` Prefix

Prefix Python commands with `uv run` (no activation needed):

```bash
# Verify installation
uv run python -c "import gtsfm; import gtsam; print('hello world')"

```


