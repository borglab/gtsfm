# Conda Setup Guide

GTSFM requires Python 3.12.x (`>=3.12,<3.13`). This guide uses Conda to manage a named Python environment and pip to install GTSFM and its declared dependencies. For an isolated tool or a `.venv`, see the [uv setup guide](uv-setup.md); both methods install the same GTSFM CLI and Studio interface.

## Prerequisites

Install Git and [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). Follow the Conda shell initialization instructions so `conda activate` works in your terminal.

Graphviz is optional for process-graph export (`sudo apt-get install graphviz` on Ubuntu or `brew install graphviz` on macOS). Node.js and a JavaScript package manager are needed only when developing the Studio frontend; see the [frontend development instructions](../../README.md#run-gtsfm).

## Prepare the checkout

For a new checkout:

```bash
git clone https://github.com/borglab/gtsfm.git
cd gtsfm
```

For an existing checkout, open a terminal in its repository root. Initialize the submodules before installing:

```bash
git submodule update --init --recursive
```

VGGT and other model implementations are bundled from these submodules when GTSFM is built. Neither Conda nor pip fetches the submodules automatically. Run the remaining setup commands from this directory.

## Create the environment

For a new environment:

```bash
conda create -n gtsfm-v1 python=3.12 pip
conda activate gtsfm-v1
```

If you already have a compatible environment, activate it instead. Use `python --version` to check that it runs Python 3.12.x. Keep the environment activated for the commands below, and activate it again in new terminals.

This guide installs dependencies from `pyproject.toml` rather than the separate `environment_*.yml` recipes. Those recipes currently differ from the package metadata: the Linux recipe pins an older PyTorch stack, the CPU recipe references Python 3.10 wheels despite selecting Python 3.12, and the macOS recipe includes CUDA-specific packages.

### Select PyTorch

pip does not use the PyTorch indexes in `[tool.uv.sources]`. To choose a specific build, install PyTorch before GTSFM. These examples use PyTorch 2.7.1 and torchvision 0.22.1, which satisfy the project's version constraints. Choose one command for your platform; see the [official PyTorch commands](https://pytorch.org/get-started/previous-versions/#v271).

Linux with an NVIDIA GPU and a driver supporting CUDA 12.8:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
```

Linux with CPU-only PyTorch:

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cpu
```

macOS (no CUDA):

```bash
python -m pip install torch==2.7.1 torchvision==0.22.1
```

## Install GTSFM

Choose one of the following methods.

### Install for use

```bash
python -m pip install .
```

This installs the `gtsfm` command inside the active Conda environment. No separate uv tool installation is needed to run Studio or the CLI.

### Set up a development environment

Use an editable installation so Python source changes are available without reinstalling:

```bash
python -m pip install -e .
python -m pip install black isort flake8 mypy pylint pytest pytest-cov coverage
```

Unlike `uv sync`, `pip install -e .` does not automatically install the project's development dependency group, so the second command supplies the development tools.

## Run GTSFM

With the Conda environment activated, check the command and detected hardware:

```bash
gtsfm --help
gtsfm hardware
python -m pip check
```

### Open GTSFM Studio

```bash
gtsfm run
```

This starts the local server and opens the browser workspace. Use `gtsfm run --no-browser` to start the server without opening a browser.

### Run a reconstruction from the terminal

Choose a configuration and a loader for your dataset, for example:

```bash
gtsfm execute -- --dataset_dir /path/to/data --config_name vggt --loader olsson
```

Use this CLI in the active Conda environment. The repository `./run` wrapper automatically invokes uv if it is installed, which can select a different environment. See the [usage guide](../../README.md#usage-guide-running-3d-reconstruction) for reconstruction options and model-weight setup.

## Optional GPU dependencies

The `complete` extra adds xformers on Linux. The `multi-gpu` extra adds Dask CUDA scheduling packages. Neither is required simply to use a single CUDA GPU. From the repository root, install the extras you need:

```bash
python -m pip install '.[complete]'
```

or:

```bash
python -m pip install '.[multi-gpu]'
```

Extras can be combined as `'.[complete,multi-gpu]'`. Add `-e` when maintaining an editable installation. An extra may change the selected PyTorch version to satisfy its requirements; check the resulting versions before installing binary extensions.

### AnySplat tracking dependencies

AnySplat's tracking stack additionally uses `torch-scatter`. Select a wheel matching PyTorch's version and CUDA build, rather than the CUDA version displayed by the driver:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

For PyTorch 2.7.x built with CUDA 12.8:

```bash
python -m pip install torch-scatter \
  --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

See the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels) for other combinations. This is required for AnySplat, not for the basic installation checks above.

## Manage development dependencies

Install additional Python packages with `python -m pip install <package>`. This changes the active environment without updating the project metadata. For a dependency needed by GTSFM itself, update `pyproject.toml` and its uv lock file as part of the code change.

Install any additional Conda-managed packages before pip packages when preparing a new environment. See [Conda's guidance on using pip](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#using-pip-in-an-environment) for managing environments that use both package managers.
