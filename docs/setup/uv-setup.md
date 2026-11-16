# UV Setup Guide

GTSFM requires Python 3.12.x (`>=3.12,<3.13`). This guide covers installation with uv, followed by browser and terminal usage. For a named Conda environment, see the [Conda setup guide](conda-setup.md); both methods install the same GTSFM CLI and Studio interface.

## Prerequisites

Install Git and [uv](https://docs.astral.sh/uv/getting-started/installation/). On Linux or macOS, the uv installer is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Follow the installer's instructions to make `uv` available in your shell. uv can download Python 3.12 when needed.

Graphviz is optional for process-graph export (`sudo apt-get install graphviz` on Ubuntu or `brew install graphviz` on macOS). Node.js and a JavaScript package manager are needed only when developing the Studio frontend; see the [frontend development instructions](../../README.md#run-gtsfm).

## Prepare the checkout

For a new checkout:

```bash
git clone https://github.com/borglab/gtsfm.git
cd gtsfm
```

For an existing checkout, open a terminal in its repository root. Initialize the submodules before either installation method below:

```bash
git submodule update --init --recursive
```

VGGT and other model implementations are bundled from these submodules when GTSFM is built. uv does not fetch the submodules automatically. Run the remaining setup commands from this directory.

## Install GTSFM

Choose one of the following methods.

### Install the command-line tool

To use GTSFM without a development environment:

```bash
uv tool install --python 3.12 .
```

This installs the `gtsfm` command in an isolated tool environment. The explicit Python option selects 3.12 for that environment; see [uv's tool Python version documentation](https://docs.astral.sh/uv/concepts/tools/#python-versions). If your shell cannot find `gtsfm`, run `uv tool update-shell` and open a new terminal.

### Set up a development environment

To work on GTSFM's source code:

```bash
uv sync --python 3.12
```

This creates or updates `.venv`, installs GTSFM in editable mode, and includes the project's development dependencies. Run commands with `uv run` to use this environment, for example `uv run gtsfm hardware`. There is no need to also install the tool.

## Run GTSFM

The examples below use the installed tool. For a development environment, prefix each command with `uv run`.

Check the command and detected hardware:

```bash
gtsfm --help
gtsfm hardware
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

See the [usage guide](../../README.md#usage-guide-running-3d-reconstruction) for reconstruction options and model-weight setup.

## Optional GPU dependencies

The following commands apply to the development environment. The project's uv source configuration selects CUDA 12.8 PyTorch wheels on Linux and CPU wheels on macOS. The `complete` extra adds xformers on Linux; it is not required simply to use a single CUDA GPU:

```bash
uv sync --python 3.12 --extra complete
```

For Dask CUDA scheduling across multiple GPUs, add the `multi-gpu` extra:

```bash
uv sync --python 3.12 --extra multi-gpu
```

These extras can be combined. Include the same `--extra` options in subsequent `uv sync` and `uv run` commands when you need those packages.

### AnySplat tracking dependencies

AnySplat's tracking stack additionally uses `torch-scatter`. Select a wheel matching the PyTorch and CUDA versions in your environment, as described in the [PyG installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html#installation-from-wheels).

Inspect those versions:

```bash
uv run python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

For PyTorch 2.7.x built with CUDA 12.8, install the matching wheel into `.venv`:

```bash
uv pip install --python .venv/bin/python torch-scatter \
  --find-links https://data.pyg.org/whl/torch-2.7.0+cu128.html
```

This package is installed separately from the project's declared dependencies; a later `uv sync` may remove it. This is required for AnySplat, not for the basic installation checks above.

## Manage development dependencies

Use `uv add <package>` to add a project dependency, `uv add --dev <package>` for a development dependency, and `uv remove <package>` to remove one. These commands update `pyproject.toml` and `uv.lock`.

After manually editing dependency declarations, run `uv lock` to update the lock file, then `uv sync` to update `.venv`. To upgrade a particular dependency within its constraints, use `uv lock --upgrade-package <package>` followed by `uv sync`.

For an installation confined to `.venv` without changing the project metadata, use `uv pip install --python .venv/bin/python <package>`. Such packages may be removed by a later sync.
