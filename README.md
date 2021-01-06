# Georgia Tech Structure from Motion (GTSFM) Library

### What is GTSFM?
GTSFM is an end-to-end SFM pipeline based on [GTSAM](https://github.com/borglab/gtsam). GTSFM was designed from the ground-up to natively support parallel computation using [Dask](https://dask.org/).

## Installation

First, create the conda environment:
```bash
conda env create -f environment.yml
```
Now, activate the conda environment. Depending upon your bash and conda set-up, this will either be via:
```bash
conda activate gtsfm-v1
```
or:
```bash
source activate gtsfm-v1
```

Now, install `gtsfm` as a module:
```bash
pip install -e .
```
Make sure that you can run `python -c "import gtsfm; print('hello world')"` in python, and you are good to go!

## Authors (alphabetically)

Ayush Baid, Frank Dellaert, Fan Jiang, Akshay Krishnan, John Lambert, Aishwarya Venkataramanan, Sushmita Warrier, Jing Wu, Xiaolong Wu

## Repository Structure

GTSFM is designed in an extremely modular way. Each module can be swapped out with a new one, as long as it implements the API of the module's abstract base class. The code is organized as follows:

- `gtsfm`: source code, organized as:
    - `averaging`
        - `rotation`: rotation averaging implementations (Shonan, Chordal, etc)
        - `translation`: translation averaging implementations (1d-SFM, etc)
    - `bundle`
    - `common`
    - `data_association`
    - `densify`
    - `frontend`: SfM front-end code, including:
        - `detector`: keypoint detector implementations (DoG, etc)
        - `descriptor`: feature descriptor implementations (SIFT, etc)
        - `matcher`: descriptor matching implementations
        - `verifier`: 2d-correspondence verifier implementations (Degensac, Superglue, etc)
    - `loader`: image data loaders
    - `utils`
- `tests`: unit tests on every function and module

## Contributing

Our CI will enforce the unit tests (`pytest tests/`), as well as formatters -- `mypy`, `isort`, and also `black`. Please be sure your contribution passes these tests first.
