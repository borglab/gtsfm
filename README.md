# Georgia Tech Structure from Motion (GTSFM) Library

[![Ubuntu CI](https://github.com/borglab/gtsfm/workflows/Python%20CI/badge.svg)](https://github.com/borglab/gtsfm/actions?query=workflow%3APython+CI)


### What is GTSFM?
GTSFM is an end-to-end SFM pipeline based on [GTSAM](https://github.com/borglab/gtsam). GTSFM was designed from the ground-up to natively support parallel computation using [Dask](https://dask.org/).

<p align="left">
  <img src="https://dask.org/_images/dask_horizontal_white_no_pad_dark_bg.png" height="50">
</p>

## License
The majority of our code is governed by a MIT license and are suitable for commercial use. However, certain implementations featured in our repo (SuperPoint, SuperGlue) are governed by a non-commercial license and may not be used commercially.

## Installation
First, we need to create a conda environment.

**Linux**
On Linux, with CUDA support:
```bash
conda env create -f environment_linux.yml
conda activate gtsfm-v1 # you may need "source activate gtsfm-v1" depending upon your bash and conda set-up
```
The Python3.8 `gtsam` wheel for Linux is available [here](https://github.com/borglab/gtsam-manylinux-build/suites/2239592652/artifacts/46493264).

**Mac**
On Mac OSX, there is no CUDA support, so run:
```bash
conda env create -f environment_mac.yml
conda activate gtsfm-v1
```
Download the Python 3.8 gtsam wheel for Mac [here](https://github.com/borglab/gtsam-manylinux-build/suites/2239592652/artifacts/46493266), and install it as
```bash
pip install ~/Downloads/gtsam-4.1.1-py3-none-any.whl
```

## Completing Installation

Now, install `gtsfm` as a module:
```bash
pip install -e .
```
Make sure that you can run `python -c "import gtsfm; import gtsam; print('hello world')"` in python, and you are good to go!

## Compiling Additional Verifiers
On Mac OSX, there is no `pydegensac` wheel in `pypi`, instead build pydegensac: 
```bash
git clone https://github.com/ducha-aiki/pydegensac.git
cd pydegensac
python setup.py bdist_wheel
pip install dist/pydegensac-0.1.2-cp38-cp38-macosx_10_15_x86_64.whl
```

## Usage Guide (Running 3d Reconstruction)

Please run
```python
python gtsfm/runner/run_scene_optimizer.py
```

## Repository Structure

GTSFM is designed in an extremely modular way. Each module can be swapped out with a new one, as long as it implements the API of the module's abstract base class. The code is organized as follows:

- `gtsfm`: source code, organized as:
    - `averaging`
        - `rotation`: rotation averaging implementations (Shonan, Chordal, etc)
        - `translation`: translation averaging implementations (1d-SFM, etc)
    - `bundle`: bundle adjustment implementations
    - `common`: basic classes used through GTSFM, such as `Keypoints`, `Image`, `SfmTrack2d`, etc
    - `data_association`: 3d point triangulation w/ or w/o RANSAC, from 2d point-tracks 
    - `densify`
    - `frontend`: SfM front-end code, including:
        - `detector`: keypoint detector implementations (DoG, etc)
        - `descriptor`: feature descriptor implementations (SIFT, etc)
        - `matcher`: descriptor matching implementations (Superglue, etc)
        - `verifier`: 2d-correspondence verifier implementations (Degensac, OA-Net, etc)
    - `loader`: image data loaders
    - `utils`: utility functions such as serialization routines and pose comparisons, etc
- `tests`: unit tests on every function and module


## Contributing
Contributions are always welcome! Please be aware of our [contribution guidelines for this project](CONTRIBUTING.md).


## Citing this work
Open-source Python implementation:
```
@misc{
    author = {Ayush Baid, Fan Jiang, Akshay Krishnan, John Lambert, Aditya Singh
       Aishwarya Venkataramanan, Sushmita Warrier, Jing Wu, Xiaolong Wu, Frank Dellaert},
    title = {GTSFM: Georgia Tech Structure from Motion},
    howpublished={\url{https://github.com/borglab/gtsfm}},
    year = {2021},
}
```
Note: authors are listed in alphabetical order.
