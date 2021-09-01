# Georgia Tech Structure from Motion (GTSFM) Library

[![Ubuntu CI](https://github.com/borglab/gtsfm/workflows/Python%20CI/badge.svg)](https://github.com/borglab/gtsfm/actions?query=workflow%3APython+CI)


### What is GTSFM?
GTSFM is an end-to-end SFM pipeline based on [GTSAM](https://github.com/borglab/gtsam). GTSFM was designed from the ground-up to natively support parallel computation using [Dask](https://dask.org/).

<p align="left">
  <img src="https://dask.org/_images/dask_horizontal_white_no_pad_dark_bg.png" height="50">
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/16724970/121294002-a4d7a400-c8ba-11eb-895e-a50305c049b6.gif" height="315" title="Olsson Lund Dataset: Door, 12 images">
  <img src="https://user-images.githubusercontent.com/16724970/121293398-8cb35500-c8b9-11eb-8898-6162cb2372e1.gif" height="315">
</p>

## License
The majority of our code is governed by a MIT license and is suitable for commercial use. However, certain implementations featured in our repo (SuperPoint, SuperGlue) are governed by a non-commercial license and may not be used commercially.

## Installation
GTSFM requires no compilation, as Python wheels are provided for GTSAM. 

To install GTSFM, first, we need to create a conda environment.

**Linux**
On Linux, with CUDA support:
```bash
conda env create -f environment_linux.yml
conda activate gtsfm-v1 # you may need "source activate gtsfm-v1" depending upon your bash and conda set-up
```
The Python3.8 `gtsam` wheel for Linux is available [here](https://github.com/borglab/gtsam-manylinux-build/suites/3489546443/artifacts/83058971).

**Mac**
On Mac OSX, there is no CUDA support, so run:
```bash
conda env create -f environment_mac.yml
conda activate gtsfm-v1
```
Download the Python 3.8 gtsam wheel for Mac [here](https://github.com/borglab/gtsam-manylinux-build/suites/3489546443/artifacts/83058973), and install it as
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

Before running reconstruction, if you intend to use modules with pre-trained weights, such as SuperPoint, SuperGlue, or PatchmatchNet, please first run:
```bash
./download_model_weights.sh
```

To run SfM with a dataset with only a image directory and EXIF, with image file names ending with "jpg", run:
```python
python gtsfm/runner/run_scene_optimizer.py --config_name {CONFIG_NAME} --dataset_root {DATASET_ROOT} --image_extension jpg --num_workers {NUM_WORKERS}
```

If you would like to compare GTSFM output with COLMAP output, please run:
```python
python gtsfm/runner/run_scene_optimizer_colmap_loader.py --config_name {CONFIG_NAME} --images_dir {IMAGES_DIR} --colmap_files_dirpath {COLMAP_FILES_DIRPATH} --image_extension jpg --num_workers {NUM_WORKERS} --max_frame_lookahead {MAX_FRAME_LOOKAHEAD}
```
where `COLMAP_FILES_DIRPATH` is a directory where .txt files such as `cameras.txt`, `images.txt`, etc have been saved.


## Repository Structure

GTSFM is designed in an extremely modular way. Each module can be swapped out with a new one, as long as it implements the API of the module's abstract base class. The code is organized as follows:

- `gtsfm`: source code, organized as:
    - `averaging`
        - `rotation`: rotation averaging implementations ([Shonan](https://arxiv.org/abs/2008.02737), Chordal, etc)
        - `translation`: translation averaging implementations ([1d-SFM](https://www.cs.cornell.edu/projects/1dsfm/docs/1DSfM_ECCV14.pdf), etc)
    - `bundle`: bundle adjustment implementations
    - `common`: basic classes used through GTSFM, such as `Keypoints`, `Image`, `SfmTrack2d`, etc
    - `data_association`: 3d point triangulation (DLT) w/ or w/o RANSAC, from 2d point-tracks 
    - `densify`
    - `frontend`: SfM front-end code, including:
        - `detector`: keypoint detector implementations (DoG, etc)
        - `descriptor`: feature descriptor implementations ([SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf), [SuperPoint](https://arxiv.org/abs/1712.07629) etc)
        - `matcher`: descriptor matching implementations ([Superglue](https://arxiv.org/abs/1911.11763), etc)
        - `verifier`: 2d-correspondence verifier implementations ([Degensac](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.466.2719&rep=rep1&type=pdf), OA-Net, etc)
    - `loader`: image data loaders
    - `utils`: utility functions such as serialization routines and pose comparisons, etc
- `tests`: unit tests on every function and module


## Contributing
Contributions are always welcome! Please be aware of our [contribution guidelines for this project](CONTRIBUTING.md).


## Citing this work
Open-source Python implementation:
```
@misc{GTSFM,
    author = {Ayush Baid and Fan Jiang and Akshay Krishnan and John Lambert and Aditya Singh and
       Aishwarya Venkataramanan and Sushmita Warrier and Jing Wu and Xiaolong Wu and Frank Dellaert},
    title = { {GTSFM}: Georgia Tech Structure from Motion},
    howpublished={\url{https://github.com/borglab/gtsfm}},
    year = {2021}
}
```
Note: authors are listed in alphabetical order.
