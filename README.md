![Alt text](gtsfm-logo.png?raw=true)

# Georgia Tech Structure from Motion (GTSFM) Library

| Platform     | Build Status  |
|:------------:| :-------------:|
| Ubuntu 20.04.3 |  ![Linux CI](https://github.com/borglab/gtsfm/workflows/Unit%20tests%20and%20python%20checks/badge.svg) |

### What is GTSFM?

GTSFM is an end-to-end SFM pipeline based on [GTSAM](https://github.com/borglab/gtsam). GTSFM was designed from the ground-up to natively support parallel computation using [Dask](https://dask.org/).

<p align="left">
  <img src="https://dask.org/_images/dask_horizontal_white_no_pad_dark_bg.png" height="50">
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/16724970/121294002-a4d7a400-c8ba-11eb-895e-a50305c049b6.gif" height="315" title="Olsson Lund Dataset: Door, 12 images">
  <img src="https://user-images.githubusercontent.com/16724970/142500100-ed3bd07b-f839-488e-a01d-823a9fbeaba4.gif" height="315">
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/25347892/146043166-c5a172d7-17e0-4779-8333-8cd5f088ea2e.gif" height="345" title="2011212_opnav_022">
  <img src="https://user-images.githubusercontent.com/25347892/146043553-5299e9d3-44c5-40a6-8ba8-ff43d2a28c8f.gif" height="345">
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

**Mac**
On Mac OSX, there is no CUDA support, so run:

```bash
conda env create -f environment_mac.yml
conda activate gtsfm-v1
```

## Completing Installation

Now, install `gtsfm` as a module:

```bash
pip install -e .
```

Make sure that you can run `python -c "import gtsfm; import gtsam; print('hello world')"` in python, and you are good to go!

## Usage Guide (Running 3d Reconstruction)

Before running reconstruction, if you intend to use modules with pre-trained weights, such as SuperPoint, SuperGlue, or PatchmatchNet, please first run:

```bash
./download_model_weights.sh
```

To run SfM with a dataset with only an image directory and EXIF, with image file names ending with "jpg", please create the following file structure like

```
└── {DATASET_NAME}
       ├── images
               ├── image1.jpg
               ├── image2.jpg
               ├── image3.jpg
```

and run

```python
python gtsfm/runner/run_scene_optimizer_olssonloader.py --config_name {CONFIG_NAME} --dataset_root {DATASET_ROOT} --image_extension jpg --num_workers {NUM_WORKERS}
```

For example, if you had 4 cores available and wanted to use the Deep Front-End (recommended) on the "door" dataset, you should run:

```bash
python gtsfm/runner/run_scene_optimizer_olssonloader.py --dataset_root tests/data/set1_lund_door --image_extension JPG --config_name deep_front_end.yaml --num_workers 4
```

(or however many workers you desire).

You can view/monitor the distributed computation using the [Dask dashboard](http://localhost:8787/status).

Currently we require EXIF data embedded into your images (or you can provide ground truth intrinsics in the expected format for an Olsson dataset, or COLMAP-exported text data, etc)

If you would like to compare GTSFM output with COLMAP output, please run:

```python
python gtsfm/runner/run_scene_optimizer_colmaploader.py --config_name {CONFIG_NAME} --images_dir {IMAGES_DIR} --colmap_files_dirpath {COLMAP_FILES_DIRPATH} --image_extension jpg --num_workers {NUM_WORKERS} --max_frame_lookahead {MAX_FRAME_LOOKAHEAD}
```

where `COLMAP_FILES_DIRPATH` is a directory where .txt files such as `cameras.txt`, `images.txt`, etc have been saved.

To visualize the result using Open3D, run:

```bash
python gtsfm/visualization/view_scene.py --rendering_library open3d --rendering_style point
```

For users that are working with the same dataset repeatedly, we provide functionality to cache front-end results for
GTSFM for very fast inference afterwards. For more information, please refer to [`gtsfm/frontend/cacher/README.md`](https://github.com/borglab/gtsfm/tree/master/gtsfm/frontend/cacher).

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
    - `cacher`: Cache implementations for different stages of the front-end.
  - `loader`: image data loaders
  - `utils`: utility functions such as serialization routines and pose comparisons, etc
- `tests`: unit tests on every function and module

## Contributing

Contributions are always welcome! Please be aware of our [contribution guidelines for this project](CONTRIBUTING.md).

## Citing this work

Open-source Python implementation:

```
@misc{GTSFM,
    author = {Ayush Baid and Travis Driver and Fan Jiang and Akshay Krishnan and John Lambert
       and Ren Liu and Aditya Singh and Neha Upadhyay and Aishwarya Venkataramanan
       and Sushmita Warrier and Jon Womack and Jing Wu and Xiaolong Wu and Frank Dellaert},
    title = { {GTSFM}: Georgia Tech Structure from Motion},
    howpublished={\url{https://github.com/borglab/gtsfm}},
    year = {2021}
}
```

Note: authors are listed in alphabetical order (by last name).

## Compiling Additional Verifiers

On Linux, we have made `pycolmap`'s LORANSAC available in [pypi](https://pypi.org/project/pycolmap/). However, on Mac, `pycolmap` must be built from scratch. See the instructions [here](https://github.com/borglab/gtsfm/blob/master/gtsfm/frontend/verifier/loransac.py#L10).
