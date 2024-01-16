![Alt text](gtsfm-logo.png?raw=true)

| Platform     | Build Status  |
|:------------:| :-------------:|
| Ubuntu 20.04.3 |  ![Linux CI](https://github.com/borglab/gtsfm/actions/workflows/test-python.yml/badge.svg?branch=master) |

Georgia Tech Structure-from-Motion (GTSfM) is an end-to-end SfM pipeline based on [GTSAM](https://github.com/borglab/gtsam). GTSfM was designed from the ground-up to natively support parallel computation using [Dask](https://dask.org/). 

For more details, please refer to our [arXiv preprint](https://arxiv.org/abs/2311.18801)

<p align="left">
  <img src="https://user-images.githubusercontent.com/16724970/121294002-a4d7a400-c8ba-11eb-895e-a50305c049b6.gif" height="315" title="Olsson Lund Dataset: Door, 12 images">
  <img src="https://user-images.githubusercontent.com/16724970/142500100-ed3bd07b-f839-488e-a01d-823a9fbeaba4.gif" height="315">
</p>

<p align="left">
  <img src="https://user-images.githubusercontent.com/25347892/146043166-c5a172d7-17e0-4779-8333-8cd5f088ea2e.gif" height="345" title="2011212_opnav_022">
  <img src="https://user-images.githubusercontent.com/25347892/146043553-5299e9d3-44c5-40a6-8ba8-ff43d2a28c8f.gif" height="345">
</p>

## License

The majority of our code is governed by an MIT license and is suitable for commercial use. However, certain implementations featured in our repo (e.g., SuperPoint, SuperGlue) are governed by a non-commercial license and may not be used commercially.

## Installation

GTSfM requires no compilation, as Python wheels are provided for GTSAM. This repository includes external repositories as Git submodules –- don't forget to pull submodules with `git submodule update --init --recursive` or clone with `git clone --recursive https://github.com/borglab/gtsfm.git`.

To run GTSfM, first, we need to create a conda environment with the required dependencies.

On **Linux**, with CUDA support, run:

```bash
conda env create -f environment_linux.yml
conda activate gtsfm-v1 # you may need "source activate gtsfm-v1" depending upon your bash and conda set-up
```

On **macOS**, there is no CUDA support, so run:

```bash
conda env create -f environment_mac.yml
conda activate gtsfm-v1
```

Now, install `gtsfm` as a module:

```bash
pip install -e .
```

Make sure that you can run `python -c "import gtsfm; import gtsam; print('hello world')"` in python, and you are good to go!

## Usage Guide (Running 3D Reconstruction)

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
python gtsfm/runner/run_scene_optimizer_olssonloader.py --config_name {CONFIG_NAME} --dataset_root {DATASET_ROOT} --num_workers {NUM_WORKERS}
```

For example, if you had 4 cores available and wanted to use the Deep Front-End (recommended) on the "door" dataset, you should run:

```bash
python gtsfm/runner/run_scene_optimizer_olssonloader.py --dataset_root tests/data/set1_lund_door --config_name deep_front_end.yaml --num_workers 4
```

(or however many workers you desire).

You can view/monitor the distributed computation using the [Dask dashboard](http://localhost:8787/status).

Currently we require EXIF data embedded into your images (or you can provide ground truth intrinsics in the expected format for an Olsson dataset, or COLMAP-exported text data, etc.)

If you would like to compare GTSfM output with COLMAP output, please run:

```python
python gtsfm/runner/run_scene_optimizer_colmaploader.py --config_name {CONFIG_NAME} --images_dir {IMAGES_DIR} --colmap_files_dirpath {COLMAP_FILES_DIRPATH} --num_workers {NUM_WORKERS} --max_frame_lookahead {MAX_FRAME_LOOKAHEAD}
```

where `COLMAP_FILES_DIRPATH` is a directory where .txt files such as `cameras.txt`, `images.txt`, etc have been saved.

To visualize the result using Open3D, run:

```bash
python gtsfm/visualization/view_scene.py
```

For users that are working with the same dataset repeatedly, we provide functionality to cache front-end results for
GTSfM for very fast inference afterwards. For more information, please refer to [`gtsfm/frontend/cacher/README.md`](https://github.com/borglab/gtsfm/tree/master/gtsfm/frontend/cacher).

For users that want to run GTSfM on a cluster of multiple machines, we provide setup instructions here: [`CLUSTER.md`](https://github.com/borglab/gtsfm/tree/master/CLUSTER.md)

The results will be stored at `--output_root`, which is the `results` folder in the repo root by default. The poses and 3D tracks are stored in COLMAP format inside the `ba_output` subdirectory of `--output_root`. These can be visualized using the COLMAP GUI as well. 

### Nerfstudio

We provide a preprocessing script to convert the camera poses estimated by GTSfM to [nerfstudio](https://docs.nerf.studio/en/latest/) format: 

```bash
python gtsfm/utils/prepare_nerfstudio.py --results_path {RESULTS_DIR} --images_dir {IMAGES_DIR}
```

The results are stored in the nerfstudio_input subdirectory inside `{RESULTS_DIR}`, which can be used directly with nerfstudio if installed:

```bash
ns-train nerfacto --data {RESULTS_DIR}/nerfstudio_input
```

## Repository Structure

GTSfM is designed in an extremely modular way. Each module can be swapped out with a new one, as long as it implements the API of the module's abstract base class. The code is organized as follows:

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

If you use GTSfM, please cite our paper: 

```
@misc{baid2023distributed,
      title={Distributed Global Structure-from-Motion with a Deep Front-End}, 
      author={Ayush Baid and John Lambert and Travis Driver and Akshay Krishnan and Hayk Stepanyan and Frank Dellaert},
      year={2023},
      eprint={2311.18801},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

Citing the open-source Python implementation:

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
