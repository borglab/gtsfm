![Alt text](gtsfm-logo.png?raw=true)

| Platform     | Build Status  |
|:------------:| :-------------:|
| Ubuntu 20.04.3 |  ![Linux CI](https://github.com/borglab/gtsfm/actions/workflows/test-python.yml/badge.svg?branch=master) |

Georgia Tech Structure-from-Motion (GTSfM) is an end-to-end SfM pipeline based on [GTSAM](https://github.com/borglab/gtsam). GTSfM was designed from the ground-up to natively support parallel computation using [Dask](https://dask.org/). 

For more details, please refer to our [arXiv preprint](https://arxiv.org/abs/2311.18801).

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

GTSfM requires no compilation, as Python wheels are provided for GTSAM.

### Initialize Git submodules

This repository includes external repositories as Git submodules, so, unless you cloned with `git clone --recursive` you need to initialize:
```bash
git submodule update --init --recursive
```

### Create a Conda Environment

To run GTSfM, first, we need to create a conda environment with the required dependencies.

[Install MiniConda](https://www.anaconda.com/docs/getting-started/miniconda/install) if needed, then:

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

### Install `gtsfm` as a module

Now, install `gtsfm` as a module:

```bash
pip install -e .
```

Make sure that you can run `python -c "import gtsfm; import gtsam; print('hello world')"` in python, and you are good to go!

## Try It on Google Colab  

For a quick hands-on example, check out this Colab notebook [![Colab notebook](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/borglab/gtsfm/blob/master/notebooks/gtsfm_colab.ipynb)


## Usage Guide (Running 3D Reconstruction)

Before running reconstruction, if you intend to use modules with pre-trained weights (e.g., **SuperPoint, SuperGlue, or PatchmatchNet**), first download the model weights by running:  

```bash
./download_model_weights.sh
```  

### Running SfM  

GTSfM provides a unified runner that supports all dataset types through Hydra configuration.

To process a dataset containing only an **image directory and EXIF metadata**, ensure your dataset follows this structure:  

```
└── {DATASET_NAME}
       ├── images
               ├── image1.jpg
               ├── image2.jpg
               ├── image3.jpg
```  

Then, run the following command:  

```bash
./run --config_name {CONFIG_NAME} --loader olsson --dataset_dir {DATASET_DIR} --num_workers {NUM_WORKERS}
```

### Command-line Options

The runner exposes five portable CLI arguments for dataset selection and universal loader configuration:

- `--loader` — which loader to use (e.g., `olsson_loader`, `colmap_loader`)
- `--dataset_dir` — path to the dataset root
- `--images_dir` — optional path to the image directory (defaults depend on loader)
- `--max_resolution` — maximum length of the image’s short side (overrides config)
- `--input_worker` — optional Dask worker address to pin image I/O (advanced; runner sets this post‑instantiation)

**All other loader‑specific settings** (anything beyond the five above) must be specified using **Hydra overrides** on the nested config node `loader.*`. This is standard Hydra behavior: use dot‑notation keys with `=` assignments.

To discover all available overrides for a given loader, open its YAML in `gtsfm/configs/loader/` and/or run:
```bash
./run --help
```

Example (deep front-end on Olsson, single worker):
```bash
./run --dataset_dir tests/data/set1_lund_door \
      --config_name deep_front_end.yaml \
      --loader olsson \
      --num_workers 1 \
      loader.max_resolution=1200
```

For a dataset with metadata formatted in the COLMAP style:
```bash
./run --dataset_dir datasets/gerrard-hall \
      --config_name deep_front_end.yaml \
      --loader colmap \
      --num_workers 5 \
      loader.use_gt_intrinsics=true \
      loader.use_gt_extrinsics=true
```

You can monitor the distributed computation using the [Dask dashboard](http://localhost:8787/status).  
**Note:** The dashboard will only display activity while tasks are actively running, but comprehensive performance reports can be found in the `dask_reports` folder.

### Required Image Metadata  

Currently, we require **EXIF data** embedded into your images. Alternatively, you can provide:  
- Ground truth intrinsics in the expected format for an **Olsson dataset**  
- **COLMAP-exported** text data  

### Comparing GTSFM Output with COLMAP Output  

To compare GTSFM output with COLMAP, use the following command:  

```bash
./run --config_name {CONFIG_NAME} --loader colmap --dataset_dir {DATASET_DIR} --num_workers {NUM_WORKERS} --max_frame_lookahead {MAX_FRAME_LOOKAHEAD}
```  

### Visualizing Results with Open3D  

To visualize the reconstructed scene using **Open3D**, run:  

```bash
python gtsfm/visualization/view_scene.py
```  

### Speeding Up Front-End Processing  

For users who work with the **same dataset repeatedly**, GTSFM allows **caching front-end results** for faster inference.  
Refer to the detailed guide:  
📄 [GTSFM Front-End Cacher README](https://github.com/borglab/gtsfm/tree/master/gtsfm/frontend/cacher)  

### Running GTSFM on a Multi-Machine Cluster  

For users who want to run GTSFM on a **cluster of multiple machines**, follow the setup instructions here:  
📄 [CLUSTER.md](https://github.com/borglab/gtsfm/tree/master/CLUSTER.md)  

### Where Are the Results Stored?  

- The output will be saved in `--output_root`, which defaults to the `results` folder in the repo root.  
- **Poses and 3D tracks** are stored in **COLMAP format** inside the `ba_output` subdirectory of `--output_root`.  
- You can **visualize** these using the **COLMAP GUI**.

### Nerfstudio

We provide a preprocessing script to convert the camera poses estimated by GTSfM to [nerfstudio](https://docs.nerf.studio/en/latest/) format:

```bash
python scripts/prepare_nerfstudio.py --results_path {RESULTS_DIR} --images_dir {IMAGES_DIR}
```

The results are stored in the nerfstudio_input subdirectory inside `{RESULTS_DIR}`, which can be used directly with nerfstudio if installed:

```bash
ns-train nerfacto --data {RESULTS_DIR}/nerfstudio_input
```

## Loader Usage Examples

The runner supports all loaders through `--loader`, `--dataset_dir`, and `--images_dir`. Any additional, loader‑specific settings are passed as **Hydra overrides** on the nested node `loader.*` (this is standard Hydra usage).

**General pattern**
```bash
./run \
  --config_name <config_file> \
  --loader <loader_type> \
  --dataset_dir <path> \
  [--images_dir <path>] \
  [--max_resolution <int>] \
  [--input_worker <address>] \
  loader.<param>=<value> \
  [loader.<param2>=<value2> ...]
```

### Available Loaders

The following loader types are supported:
- `colmap` - COLMAP format datasets
- `hilti` - Hilti SLAM challenge datasets  
- `astrovision` - AstroVision space datasets
- `olsson` - Olsson format datasets
- `argoverse` - Argoverse autonomous driving datasets
- `mobilebrick` - MobileBrick datasets
- `one_d_sfm` - 1DSFM format datasets
- `tanks_and_temples` - Tanks and Temples benchmark datasets
- `yfcc_imb` - YFCC Image Matching Benchmark datasets

For the complete list of available arguments for each loader, run:
```bash
./run --help
```

### Example: Olsson Loader (images + EXIF)
```bash
./run \
  --config_name sift_front_end.yaml \
  --loader olsson \
  --dataset_dir /path/to/olsson_dataset \
  loader.max_resolution=1200
```

### Example: Colmap Loader (COLMAP text export)
```bash
./run \
  --config_name sift_front_end.yaml \
  --loader colmap \
  --dataset_dir /path/to/colmap_dataset \
  loader.use_gt_intrinsics=true \
  loader.use_gt_extrinsics=true
```

> Tip: consult `gtsfm/configs/loader/<loader_name>.yaml` for the full set of fields supported by each loader.

## Repository Structure

GTSfM is designed in a modular way. Each module can be swapped out with a new one, as long as it implements the API of the module's abstract base class. The code is organized as follows:

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
@misc{Baid23_distributedDeepSfm,
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
