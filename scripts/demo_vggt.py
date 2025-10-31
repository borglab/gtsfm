"""
A minimal demo for running vggt and export results in colmap format.

python demo_vggt.py --scene_dir PATH/TO/DIR

Original code from https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py

Modified by Xinan Zhang
"""

import argparse
import gc
import glob
import os
import random
import time

import numpy as np
import torch
import trimesh

from gtsfm.utils.vggt import (  # type: ignore[attr-defined]
    VGGTReconstructionConfig,
    default_vggt_device,
    default_vggt_dtype,
    load_and_preprocess_images_square,
    load_vggt_model,
    run_vggt_reconstruction,
)

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Optional (nice CPU memory readout)
try:
    import psutil

    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False


def _sync():
    # Make sure kernels finish before timing/memory reads
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_peak_memory():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def get_peak_memory_str():
    if torch.cuda.is_available():
        alloc = torch.cuda.max_memory_allocated() / (1024**2)
        reserv = torch.cuda.max_memory_reserved() / (1024**2)
        return f"GPU peak allocated: {alloc:.1f} MB | reserved: {reserv:.1f} MB"
    else:
        if _HAS_PSUTIL:
            rss = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
            return f"CPU RSS (approx peak during run): {rss:.1f} MB"
        return "CPU memory: psutil not installed"


class Timer:
    def __init__(self, label):
        self.label = label
        self.t0 = None
        self.elapsed = None

    def __enter__(self):
        _sync()
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        _sync()
        self.elapsed = time.perf_counter() - self.t0
        print(f"[TIMER] {self.label}: {self.elapsed:.2f} s")


# TODO: add support for masks
# TODO: add iterative BA
# TODO: add support for radial distortion, which needs extra_params
# TODO: test with more cases
# TODO: test different camera types


def add_common_vggt_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Register shared VGGT CLI options on the provided parser."""
    parser.add_argument("--seed", type=int, default=42, help="Seed for torch/numpy/python RNGs.")
    parser.add_argument(
        "--use_ba",
        action="store_true",
        default=False,
        help="Enable VGGSfM tracking + COLMAP-style BA (requires LightGlue with ALIKED).",
    )
    parser.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help="Reprojection error threshold (in pixels) when filtering VGGSfM tracks for BA.",
    )
    parser.add_argument(
        "--shared_camera",
        action="store_true",
        default=False,
        help="Use a single shared camera model for all frames instead of per-view cameras.",
    )
    parser.add_argument(
        "--camera_type",
        type=str,
        default="SIMPLE_PINHOLE",
        help="COLMAP camera model to export when BA is enabled (e.g. SIMPLE_PINHOLE, PINHOLE).",
    )
    parser.add_argument(
        "--vis_thresh",
        type=float,
        default=0.2,
        help="Minimum visibility confidence for a VGGSfM track to survive BA filtering.",
    )
    parser.add_argument(
        "--query_frame_num",
        type=int,
        default=8,
        help="Number of frames to query per tracking pass (higher increases runtime and coverage).",
    )
    parser.add_argument(
        "--max_query_pts",
        type=int,
        default=4096,
        help="Maximum number of VGGSfM query points per frame when BA is enabled.",
    )
    parser.add_argument(
        "--fine_tracking",
        action="store_true",
        default=True,
        help="Enable VGGSfM fine tracking refinement (slower but more accurate).",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=5.0,
        help="Depth confidence threshold for the feed-forward path (used when BA is disabled).",
    )
    return parser


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT Demo")
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Path to the scene root that contains an `images/` directory to reconstruct.",
    )
    add_common_vggt_args(parser)
    return parser.parse_args()


def demo_fn(args: argparse.Namespace) -> bool:
    print("Arguments:", vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    device = default_vggt_device()
    dtype = default_vggt_dtype(device)
    print(f"Using device: {device.type}")
    print(f"Using dtype: {dtype}")

    model = load_vggt_model(device=device, dtype=dtype)
    print("Model loaded")

    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_path_list:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}")

    config = VGGTReconstructionConfig(
        use_ba=args.use_ba,
        vggt_fixed_resolution=vggt_fixed_resolution,
        img_load_resolution=img_load_resolution,
        max_query_pts=args.max_query_pts,
        query_frame_num=args.query_frame_num,
        fine_tracking=args.fine_tracking,
        vis_thresh=args.vis_thresh,
        max_reproj_error=args.max_reproj_error,
        confidence_threshold=args.confidence_threshold,
        shared_camera=args.shared_camera,
        use_colmap_ba=args.use_ba,
        camera_type_ba=args.camera_type,
    )

    image_indices = list(range(1, len(image_path_list) + 1))

    reset_peak_memory()
    with Timer("VGGT reconstruction"):
    result = run_vggt_reconstruction(
        images,
        image_indices=image_indices,
        image_names=base_image_path_list,
        original_coords=original_coords,
        config=config,
        device=device,
        dtype=dtype,
        model=model,
        total_num_images=len(image_path_list),
    )
    print(get_peak_memory_str())

    if result.fallback_reason:
        print(result.fallback_reason)

    output_dir_name = "sparse_w_ba" if result.used_ba else "sparse_wo_ba"
    sparse_reconstruction_dir = os.path.join(args.scene_dir, output_dir_name)
    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    result.gtsfm_data.export_as_colmap_text(sparse_reconstruction_dir)

    if result.points_rgb is not None and result.points_3d.size > 0:
        try:
            trimesh.PointCloud(result.points_3d, colors=result.points_rgb).export(
                os.path.join(sparse_reconstruction_dir, "points.ply")
            )
        except Exception as exc:
            print(f"Failed to export point cloud: {exc}")
    else:
        print("Skipping point cloud export (no RGB information available).")

    return True


if __name__ == "__main__":
    args = parse_args()
    with torch.no_grad():
        demo_fn(args)

"""
VGGT Runner Script
=================

A script to run the VGGT model for 3D reconstruction from image sequences.

Directory Structure
------------------
Input:
    input_folder/
    └── images/            # Source images for reconstruction

Output:
    output_folder/
    ├── images/
    ├── sparse/           # Reconstruction results
    │   ├── cameras.bin   # Camera parameters (COLMAP format)
    │   ├── images.bin    # Pose for each image (COLMAP format)
    │   ├── points3D.bin  # 3D points (COLMAP format)
    │   └── points.ply    # Point cloud visualization file 
    └── visuals/          # Visualization outputs TODO

Key Features
-----------
• Dual-mode Support: Run reconstructions using either VGGT or VGGT+BA
• Resolution Preservation: Maintains original image resolution in camera parameters and tracks
• COLMAP Compatibility: Exports results in standard COLMAP sparse reconstruction format
"""
