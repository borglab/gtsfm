"""Standalone VGGT probe script for investigating cluster reconstructions.

Given a set of images (by camera indices or explicit file paths), runs VGGT
inference and exports a COLMAP-format reconstruction with summary statistics.

Typical workflow: inspect the cluster-tree HTML visualization, identify an
interesting cluster, copy the camera indices, and probe it with this script.

Usage examples::

    # From camera indices (resolves filenames from sorted directory listing):
    python scripts/probe_vggt.py \\
        --images_dir results/gendermarket_results_2/images/ \\
        --camera_indices 42 67 82 188 269 432 \\
        --output_dir /tmp/vggt_probe_cluster_3_1/

    # From explicit image files:
    python scripts/probe_vggt.py \\
        --image_files img001.jpg img002.jpg img003.jpg \\
        --output_dir /tmp/vggt_probe/

    # With custom VGGT parameters:
    python scripts/probe_vggt.py \\
        --images_dir results/scene/images/ \\
        --camera_indices 10 20 30 \\
        --output_dir /tmp/probe/ \\
        --run_ba \\
        --max_query_pts 4096 \\
        --query_frame_num 8
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Reuse common CLI helpers from the existing demo script when available.
# ---------------------------------------------------------------------------
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import gtsfm.frontend.vggt as vggt
from gtsfm.frontend.vggt import VggtConfiguration
from gtsfm.utils import torch as torch_utils

# Configure CUDA settings for best throughput.
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_image_paths(args: argparse.Namespace) -> tuple[list[str], list[int]]:
    """Return (image_paths, image_indices) from the CLI arguments.

    Two mutually-exclusive input modes:
      1. ``--image_files``  explicit file paths.
      2. ``--images_dir`` + ``--camera_indices``  resolve via sorted listing.

    Returns:
        image_paths: Absolute paths to each image file.
        image_indices: Global camera indices (1-based for COLMAP compat).
    """
    if args.image_files:
        paths = [os.path.abspath(p) for p in args.image_files]
        for p in paths:
            if not os.path.isfile(p):
                raise FileNotFoundError(f"Image file not found: {p}")
        indices = list(range(1, len(paths) + 1))
        return paths, indices

    if not args.images_dir:
        raise ValueError("Provide either --image_files or --images_dir + --camera_indices.")

    images_dir = os.path.abspath(args.images_dir)
    all_files = sorted(glob.glob(os.path.join(images_dir, "*")))
    # Filter to common image extensions.
    _IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    all_files = [f for f in all_files if os.path.splitext(f)[1].lower() in _IMG_EXTS]
    if not all_files:
        raise FileNotFoundError(f"No image files found in {images_dir}")

    if args.camera_indices:
        indices_0based = args.camera_indices
    else:
        # No indices given  use all images in the directory.
        indices_0based = list(range(len(all_files)))

    paths: list[str] = []
    for idx in indices_0based:
        if idx < 0 or idx >= len(all_files):
            raise IndexError(
                f"Camera index {idx} is out of range. "
                f"Directory contains {len(all_files)} images (0-{len(all_files) - 1})."
            )
        paths.append(all_files[idx])

    # Use 1-based indices for COLMAP convention.
    global_indices = [idx + 1 for idx in indices_0based]
    return paths, global_indices


def _print_summary(result) -> None:
    """Print a human-readable reconstruction summary."""
    data = result.gtsfm_data
    camera_indices = data.get_valid_camera_indices()
    num_cameras = len(camera_indices)
    num_tracks = data.number_tracks()

    print("\n" + "=" * 60)
    print("VGGT Probe  Reconstruction Summary")
    print("=" * 60)
    print(f"  Cameras recovered : {num_cameras}")
    print(f"  Tracks (3D pts)   : {num_tracks}")

    if num_tracks > 0:
        mean_len, median_len = data.get_track_length_statistics()
        print(f"  Mean track length : {mean_len:.2f}")
        print(f"  Median track len  : {median_len:.2f}")

        reproj_errors = data.get_scene_reprojection_errors()
        if reproj_errors.size > 0 and not np.all(np.isnan(reproj_errors)):
            print(f"  Reproj error mean : {np.nanmean(reproj_errors):.3f} px")
            print(f"  Reproj error med  : {np.nanmedian(reproj_errors):.3f} px")
            print(f"  Reproj error max  : {np.nanmax(reproj_errors):.3f} px")

    print(f"  Dense points      : {result.points_3d.shape[0]}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_probe(args: argparse.Namespace) -> None:
    """Load images, run VGGT, export COLMAP output, and print summary."""
    image_paths, image_indices = _resolve_image_paths(args)
    image_names = [os.path.basename(p) for p in image_paths]
    print(f"Probing {len(image_paths)} images:")
    for idx, name in zip(image_indices, image_names):
        print(f"  [{idx}] {name}")

    # --- Seed everything for reproducibility --------------------------------
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # --- Device / dtype selection -------------------------------------------
    device = torch_utils.default_device()
    dtype = vggt.default_dtype(device)
    print(f"Device: {device.type} | dtype: {dtype}")

    # --- Load VGGT model ----------------------------------------------------
    t0 = time.perf_counter()
    model = vggt.load_model(device=device, dtype=dtype)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # --- Load & preprocess images -------------------------------------------
    img_load_resolution = args.max_resolution
    images, original_coords = vggt.load_and_preprocess_images_square(
        image_paths, img_load_resolution
    )
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images (load resolution={img_load_resolution})")

    # --- Build VGGT configuration -------------------------------------------
    config = VggtConfiguration(
        seed=args.seed,
        confidence_threshold=args.confidence_threshold,
        tracking=args.tracking,
        max_query_pts=args.max_query_pts,
        query_frame_num=args.query_frame_num,
        track_vis_thresh=args.vis_thresh,
        max_reproj_error=args.max_reproj_error,
        run_bundle_adjustment_on_leaf=args.run_ba,
        store_pre_ba_result=args.run_ba,
    )

    # --- Run VGGT reconstruction --------------------------------------------
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()

    t0 = time.perf_counter()
    with torch.no_grad():
        result = vggt.run_reconstruction(
            images,
            image_indices=image_indices,
            image_names=image_names,
            original_coords=original_coords,
            config=config,
            model=model,
        )
    elapsed = time.perf_counter() - t0
    print(f"VGGT reconstruction completed in {elapsed:.2f}s")

    if torch.cuda.is_available():
        alloc_mb = torch.cuda.max_memory_allocated() / (1024**2)
        print(f"GPU peak memory: {alloc_mb:.0f} MB")

    # --- Export results -----------------------------------------------------
    output_dir = os.path.abspath(args.output_dir)
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)

    result.gtsfm_data.export_as_colmap_text(sparse_dir)
    print(f"COLMAP text saved to {sparse_dir}/")

    # Dense point cloud as PLY (if trimesh available).
    if result.points_3d.size > 0 and result.points_rgb is not None:
        try:
            import trimesh

            ply_path = os.path.join(sparse_dir, "points.ply")
            trimesh.PointCloud(result.points_3d, colors=result.points_rgb).export(ply_path)
            print(f"Dense point cloud saved to {ply_path}")
        except ImportError:
            print("(trimesh not installed  skipping PLY export)")
        except Exception as exc:
            print(f"PLY export failed: {exc}")
    elif result.points_3d.size == 0:
        print("VGGT produced no confident dense 3D points.")

    # Pre-BA result (if BA was enabled).
    if result.pre_ba_data is not None:
        pre_ba_dir = os.path.join(output_dir, "sparse_pre_ba")
        os.makedirs(pre_ba_dir, exist_ok=True)
        result.pre_ba_data.export_as_colmap_text(pre_ba_dir)
        print(f"Pre-BA reconstruction saved to {pre_ba_dir}/")

    # --- Copy source images into output for convenience --------------------
    images_out_dir = os.path.join(output_dir, "images")
    if not os.path.exists(images_out_dir):
        import shutil

        os.makedirs(images_out_dir, exist_ok=True)
        for src_path in image_paths:
            shutil.copy2(src_path, images_out_dir)
        print(f"Source images copied to {images_out_dir}/")

    # --- Print summary ------------------------------------------------------
    _print_summary(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe VGGT on a subset of images and export COLMAP reconstruction.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources (mutually exclusive).
    input_group = parser.add_argument_group("Input (choose one mode)")
    input_group.add_argument(
        "--image_files",
        nargs="+",
        default=None,
        help="Explicit image file paths to reconstruct.",
    )
    input_group.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Directory containing images. Used with --camera_indices.",
    )
    input_group.add_argument(
        "--camera_indices",
        nargs="+",
        type=int,
        default=None,
        help="0-based camera indices to select from --images_dir (sorted listing).",
    )

    # Output.
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to write COLMAP text output and PLY point cloud.",
    )

    # VGGT parameters.
    vggt_group = parser.add_argument_group("VGGT parameters")
    vggt_group.add_argument("--seed", type=int, default=42, help="RNG seed.")
    vggt_group.add_argument(
        "--max_resolution",
        type=int,
        default=1024,
        help="Resolution for loading images before VGGT preprocessing (default: 1024).",
    )
    vggt_group.add_argument(
        "--confidence_threshold",
        type=float,
        default=5.0,
        help="Depth confidence threshold for dense point filtering.",
    )
    vggt_group.add_argument(
        "--tracking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable VGGT tracking (--tracking / --no-tracking).",
    )
    vggt_group.add_argument(
        "--max_query_pts",
        type=int,
        default=4096,
        help="Max query points per frame for tracking.",
    )
    vggt_group.add_argument(
        "--query_frame_num",
        type=int,
        default=8,
        help="Number of query frames for tracking.",
    )
    vggt_group.add_argument(
        "--vis_thresh",
        type=float,
        default=0.2,
        help="Minimum visibility confidence for track filtering.",
    )
    vggt_group.add_argument(
        "--max_reproj_error",
        type=float,
        default=8.0,
        help="Reprojection error threshold (pixels) for track/BA filtering.",
    )
    vggt_group.add_argument(
        "--run_ba",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run bundle adjustment on the reconstruction (--run_ba / --no-run_ba).",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_probe(args)
