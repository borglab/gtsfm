"""
A minimal demo for running VGGT and exporting COLMAP reconstructions per cluster.

python demo_vggt_w_partition.py --scene_dir PATH/TO/DIR

Original code from https://github.com/facebookresearch/vggt/blob/main/demo_colmap.py

Modified by Xinan Zhang
"""

from __future__ import annotations

import argparse
import glob
import os
import random
import shutil
from typing import Dict, List

import numpy as np
import torch
import trimesh
from demo_vggt import Timer, add_common_vggt_args

import gtsfm.frontend.vggt as vggt
from gtsfm.frontend.vggt import VGGTReconstructionConfig
from gtsfm.utils import torch as torch_utils

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT Demo With Clustering")
    parser.add_argument(
        "--scene_dir",
        type=str,
        required=True,
        help="Original dataset directory containing an `images/` subfolder used to populate clusters.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./vggt_output",
        help="Destination directory where per-cluster `images/` folders and reconstructions will be written.",
    )
    add_common_vggt_args(parser)
    return parser.parse_args()


def demo_fn(cluster_key: str, image_indices: List[int], args: argparse.Namespace) -> bool:

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"[{cluster_key}] Setting seed as: {args.seed}")

    device = torch_utils.default_device()
    dtype = vggt.default_dtype(device)
    print(f"[{cluster_key}] Using device: {device.type}")
    print(f"[{cluster_key}] Using dtype: {dtype}")

    model = vggt.load_model(device=device, dtype=dtype)
    print(f"[{cluster_key}] Model loaded")

    image_dir = os.path.join(args.output_dir, cluster_key, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if not image_path_list:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    img_load_resolution = 1024
    vggt_fixed_resolution = 518

    images, original_coords = vggt.load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"[{cluster_key}] Loaded {len(images)} images from {image_dir}")

    config = VGGTReconstructionConfig(
        vggt_fixed_resolution=vggt_fixed_resolution,
        img_load_resolution=img_load_resolution,
        max_query_pts=args.max_query_pts,
        query_frame_num=args.query_frame_num,
        fine_tracking=args.fine_tracking,
        vis_thresh=args.vis_thresh,
        max_reproj_error=args.max_reproj_error,
        confidence_threshold=args.confidence_threshold,
        shared_camera=args.shared_camera,
    )

    with Timer(f"[{cluster_key}] VGGT reconstruction"):
        result = vggt.run_reconstruction(
            images,
            image_indices=image_indices,
            image_names=base_image_path_list,
            original_coords=original_coords,
            config=config,
            device=device,
            dtype=dtype,
            model=model,
            total_num_images=max(image_indices) + 1,
        )

    if result.points_3d.size == 0:
        print(f"[{cluster_key}] VGGT produced no confident 3D structure.")

    sparse_subdir = "sparse_wo_ba"
    sparse_reconstruction_dir = os.path.join(args.output_dir, cluster_key, sparse_subdir)
    print(f"[{cluster_key}] Saving reconstruction to {sparse_reconstruction_dir}")
    os.makedirs(sparse_reconstruction_dir, exist_ok=True)
    result.gtsfm_data.export_as_colmap_text(sparse_reconstruction_dir)

    if result.points_rgb is not None and result.points_3d.size > 0:
        try:
            trimesh.PointCloud(result.points_3d, colors=result.points_rgb).export(
                os.path.join(sparse_reconstruction_dir, "points.ply")
            )
        except Exception as exc:  # pragma: no cover - export helper
            print(f"[{cluster_key}] Failed to export point cloud: {exc}")
    else:
        print(f"[{cluster_key}] Skipping point cloud export (no RGB information available).")

    return True


def prepare_cluster_tree_data(clusters: Dict[str, List[int]], source_directory: str, output_dir: str) -> None:
    """Prepare per-cluster image folders by copying images based on indices."""

    def copy_files_by_indices(src_dir: str, dst_dir: str, indices: List[int]) -> None:
        files = sorted(os.listdir(src_dir))
        os.makedirs(dst_dir, exist_ok=True)
        for idx in indices:
            if 0 <= idx < len(files):
                src_path = os.path.join(src_dir, files[idx])
                dst_path = os.path.join(dst_dir, files[idx])
                shutil.copy2(src_path, dst_path)
                print(f"Copied: {files[idx]}")
            else:
                print(f"Index {idx} is out of range. Skipped.")

    for name, cluster_indices in clusters.items():
        destination_directory = os.path.join(output_dir, name, "images")
        copy_files_by_indices(source_directory, destination_directory, cluster_indices)


if __name__ == "__main__":
    args = parse_args()
    print("Arguments:", vars(args))

    # Hard coded clusters
    # fmt: off
    clusters: Dict[str, List[int]] = {
        "c": [65, 66, 67, 68, 144, 145, 248, 249, 277, 279, 280],
        "c_1": [204, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 279],
        "c_1_1": [202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239],
        "c_1_2": [248, 261, 262, 263, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280],
        "c_2": [147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 206],
        "c_2_1": [183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216],
        "c_2_2": [127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161],
        "c_3": [73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124],
        "c_3_1": [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53],
        "c_3_1_1": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        "c_3_1_2": [39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70],
        "c_3_2": [56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87],
        "c_3_3": [110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141],
    }
    # fmt: on

    prepare_cluster_tree_data(clusters, args.scene_dir, args.output_dir)

    with torch.no_grad():
        for cluster_key, cluster_indices in clusters.items():
            print(f"Running VGGT on cluster {cluster_key}")
            demo_fn(cluster_key, cluster_indices, args)
