"""Base class for Loaders.

Authors: Frank Dellaert
"""

from __future__ import annotations  # not needed after 3.11

from argparse import ArgumentParser, Namespace
from typing import List, Optional


def add_loader_args(parser: ArgumentParser) -> None:
    """Register all loader-related CLI arguments on the given parser."""
    # Loader selection
    parser.add_argument(
        "--loader",
        type=str,
        default="olsson_loader",
        help=(
            "Loader type. Available options: colmap_loader, hilti_loader, astrovision_loader, "
            "olsson_loader, argoverse_loader, mobilebrick_loader, one_d_sfm_loader, "
            "tanks_and_temples_loader, yfcc_imb_loader. Default: olsson_loader"
        ),
    )

    # Standardized loader arguments
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset directory")
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Path to images directory (optional, defaults depend on loader)",
    )
    parser.add_argument("--max_length", type=int, help="Maximum number of images/timestamps to process")
    parser.add_argument("--scene_name", type=str, help="Name of the scene (for Tanks and Temples, etc.)")

    # Argoverse-specific arguments
    parser.add_argument("--log_id", type=str, help="Unique ID of vehicle log (Argoverse)")
    parser.add_argument("--stride", type=int, help="Sampling rate, e.g. every N images (Argoverse)")
    parser.add_argument("--max_num_imgs", type=int, help="Maximum number of images to load (Argoverse)")
    parser.add_argument("--max_lookahead_sec", type=float, help="Maximum lookahead in seconds (Argoverse)")
    parser.add_argument("--camera_name", type=str, help="Camera name to use (Argoverse)")

    # MobileBrick/COLMAP-specific arguments
    parser.add_argument(
        "--use_gt_intrinsics",
        action="store_true",
        help="Use ground truth intrinsics (MobileBrick/COLMAP)",
    )
    parser.add_argument(
        "--use_gt_extrinsics",
        action="store_true",
        help="Use ground truth extrinsics (COLMAP)",
    )

    # 1DSFM-specific arguments
    parser.add_argument("--enable_no_exif", action="store_true", help="Read images without EXIF (1DSFM)")
    parser.add_argument(
        "--default_focal_length_factor",
        type=float,
        help="Default focal length factor (1DSFM)",
    )

    # Tanks and Temples-specific arguments
    parser.add_argument("--poses_fpath", type=str, help="Path to poses file (Tanks and Temples)")
    parser.add_argument(
        "--bounding_polyhedron_json_fpath",
        type=str,
        help="Path to bounding polyhedron JSON (Tanks and Temples)",
    )
    parser.add_argument("--ply_alignment_fpath", type=str, help="Path to PLY alignment file (Tanks and Temples)")
    parser.add_argument("--lidar_ply_fpath", type=str, help="Path to LiDAR PLY file (Tanks and Temples)")
    parser.add_argument("--colmap_ply_fpath", type=str, help="Path to COLMAP PLY file (Tanks and Temples)")
    parser.add_argument("--max_num_images", type=int, help="Maximum number of images (Tanks and Temples)")

    # YFCC IMB-specific arguments
    parser.add_argument("--co_visibility_threshold", type=float, help="Co-visibility threshold (YFCC IMB)")


def build_loader_overrides(args: Namespace, default_max_resolution: Optional[int] = None) -> List[str]:
    """Construct Hydra overrides for the selected loader based on parsed CLI args.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args from the runner.
    default_max_resolution : Optional[int]
        Value to force into `SceneOptimizer.loader.max_resolution`. If None, no override is added.

    Returns
    -------
    List[str]
        Hydra overrides to extend the top-level `overrides` list in the runner.
    """
    overrides: List[str] = []

    # Loader choice
    if getattr(args, "loader", None):
        overrides.append(f"+loader@SceneOptimizer.loader={args.loader}")

    # Standardized loader parameter overrides
    if getattr(args, "dataset_dir", None):
        overrides.append(f"SceneOptimizer.loader.dataset_dir={args.dataset_dir}")
    if getattr(args, "images_dir", None):
        overrides.append(f"SceneOptimizer.loader.images_dir={args.images_dir}")
    if getattr(args, "max_length", None) is not None:
        overrides.append(f"SceneOptimizer.loader.max_length={args.max_length}")
    if getattr(args, "scene_name", None):
        overrides.append(f"SceneOptimizer.loader.scene_name={args.scene_name}")

    # Argoverse-specific overrides
    if getattr(args, "log_id", None) and args.loader == "argoverse_loader":
        overrides.append(f"SceneOptimizer.loader.log_id={args.log_id}")
    if getattr(args, "stride", None) is not None and args.loader == "argoverse_loader":
        overrides.append(f"SceneOptimizer.loader.stride={args.stride}")
    if getattr(args, "max_num_imgs", None) is not None and args.loader == "argoverse_loader":
        overrides.append(f"SceneOptimizer.loader.max_num_imgs={args.max_num_imgs}")
    if getattr(args, "max_lookahead_sec", None) is not None and args.loader == "argoverse_loader":
        overrides.append(f"SceneOptimizer.loader.max_lookahead_sec={args.max_lookahead_sec}")
    if getattr(args, "camera_name", None) and args.loader == "argoverse_loader":
        overrides.append(f"SceneOptimizer.loader.camera_name={args.camera_name}")

    # MobileBrick/COLMAP-specific overrides
    if getattr(args, "use_gt_intrinsics", False) and args.loader in ["mobilebrick_loader", "colmap_loader"]:
        overrides.append(f"SceneOptimizer.loader.use_gt_intrinsics={args.use_gt_intrinsics}")
    if getattr(args, "use_gt_extrinsics", False) and args.loader == "colmap_loader":
        overrides.append(f"SceneOptimizer.loader.use_gt_extrinsics={args.use_gt_extrinsics}")

    # Argoverse/Hilti max_frame_lookahead (this flag lives outside loader args in runner)
    if getattr(args, "max_frame_lookahead", None) is not None and args.loader in ["argoverse_loader", "hilti_loader"]:
        overrides.append(f"SceneOptimizer.loader.max_frame_lookahead={args.max_frame_lookahead}")

    # 1DSFM-specific overrides
    if getattr(args, "enable_no_exif", False) and args.loader == "one_d_sfm_loader":
        overrides.append(f"SceneOptimizer.loader.enable_no_exif={args.enable_no_exif}")
    if getattr(args, "default_focal_length_factor", None) is not None and args.loader == "one_d_sfm_loader":
        overrides.append(f"SceneOptimizer.loader.default_focal_length_factor={args.default_focal_length_factor}")

    # Tanks and Temples-specific overrides
    if getattr(args, "poses_fpath", None) and args.loader == "tanks_and_temples_loader":
        overrides.append(f"SceneOptimizer.loader.poses_fpath={args.poses_fpath}")
    if getattr(args, "bounding_polyhedron_json_fpath", None) and args.loader == "tanks_and_temples_loader":
        overrides.append(f"SceneOptimizer.loader.bounding_polyhedron_json_fpath={args.bounding_polyhedron_json_fpath}")
    if getattr(args, "ply_alignment_fpath", None) and args.loader == "tanks_and_temples_loader":
        overrides.append(f"SceneOptimizer.loader.ply_alignment_fpath={args.ply_alignment_fpath}")
    if getattr(args, "lidar_ply_fpath", None) and args.loader == "tanks_and_temples_loader":
        overrides.append(f"SceneOptimizer.loader.lidar_ply_fpath={args.lidar_ply_fpath}")
    if getattr(args, "colmap_ply_fpath", None) and args.loader == "tanks_and_temples_loader":
        overrides.append(f"SceneOptimizer.loader.colmap_ply_fpath={args.colmap_ply_fpath}")
    if getattr(args, "max_num_images", None) is not None and args.loader == "tanks_and_temples_loader":
        overrides.append(f"SceneOptimizer.loader.max_num_images={args.max_num_images}")

    # YFCC IMB-specific overrides
    if getattr(args, "co_visibility_threshold", None) is not None and args.loader == "yfcc_imb_loader":
        overrides.append(f"SceneOptimizer.loader.co_visibility_threshold={args.co_visibility_threshold}")

    # Common convenience override
    if default_max_resolution is not None:
        overrides.append(f"SceneOptimizer.loader.max_resolution={default_max_resolution}")

    return overrides
