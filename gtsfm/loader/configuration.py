"""Base class for Loaders.

Authors: Frank Dellaert
"""

from __future__ import annotations  # not needed after 3.11

from argparse import ArgumentParser, Namespace
from typing import List, Optional


def add_loader_args(parser: ArgumentParser) -> None:
    """Register *portable* loader-related CLI arguments on the given parser.

    Only universal knobs live here; all other loader-specific settings must be passed as Hydra overrides
    on `SceneOptimizer.loader.*`.
    """
    # Loader selection (Hydra config key under gtsfm/configs/loader/)
    parser.add_argument(
        "--loader",
        type=str,
        default="olsson_loader",
        help=(
            "Loader type. Available options include: colmap_loader, hilti_loader, astrovision_loader, "
            "olsson_loader, argoverse_loader, mobilebrick_loader, one_d_sfm_loader, "
            "tanks_and_temples_loader, yfcc_imb_loader. Default: olsson_loader"
        ),
    )

    # Portable dataset location arguments
    parser.add_argument("--dataset_dir", type=str, help="Path to dataset directory/root for the chosen loader")
    parser.add_argument(
        "--images_dir",
        type=str,
        help="Path to images directory (optional; defaults depend on the loader)",
    )

    # Universal loader knobs shared by all loaders (from LoaderBase)
    parser.add_argument(
        "--max_resolution",
        type=int,
        help="Maximum length of the image's short side (overrides config's SceneOptimizer.loader.max_resolution)",
    )
    parser.add_argument(
        "--input_worker",
        type=str,
        help="Optional Dask worker address to pin image I/O to (e.g. 'tcp://IP:PORT'). "
        "If provided, the runner will set loader._input_worker post-instantiation.",
    )


def build_loader_overrides(args: Namespace, default_max_resolution: Optional[int] = None) -> List[str]:
    """Construct Hydra overrides for portable loader settings."""
    overrides: List[str] = []

    # Loader choice (swap the nested node)
    if getattr(args, "loader", None):
        overrides.append(f"+loader@SceneOptimizer.loader={args.loader}")

    # Dataset locations
    if getattr(args, "dataset_dir", None):
        overrides.append(f"SceneOptimizer.loader.dataset_dir={args.dataset_dir}")
    if getattr(args, "images_dir", None):
        overrides.append(f"SceneOptimizer.loader.images_dir={args.images_dir}")

    # Max resolution: prefer explicit CLI, otherwise use provided default
    if getattr(args, "max_resolution", None) is not None:
        overrides.append(f"SceneOptimizer.loader.max_resolution={args.max_resolution}")
    elif default_max_resolution is not None:
        overrides.append(f"SceneOptimizer.loader.max_resolution={default_max_resolution}")

    # Note: input_worker cannot be passed via Hydra for loaders whose __init__ doesn't accept it.
    # The runner sets loader._input_worker post-instantiation if args.input_worker is provided.

    return overrides
