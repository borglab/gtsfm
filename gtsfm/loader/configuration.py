"""Base class for Loaders.

Authors: Frank Dellaert
"""

from __future__ import annotations  # not needed after 3.11

from argparse import ArgumentParser, Namespace
from typing import List, Optional


def add_loader_args(parser: ArgumentParser) -> None:
    """Register *portable* loader-related CLI arguments on the given parser.

    Only universal knobs live here; all other loader-specific settings must be passed as Hydra overrides
    on `loader.*`.
    """
    # Loader selection (Hydra config key under gtsfm/configs/loader/)
    parser.add_argument(
        "--loader",
        type=str,
        default=None,
        help=(
            "Loader type. Available options include: argoverse, astrovision, colmap, "
            "hilti, mobilebrick, olsson, one_d_sfm, "
            "tanks_and_temples, yfcc_imb. Falls back to config when omitted."
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
        default=760,
        help="integer representing maximum length of image's short side"
        " e.g. for 1080p (1920 x 1080), max_resolution would be 1080",
    )
    parser.add_argument(
        "--input_worker",
        type=str,
        help="Optional Dask worker address to pin image I/O to (e.g. 'tcp://IP:PORT'). "
        "If provided, the runner will set loader._input_worker post-instantiation.",
    )


def build_loader_overrides(
    args: Namespace, default_max_resolution: Optional[int] = None, default_input_worker: Optional[str] = None
) -> List[str]:
    """Construct Hydra overrides for portable loader settings."""
    overrides: List[str] = []

    # Loader choice (swap the nested node)
    if getattr(args, "loader", None):
        overrides.append(f"+loader@loader={args.loader}")

    # Dataset locations
    if getattr(args, "dataset_dir", None):
        overrides.append(f"loader.dataset_dir={args.dataset_dir}")
    if getattr(args, "images_dir", None):
        overrides.append(f"loader.images_dir={args.images_dir}")

    # Max resolution: prefer explicit CLI, otherwise use provided default
    if getattr(args, "max_resolution", None) is not None:
        overrides.append(f"loader.max_resolution={args.max_resolution}")
    elif default_max_resolution is not None:
        overrides.append(f"loader.max_resolution={default_max_resolution}")

    # Max resolution: prefer explicit CLI, otherwise use provided default
    if getattr(args, "input_worker", None) is not None:
        overrides.append(f"loader.input_worker={args.input_worker}")
    elif default_input_worker is not None:
        overrides.append(f"loader.input_worker={default_input_worker}")

    return overrides
