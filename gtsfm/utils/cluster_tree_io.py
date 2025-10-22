"""
Module for COLMAP scene data I/O stored as a tree structure.
Authors: Frank Dellaert
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gtsam import Cal3Bundler, Pose3, SfmTrack  # type: ignore

from gtsfm.utils.io import read_scene_data_from_colmap_format
from gtsfm.utils.tree import Tree


@dataclass(frozen=True)
class ColmapScene:
    """Data class to hold COLMAP scene data.
    img_fnames: File names of images in images_gtsfm.
    wTi_list: List of N camera poses when each image was taken.
    calibrations: List of N camera calibrations corresponding to the N images in images_gtsfm.
    sfm_tracks: Tracks of points in points3D.
    point_cloud: (N,3) array representing xyz coordinates of 3d points.
    rgb: Uint8 array of shape (N,3) representing per-point colors.
    img_dims: List of dimensions of each img (H, W).
    """

    wTi_list: list[Pose3]
    img_fnames: list[str]
    calibrations: list[Cal3Bundler]
    point_cloud: np.ndarray
    rgb: np.ndarray
    img_dims: list[tuple[int, int]]
    sfm_tracks: list[SfmTrack] | None = None

    @classmethod
    def read_from_disk(cls, root: str) -> "ColmapScene":
        """Read COLMAP scene data from disk at the given directory."""
        return cls(*read_scene_data_from_colmap_format(root))

    def num_points(self) -> int:
        """Return the number of 3D points in the scene."""
        return self.point_cloud.shape[0]

    def is_valid(self) -> bool:
        """Validate the integrity of the ColmapScene data."""
        return (
            len(self.img_fnames) != 0
            and self.point_cloud.shape[1] == 3
            and self.rgb.shape[1] == 3
            and self.rgb.shape[0] == self.num_points()
        )

    def __repr__(self) -> str:
        return f"ColmapScene(num_images={len(self.img_fnames)}, num_points={self.num_points()})"


def read_dir_hierarchy_as_tree(base_dir: str) -> Tree[Path]:
    """Read a hierarchy of directories into a Tree[Path].

    Each node's value is the absolute Path of that directory. Children are subdirectories.
    Files are ignored.

    Args:
        base_dir: Root directory to scan.

    Returns:
        A Tree whose root value is Path(base_dir) and whose children represent subdirectories.
    """
    root_path = Path(base_dir)
    # If base_dir does not exist, raise an error
    if not root_path.exists():
        raise FileNotFoundError(f"Base directory '{base_dir}' does not exist.")

    def build_dir_tree(p: Path) -> Tree[Path]:
        children: list[Tree[Path]] = []
        for entry in sorted(p.iterdir(), key=lambda e: e.name):
            if entry.is_dir():
                children.append(build_dir_tree(entry))
        return Tree(value=p, children=tuple(children))

    return build_dir_tree(root_path)


NamedColmapScene = tuple[Path, ColmapScene | None]


def read_colmap_hierarchy_as_tree(base_dir: str, name: str = "ba_output") -> Tree[NamedColmapScene] | None:
    """Read a COLMAP hierarchy stored on disk as a tree.
    Args:
        base_dir: Root directory containing the COLMAP hierarchy.
        name: Name of the subdirectory in each node where COLMAP data is stored.

    Returns:
        A Tree whose nodes contain (Path, ColmapScene|None) tuples.
        Nodes without COLMAP data have None as the second element.
        If no COLMAP data is found in the entire hierarchy, returns None.
    """
    dir_tree = read_dir_hierarchy_as_tree(base_dir)

    def transform(p: Path) -> NamedColmapScene:
        ba_dir = p / name
        if ba_dir.is_dir():
            scene = ColmapScene.read_from_disk(str(ba_dir))
            return (p, scene)
        return (p, None)

    # Map the directory tree into a Tree[(str, ColmapScene|None)]
    mapped_tree = dir_tree.map(transform)

    # Prune nodes without scene data
    return mapped_tree.prune(lambda x: x[1] is not None)
