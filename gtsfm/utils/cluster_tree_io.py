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

    def build_dir_tree(p: Path) -> Tree[Path]:
        # Only include directories. Sort for determinism in tests.
        children: list[Tree[Path]] = []
        try:
            for entry in sorted(p.iterdir(), key=lambda e: e.name):
                if entry.is_dir():
                    children.append(build_dir_tree(entry))
        except FileNotFoundError:
            # If base_dir does not exist, still return a node with no children.
            pass
        return Tree(value=p, children=tuple(children))

    return build_dir_tree(root_path)


NamedColmapScene = tuple[Path, ColmapScene | None]


def print_tree_cameras_and_points(tree: Tree[NamedColmapScene] | None) -> None:
    """Prints each node in the tree with #cameras and #points if Scene, else None."""
    if tree is None:
        print("No tree found.")
        return
    for node in tree.traverse():
        path, scene = node.value
        if scene is None:
            print(f"{path}: None")
        else:
            num_cameras = len(scene.img_fnames)
            num_points = scene.point_cloud.shape[0] if scene.point_cloud is not None else 0
            print(f"{path}: {num_cameras} cameras, {num_points} points")


def read_colmap_hierarchy_as_tree(base_dir: str, name: str = "ba_output") -> Tree[NamedColmapScene] | None:
    dir_tree = read_dir_hierarchy_as_tree(base_dir)

    found_any = False

    def transform(p: Path) -> NamedColmapScene:
        nonlocal found_any
        ba_dir = p / name
        if ba_dir.is_dir():
            found_any = True
            scene = ColmapScene.read_from_disk(str(ba_dir))
            return (p, scene)
        return (p, None)

    # Map the directory tree into a Tree[(str, ColmapScene|None)]
    mapped_tree = dir_tree.map(transform)

    def prune_empty(node: Tree[NamedColmapScene]) -> Tree[NamedColmapScene] | None:
        label, scene = node.value
        # Prune children first.
        pruned_children = [c for c in (prune_empty(ch) for ch in node.children) if c is not None]
        # Keep this node if it is the root, or it has a scene, or it has any pruned children.
        if label == "" or scene is not None or pruned_children:
            return Tree(value=node.value, children=tuple(pruned_children))
        # Otherwise, drop internal nodes with no scene descendants.
        return None

    result = prune_empty(mapped_tree)
    return result if (result is not None and found_any) else None
