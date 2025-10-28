"""
Module for COLMAP scene data I/O stored as a tree structure.
Authors: Frank Dellaert
"""

import random
from pathlib import Path

import gtsfm.utils.alignment as alignment_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.utils.tree import PreOrderIter, Tree


def read_dir_hierarchy_as_tree(base_dir: str | Path) -> Tree[Path]:
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


LocalScene = tuple[Path, GtsfmData | None]
SceneTree = Tree[LocalScene]  # Tree whose nodes contain (Path, GtsfmData|None) tuples.


def downsample(tree: SceneTree, **kwargs) -> SceneTree:
    """Test reading the COLMAP hierarchy as a tree of (Path, GtsfmData) with downsampling."""

    def f(path_scene) -> LocalScene:
        path, scene = path_scene
        return (path, scene.downsample(**kwargs) if scene else None)

    return tree.map(f)


def number_tracks(tree: SceneTree) -> int:
    """Total number of tracks in entire tree."""

    def f(path_scene: LocalScene, children):
        sum_below = sum(children)
        return sum_below + (path_scene[1].number_tracks() if path_scene[1] else 0)

    return tree.fold(f)


def color_by_cluster(tree: SceneTree) -> SceneTree:
    """Color each cluster in the scene tree with a different random color."""
    assert tree is not None, "color_by_cluster: input tree is None"

    def color_one(path_scene):
        path, scene = path_scene
        new_tracks = scene.tracks().copy()
        r = random.randint(0, 255)
        g = random.randint(0, 255)
        b = random.randint(0, 255)
        for track in new_tracks:
            track.r = r
            track.g = g
            track.b = b
        new_data = GtsfmData(scene.number_images(), scene.cameras(), new_tracks)
        return path, new_data

    return tree.map(color_one)


def reorder(tree: SceneTree, cluster_tree: ClusterTree) -> SceneTree:
    """Reorder cameras in each scene according to the visibility graph in the cluster tree."""
    assert tree is not None, "reorder: input tree is None"

    def reorder(vg_path_scene):
        vg, (path, scene) = vg_path_scene
        keys = sorted(visibility_graph_keys(vg))
        new_cameras = {k: scene.get_camera(i) for i, k in enumerate(keys)}
        new_data = GtsfmData(len(new_cameras), new_cameras, [])
        # HACK(Frank): don't remap tracks for now, just copy them over.
        new_data._tracks = scene.tracks().copy()
        return path, new_data

    return Tree.zip(cluster_tree, tree).map(reorder)


def merge(tree: SceneTree) -> GtsfmData:
    """Merge a cluster tree of ColmapScenes into a single GtsfmData.

    Args:
        tree: Tree whose values are (Path, GtsfmData) tuples.

    Returns:
        Merged GtsfmData.
    """

    def f(path_scene: LocalScene, merged_children: tuple[GtsfmData, ...], verbose: bool = False) -> GtsfmData:
        path, scene = path_scene
        if scene is None:
            raise ValueError(f"merge_colmap_tree: scene at path {path} is None.")
        if len(merged_children) == 0:
            return scene
        merged_scene = scene
        for child in merged_children:
            try:
                aSb = alignment_utils.estimate_sim3_from_pose_maps(merged_scene.poses(), child.poses())
                merged_scene = merged_scene.merged_with(child, aSb)
            except Exception as e:
                child_keys = list(child.cameras().keys())
                if verbose:
                    print(
                        f"Failed to merge child scene with keys {sorted(child_keys)} into parent at {path} "
                        f"with keys {sorted(scene.cameras().keys())}: {e}"
                    )
        return merged_scene

    return tree.fold(f)


def read_colmap(base_dir: str | Path, name: str = "ba_output") -> SceneTree | None:
    """Read a COLMAP hierarchy stored on disk as a tree.
    Args:
        base_dir: Root directory containing the COLMAP hierarchy.
        name: Name of the subdirectory in each node where COLMAP data is stored.

    Returns:
        A Tree whose nodes contain (Path, GtsfmData|None) tuples.
        Nodes without COLMAP data have None as the second element.
        If no COLMAP data is found in the entire hierarchy, returns None.
    """
    dir_tree = read_dir_hierarchy_as_tree(base_dir)

    def transform(p: Path) -> LocalScene:
        ba_dir = p / name
        relative_path = p.relative_to(base_dir)
        if ba_dir.is_dir():
            scene = GtsfmData.read_colmap(str(ba_dir))
            return (relative_path, scene)
        return (relative_path, None)

    # Map the directory tree into a Tree[(str, GtsfmData|None)]
    mapped_tree = dir_tree.map(transform)

    # Prune nodes without scene data
    return mapped_tree.prune(lambda x: x[1] is not None)


def write_colmap(tree: SceneTree | None, output_base_dir: str | Path, name: str = "ba_output", **kwargs) -> None:
    """Write a COLMAP hierarchy stored as a tree in the exact nested directory structure on disk.
    Args:
        tree: Tree whose nodes contain (Path, GtsfmData|None) tuples.
        output_base_dir: Root directory where the COLMAP hierarchy will be written.
        name: Name of the subdirectory in each node where COLMAP data will be stored.
        **kwargs: Additional keyword arguments to pass to GtsfmData.export_as_colmap_text().
    """
    if tree is None:
        return

    output_base_path = Path(output_base_dir)
    output_base_path.mkdir(parents=True, exist_ok=True)

    for node in PreOrderIter(tree):
        relative_path, scene = node.value
        if scene is not None:
            path = output_base_path / relative_path
            ba_output_dir = path / name
            scene.export_as_colmap_text(ba_output_dir, **kwargs)
