"""
Module for COLMAP scene data I/O stored as a tree structure.
Authors: Frank Dellaert
"""

from pathlib import Path

from gtsfm.common.gtsfm_data import GtsfmData
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


def read_colmap_hierarchy_as_tree(base_dir: str | Path, name: str = "ba_output") -> SceneTree | None:
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


def write_colmap_hierarchy_as_tree(tree: SceneTree | None, output_base_dir: str | Path, **kwargs) -> None:
    """Write a COLMAP hierarchy stored as a tree in the exact nested directory structure on disk.
    Args:
        tree: Tree whose nodes contain (Path, GtsfmData|None) tuples.
        output_base_dir: Root directory where the COLMAP hierarchy will be written.
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
            ba_output_dir = path / "ba_output"
            scene.export_as_colmap_text(ba_output_dir, **kwargs)
