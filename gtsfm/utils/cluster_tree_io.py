"""
Module for COLMAP scene data I/O stored as a tree structure.
Authors: Frank Dellaert
"""

from pathlib import Path

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils.tree import Tree


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


NamedGtsfmData = tuple[Path, GtsfmData | None]


def read_colmap_hierarchy_as_tree(base_dir: str | Path, name: str = "ba_output") -> Tree[NamedGtsfmData] | None:
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

    def transform(p: Path) -> NamedGtsfmData:
        ba_dir = p / name
        if ba_dir.is_dir():
            scene = GtsfmData.read_colmap(str(ba_dir))
            return (p, scene)
        return (p, None)

    # Map the directory tree into a Tree[(str, GtsfmData|None)]
    mapped_tree = dir_tree.map(transform)

    # Prune nodes without scene data
    return mapped_tree.prune(lambda x: x[1] is not None)
