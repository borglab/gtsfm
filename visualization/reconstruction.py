"""Lightweight inspection of saved reconstructions, without loading CUDA or GTSAM."""

from functools import lru_cache
from pathlib import Path

from visualization.photos import ScenePhotos


def result_revision(paths: list[Path | None]) -> str:
    """Identify a result's current geometry, metadata, and splats without reading their contents.

    Args:
        paths: Ordered result files, including optional or missing files.

    Returns:
        A cache revision that changes when any file's timestamp or size changes.
    """
    stamps = []
    for path in paths:
        try:
            stat = path.stat() if path is not None else None
        except FileNotFoundError:
            stat = None
        stamps.append(f"{stat.st_mtime_ns}:{stat.st_size}" if stat is not None else "missing")
    return "-".join(stamps)


@lru_cache(maxsize=512)
def _point_count(path: Path, modified: int, size: int) -> int:
    del modified, size  # Invalidate cached counts when a result changes.
    with path.open(encoding="utf-8") as stream:
        return sum(1 for line in stream if line.strip() and not line.lstrip().startswith("#"))


def point_count(scene: Path) -> int:
    path = scene / "points3D.txt"
    stat = path.stat()
    return _point_count(path, stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=512)
def _splat_count(path: Path, modified: int, size: int) -> int | None:
    del modified, size
    with path.open("rb") as stream:
        for _ in range(256):
            line = stream.readline(4096).strip()
            if line.startswith(b"element vertex "):
                return int(line.split()[-1])
            if not line or line == b"end_header":
                break
    return None


def splat_count(path: Path) -> int | None:
    stat = path.stat()
    return _splat_count(path, stat.st_mtime_ns, stat.st_size)


def splat_sources(results_root: Path, relative: str) -> tuple[Path, dict[str, Path]]:
    """Validate a nonempty text reconstruction and resolve all registered photos."""
    photos = ScenePhotos(results_root)
    scene = photos.scene(relative)
    for name in ("cameras.txt", "points3D.txt"):
        path = scene / name
        if not path.resolve().is_relative_to(photos.root) or not path.is_file():
            raise ValueError(f"Scene is missing a workspace-local {name}")
    if point_count(scene) == 0:
        raise ValueError("This reconstruction has no 3D points")
    names = photos.names(scene)
    if not names:
        raise ValueError("This reconstruction has no registered cameras")
    roots = photos.roots(scene)
    sources = {}
    for name in names:
        source = photos.source(roots, name)
        if source is None:
            raise ValueError(
                f"Original image is unavailable: {name}. Restore the source dataset to its saved location."
            )
        sources[name] = source
    return scene, sources
