"""Dataset path helpers for Studio uploads and curated examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".avif", ".bmp", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve_relative_loader_paths(
    dataset_dir: Path, loader_options: dict[str, Any], images_dir: str | None
) -> tuple[dict[str, Any], str | None]:
    """Resolve configured relative paths against the active dataset root."""

    resolved = dict(loader_options)
    for name, value in list(resolved.items()):
        if not isinstance(value, str) or not (name.endswith("_path") or name.endswith("_fpath")):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            resolved[name] = str((dataset_dir / path).resolve())
    if images_dir:
        image_path = Path(images_dir).expanduser()
        images_dir = (
            str((dataset_dir / image_path).resolve()) if not image_path.is_absolute() else str(image_path.resolve())
        )
    return resolved, images_dir
