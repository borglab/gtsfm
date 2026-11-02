"""Dataset path helpers for Studio uploads and curated examples."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path, PurePosixPath
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
        if value == "":
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


def extract_input_archive(archive_path: Path, destination: Path) -> tuple[int, int]:
    """Extract regular files from a tar archive without allowing path traversal or links."""

    file_count = 0
    total_bytes = 0
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name.replace("\\", "/"))
            if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise ValueError(f"Unsafe path in uploaded archive: {member.name}")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported entry in uploaded archive: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read uploaded archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
            file_count += 1
            total_bytes += target.stat().st_size
    return file_count, total_bytes
