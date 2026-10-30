"""Resolve reconstruction photos and serve bounded, cached display-size images.

Only image names registered in a scene's COLMAP model can be requested. Dataset
locations come from the saved run configuration, never from a client file path.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import yaml
from PIL import Image, ImageOps


@lru_cache(maxsize=32)
def _image_records(path: Path, modified: int) -> tuple[tuple[int, str], ...]:
    del modified  # Part of the cache key: rewriting a reconstruction invalidates it.
    names = []
    pose_line = True
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("#"):
            continue
        if pose_line:
            if not line.strip():
                continue
            fields = line.split(maxsplit=9)
            if len(fields) != 10:
                raise ValueError("Invalid COLMAP image record")
            names.append((int(fields[0]), fields[9]))
        pose_line = not pose_line
    return tuple(names)


@lru_cache(maxsize=32)
def _loader_config(path: Path, modified: int) -> dict[str, Any]:
    del modified
    try:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError("Invalid saved run configuration") from exc
    return config.get("loader", {}) if isinstance(config, dict) else {}


@lru_cache(maxsize=32)
def _loader_image_names(path: Path, modified: int) -> tuple[str, ...]:
    del modified
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("Invalid recorded loader image names")
    return tuple(names)


class ScenePhotos:
    """Small shared JPEG cache with at most two concurrent image decoders."""

    def __init__(self, results_root: Path, cache_bytes: int = 32 * 1024 * 1024):
        self.root = results_root.resolve()
        self.cache_bytes = cache_bytes
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self._decoders = threading.BoundedSemaphore(2)

    def scene(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Scene is outside the results workspace")
        images = path / "images.txt"
        if not images.resolve().is_relative_to(self.root):
            raise ValueError("Scene is outside the results workspace")
        if not images.is_file():
            raise FileNotFoundError("Scene has no registered cameras")
        return path

    def names(self, scene: Path) -> tuple[str, ...]:
        return tuple(name for _, name in self.records(scene))

    def records(self, scene: Path) -> tuple[tuple[int, str], ...]:
        path = scene / "images.txt"
        return _image_records(path, path.stat().st_mtime_ns)

    def roots(self, scene: Path) -> list[Path]:
        """Use the nearest run config, or adjacent images for a portable COLMAP export."""
        locations = {}
        for parent in (scene, *scene.parents):
            if not parent.is_relative_to(self.root):
                break
            portable_images = parent / "source-images"
            if portable_images.is_dir():
                return [portable_images]
            mapping = parent / "studio-inputs.json"
            if mapping.is_file() and mapping.resolve().is_relative_to(self.root):
                locations = json.loads(mapping.read_text(encoding="utf-8"))
                break
        current = scene
        while current.is_relative_to(self.root):
            config = current / "config.yaml"
            if config.is_file() and config.resolve().is_relative_to(self.root):
                loader = _loader_config(config, config.stat().st_mtime_ns)
                if not isinstance(loader, dict):
                    break
                loader = {**loader, **locations}
                dataset = loader.get("dataset_dir")
                images = loader.get("images_dir")
                dataset_root = Path(dataset).expanduser() if dataset else current
                # New runs save absolute paths; relative paths in portable exports
                # are interpreted relative to the saved configuration, then the
                # server working directory for older CLI runs.
                if not dataset_root.is_absolute():
                    portable = current / dataset_root
                    dataset_root = portable if portable.is_dir() else Path.cwd() / dataset_root
                if images:
                    image_root = Path(images).expanduser()
                    return [(image_root if image_root.is_absolute() else dataset_root / image_root).resolve()]
                kind = str(loader.get("_target_", "")).lower()
                if "mobilebrick" in kind:
                    return [(dataset_root / "image").resolve()]
                if "argoverse" in kind:
                    return [
                        (
                            dataset_root
                            / str(loader.get("log_id", ""))
                            / str(loader.get("camera_name", "ring_front_center"))
                        ).resolve()
                    ]
                return [(dataset_root / "images").resolve(), dataset_root.resolve()]
            if current == self.root:
                break
            current = current.parent
        return [scene / "images", scene.parent / "images"]

    def source_name(self, scene: Path, camera_id: int, name: str) -> str:
        """Resolve the exporter's placeholder by its global loader index."""
        match = re.fullmatch(r"image_(\d{6,})\.jpg", name)
        if match and int(match[1]) == camera_id:
            for parent in (scene, *scene.parents):
                if not parent.is_relative_to(self.root):
                    break
                mapping = parent / "image_filenames.json"
                if mapping.is_file() and mapping.resolve().is_relative_to(self.root):
                    names = _loader_image_names(mapping, mapping.stat().st_mtime_ns)
                    index = int(match[1])
                    if index < len(names) and isinstance(names[index], str):
                        return names[index]
                    break
        return name

    @staticmethod
    def source(roots: list[Path], name: str) -> Path | None:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            return None
        for root in roots:
            root = root.resolve()
            # YFCC records extensionless image identifiers in its COLMAP export.
            candidates = [relative] if relative.suffix else [relative, relative.with_suffix(".jpg")]
            for candidate in candidates:
                path = (root / candidate).resolve()
                if path.is_relative_to(root) and path.is_file():
                    return path
        return None

    def manifest(self, relative: str) -> dict[str, Any]:
        scene = self.scene(relative)
        roots = self.roots(scene)
        items = []
        for index, (camera_id, name) in enumerate(self.records(scene)):
            source = self.source(roots, self.source_name(scene, camera_id, name))
            available = source is not None
            version = self.etag(source, 256).strip('"') if source else "missing"
            query = urlencode({"scene": relative, "index": index, "v": version})
            items.append(
                {
                    "name": name,
                    "thumbnail": f"/api/photos/image?{query}&size=256" if available else None,
                    "preview": f"/api/photos/image?{query}&size=1600" if available else None,
                }
            )
        return {"items": items}

    def image_source(self, relative: str, index: int) -> Path:
        scene = self.scene(relative)
        records = self.records(scene)
        if index < 0 or index >= len(records):
            raise FileNotFoundError("Unknown scene image")
        camera_id, name = records[index]
        path = self.source(self.roots(scene), self.source_name(scene, camera_id, name))
        if path is None:
            raise FileNotFoundError("Original image is unavailable; restore the source dataset to its saved location")
        return path

    @staticmethod
    def etag(path: Path, size: int) -> str:
        stat = path.stat()
        key = f"{path}:{stat.st_mtime_ns}:{stat.st_size}:{size}:v1"
        return '"' + hashlib.sha256(key.encode()).hexdigest() + '"'

    def resized(self, path: Path, size: int, etag: str) -> bytes:
        with self._decoders:
            with self._lock:
                cached = self._cache.get(etag)
                if cached is not None:
                    self._cache.move_to_end(etag)
                    return cached
            with Image.open(path) as image:
                # JPEG draft decoding avoids expanding large originals at full size.
                image.draft("RGB", (size, size))
                image = ImageOps.exif_transpose(image)
                image.thumbnail((size, size), Image.Resampling.LANCZOS)
                output = io.BytesIO()
                image.convert("RGB").save(output, format="JPEG", quality=82)
                result = output.getvalue()
            with self._lock:
                previous = self._cache.pop(etag, None)
                self._bytes += len(result) - (len(previous) if previous else 0)
                self._cache[etag] = result
                while self._bytes > self.cache_bytes and self._cache:
                    _, evicted = self._cache.popitem(last=False)
                    self._bytes -= len(evicted)
            return result
