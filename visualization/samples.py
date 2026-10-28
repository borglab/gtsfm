"""Curated GTSFM GitHub sample datasets for the browser workspace."""

from __future__ import annotations

import json
import os
import shutil
import ssl
import threading
import urllib.error
import urllib.request
import uuid
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import quote

import certifi


GITHUB_REPOSITORY = "borglab/gtsfm"
GITHUB_BRANCH = "master"
GITHUB_CONTENTS_API = f"https://api.github.com/repos/{GITHUB_REPOSITORY}/contents"
SOURCE_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
_DOWNLOAD_LOCK = threading.Lock()
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())

SAMPLE_DATASETS: tuple[dict[str, Any], ...] = (
    {
        "id": "one-d-sfm",
        "label": "1DSfM Images",
        "description": "4-image internet-photo scene without reconstruction metadata.",
        "image_count": 4,
        "source_path": "tests/data/1dsfm",
        "recommendations": {
            "loader": "one_d_sfm",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {"enable_no_exif": True, "default_focal_length_factor": 1.2},
        },
    },
    {
        "id": "argoverse",
        "label": "Argoverse Tracking",
        "description": "2-frame vehicle log with calibration, poses, and timestamped camera images.",
        "image_count": 2,
        "source_path": "tests/data/argoverse/train1",
        "recommendations": {
            "loader": "argoverse",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {
                "log_id": "273c1883-673a-36bf-b124-88311b1a80be",
                "stride": 1,
                "max_num_imgs": 2,
                "camera_name": "ring_front_center",
            },
        },
    },
    {
        "id": "astrovision-vesta",
        "label": "AstroVision Vesta",
        "description": "4-image grayscale scene with COLMAP binary geometry and a Vesta mesh.",
        "image_count": 4,
        "source_path": "tests/data/astrovision/test_2011212_opnav_022",
        "recommendations": {
            "loader": "astrovision",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {"gt_scene_mesh_path": "vesta_5002.ply", "use_gt_extrinsics": True},
        },
    },
    {
        "id": "lund-door",
        "label": "Lund Door",
        "description": "12-image reference scene with camera metadata and ground truth.",
        "image_count": 12,
        "source_path": "tests/data/set1_lund_door",
        "recommendations": {
            "loader": "olsson",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {},
        },
    },
    {
        "id": "crane-mast",
        "label": "Crane Mast",
        "description": "2-image scene with COLMAP cameras, poses, and sparse points.",
        "image_count": 2,
        "source_path": "tests/data/crane_mast_8imgs_colmap_output",
        "recommendations": {
            "loader": "colmap",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {"use_gt_intrinsics": True, "use_gt_extrinsics": True},
        },
    },
    {
        "id": "hilti-exp4",
        "label": "Hilti Exp4",
        "description": "16 synchronized multi-camera images with calibration and LiDAR priors.",
        "image_count": 16,
        "source_path": "tests/data/hilti_exp4_small",
        "recommendations": {
            "loader": "hilti",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {"max_length": 3},
        },
    },
    {
        "id": "imb-reichstag",
        "label": "IMB Reichstag",
        "description": "10-image YFCC scene with HDF5 calibration and visibility-pair metadata.",
        "image_count": 10,
        "source_path": "tests/data/imb_reichstag",
        "recommendations": {
            "loader": "yfcc_imb",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {"co_visibility_threshold": 0.1},
        },
    },
    {
        "id": "mobilebrick",
        "label": "MobileBrick",
        "description": "5-image object scene with MobileBrick intrinsics and poses.",
        "image_count": 5,
        "source_path": "tests/data/mobilebrick",
        "recommendations": {
            "loader": "mobilebrick",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {"use_gt_intrinsics": True, "max_frame_lookahead": 5},
        },
    },
    {
        "id": "tanks-temples-barn",
        "label": "Tanks and Temples Barn",
        "description": "3-image Barn scene with Redwood poses, bounds, and alignment metadata.",
        "image_count": 3,
        "source_path": "tests/data/tanks_and_temples_barn",
        "recommendations": {
            "loader": "tanks_and_temples",
            "config_name": "vggt",
            "max_resolution": 518,
            "loader_options": {
                "poses_fpath": "Barn_COLMAP_SfM.log",
                "bounding_polyhedron_json_fpath": "Barn.json",
                "ply_alignment_fpath": "Barn_trans.txt",
                "max_num_images": 3,
            },
        },
    },
)


class SampleDownloadError(RuntimeError):
    """Raised when a curated sample cannot be prepared."""


def _sample(sample_id: str) -> dict[str, Any]:
    for item in SAMPLE_DATASETS:
        if item["id"] == sample_id:
            return item
    raise KeyError(sample_id)


def _source_url(source_path: str) -> str:
    return f"https://github.com/{GITHUB_REPOSITORY}/tree/{GITHUB_BRANCH}/{source_path}"


def _local_source(sample: dict[str, Any]) -> Path | None:
    path = SOURCE_REPOSITORY_ROOT.joinpath(*PurePosixPath(sample["source_path"]).parts)
    return path.resolve() if path.is_dir() else None


def _cached_source(sample: dict[str, Any], cache_root: Path) -> Path | None:
    path = cache_root / str(sample["id"])
    return path.resolve() if path.is_dir() and (path / ".gtsfm-sample.json").is_file() else None


def sample_catalog(cache_root: Path) -> list[dict[str, Any]]:
    """Return public sample metadata and whether each scene is ready locally."""

    items: list[dict[str, Any]] = []
    for sample in SAMPLE_DATASETS:
        public = {key: value for key, value in sample.items() if key != "source_path"}
        public["source_url"] = _source_url(str(sample["source_path"]))
        public["prepared"] = _local_source(sample) is not None or _cached_source(sample, cache_root) is not None
        items.append(public)
    return items


def _github_json(source_path: str) -> list[dict[str, Any]]:
    url = f"{GITHUB_CONTENTS_API}/{quote(source_path, safe='/')}?ref={quote(GITHUB_BRANCH)}"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "gtsfm-studio",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=60, context=_SSL_CONTEXT) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, list):
        raise SampleDownloadError("GitHub returned an invalid directory listing")
    return payload


def _download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "gtsfm-studio"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=120, context=_SSL_CONTEXT) as response, destination.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)


def _download_directory(source_root: str, destination_root: Path) -> None:
    pending = [source_root]
    while pending:
        current = pending.pop()
        for item in _github_json(current):
            item_path = str(item.get("path") or "")
            relative = PurePosixPath(item_path).relative_to(PurePosixPath(source_root))
            if any(part in {"", ".", ".."} for part in relative.parts):
                raise SampleDownloadError(f"GitHub returned an unsafe sample path: {item_path}")
            item_type = item.get("type")
            if item_type == "dir":
                pending.append(item_path)
            elif item_type == "file":
                download_url = item.get("download_url")
                if not isinstance(download_url, str) or not download_url.startswith("https://"):
                    raise SampleDownloadError(f"GitHub did not provide a download URL for {item_path}")
                _download_file(download_url, destination_root.joinpath(*relative.parts))


def prepare_sample(sample_id: str, cache_root: Path) -> dict[str, Any]:
    """Return a local sample path, downloading the fixed GitHub directory when necessary."""

    sample = _sample(sample_id)
    local = _local_source(sample)
    if local is not None:
        return {
            "path": str(local),
            "sample": sample_catalog(cache_root)[[x["id"] for x in SAMPLE_DATASETS].index(sample_id)],
        }

    cache_root.mkdir(parents=True, exist_ok=True)
    with _DOWNLOAD_LOCK:
        cached = _cached_source(sample, cache_root)
        if cached is None:
            temporary = cache_root / f".{sample_id}-{uuid.uuid4().hex[:8]}"
            destination = cache_root / sample_id
            try:
                _download_directory(str(sample["source_path"]), temporary)
                (temporary / ".gtsfm-sample.json").write_text(
                    json.dumps({"id": sample_id, "source": _source_url(str(sample["source_path"]))}),
                    encoding="utf-8",
                )
                os.replace(temporary, destination)
            except (
                SampleDownloadError,
                OSError,
                ValueError,
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
            ) as exc:
                shutil.rmtree(temporary, ignore_errors=True)
                raise SampleDownloadError(f"Unable to download {sample['label']} from GitHub: {exc}") from exc
            cached = destination.resolve()

    public = next(item for item in sample_catalog(cache_root) if item["id"] == sample_id)
    return {"path": str(cached), "sample": public}
