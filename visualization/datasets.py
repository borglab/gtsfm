"""Dataset-layout detection for Studio uploads and curated examples."""

from __future__ import annotations

from pathlib import Path
from typing import Any


IMAGE_SUFFIXES = {".avif", ".bmp", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _relative_files(root: Path) -> list[Path]:
    return [path.relative_to(root) for path in root.rglob("*") if path.is_file()]


def detect_dataset_format(root: Path) -> dict[str, Any]:
    """Infer the most likely GTSFM loader from a dataset's directory structure.

    The detector deliberately uses structural signatures rather than directory
    names. Its result is a recommendation; callers may still select a loader
    manually. Relative path-valued loader options remain portable when a
    dataset is copied to a remote workspace.
    """

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Dataset directory does not exist: {root}")

    files = _relative_files(root)
    lowered = {path.as_posix().lower() for path in files}
    top_dir_names = {path.parts[0] for path in files if len(path.parts) > 1}
    top_dirs = {name.lower() for name in top_dir_names}
    basenames = {path.name.lower() for path in files}
    image_files = [path for path in files if path.suffix.lower() in IMAGE_SUFFIXES]

    def result(
        loader: str,
        confidence: float,
        reason: str,
        *,
        dataset_subdir: str | None = None,
        images_dir: str | None = None,
        loader_options: dict[str, Any] | None = None,
        alternatives: list[str] | None = None,
    ) -> dict[str, Any]:
        return {
            "loader": loader,
            "confidence": confidence,
            "reason": reason,
            "dataset_subdir": dataset_subdir,
            "images_dir": images_dir,
            "loader_options": loader_options or {},
            "alternatives": alternatives or [],
        }

    # MobileBrick has the strongest and most distinctive three-directory signature.
    if {"image", "intrinsic", "pose"}.issubset(top_dirs):
        return result(
            "mobilebrick",
            0.99,
            "Found MobileBrick image, intrinsic, and pose directories.",
            images_dir="image",
            loader_options={"use_gt_intrinsics": True, "max_frame_lookahead": 5},
        )

    # Tanks and Temples contains a Redwood pose log, bounding volume, alignment, and scene image directory.
    pose_logs = sorted(path for path in files if path.name.endswith("_COLMAP_SfM.log"))
    alignments = sorted(path for path in files if path.name.endswith("_trans.txt"))
    scene_json = sorted(path for path in files if path.suffix.lower() == ".json")
    if pose_logs and alignments and scene_json:
        scene_name = pose_logs[0].name.removesuffix("_COLMAP_SfM.log")
        image_dir = next(
            (name for name in top_dir_names if name.lower() == scene_name.lower()),
            next((path.parts[0] for path in image_files if len(path.parts) > 1), "images"),
        )
        return result(
            "tanks_and_temples",
            0.99,
            "Found the Tanks and Temples pose log, bounding volume, and alignment transform.",
            images_dir=str(image_dir),
            loader_options={
                "poses_fpath": pose_logs[0].as_posix(),
                "bounding_polyhedron_json_fpath": scene_json[0].as_posix(),
                "ply_alignment_fpath": alignments[0].as_posix(),
            },
        )

    # Hilti calibration and LiDAR priors are unique even when the sample contains only a few rig timestamps.
    if {"calibration", "images", "lidar"}.issubset(top_dirs) and "lidar/fastlio2.g2o" in lowered:
        return result("hilti", 0.99, "Found Hilti calibration, synchronized images, and LiDAR priors.")

    # YFCC IMB datasets pair HDF5 calibrations with precomputed visibility-pair arrays.
    if {"calibration", "images", "new-vis-pairs"}.issubset(top_dirs) and any(
        path.startswith("new-vis-pairs/keys-th-") and path.endswith(".npy") for path in lowered
    ):
        return result("yfcc_imb", 0.99, "Found YFCC calibration files and IMB visibility-pair arrays.")

    # Argoverse logs are identified by a vehicle calibration file and timestamped camera/pose directories.
    calibration_paths = sorted(path for path in files if path.name == "vehicle_calibration_info.json")
    if calibration_paths:
        calibration = calibration_paths[0]
        log_dir = calibration.parent
        dataset_prefix = log_dir.parent
        # The loader expects dataset_dir to contain log_id directly. A non-empty
        # prefix is surfaced so the caller can adjust the effective root.
        return result(
            "argoverse",
            0.99,
            "Found an Argoverse log with vehicle calibration, poses, and ring-camera images.",
            dataset_subdir=dataset_prefix.as_posix() if dataset_prefix.parts else None,
            loader_options={"log_id": log_dir.name, "stride": 1, "max_num_imgs": len(image_files)},
        )

    # Olsson datasets may also contain a nested COLMAP ground-truth export.
    # Their top-level data.mat is the authoritative loader signature.
    if "data.mat" in basenames and image_files:
        return result("olsson", 0.98, "Found an Olsson data.mat reconstruction with source images.")

    # AstroVision is a COLMAP binary model with an accompanying target-body mesh.
    colmap_binary = {"cameras.bin", "images.bin", "points3d.bin"}.issubset(basenames)
    if colmap_binary and any(
        path.name.lower().startswith("vesta_") and path.suffix.lower() == ".ply" for path in files
    ):
        mesh = next(path for path in files if path.name.lower().startswith("vesta_") and path.suffix.lower() == ".ply")
        return result(
            "astrovision",
            0.99,
            "Found an AstroVision COLMAP binary model and target-body mesh.",
            loader_options={"gt_scene_mesh_path": mesh.as_posix(), "use_gt_extrinsics": True},
        )

    # Generic COLMAP text and binary models share the same three canonical files.
    colmap_text = {"cameras.txt", "images.txt", "points3d.txt"}.issubset(basenames)
    if colmap_text or colmap_binary:
        return result(
            "colmap",
            0.98,
            f"Found a complete COLMAP {'text' if colmap_text else 'binary'} sparse model.",
            loader_options={"use_gt_intrinsics": True, "use_gt_extrinsics": True},
            alternatives=["astrovision"] if colmap_binary else [],
        )

    # Plain image collections are valid but structurally ambiguous. OneDSFM is
    # the safer automatic choice because it tolerates images without EXIF.
    if image_files:
        if "images" in top_dirs:
            images_dir = None
        elif len({path.parts[0] for path in image_files if len(path.parts) > 1}) == 1:
            images_dir = next(path.parts[0] for path in image_files if len(path.parts) > 1)
        else:
            images_dir = "."
        return result(
            "one_d_sfm",
            0.65,
            "Found a generic image collection; selected the EXIF-optional 1DSfM loader.",
            images_dir=images_dir,
            loader_options={"enable_no_exif": True, "default_focal_length_factor": 1.2},
            alternatives=["olsson"],
        )

    return result(
        "olsson",
        0.0,
        "No recognized image-dataset structure was found. Choose a format manually.",
        alternatives=["colmap", "one_d_sfm", "mobilebrick"],
    )


def resolve_relative_loader_paths(
    dataset_dir: Path, loader_options: dict[str, Any], images_dir: str | None
) -> tuple[dict[str, Any], str | None]:
    """Resolve detector-produced relative paths against the active dataset root."""

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
