"""Tests for browser dataset-format detection."""

from pathlib import Path

from visualization.datasets import detect_dataset_format


def _touch(root: Path, *relative_paths: str) -> None:
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")


def test_detects_mobilebrick_layout(tmp_path: Path) -> None:
    _touch(tmp_path, "image/000.jpg", "intrinsic/000.txt", "pose/000.txt")

    detected = detect_dataset_format(tmp_path)

    assert detected["loader"] == "mobilebrick"
    assert detected["images_dir"] == "image"
    assert detected["loader_options"]["use_gt_intrinsics"] is True


def test_detects_colmap_text_layout(tmp_path: Path) -> None:
    _touch(tmp_path, "cameras.txt", "images.txt", "points3D.txt", "images/one.jpg")

    detected = detect_dataset_format(tmp_path)

    assert detected["loader"] == "colmap"
    assert detected["confidence"] > 0.9


def test_olsson_signature_wins_over_nested_colmap_ground_truth(tmp_path: Path) -> None:
    _touch(
        tmp_path,
        "data.mat",
        "images/one.jpg",
        "colmap_ground_truth/cameras.txt",
        "colmap_ground_truth/images.txt",
        "colmap_ground_truth/points3D.txt",
    )

    detected = detect_dataset_format(tmp_path)

    assert detected["loader"] == "olsson"


def test_detects_tanks_and_temples_with_portable_paths(tmp_path: Path) -> None:
    _touch(tmp_path, "Barn_COLMAP_SfM.log", "Barn.json", "Barn_trans.txt", "Barn/000001.jpg")

    detected = detect_dataset_format(tmp_path)

    assert detected["loader"] == "tanks_and_temples"
    assert detected["images_dir"] == "Barn"
    assert detected["loader_options"]["poses_fpath"] == "Barn_COLMAP_SfM.log"


def test_generic_images_use_exif_optional_loader(tmp_path: Path) -> None:
    _touch(tmp_path, "photos/one.jpg", "photos/two.png")

    detected = detect_dataset_format(tmp_path)

    assert detected["loader"] == "one_d_sfm"
    assert detected["images_dir"] == "photos"
    assert detected["confidence"] < 0.9


def test_detects_argoverse_dataset_subdirectory(tmp_path: Path) -> None:
    log_id = "273c1883-673a-36bf-b124-88311b1a80be"
    _touch(
        tmp_path,
        f"train1/{log_id}/vehicle_calibration_info.json",
        f"train1/{log_id}/poses/city_SE3_egovehicle_1.json",
        f"train1/{log_id}/ring_front_center/ring_front_center_1.jpg",
    )

    detected = detect_dataset_format(tmp_path)

    assert detected["loader"] == "argoverse"
    assert detected["dataset_subdir"] == "train1"
    assert detected["loader_options"]["log_id"] == log_id
