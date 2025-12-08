"""Sanity tests for AnySplat extrinsics"""

from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
from gtsam import Point3, Pose3, Rot3

from gtsfm.cluster_optimizer.cluster_anysplat import ClusterAnySplat
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.utils import align, transform
from gtsfm.utils.geometry_comparisons import compute_relative_rotation_angle

GTSFM_LUND_DOOR_ROOT = Path("tests/data/set1_lund_door")
GTSFM_NUM_LUND_IMAGES = 12


def pose3_from_w2c(matrix: torch.Tensor) -> Pose3:
    """Convert a camera→world matrix to a Pose3 (world pose of the camera)."""
    rotation = Rot3(matrix[:3, :3].numpy())
    translation = Point3(*matrix[:3, 3].numpy())
    return Pose3(rotation, translation)


def collect_pose_pairs_from_runtime() -> Optional[tuple[list[Pose3], list[Pose3]]]:
    """Run AnySplat on the Lund Door subset and pair it with COLMAP camera poses."""

    dataset_root = Path(GTSFM_LUND_DOOR_ROOT)
    colmap_dir = dataset_root / "colmap_ground_truth"
    images_dir = dataset_root / "images"
    if not colmap_dir.exists() or not images_dir.exists():
        pytest.skip(f"Lund Door dataset not found under {dataset_root}")

    num_images = GTSFM_NUM_LUND_IMAGES
    loader = ColmapLoader(str(colmap_dir), str(images_dir))

    if len(loader) == 0:
        pytest.skip("No images available in the Lund Door dataset")

    selected_indices = list(range(min(num_images, len(loader))))
    images = {new_idx: loader.get_image_full_res(idx) for new_idx, idx in enumerate(selected_indices)}

    anysplat_cluster = ClusterAnySplat(
        tracking=False,
        reproj_error_thresh=None,
        run_bundle_adjustment_on_leaf=False,
    )

    with torch.no_grad():
        anysplat_result = anysplat_cluster._generate_splats(images)

    extrinsics = anysplat_result.pred_context_pose["extrinsic"][0].cpu().numpy()
    anysplat_pose_list = []
    mvo_pose_list = []
    for pose_matrix, loader_idx in zip(extrinsics, selected_indices):
        pose_gt = loader.get_camera_pose(loader_idx)
        anysplat_pose_list.append(pose3_from_w2c(torch.from_numpy(pose_matrix)))
        mvo_pose_list.append(pose_gt)

    if not anysplat_pose_list or not mvo_pose_list:
        pytest.skip("Failed to retrieve valid pose pairs from runtime execution")
    return anysplat_pose_list, mvo_pose_list


def test_sim3_alignment_recovers_similarity() -> None:
    """Sim(3) alignment shows AnySplat and MVO camera poses differ only by a similarity transform."""
    runtime_pairs = collect_pose_pairs_from_runtime()
    anysplat_poses, mvo_poses = runtime_pairs

    pre_translation_errors = np.array(
        [np.linalg.norm(a.translation() - b.translation()) for a, b in zip(anysplat_poses, mvo_poses)]
    )
    print("Max pre translation error", np.max(pre_translation_errors))
    assert np.max(pre_translation_errors) > 1e-2

    recovered_sim3 = align.sim3_from_Pose3s_robust(anysplat_poses, mvo_poses)
    aligned_mvo = transform.Pose3s_with_sim3(recovered_sim3, mvo_poses)

    trans_tol = 5e-2
    rot_tol_deg = 1

    post_translation_errors = np.array(
        [np.linalg.norm(a.translation() - b.translation()) for a, b in zip(anysplat_poses, aligned_mvo)]
    )
    print("Max post translation error", np.max(post_translation_errors))
    assert np.max(post_translation_errors) < trans_tol

    post_rotation_errors = np.array(
        [compute_relative_rotation_angle(a.rotation(), b.rotation()) for a, b in zip(anysplat_poses, aligned_mvo)]
    )
    print("Max rotation error", np.max(post_rotation_errors))
    assert np.max(post_rotation_errors) < rot_tol_deg
    assert np.max(pre_translation_errors) > 10 * np.max(post_translation_errors)


test_sim3_alignment_recovers_similarity()
