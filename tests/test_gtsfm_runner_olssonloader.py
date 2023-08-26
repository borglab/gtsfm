"""Unit tests for the scene-optimizer class.

Authors: Ayush Baid, John Lambert
"""
from pathlib import Path

import numpy as np
import pytest
from gtsam import EssentialMatrix, Rot3, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.runner.run_scene_optimizer_olssonloader import GtsfmRunnerOlssonLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"
TEST_DATA_WITH_GT = DATA_ROOT_PATH / "set1_lund_door"
TEST_DATA_NO_GT = DATA_ROOT_PATH / "set3_lund_door_nointrinsics_noextrinsics"


@pytest.mark.parametrize("dataset_path", [TEST_DATA_WITH_GT, TEST_DATA_NO_GT])
def test_gtsfm_runner_olssonloader(dataset_path):
    print("Running {}", dataset_path)
    runner = GtsfmRunnerOlssonLoader(override_args=["--dataset_root", str(dataset_path)])
    runner.parsed_args.dataset_root = str(dataset_path)
    sfm_result = runner.run()

    assert isinstance(sfm_result, GtsfmData)

    if dataset_path == TEST_DATA_NO_GT:
        assert len(sfm_result.get_valid_camera_indices()) == len(runner.loader)
    else:
        # compare the camera poses
        computed_poses = sfm_result.get_camera_poses()

        # get active cameras from largest connected component, may be <len(loader)
        connected_camera_idxs = sfm_result.get_valid_camera_indices()
        expected_poses = [runner.loader.get_camera_pose(i) for i in connected_camera_idxs]

        assert comp_utils.compare_global_poses(computed_poses, expected_poses, trans_err_atol=1.0, trans_err_rtol=0.1)


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))
