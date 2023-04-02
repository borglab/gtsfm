"""Unit tests for the scene-optimizer class.

Authors: Ayush Baid, John Lambert
"""
from pathlib import Path
import pytest

import dask
import hydra
import numpy as np
from dask.distributed import Client, LocalCluster
from gtsam import EssentialMatrix, Rot3, Unit3
from hydra.utils import instantiate

import gtsfm.utils.geometry_comparisons as comp_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.retriever.exhaustive_retriever import ExhaustiveRetriever
from gtsfm.retriever.retriever_base import ImageMatchingRegime
from gtsfm.scene_optimizer import SceneOptimizer

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"
TEST_DATA_WITH_GT = DATA_ROOT_PATH / "set1_lund_door"
TEST_DATA_NO_GT = DATA_ROOT_PATH / "set3_lund_door_nointrinsics_noextrinsics"


@pytest.mark.parametrize("dataset_path", [TEST_DATA_WITH_GT, TEST_DATA_NO_GT])
def test_create_computation_graph_with_ground_truth(dataset_path):
    print("running {}", dataset_path)
    loader = OlssonLoader(str(dataset_path), image_extension="JPG")
    assert len(loader)
    """Will test Dask multi-processing capabilities and ability to serialize all objects."""
    with hydra.initialize_config_module(config_module="gtsfm.configs"):

        # Config is relative to the gtsfm module
        cfg = hydra.compose(config_name="scene_optimizer_unit_test_config.yaml")
        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

        # Create dask client
        cluster = LocalCluster(n_workers=1, threads_per_worker=4)
        client = Client(cluster)

        matching_regime = ImageMatchingRegime.EXHAUSTIVE
        retriever = ExhaustiveRetriever()

        pairs_graph = retriever.create_computation_graph(loader)

        image_pair_indices = pairs_graph.compute()

        (
            delayed_keypoints,
            delayed_putative_corr_idxs_dict,
        ) = scene_optimizer.correspondence_generator.create_computation_graph(
            delayed_images=loader.create_computation_graph_for_images(),
            image_shapes=loader.create_computation_graph_for_image_shapes(),
            image_pair_indices=image_pair_indices,
        )

        keypoints_list, putative_corr_idxs_dict = dask.compute(delayed_keypoints, delayed_putative_corr_idxs_dict)

        delayed_sfm_result, delayed_io = scene_optimizer.create_computation_graph(
            keypoints_list=keypoints_list,
            putative_corr_idxs_dict=putative_corr_idxs_dict,
            num_images=len(loader),
            image_pair_indices=image_pair_indices,
            image_graph=loader.create_computation_graph_for_images(),
            all_intrinsics=loader.create_computation_graph_for_intrinsics(),
            image_shapes=loader.create_computation_graph_for_image_shapes(),
            absolute_pose_priors=loader.get_absolute_pose_priors(),
            relative_pose_priors=loader.get_relative_pose_priors(image_pair_indices),
            cameras_gt=loader.create_computation_graph_for_gt_cameras(),
            gt_wTi_list=loader.get_gt_poses(),
            matching_regime=ImageMatchingRegime(matching_regime),
        )

        sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

        client.close()

        assert isinstance(sfm_result, GtsfmData)

        if dataset_path == TEST_DATA_NO_GT:
            assert len(sfm_result.get_valid_camera_indices()) == len(loader)
        else:
            # compare the camera poses
            computed_poses = sfm_result.get_camera_poses()

            # get active cameras from largest connected component, may be <len(loader)
            connected_camera_idxs = sfm_result.get_valid_camera_indices()
            expected_poses = [loader.get_camera_pose(i) for i in connected_camera_idxs]

            assert comp_utils.compare_global_poses(
                computed_poses, expected_poses, trans_err_atol=1.0, trans_err_rtol=0.1
            )


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))
