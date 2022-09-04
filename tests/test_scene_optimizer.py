"""Unit tests for the scene-optimizer class.

Authors: Ayush Baid, John Lambert
"""
import unittest
from pathlib import Path

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


class TestSceneOptimizer(unittest.TestCase):
    """Unit test for SceneOptimizer, which runs SfM for a scene."""

    def setUp(self) -> None:
        self.loader = OlssonLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")
        assert len(self.loader)

    def test_create_computation_graph(self):
        """Will test Dask multi-processing capabilities and ability to serialize all objects."""
        self.loader = OlssonLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")

        with hydra.initialize_config_module(config_module="gtsfm.configs"):

            # config is relative to the gtsfm module
            cfg = hydra.compose(config_name="scene_optimizer_unit_test_config.yaml")
            scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

            # create dask client
            cluster = LocalCluster(n_workers=1, threads_per_worker=4)

            matching_regime = ImageMatchingRegime.EXHAUSTIVE
            retriever = ExhaustiveRetriever()

            pairs_graph = retriever.create_computation_graph(self.loader)
            with Client(cluster):
                image_pair_indices = pairs_graph.compute()

            (
                delayed_keypoints,
                delayed_putative_corr_idxs_dict,
            ) = scene_optimizer.correspondence_generator.create_computation_graph(
                delayed_images=self.loader.create_computation_graph_for_images(),
                image_shapes=self.loader.get_image_shapes(),
                image_pair_indices=image_pair_indices,
            )

            with Client(cluster):
                keypoints_list, putative_corr_idxs_dict = dask.compute(
                    delayed_keypoints, delayed_putative_corr_idxs_dict
                )

            # generate the dask computation graph
            delayed_sfm_result, delayed_io = scene_optimizer.create_computation_graph(
                keypoints_list=keypoints_list,
                putative_corr_idxs_dict=putative_corr_idxs_dict,
                num_images=len(self.loader),
                image_pair_indices=image_pair_indices,
                image_graph=self.loader.create_computation_graph_for_images(),
                all_intrinsics=self.loader.get_all_intrinsics(),
                image_shapes=self.loader.get_image_shapes(),
                absolute_pose_priors=self.loader.get_absolute_pose_priors(),
                relative_pose_priors=self.loader.get_relative_pose_priors(image_pair_indices),
                cameras_gt=self.loader.get_gt_cameras(),
                gt_wTi_list=self.loader.get_gt_poses(),
                matching_regime=ImageMatchingRegime(matching_regime),
            )

            with Client(cluster):
                sfm_result, *io = dask.compute(delayed_sfm_result, *delayed_io)

            self.assertIsInstance(sfm_result, GtsfmData)

            # compare the camera poses
            computed_poses = sfm_result.get_camera_poses()

            # get active cameras from largest connected component, may be <len(self.loader)
            connected_camera_idxs = sfm_result.get_valid_camera_indices()
            expected_poses = [self.loader.get_camera_pose(i) for i in connected_camera_idxs]

            self.assertTrue(
                comp_utils.compare_global_poses(computed_poses, expected_poses, trans_err_atol=1.0, trans_err_rtol=0.1)
            )


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
