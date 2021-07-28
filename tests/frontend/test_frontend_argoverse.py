"""Unit tests for the the GTSFM frontend.

Authors: John Lambert
"""
import unittest
from pathlib import Path
from typing import Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Pose3
from scipy.spatial.transform import Rotation

from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
from gtsfm.frontend.verifier.degensac import Degensac
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.loader.argoverse_dataset_loader import ArgoverseDatasetLoader
from gtsfm.scene_optimizer import FeatureExtractor, TwoViewEstimator

TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"


class TestFrontend(unittest.TestCase):
    """Tests a combined FeatureExtractor and TwoViewEstimator using an Argoverse image pair."""

    def setUp(self) -> None:
        """ """
        self.loader = ArgoverseDatasetLoader(
            dataset_dir=TEST_DATA_ROOT_PATH / "argoverse" / "train1",
            log_id="273c1883-673a-36bf-b124-88311b1a80be",
            stride=1,
            max_num_imgs=2,
            max_lookahead_sec=50,
            camera_name="ring_front_center",
        )
        assert len(self.loader)

    def __get_frontend_computation_graph(
        self, feature_extractor: FeatureExtractor, two_view_estimator: TwoViewEstimator,
    ) -> Tuple[Delayed, Delayed]:
        """Copied from SceneOptimizer class, without back-end code"""
        image_pair_indices = self.loader.get_valid_pairs()
        image_graph = self.loader.create_computation_graph_for_images()
        camera_intrinsics_graph = self.loader.create_computation_graph_for_intrinsics()
        image_shape_graph = self.loader.create_computation_graph_for_image_shapes()

        # detection and description graph
        keypoints_graph_list = []
        descriptors_graph_list = []
        for delayed_image in image_graph:
            (delayed_dets, delayed_descs,) = feature_extractor.create_computation_graph(delayed_image)
            keypoints_graph_list += [delayed_dets]
            descriptors_graph_list += [delayed_descs]

        # estimate two-view geometry and get indices of verified correspondences.
        i2Ri1_graph_dict = {}
        i2Ui1_graph_dict = {}
        for (i1, i2) in image_pair_indices:
            (i2Ri1, i2Ui1, v_corr_idxs, two_view_report) = two_view_estimator.create_computation_graph(
                keypoints_graph_list[i1],
                keypoints_graph_list[i2],
                descriptors_graph_list[i1],
                descriptors_graph_list[i2],
                camera_intrinsics_graph[i1],
                camera_intrinsics_graph[i2],
                image_shape_graph[i1],
                image_shape_graph[i2],
            )
            i2Ri1_graph_dict[(i1, i2)] = i2Ri1
            i2Ui1_graph_dict[(i1, i2)] = i2Ui1

        return i2Ri1_graph_dict, i2Ui1_graph_dict

    def test_sift_twoway_ransac(self):
        """Check DoG + SIFT + 2-way Matcher + RANSAC-5pt frontend."""
        det_desc = SIFTDetectorDescriptor()
        feature_extractor = FeatureExtractor(det_desc)
        two_view_estimator = TwoViewEstimator(
            matcher=TwoWayMatcher(),
            verifier=Ransac(
                use_intrinsics_in_verification=True,
                estimation_threshold_px=4,
                min_allowed_inlier_ratio_est_model=0.1
            ),
            eval_threshold_px=4,
            estimation_threshold_px=0.5,
            min_num_inliers_acceptance=15
        )
        self.__compare_frontend_result_error(
            feature_extractor, two_view_estimator, euler_angle_err_tol=1.4, translation_err_tol=0.026,
        )

    # def test_sift_twoway_degensac(self):
    #     """Check DoG + SIFT + 2-way Matcher + DEGENSAC-8pt frontend."""
    #     det_desc = SIFTDetectorDescriptor()
    #     feature_extractor = FeatureExtractor(det_desc)
    #     two_view_estimator = TwoViewEstimator(
    #         matcher=TwoWayMatcher(),
    #         verifier=Degensac(
    #             use_intrinsics_in_verification=False,
    #             estimation_threshold_px=0.5,
    #             min_allowed_inlier_ratio_est_model=0.05
    #         ),
    #         eval_threshold_px=4,
    #         estimation_threshold_px=0.5,
    #         min_num_inliers_acceptance=15
    #     )
    #     self.__compare_frontend_result_error(
    #         feature_extractor, two_view_estimator, euler_angle_err_tol=0.95, translation_err_tol=0.03,
    #     )

    def __compare_frontend_result_error(
        self,
        feature_extractor: FeatureExtractor,
        two_view_estimator: TwoViewEstimator,
        euler_angle_err_tol: float,
        translation_err_tol: float,
    ) -> None:
        """Compare recovered relative rotation and translation with ground truth."""
        (i2Ri1_graph_dict, i2Ui1_graph_dict,) = self.__get_frontend_computation_graph(
            feature_extractor, two_view_estimator
        )

        with dask.config.set(scheduler="single-threaded"):
            i2Ri1_results, i2ti1_results = dask.compute(i2Ri1_graph_dict, i2Ui1_graph_dict)

        i1, i2 = 0, 1
        i2Ri1 = i2Ri1_results[(i1, i2)]
        i2Ui1 = i2ti1_results[(i1, i2)]

        # Ground truth is provided in inverse format, so invert SE(3) object
        i2Ti1 = Pose3(i2Ri1, i2Ui1.point3())
        i1Ti2 = i2Ti1.inverse()
        i1ti2 = i1Ti2.translation()
        i1Ri2 = i1Ti2.rotation().matrix()

        euler_angles = Rotation.from_matrix(i1Ri2).as_euler("zyx", degrees=True)
        gt_euler_angles = np.array([-0.37, 32.47, -0.42])
        np.testing.assert_allclose(gt_euler_angles, euler_angles, atol=euler_angle_err_tol)

        gt_i1ti2 = np.array([0.21, -0.0024, 0.976])
        np.testing.assert_allclose(gt_i1ti2, i1ti2, atol=translation_err_tol)


if __name__ == "__main__":
    unittest.main()
