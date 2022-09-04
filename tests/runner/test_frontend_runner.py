"""Unit tests for the the GTSFM frontend runner.

Authors: John Lambert
"""

import unittest
from pathlib import Path
from typing import Union

import numpy as np
from gtsam import Pose3
from scipy.spatial.transform import Rotation

import gtsfm.runner.frontend_runner as frontend_runner
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.frontend.correspondence_generator.det_desc_correspondence_generator import DetDescCorrespondenceGenerator
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.frontend.detector_descriptor.superpoint import SuperPointDetectorDescriptor
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher
from gtsfm.frontend.verifier.loransac import LoRansac
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.loader.argoverse_dataset_loader import ArgoverseDatasetLoader
from gtsfm.two_view_estimator import TwoViewEstimator

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

    # TODO(johnwlambert): make SIFT-based unit test below non-flaky (should be deterministic).
    # def test_sift_twoway_ransac(self) -> None:
    #     """Check DoG + SIFT + 2-way Matcher + RANSAC-5pt frontend."""
    #     det_desc = SIFTDetectorDescriptor()
    #     correspondence_generator = DetDescCorrespondenceGenerator(
    #         feature_extractor=FeatureExtractor(det_desc),
    #         matcher=TwoWayMatcher(),
    #     )

    #     two_view_estimator = TwoViewEstimator(
    #         verifier=Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
    #         eval_threshold_px=4,
    #         bundle_adjust_2view=False,
    #         inlier_support_processor=InlierSupportProcessor(
    #             min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1
    #         ),
    #     )
    #     self.__compare_frontend_result_error(
    #         correspondence_generator=correspondence_generator,
    #         two_view_estimator=two_view_estimator,
    #         euler_angle_err_tol=1.4,
    #         translation_err_tol=0.026,
    #     )

    def test_superpoint_superglue_twoway_ransac(self):
        """Check SuperPoint + SuperGlue + OpenCV RANSAC-5pt frontend (Essential matrix estimation)."""
        det_desc = SuperPointDetectorDescriptor()

        correspondence_generator = DetDescCorrespondenceGenerator(
            feature_extractor=FeatureExtractor(det_desc), matcher=SuperGlueMatcher(use_outdoor_model=True)
        )
        two_view_estimator = TwoViewEstimator(
            verifier=Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
            eval_threshold_px=4,
            bundle_adjust_2view=False,
            inlier_support_processor=InlierSupportProcessor(
                min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1
            ),
        )
        self.__compare_frontend_result_error(
            correspondence_generator=correspondence_generator,
            two_view_estimator=two_view_estimator,
            euler_angle_err_tol=1.4,
            translation_err_tol=0.026,
        )

    def test_superpoint_superglue_twoway_loransac_essential(self) -> None:
        """Check SuperPoint + SuperGlue + LORANSAC-5pt frontend (Essential matrix estimation)."""
        det_desc = SuperPointDetectorDescriptor()
        correspondence_generator = DetDescCorrespondenceGenerator(
            feature_extractor=FeatureExtractor(det_desc), matcher=SuperGlueMatcher(use_outdoor_model=True)
        )
        two_view_estimator = TwoViewEstimator(
            verifier=LoRansac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
            eval_threshold_px=4,
            bundle_adjust_2view=False,
            inlier_support_processor=InlierSupportProcessor(
                min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1
            ),
        )
        self.__compare_frontend_result_error(
            correspondence_generator=correspondence_generator,
            two_view_estimator=two_view_estimator,
            euler_angle_err_tol=1.4,
            translation_err_tol=0.026,
        )

    def test_superpoint_superglue_twoway_loransac_fundamental(self) -> None:
        """Check SuperPoint + SuperGlue + LORANSAC-8pt frontend (Fundamental matrix estimation)."""
        det_desc = SuperPointDetectorDescriptor()
        correspondence_generator = DetDescCorrespondenceGenerator(
            feature_extractor=FeatureExtractor(det_desc), matcher=SuperGlueMatcher(use_outdoor_model=True)
        )

        two_view_estimator = TwoViewEstimator(
            verifier=LoRansac(use_intrinsics_in_verification=False, estimation_threshold_px=4),
            eval_threshold_px=4,
            bundle_adjust_2view=False,
            inlier_support_processor=InlierSupportProcessor(
                min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1
            ),
        )
        self.__compare_frontend_result_error(
            correspondence_generator=correspondence_generator,
            two_view_estimator=two_view_estimator,
            euler_angle_err_tol=1.4,
            translation_err_tol=0.026,
        )

    # TODO(johnwlambert): make SIFT-based unit test below non-flaky (should be deterministic).
    # def test_sift_twoway_degensac(self) -> None:
    #     """Check DoG + SIFT + 2-way Matcher + DEGENSAC-8pt frontend."""
    #     det_desc = SIFTDetectorDescriptor()
    #     correspondence_generator = DetDescCorrespondenceGenerator(
    #         feature_extractor=FeatureExtractor(det_desc),
    #         matcher=TwoWayMatcher(),
    #     )
    #     two_view_estimator = TwoViewEstimator(
    #         verifier=Degensac(
    #             use_intrinsics_in_verification=False,
    #             estimation_threshold_px=0.5,
    #         ),
    #         eval_threshold_px=4,
    #         bundle_adjust_2view=False,
    #         inlier_support_processor=InlierSupportProcessor(
    #             min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.05
    #         ),
    #     )
    #     self.__compare_frontend_result_error(
    #         correspondence_generator=correspondence_generator,
    #         two_view_estimator=two_view_estimator,
    #         euler_angle_err_tol=0.95,
    #         translation_err_tol=0.03,
    #     )

    def __compare_frontend_result_error(
        self,
        correspondence_generator: Union[DetDescCorrespondenceGenerator, ImageCorrespondenceGenerator],
        two_view_estimator: TwoViewEstimator,
        euler_angle_err_tol: float,
        translation_err_tol: float,
    ) -> None:
        """Compare recovered relative rotation and translation with ground truth."""
        _, i2Ri1_dict, i2Ui1_dict, _ = frontend_runner.run_frontend(
            loader=self.loader, correspondence_generator=correspondence_generator, two_view_estimator=two_view_estimator
        )

        i1, i2 = 0, 1
        i2Ri1 = i2Ri1_dict[(i1, i2)]
        i2Ui1 = i2Ui1_dict[(i1, i2)]

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
