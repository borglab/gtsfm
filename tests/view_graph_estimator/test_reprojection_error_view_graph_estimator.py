"""Unit tests on 3-view BA + reprojection error view graph estimator.

Authors: John Lambert
"""
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from gtsam import Cal3Bundler

import gtsfm.runner.frontend_runner as frontend_runner
from gtsfm.common.keypoints import Keypoints
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.frontend.detector_descriptor.superpoint import SuperPointDetectorDescriptor
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.two_view_estimator import TwoViewEstimator

from gtsfm.frontend.cacher.matcher_cacher import MatcherCacher
from gtsfm.frontend.cacher.detector_descriptor_cacher import DetectorDescriptorCacher

from gtsfm.view_graph_estimator.reprojection_error_view_graph_estimator import ReprojectionErrorViewGraphEstimator

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestReprojectionErrorViewGraphEstimator(unittest.TestCase):
    def setUp(self) -> None:
        """ """
        det_desc = SuperPointDetectorDescriptor()

        self.feature_extractor = FeatureExtractor(
            detector_descriptor=DetectorDescriptorCacher(detector_descriptor_obj=det_desc)
        )
        # feature_extractor = FeatureExtractor(det_desc)
        self.two_view_estimator = TwoViewEstimator(
            # matcher=SuperGlueMatcher(use_outdoor_model=True),
            matcher=MatcherCacher(matcher_obj=SuperGlueMatcher(use_outdoor_model=True)),
            verifier=Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
            inlier_support_processor=InlierSupportProcessor(
                min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1
            ),
            bundle_adjust_2view=False,
            eval_threshold_px=4,
            bundle_adjust_2view_maxiters=0,
        )

    def test_optimize_three_views_averaging_door(self) -> None:
        """ """
        dataset_root = TEST_DATA_ROOT / "set1_lund_door"
        image_extension = "JPG"

        loader = OlssonLoader(dataset_root, image_extension=image_extension)
        camera_intrinsics = [loader.get_camera_intrinsics(i) for i in range(11)]

        keypoints_list, i2Ri1_dict, i2Ui1_dict, corr_idxs_dict = frontend_runner.run_frontend(
            loader, self.feature_extractor, self.two_view_estimator
        )

        vge = ReprojectionErrorViewGraphEstimator()
        wTi_list_gt = [loader.get_camera_pose(i) for i in range(11)]

        # only using first 11 of 12 images, since we need to have N gt cameras for N estimated cameras
        # and we use num_images=(max_index+1) to determine this.
        cameras_gt = [loader.get_camera(i) for i in range(11)]

        # choose an arbitrary known triplet of cameras
        i0, i1, i2 = 0, 5, 10

        two_view_reports = {k: None for k in i2Ri1_dict.keys()}
        i2Ri1_dict_subscene, i2Ui1_dict_subscene, corr_idxs_i1_i2_subscene, _ = vge._filter_with_edges(
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            corr_idxs_i1i2=corr_idxs_dict,
            two_view_reports=two_view_reports,
            edges_to_select=[(i0, i1), (i1, i2), (i0, i2)],
        )

        wTi_list, reproj_errors, ra_metrics, ta_metrics, ba_metrics = vge.optimize_three_views_averaging(
            i2Ri1_dict=i2Ri1_dict_subscene,
            i2Ui1_dict=i2Ui1_dict_subscene,
            calibrations=camera_intrinsics,
            corr_idxs_i1_i2=corr_idxs_i1_i2_subscene,
            keypoints_list=keypoints_list,
            cameras_gt=cameras_gt,
        )

        # import matplotlib.pyplot as plt
        # plt.hist(reproj_errors, bins=100)
        # plt.show()

        assert ra_metrics._metrics[1].name == "rotation_error_angle_deg"
        assert np.less(ra_metrics._metrics[1].data, 0.5).all()

        assert ba_metrics._metrics[4].name == "rotation_error_angle_deg"
        assert np.less(ba_metrics._metrics[4].data, 0.1).all()

        assert ta_metrics._metrics[9].name == "translation_error_distance"
        assert np.less(ta_metrics._metrics[9].data, 0.1).all()

        assert ba_metrics._metrics[5].name == "translation_error_distance"
        assert np.less(ba_metrics._metrics[5].data, 0.01).all()

        # import gtsfm.utils.geometry_comparisons as geometry_comparisons
        # wTi_list_aligned = geometry_comparisons.align_poses_sim3_ignore_missing(aTi_list=wTi_list_gt, bTi_list=wTi_list)

def test_view_graph_estimator_run_door() -> None:
    """ """

    det_desc = SuperPointDetectorDescriptor()

    feature_extractor = FeatureExtractor(
        detector_descriptor=DetectorDescriptorCacher(detector_descriptor_obj=det_desc)
    )
    # feature_extractor = FeatureExtractor(det_desc)
    two_view_estimator = TwoViewEstimator(
        # matcher=SuperGlueMatcher(use_outdoor_model=True),
        matcher=MatcherCacher(matcher_obj=SuperGlueMatcher(use_outdoor_model=True)),
        verifier=Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
        inlier_support_processor=InlierSupportProcessor(
            min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1
        ),
        bundle_adjust_2view=False,
        eval_threshold_px=4,
        bundle_adjust_2view_maxiters=0,
    )

    dataset_root = TEST_DATA_ROOT / "set1_lund_door"
    image_extension = "JPG"
    loader = OlssonLoader(dataset_root, image_extension=image_extension)
    num_images = 12
    
    camera_intrinsics = [loader.get_camera_intrinsics(i) for i in range(num_images)]

    keypoints_list, i2Ri1_dict, i2Ui1_dict, corr_idxs_dict = frontend_runner.run_frontend(
        loader, feature_extractor, two_view_estimator
    )

    two_view_reports = {k: None for k in i2Ri1_dict.keys()}

    vge = ReprojectionErrorViewGraphEstimator()
    result = vge.run(
        i2Ri1_dict=i2Ri1_dict,
        i2Ui1_dict=i2Ui1_dict,
        calibrations=camera_intrinsics,
        corr_idxs_i1i2=corr_idxs_dict,
        two_view_reports=two_view_reports,
        keypoints=keypoints_list,
    )
    # import pdb; pdb.set_trace()


if __name__ == "__main__":
    #unittest.main()

    test_view_graph_estimator_run_door()
