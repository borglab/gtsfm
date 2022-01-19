"""Unit tests on GRIC-based estimator for pycolmap.

Authors: John Lambert
"""

import unittest
from pathlib import Path

import matplotlib.pyplot as plt
from gtsam import Unit3

import gtsfm.runner.frontend_runner as frontend_runner
import gtsfm.utils.geometry_comparisons as geometry_comparisons
import tests.frontend.verifier.test_verifier_base as test_verifier_base
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.frontend.detector_descriptor.superpoint import SuperPointDetectorDescriptor
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher
from gtsfm.frontend.verifier.colmap_two_view_geometry_verifier import ColmapTwoViewGeometryVerifier
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.loader.argoverse_dataset_loader import ArgoverseDatasetLoader
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.two_view_estimator import TwoViewEstimator

TEST_DATA_ROOT = Path(__file__).parent.parent.parent.resolve() / "data"


# def test_pose_from_homography_matrix_notre_dame() -> None:
#     """Purely planar scene.
#     Check SuperPoint + SuperGlue + OpenCV RANSAC-5pt frontend (Essential matrix estimation).
#     Essential matrix decomposition on the following image pair yields:
#     {
#         "i1": 11,
#         "i2": 12,
#         "i1_filename": "beggs_2603656317.jpg",
#         "i2_filename": "beggs_2604036425.jpg",
#         "rotation_angular_error": 0.28,
#         "translation_angular_error": 72.44,
#         "num_inliers_gt_model": 1394,
#         "inlier_ratio_gt_model": 1.0,
#         "inlier_ratio_est_model": 0.95,
#         "num_inliers_est_model": 1394
#     }
#     """
#     images_dir = TEST_DATA_ROOT / "notre-dame-20" / "images"
#     colmap_files_dirpath = TEST_DATA_ROOT / "notre-dame-20" / "notre-dame-20-colmap"

#     # images_dir = "/Users/jlambert/Downloads/Statue_of_Liberty/images"
#     # images_dir = "/Users/jlambert/Downloads/gtsfm/tests/data/statue-liberty/images"
#     # colmap_files_dirpath = "/Users/jlambert/Downloads/Statue_of_Liberty/statue-liberty-146-colmap-gt"

#     # images_dir = "/Users/jlambert/Downloads/gtsfm/tests/data/door-images-sample-large-baseline"
#     # # images_dir = "/Users/jlambert/Downloads/gtsfm/tests/data/set1_lund_door/images"
#     # colmap_files_dirpath = "/Users/jlambert/Downloads/gtsfm/tests/data/set1_lund_door/colmap_ground_truth"

#     # GRIC (1.92, 0.71) vs. RASNAC (5.23, 4.0)
#     images_dir = "/Users/jlambert/Downloads/skydio8/images"
#     colmap_files_dirpath = "/Users/jlambert/Downloads/skydio8/colmap-skydio-8"

#     import pdb; pdb.set_trace()
#     loader = ColmapLoader(colmap_files_dirpath=colmap_files_dirpath, images_dir=images_dir, max_frame_lookahead=20)
    
#     # # Notre Dame-20 Dataset
#     # fname1 = "beggs_2603656317.jpg"
#     # fname2 = "beggs_2604036425.jpg"

#     # Door Dataset
#     # fname1 = "DSC_0001.JPG"
#     # fname2 = "DSC_0012.JPG"

#     # Statue of Liberty
#     # fname1 = "IM_167.jpg"
#     # fname2 = "IM_181.jpg"

#     fname1 = "crane_mast_1.jpg"
#     fname2 = "crane_mast_3.jpg"

#     i1 = loader.get_image_index_from_filename(fname1)
#     i2 = loader.get_image_index_from_filename(fname2)

#     # Door Dataset
#     # assert i1 == 0
#     # assert i2 == 11

#     wTi1 = loader.get_camera_pose(i1)
#     wTi2 = loader.get_camera_pose(i2)

#     i2Ti1_gt = wTi2.between(wTi1)

#     img_i1 = loader.get_image(i1)
#     img_i2 = loader.get_image(i2)

#     det_desc = SuperPointDetectorDescriptor()
#     feature_extractor = FeatureExtractor(det_desc)
#     two_view_estimator = TwoViewEstimator(
#         matcher=SuperGlueMatcher(use_outdoor_model=True),
#         verifier=ColmapTwoViewGeometryVerifier(use_intrinsics_in_verification=True, estimation_threshold_px=4),
#         # verifier=Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
#         eval_threshold_px=4,
#         bundle_adjust_2view=False,
#         inlier_support_processor=InlierSupportProcessor(
#             min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1
#         )
#     )
#     keypoints_list, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict = frontend_runner.run_frontend(loader, feature_extractor, two_view_estimator)
    
#     err_rot = geometry_comparisons.compute_relative_rotation_angle(i2Ri1_dict[(i1,i2)], i2Ti1_gt.rotation())
#     err_trans = geometry_comparisons.compute_relative_unit_translation_angle(U_1=i2Ui1_dict[(i1,i2)], U_2=Unit3(i2Ti1_gt.translation()))
#     import pdb; pdb.set_trace()

#     import gtsfm.utils.viz as viz_utils
#     result = viz_utils.plot_twoview_correspondences(
#         image_i1=img_i1,
#         image_i2=img_i2,
#         kps_i1=keypoints_list[i1],
#         kps_i2=keypoints_list[i2],
#         corr_idxs_i1i2=v_corr_idxs_dict[(i1,i2)],
#         inlier_mask=None,
#         max_corrs=300
#     )
#     import pdb; pdb.set_trace()
#     plt.imshow(result.value_array)
#     plt.show()


# """
# Another homography dataset (planar graffiti scene).
# https://www.robots.ox.ac.uk/~vgg/data/affine/
# """



# def test_pose_from_homography_skydio() -> None:
#     """
#     311: S1014913.JPG
#     324: S1014926.JPG
#     11.16 degrees of rotation error, and 156.54 errors of translation error w/o homography consideration
#     seems planar.
#     """
#     # TODO(johnwlambert): add additional test for these planar matches.
#     pass


if __name__ == "__main__":
    #test_pose_from_homography_matrix_notre_dame()
    #test_pose_from_homography_skydio()

    unittest.main()

