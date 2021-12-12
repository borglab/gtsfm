"""Utility for point transfer using Fundamental matrices.

Author: John Lambert
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import EssentialMatrix

import gtsfm.runner.frontend_runner as frontend_runner
import gtsfm.utils.features as feature_utils
import gtsfm.utils.verification as verification_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.frontend.detector_descriptor.superpoint import SuperPointDetectorDescriptor
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.two_view_estimator import TwoViewEstimator


def fmat_point_transfer(
    i3Fi1: np.ndarray,
    i3Fi2: np.ndarray,
    matched_keypoints_i1: np.ndarray,
    matched_keypoints_i2: np.ndarray,
    matched_keypoints_i3: np.ndarray,
) -> np.ndarray:
    """Transfer points to a 3rd image, using intersection of epipolar lines (via Fundamental matrices).

    See Hartley Zisserman, p. 398
    http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf

    or Sweeney15iccv: https://sites.cs.ucsb.edu/~holl/pubs/Sweeney-2015-ICCV.pdf, section 3.1

    Args:
        match_keypoints_i1: (N,2) array representing measurements in view 1 for each of N tracks (in 3 views).
        match_keypoints_i2: (N,2) array representing measurements in view 2 for each of N tracks (in 3 views).
        match_keypoints_i3: (N,2) array representing measurements in view 3 for each of N tracks (in 3 views).

    Returns:
        errors, as distances in the image plane.
    """
    matched_h_keypoints_i1 = feature_utils.convert_to_homogenous_coordinates(matched_keypoints_i1)
    matched_h_keypoints_i2 = feature_utils.convert_to_homogenous_coordinates(matched_keypoints_i2)
    matched_h_keypoints_i3 = feature_utils.convert_to_homogenous_coordinates(matched_keypoints_i3)

    J = len(matched_keypoints_i1)
    dists = np.zeros(J)

    for j in range(J):
        p1 = matched_h_keypoints_i1[j]
        p2 = matched_h_keypoints_i2[j]
        p3 = matched_h_keypoints_i3[j]

        l = i3Fi1 @ p1
        l_ = i3Fi2 @ p2

        p3_expected = np.cross(l, l_)
        p3_expected = p3_expected[:2] / p3_expected[2]
        error_2d = np.linalg.norm(p3[:2] - p3_expected)
        print(f"Error: {error_2d:.2f}")
        dists[j] = error_2d

    return dists


def test_fmat_point_transfer() -> None:
    """ """
    loader = OlssonLoader("/Users/johnlambert/Downloads/door-trifocal-example", image_extension="JPG")

    det_desc = SuperPointDetectorDescriptor()

    from gtsfm.frontend.cacher.matcher_cacher import MatcherCacher
    from gtsfm.frontend.cacher.detector_descriptor_cacher import DetectorDescriptorCacher
    feature_extractor =FeatureExtractor(
        detector_descriptor=DetectorDescriptorCacher(detector_descriptor_obj=det_desc)
    )
    #feature_extractor = FeatureExtractor(det_desc)
    two_view_estimator = TwoViewEstimator(
        #matcher=SuperGlueMatcher(use_outdoor_model=True),
        matcher=MatcherCacher(matcher_obj=SuperGlueMatcher(use_outdoor_model=True)),
        verifier=Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
        inlier_support_processor=InlierSupportProcessor(min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1),
        bundle_adjust_2view=False,
        eval_threshold_px=4,
        bundle_adjust_2view_maxiters=0,
    )

    keypoints_list, i2Ri1_dict, i2Ui1_dict, corr_idxs_dict = frontend_runner.run_frontend(
        loader, feature_extractor, two_view_estimator
    )
    tracks_2d = SfmTrack2d.generate_tracks_from_pairwise_matches(corr_idxs_dict, keypoints_list)

    matched_keypoints_i1 = []
    matched_keypoints_i2 = []
    matched_keypoints_i3 = []

    for track in tracks_2d:
        if len(track.measurements) == 1:
            import pdb; pdb.set_trace()

        if len(track.measurements) != 3:
            continue

        # should be sorted?
        assert track.measurements[0].i == 0
        assert track.measurements[1].i == 1
        assert track.measurements[2].i == 2

        matched_keypoints_i1.append(track.measurements[0].uv)
        matched_keypoints_i2.append(track.measurements[1].uv)
        matched_keypoints_i3.append(track.measurements[2].uv)

    i1, i2, i3 = 0, 1, 2
    img_i1 = loader.get_image(i1)
    img_i2 = loader.get_image(i2)
    img_i3 = loader.get_image(i3)

    i3Ei1 = EssentialMatrix(i2Ri1_dict[(i1,i3)], i2Ui1_dict[(i1,i3)])
    i3Ei2 = EssentialMatrix(i2Ri1_dict[(i2,i3)], i2Ui1_dict[(i2,i3)])

    camera_intrinsics_i1 = loader.get_camera_intrinsics(i1)
    camera_intrinsics_i2 = loader.get_camera_intrinsics(i2)
    camera_intrinsics_i3 = loader.get_camera_intrinsics(i3)

    i3Fi1 = verification_utils.essential_to_fundamental_matrix(i3Ei1, camera_intrinsics_i1, camera_intrinsics_i3)
    i3Fi2 = verification_utils.essential_to_fundamental_matrix(i3Ei2, camera_intrinsics_i2, camera_intrinsics_i3)

    matched_keypoints_i1 = np.array(matched_keypoints_i1)
    matched_keypoints_i2 = np.array(matched_keypoints_i2)
    matched_keypoints_i3 = np.array(matched_keypoints_i3)

    #import pdb; pdb.set_trace()

    dists = fmat_point_transfer(i3Fi1, i3Fi2, matched_keypoints_i1, matched_keypoints_i2, matched_keypoints_i3)

    mask = dists > 5000

    draw_epipolar_lines(
        F=i3Fi1,
        img_left=img_i1.value_array,
        img_right=img_i3.value_array,
        pts_left=matched_keypoints_i1[mask],
        pts_right=matched_keypoints_i3[mask]
    )

    # draw_epipolar_lines(
    #     F=i3Fi2,
    #     img_left=img_i2.value_array,
    #     img_right=img_i3.value_array,
    #     pts_left=matched_keypoints_i2[-2:],
    #     pts_right=matched_keypoints_i3[-2:]
    # )

    plt.show()


def convert_to_homogenous_coordinates(coords: np.ndarray) -> np.ndarray:
    """Convert coordinates to homogenous system (by appending a column of ones)."""
    N = coords.shape[0]
    return np.hstack((coords, np.ones((N, 1))))


def draw_epipolar_lines(
    F: np.ndarray,
    img_left: np.ndarray,
    img_right: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    figsize=(10, 8)
) -> None:
    """Draw the epipolar lines given the fundamental matrix, left & right images and left & right datapoints.

    Args:
        F: a 3 x 3 numpy array representing the fundamental matrix, such that
            p_right^T @ F @ p_left = 0 for correct correspondences
        img_left: array representing image 1.
        img_right: array representing image 2.
        pts_left: array of shape (N,2) representing image 1 datapoints.
        pts_right: array of shape (N,2) representing image 2 datapoints.
    """
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=figsize)

    # defn of epipolar line in the right image, corresponding to left point p
    # l_e = F @ p_left
    draw_lines_in_single_image(ax=ax[1], i2Fi1=F, pts_i1=pts_left, pts_i2=pts_right, img_i2=img_right)
    
    # defn of epipolar line in the left image, corresponding to point p in the right image
    # l_e = F.T @ p_right
    draw_lines_in_single_image(ax=ax[0], i2Fi1=F.T, pts_i1=pts_right, pts_i2=pts_left, img_i2=img_left)



def draw_lines_in_single_image(ax, i2Fi1: np.ndarray, pts_i1: np.ndarray, pts_i2: np.ndarray, img_i2: np.ndarray) -> None:
    """Draw epipolar lines in image 2, where each epipolar line corresponds to a point from image 1.

    Args:
        ax: matplotlib drawing axis.
        i2Fi1: Fundamental matrix, mapping points in image 1, to lines in image2.
        pts_i1: array of shape (N,2) representing image 1 datapoints.
        pts_i2:  array of shape (N,2) representing image 2 datapoints.
        img_i2: array of shape (H,W,3) representing image 2.
    """
    img_h, img_w = img_i2.shape[:2]

    # corner points, as homogeneous (x,y,1)
    p_ul = np.asarray([0, 0, 1])
    p_ur = np.asarray([img_w, 0, 1])
    p_bl = np.asarray([0, img_h, 1])
    p_br = np.asarray([img_w, img_h, 1])

    # The equation of the line through two points
    # can be determined by taking the ‘cross product’
    # of their homogeneous coordinates.

    # left and right border lines (applicable for both images)
    l_l = np.cross(p_ul, p_bl)
    l_r = np.cross(p_ur, p_br)

    # convert to homogeneous
    pts_i1 = convert_to_homogenous_coordinates(pts_i1)

    ax.imshow(img_i2)
    ax.autoscale(False)
    ax.scatter(pts_i2[:, 0], pts_i2[:, 1], marker='o', s=20, c='yellow', edgecolors='red')
    
    for p_i1 in pts_i1:
        # get defn of epipolar line in  image, corresponding to left point p
        l_e = np.dot(i2Fi1, p_i1).squeeze()
        # find where epipolar line in right image crosses the left and image borders
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        # convert back from homogeneous to cartesian by dividing by 3rd entry
        # draw line between point on left border, and on the right border
        x = [p_l[0]/p_l[2], p_r[0]/p_r[2]]
        y = [p_l[1]/p_l[2], p_r[1]/p_r[2]]
        ax.plot(x, y, linewidth=1, c='blue')


if __name__ == "__main__":
    test_fmat_point_transfer()
