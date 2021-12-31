"""Utility for point transfer using Fundamental matrices.

Author: John Lambert
"""

import itertools
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Rot3, Unit3

import gtsfm.averaging.rotation.cycle_consistency as cycle_consistency
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
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.two_view_estimator import TwoViewEstimator
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport

from gtsfm.frontend.cacher.matcher_cacher import MatcherCacher
from gtsfm.frontend.cacher.detector_descriptor_cacher import DetectorDescriptorCacher
import gtsfm.averaging.rotation.cycle_consistency as cycle_consistency


def fmat_point_transfer(
    i3Fi1: np.ndarray,
    i3Fi2: np.ndarray,
    correspondences: np.ndarray,
) -> np.ndarray:
    """Transfer points to a 3rd image, using intersection of epipolar lines (via Fundamental matrices), and measure error.

    See Hartley Zisserman, p. 398
    http://www.r-5.org/files/books/computers/algo-list/image-processing/vision/Richard_Hartley_Andrew_Zisserman-Multiple_View_Geometry_in_Computer_Vision-EN.pdf

    or Sweeney15iccv: https://sites.cs.ucsb.edu/~holl/pubs/Sweeney-2015-ICCV.pdf, section 3.1

    Args:
        correspondences: (N,6) array representing measurements in view 1, view 2, view 3 for each of N tracks (in 3 views).

    Returns:
        errors, as distances in the image plane.
    """
    matched_keypoints_i1 = correspondences[:, :2]
    matched_keypoints_i2 = correspondences[:, 2:4]
    matched_keypoints_i3 = correspondences[:, 4:]

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

        # if effectively collinear, ignore this correspondence.
        if np.isclose(p3_expected[2], 0, atol=1e-8):
            dists[j] = 0.0
            continue

        p3_expected = p3_expected[:2] / p3_expected[2]
        error_2d = np.linalg.norm(p3[:2] - p3_expected)
        # print(f"Error: {error_2d:.2f}")
        dists[j] = error_2d

        # if error_2d > 100:
        #     import pdb; pdb.set_trace()

    return dists


MIN_N_REQUIRED_INLIERS = 50


def filter_to_cycle_consistent_edges(
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints_list: List[Keypoints],
    camera_intrinsics_dict: Dict[int, Cal3Bundler],
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport] = None,
    loader: Optional[LoaderBase] = None,
    visualize: bool = True,
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], Dict[Tuple[int, int], np.ndarray]]:
    """Remove noisy edges in a graph ...

    Args:
        i2Ri1_dict: mapping from image pair indices (i1,i2) to relative rotation i2Ri1.
        i2Ui1_dict: mapping from image pair indices (i1,i2) to relative translation direction i2Ui1.
            Should have same keys as i2Ri1_dict.
        corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value as matching keypoint indices.
            (for verified correspondences)
        keypoints_list:
        loader:
        visualize:

    Returns:
        i2Ri1_dict_consistent: subset of i2Ri1_dict, i.e. only including edges that belonged to some triplet
            and had cycle error below the predefined threshold.
        i2Ui1_dict_consistent: subset of i2Ui1_dict, as above.
        v_corr_idxs_dict_consistent: subset of v_corr_idxs_dict above.
    """
    cycle_consistent_edges = []

    tracks_2d = SfmTrack2d.generate_tracks_from_pairwise_matches(corr_idxs_dict, keypoints_list)

    matched_keypoints_i1 = []
    matched_keypoints_i2 = []
    matched_keypoints_i3 = []

    print("Num tracks: ", len(tracks_2d))

    # each triplet should has its own array of matched keypoints.
    triplet_tracks = defaultdict(list)

    for track in tracks_2d:

        # print("Track length: ", len(track.measurements))

        if len(track.measurements) == 1:
            import pdb

            pdb.set_trace()

        if len(track.measurements) < 3:
            continue

        # derive all triplets from length-N track
        measurement_idxs = range(len(track.measurements))
        measurement_triplets = list(itertools.combinations(measurement_idxs, 3))

        # print("From ", track, " generate ", triplets)

        for (k1, k2, k3) in measurement_triplets:
            # get triplet of image indices
            triplet = np.array([track.measurements[k1].i, track.measurements[k2].i, track.measurements[k3].i])
            sort_idxs = np.argsort(triplet)
            i1, i2, i3 = triplet[sort_idxs]

            correspondence = np.array([track.measurements[k1].uv, track.measurements[k2].uv, track.measurements[k3].uv])
            triplet_tracks[(i1, i2, i3)] += [correspondence[sort_idxs].flatten()]

        # # should be sorted?
        # assert i1 == 0
        # assert i2 == 1
        # assert i3 == 2

    inlier_errors_trifocal = []
    outlier_errors_trifocal = []
    inlier_errors_wrt_gt = []
    outlier_errors_wrt_gt = []

    print(f"Found {len(triplet_tracks)} triplets")
    print(triplet_tracks.keys())

    import gtsfm.frontend.trifocal as trifocal

    # TODO: discover all triplets.
    triplet_counter = 0
    for (i1, i2, i3), correspondences in triplet_tracks.items():

        triplet_edges = [(i1, i2), (i1, i3), (i2, i3)]
        if any([i2Ri1_dict[e] is None for e in triplet_edges]):
            continue

        if any([i2Ui1_dict[e] is None for e in triplet_edges]):
            continue

        print(f"On triplet {triplet_counter}/{len(triplet_tracks.keys())}")
        triplet_counter += 1

        _, max_rot_error, _ = cycle_consistency.compute_cycle_error(i2Ri1_dict, (i1, i2, i3), two_view_reports_dict)

        correspondences = np.array(correspondences)

        # convert (R,t) to F matrices for each image pair.
        i3Ei1 = EssentialMatrix(i2Ri1_dict[(i1, i3)], i2Ui1_dict[(i1, i3)])
        i3Ei2 = EssentialMatrix(i2Ri1_dict[(i2, i3)], i2Ui1_dict[(i2, i3)])

        camera_intrinsics_i1 = camera_intrinsics_dict[i1]
        camera_intrinsics_i2 = camera_intrinsics_dict[i2]
        camera_intrinsics_i3 = camera_intrinsics_dict[i3]

        i3Fi1 = verification_utils.essential_to_fundamental_matrix(i3Ei1, camera_intrinsics_i1, camera_intrinsics_i3)
        i3Fi2 = verification_utils.essential_to_fundamental_matrix(i3Ei2, camera_intrinsics_i2, camera_intrinsics_i3)

        # matched_keypoints_i1 = correspondences[:,:2]
        # matched_keypoints_i2 = correspondences[:,2:4]
        # matched_keypoints_i3 = correspondences[:,4:]

        # print(matched_keypoints_i1[:20].T.tolist())
        # print(matched_keypoints_i2[:20].T.tolist())
        # print(matched_keypoints_i3[:20].T.tolist())

        # print("Corr: ", correspondences.shape)
        if correspondences.shape[0] < 6:
            continue
        _, dists = trifocal.compute_trifocal_tensor_inliers(correspondences)

        # dists = fmat_point_transfer(i3Fi1, i3Fi2, correspondences)

        n_inliers = (np.absolute(dists) < 0.01).sum()
        print(f"Found {n_inliers} inliers.")
        if n_inliers >= MIN_N_REQUIRED_INLIERS:
            for e in triplet_edges:
                cycle_consistent_edges.append(e)

            inlier_errors_trifocal.append(n_inliers)
            inlier_errors_wrt_gt.append(max_rot_error)
        else:
            outlier_errors_trifocal.append(n_inliers)
            outlier_errors_wrt_gt.append(max_rot_error)

        continue
        plt.hist(dists, bins=30)
        plt.show()
        #
        # import pdb; pdb.set_trace()

        # mask = np.logical_and( 0 < dists, dists < 50)
        mask = dists > 100

        if visualize:

            img_i1 = loader.get_image(i1)
            img_i2 = loader.get_image(i2)
            img_i3 = loader.get_image(i3)

            # why degenerate for Door, indices 468, 564 ?
            draw_epipolar_lines_image_pair(
                F=i3Fi1,
                img_left=img_i1.value_array,
                img_right=img_i3.value_array,
                pts_left=matched_keypoints_i1[mask],
                pts_right=matched_keypoints_i3[mask],
            )
            plt.show()

            draw_epipolar_lines_image_pair(
                F=i3Fi2,
                img_left=img_i2.value_array,
                img_right=img_i3.value_array,
                pts_left=matched_keypoints_i2[mask],
                pts_right=matched_keypoints_i3[mask],
            )
            plt.show()

            draw_epipolar_lines_image_triplet(
                img_i1.value_array,
                img_i2.value_array,
                img_i3.value_array,
                matched_keypoints_i1[mask],
                matched_keypoints_i2[mask],
                matched_keypoints_i3[mask],
                i3Fi1,
                i3Fi2,
            )

            plt.show()

    plot_(inlier_errors_trifocal, outlier_errors_trifocal, inlier_errors_wrt_gt, outlier_errors_wrt_gt)

    # find cycle consistent ones
    i2Ri1_dict_cc = {}
    i2Ui1_dict_cc = {}
    corr_idxs_dict_cc = {}

    for (i1, i2) in cycle_consistent_edges:
        i2Ri1_dict_cc[(i1, i2)] = i2Ri1_dict[(i1, i2)]
        i2Ui1_dict_cc[(i1, i2)] = i2Ui1_dict[(i1, i2)]
        corr_idxs_dict_cc[(i1, i2)] = corr_idxs_dict[(i1, i2)]

    return i2Ri1_dict_cc, i2Ui1_dict_cc, corr_idxs_dict_cc


def plot_(inlier_errors_trifocal, outlier_errors_trifocal, inlier_errors_wrt_gt, outlier_errors_wrt_gt):
    """ """

    plt.scatter(
        inlier_errors_trifocal,
        inlier_errors_wrt_gt,
        10,
        color="g",
        marker=".",
        label=f"inliers @ ",
    )
    plt.scatter(
        outlier_errors_trifocal,
        outlier_errors_wrt_gt,
        10,
        color="r",
        marker=".",
        label=f"outliers @ ",
    )
    plt.xlabel("Num inliers (by Trifocal error)")
    plt.ylabel("Rotation error w.r.t GT")
    plt.axis("equal")
    plt.legend(loc="lower right")
    plt.savefig(os.path.join("plots", "gt_err_vs_trifocal_error.jpg"), dpi=400)
    plt.close("all")


def convert_to_homogenous_coordinates(coords: np.ndarray) -> np.ndarray:
    """Convert coordinates to homogenous system (by appending a column of ones)."""
    N = coords.shape[0]
    return np.hstack((coords, np.ones((N, 1))))


def draw_epipolar_lines_image_triplet(
    img_i1: np.ndarray,
    img_i2: np.ndarray,
    img_i3: np.ndarray,
    pts_i1: np.ndarray,
    pts_i2: np.ndarray,
    pts_i3: np.ndarray,
    i3Fi1: np.ndarray,
    i3Fi2: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """Draw the lines in image 3 that correspond to points in image 1, and to points in image 2."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    draw_lines_in_single_image(ax=ax, i2Fi1=i3Fi1, pts_i1=pts_i1, pts_i2=pts_i3, img_i2=img_i3)
    draw_lines_in_single_image(ax=ax, i2Fi1=i3Fi2, pts_i1=pts_i2, pts_i2=pts_i3, img_i2=img_i3)


def draw_epipolar_lines_image_pair(
    F: np.ndarray,
    img_left: np.ndarray,
    img_right: np.ndarray,
    pts_left: np.ndarray,
    pts_right: np.ndarray,
    figsize: Tuple[int, int] = (10, 8),
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


def draw_lines_in_single_image(
    ax, i2Fi1: np.ndarray, pts_i1: np.ndarray, pts_i2: np.ndarray, img_i2: np.ndarray
) -> None:
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
    ax.scatter(pts_i2[:, 0], pts_i2[:, 1], marker="o", s=20, c="yellow", edgecolors="red")

    for p_i1 in pts_i1:
        # get defn of epipolar line in  image, corresponding to left point p
        l_e = np.dot(i2Fi1, p_i1).squeeze()
        # find where epipolar line in right image crosses the left and image borders
        p_l = np.cross(l_e, l_l)
        p_r = np.cross(l_e, l_r)
        # convert back from homogeneous to cartesian by dividing by 3rd entry
        # draw line between point on left border, and on the right border
        x = [p_l[0] / p_l[2], p_r[0] / p_r[2]]
        y = [p_l[1] / p_l[2], p_r[1] / p_r[2]]
        ax.plot(x, y, linewidth=1, c="blue")


def test_fmat_point_transfer() -> None:
    """ """
    # dataset_root = "/Users/johnlambert/Downloads/door-trifocal-example"
    # image_extension = "JPG"

    # dataset_root = "/Users/johnlambert/Downloads/skydio-8-trifocal-example"
    # image_extension = "jpg"

    dataset_root = "/Users/johnlambert/Downloads/skydio-501-trifocal-example"
    image_extension = "JPG"

    # dataset_root = "/Users/johnlambert/Downloads/skydio-501-trifocal-example-no-covis"
    # image_extension = "JPG"

    # dataset_root = "/Users/johnlambert/Downloads/skydio-32-trifocal-example"
    # image_extension = "JPG"

    loader = OlssonLoader(dataset_root, image_extension=image_extension)

    det_desc = SuperPointDetectorDescriptor()

    feature_extractor = FeatureExtractor(detector_descriptor=DetectorDescriptorCacher(detector_descriptor_obj=det_desc))
    # feature_extractor = FeatureExtractor(det_desc)
    two_view_estimator = TwoViewEstimator(
        # matcher=SuperGlueMatcher(use_outdoor_model=True),
        matcher=MatcherCacher(matcher_obj=SuperGlueMatcher(use_outdoor_model=True)),
        verifier=Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=4),
        inlier_support_processor=InlierSupportProcessor(min_num_inliers_est_model=15, min_inlier_ratio_est_model=0.1),
        bundle_adjust_2view=False,
        eval_threshold_px=4,
        bundle_adjust_2view_maxiters=0,
    )

    camera_intrinsics_dict = {i: loader.get_camera_intrinsics(i) for i in [0, 1, 2]}

    keypoints_list, i2Ri1_dict, i2Ui1_dict, corr_idxs_dict = frontend_runner.run_frontend(
        loader, feature_extractor, two_view_estimator
    )

    filter_to_cycle_consistent_edges(
        i2Ri1_dict,
        i2Ui1_dict,
        corr_idxs_dict,
        keypoints_list,
        camera_intrinsics_dict=camera_intrinsics_dict,
        loader=loader,
    )


if __name__ == "__main__":
    test_fmat_point_transfer()
