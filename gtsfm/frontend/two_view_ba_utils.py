"""Functions to run Bundle Adjustment on a pair of images.

Authors: Ayush Baid, John Lambert.
"""
import timeit
from typing import Dict, List, Optional, Tuple

import numpy as np
from gtsam import Pose3, Rot3, SfmTrack, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import Point3dInitializer, TriangulationOptions

logger = logger_utils.get_logger()


def triangulate_two_view_correspondences(
    triangulation_options: TriangulationOptions,
    cameras: Dict[int, gtsfm_types.CAMERA_TYPE],
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_ind: np.ndarray,
) -> Tuple[List[int], List[SfmTrack]]:
    """Triangulate 2-view correspondences to form 3D tracks.

    Args:
        camera_i1: Camera for 1st view.
        camera_i2: Camera for 2nd view.
        keypoints_i1: Keypoints for 1st view.
        keypoints_i2: Keypoints for 2nd view.
        corr_ind: Indices of corresponding keypoints.

    Returns:
        Indices of successfully triangulated tracks.
        Triangulated 3D points as tracks.
    """
    assert len(cameras) == 2
    point3d_initializer = Point3dInitializer(cameras, triangulation_options)
    triangulated_indices: List[int] = []
    triangulated_tracks: List[SfmTrack] = []
    for j, (idx1, idx2) in enumerate(corr_ind):
        track2d = SfmTrack2d(
            [SfmMeasurement(0, keypoints_i1.coordinates[idx1]), SfmMeasurement(1, keypoints_i2.coordinates[idx2])]
        )
        track, _, _ = point3d_initializer.triangulate(track2d)
        if track is not None:
            triangulated_indices.append(j)
            triangulated_tracks.append(track)

    return triangulated_indices, triangulated_tracks


def bundle_adjust(
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    verified_corr_idxs: np.ndarray,
    camera_intrinsics_i1: gtsfm_types.CALIBRATION_TYPE,
    camera_intrinsics_i2: gtsfm_types.CALIBRATION_TYPE,
    i2Ri1_initial: Optional[Rot3],
    i2Ui1_initial: Optional[Unit3],
    i2Ti1_prior: Optional[PosePrior],
    triangulation_options: TriangulationOptions,
    ba_optimizer: BundleAdjustmentOptimizer,
) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
    """Refine the relative pose using bundle adjustment on the 2-view scene.

    Args:
        keypoints_i1: Keypoints from image i1.
        keypoints_i2: Keypoints from image i2.
        verified_corr_idxs: Indices of verified correspondences between i1 and i2.
        camera_intrinsics_i1: Intrinsics for i1.
        camera_intrinsics_i2: Intrinsics for i2.
        i2Ri1_initial: The relative rotation to be used as initial rotation between cameras.
        i2Ui1_initial: The relative unit direction, to be used to initialize initial translation between cameras.
        i2Ti1_prior: Prior on the relative pose for cameras (i1, i2).
    Returns:
        Optimized relative rotation i2Ri1.
        Optimized unit translation i2Ui1.
        Optimized verified_corr_idxs.
    """
    # Choose initial pose estimate for triangulation and BA (prior gets priority).
    i2Ti1_initial = i2Ti1_prior.value if i2Ti1_prior is not None else None
    if i2Ti1_initial is None and i2Ri1_initial is not None and i2Ui1_initial is not None:
        i2Ti1_initial = Pose3(i2Ri1_initial, i2Ui1_initial.point3())
    if i2Ti1_initial is None:
        return None, None, verified_corr_idxs

    # Set the i1 camera pose as the global coordinate system.
    camera_class = gtsfm_types.get_camera_class_for_calibration(camera_intrinsics_i1)
    cameras = {
        0: camera_class(Pose3(), camera_intrinsics_i1),
        1: camera_class(i2Ti1_initial.inverse(), camera_intrinsics_i2),
    }

    # Triangulate!
    start_time = timeit.default_timer()
    triangulated_indices, triangulated_tracks = triangulate_two_view_correspondences(
        triangulation_options, cameras, keypoints_i1, keypoints_i2, verified_corr_idxs
    )
    logger.debug("Performed DA in %.6f seconds.", timeit.default_timer() - start_time)
    logger.debug("Triangulated %d correspondences out of %d.", len(triangulated_tracks), len(verified_corr_idxs))

    if len(triangulated_tracks) == 0:
        return i2Ti1_initial.rotation(), Unit3(i2Ti1_initial.translation()), np.array([], dtype=np.uint32)

    # Build BA inputs.
    start_time = timeit.default_timer()
    ba_input = GtsfmData(number_images=2, cameras=cameras, tracks=triangulated_tracks)
    relative_pose_prior_for_ba = {(0, 1): i2Ti1_prior} if i2Ti1_prior is not None else {}

    # Optimize!
    _, ba_output, valid_mask = ba_optimizer.run_ba(
        ba_input, absolute_pose_priors=[], relative_pose_priors=relative_pose_prior_for_ba, verbose=False
    )

    # Unpack results.
    valid_corr_idxs = verified_corr_idxs[triangulated_indices][valid_mask]
    wTi1, wTi2 = ba_output.get_camera_poses()  # extract the camera poses
    if wTi1 is None or wTi2 is None:
        logger.warning("2-view BA failed...")
        return i2Ri1_initial, i2Ui1_initial, valid_corr_idxs
    i2Ti1_optimized = wTi2.between(wTi1)
    logger.debug("Performed 2-view BA in %.6f seconds.", timeit.default_timer() - start_time)

    return i2Ti1_optimized.rotation(), Unit3(i2Ti1_optimized.translation()), valid_corr_idxs
