import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.io import savemat, loadmat
from gtsam import Pose3, Rot3, Unit3, Point3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.features as feature_utils
import gtsfm.utils.cache as cache_utils
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.common.keypoints import Keypoints
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.averaging.translation.averaging_1dsfm import get_valid_measurements_in_world_frame, compute_metrics


logger = logger_utils.get_logger()


class TranslationAveragingCReTA(TranslationAveragingBase):

    def run_translation_averaging(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray], 
        keypoints_list: List[Keypoints],
        tracks_2d: Optional[List[SfmTrack2d]] = None,
        intrinsics: Optional[List[Optional[gtsfm_types.CALIBRATION_TYPE]]] = None,
        absolute_pose_priors: List[Optional[PosePrior]] = [],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior] = {},
        scale_factor: float = 1.0,
        gt_wTi_list: List[Optional[Pose3]] = [],
    ) -> Tuple[List[Optional[Pose3]], Optional[GtsfmMetricsGroup], Optional[List[Tuple[int, int]]]]:
        """Run the translation averaging, and combine the estimated global translations with global rotations.

        Args:
            num_images: Number of camera poses.
            i2Ui1_dict: Relative unit-trans as dictionary (i1, i2): i2Ui1.
            wRi_list: Global rotations for each camera pose in the world coordinates.
            tracks_2d: 2d tracks.
            intrinsics: List of camera intrinsics.
            absolute_pose_priors: Priors on the camera poses (not delayed).
            i2Ti1_priors: Priors on the pose between camera pairs (not delayed) as (i1, i2): i2Ti1.
            scale_factor: Non-negative global scaling factor.
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.

        Returns:
            Global camera poses wTi. The number of entries in the list is `num_images`. The list
                may contain `None` where the global translations could not be computed (either underconstrained system
                or ill-constrained system).
            A GtsfmMetricsGroup with translation averaging metrics.
            Indices of inlier measurements (list of camera pair indices).
        """
        # Generate CReTA inputs.
        """[From CReTA_demo.m]
        The data should contain the following variables:
        Graph has N nodes and M edges.
        RT: 3XM matrix of relative translation directions (Tij=Rj*(Ti-Tj)).  [i2Ui1 in our notation]
        edges: Mx2 matrix of camera pairs.RT
        matches: Mx1 cell each containing Kx4 matrix.
          Each matrix contains the point correspondences for each edge.
          Each row in the matrix contains (xi,yi,xj,yj) where xi,yi and xj,yj
          are coordinates for images i,j respectively afer correcting for camera intrinsics.
        R_avg: 3X3XN matrix of absolute rotations.  [iRw in our notation]
        maxImages: Maimum No. of images in the dataset (used for indexing)
        NOTE: Ensure that the graph is connected and has the maximal parallel
        rigid component.
        """
        w_i2Ui1_dict, valid_cameras = get_valid_measurements_in_world_frame(i2Ui1_dict, wRi_list)
        RT = np.array([i2Ui1.point3().flatten() for i2Ui1 in i2Ui1_dict.values()]).T
        R_avg = np.array([wRi_list[ii].matrix() for ii in valid_cameras]).T

        edges = np.array([(i1, i2) for i1, i2 in i2Ui1_dict.keys()])
        matches = []
        for (i1, i2), i2Ui1 in i2Ui1_dict.items():
            if i2Ui1 is None:
                continue
            # Calibrate keypoints.
            uv_norm_i1 = feature_utils.normalize_coordinates(keypoints_list[i1].coordinates, intrinsics[i1])
            uv_norm_i2 = feature_utils.normalize_coordinates(keypoints_list[i2].coordinates, intrinsics[i2])

            corr = corr_idxs_dict[(i1, i2)]
            matches_i12i = np.hstack((uv_norm_i1[corr[:, 0]], uv_norm_i2[corr[:, 1]]))
            matches.append(matches_i12i)
        matches_hash = cache_utils.generate_hash_for_numpy_array(np.concatenate(matches))

        # Write inputs for MATLAB script.
        savemat(f"ta_input_{matches_hash}.mat", {"RT": RT, "edges": edges, "R_avg": R_avg, "matches": matches})
        if not os.path.isfile(f"ta_output_{matches_hash}.mat"):
            raise FileExistsError("Could not find CReTA translation averaging solution.")

        # Read in MATLAB results.
        res = loadmat(f"ta_output_{matches_hash}.mat")
        wti_list = [Point3(res["T_avg"][:, i].flatten()) for i in range(res["T_avg"].shape[1])]
        inlier_edges = [(int(i1 - 1), int(i2 - 1)) for i1, i2 in res["edges"]]

        # Compute the metrics.
        ta_metrics = compute_metrics(set(inlier_edges), i2Ui1_dict, wRi_list, wti_list, gt_wTi_list)

        num_translations = sum([1 for wti in wti_list if wti is not None])
        logger.info("Estimated %d translations out of %d images.", num_translations, num_images)

        # Combine estimated global rotations and global translations to Pose(3) objects.
        wTi_list = [
            Pose3(wRi, wti) if wRi is not None and wti is not None else None for wRi, wti in zip(wRi_list, wti_list)
        ]
        logger.info("Translation averaging took %.4f seconds.", 0.0)
        ta_metrics.add_metric(GtsfmMetric("total_duration_sec", 0.0))
        ta_metrics.add_metric(GtsfmMetric("outlier_rejection_duration_sec", 0.0))
        ta_metrics.add_metric(GtsfmMetric("optimization_duration_sec", 0.0))

        return wTi_list, ta_metrics, inlier_edges
