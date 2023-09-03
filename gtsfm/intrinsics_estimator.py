import dataclasses
import logging
import timeit
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import dask
import gtsam
import numpy as np
from dask.delayed import Delayed
from gtsam import (
    CameraSetCal3Bundler,
    CameraSetCal3Fisheye,
    PinholeCameraCal3Bundler,
    Point2Vector,
    Pose3,
    Rot3,
    SfmTrack,
    Unit3,
    Cal3Bundler,
)

import gtsfm.utils.logger as logger_utils
from gtsfm.frontend.verifier.ransac import Ransac
import gtsfm.utils.verification as verify_utils

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

PRE_BA_REPORT_TAG = "PRE_BA_2VIEW_REPORT"
POST_BA_REPORT_TAG = "POST_BA_2VIEW_REPORT"
POST_ISP_REPORT_TAG = "POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT"
VIEWGRAPH_REPORT_TAG = "VIEWGRAPH_2VIEW_REPORT"


class IntrinsicsEstimator:
    """Estimate the focal length from two view correspondences."""

    def __init__(
        self,
        verification_threshold_px: float,
        per_camera_intrinsics: bool = False,
        max_num_points_for_estimation: int = 200,
    ) -> None:
        """Initializes the intrinsics estimator."""
        self._F_verifier = Ransac(
            use_intrinsics_in_verification=False, estimation_threshold_px=verification_threshold_px
        )
        self.per_camera_intrinsics = per_camera_intrinsics
        self.max_num_points_for_estimation = max_num_points_for_estimation
        self.min_focal_length_ratio = 0.5
        self.max_focal_length_ratio = 1.5
        self.num_focal_length_steps = 20

        self.max_reprojection_error_px = 20

    def compute_intrinsics_triang(
        self,
        intrinsics_i1,
        intrinsics_i2,
        keypoints_i1,
        keypoints_i2,
        putative_corr_idxs,
    ):
        if self.per_camera_intrinsics:
            raise ValueError("not implemented yet")
        f1_range = np.linspace(start=0.5, stop=1.5, num=30) * intrinsics_i1.fx()
        focal_candidate_pairs = [(f1, f1) for f1 in f1_range]

        i2Fi1, mask = self._F_verifier.estimate_F(keypoints_i1, keypoints_i2, putative_corr_idxs)
        if mask is None:
            return None, None, None, None
        verified_corr = putative_corr_idxs[mask.squeeze()]

        errors = []
        valid_candidates = []

        for f1, f2 in focal_candidate_pairs:
            new_intrin_i1 = gtsam.Cal3Bundler(
                f1,
                intrinsics_i1.k1(),
                intrinsics_i1.k2(),
                intrinsics_i1.px(),
                intrinsics_i1.py(),
            )
            new_intrin_i2 = gtsam.Cal3Bundler(
                f2,
                intrinsics_i2.k1(),
                intrinsics_i2.k2(),
                intrinsics_i2.px(),
                intrinsics_i2.py(),
            )
            i2Ei1 = verify_utils.fundamental_to_essential_matrix(i2Fi1, new_intrin_i1, new_intrin_i2)
            i2Ri1, i2Ui1 = verify_utils.recover_relative_pose_from_essential_matrix(
                i2Ei1,
                keypoints_i1.coordinates[verified_corr[:, 0]],
                keypoints_i2.coordinates[verified_corr[:, 1]],
                new_intrin_i1,
                new_intrin_i2,
            )
            wTi1 = gtsam.Pose3(i2Ri1, i2Ui1.point3())
            wTi2 = gtsam.Pose3(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(0, 0, 0)))
            camposes = gtsam.Pose3Vector([wTi1, wTi2])

            this_errors = []
            for corr in verified_corr[: self.max_num_points_for_estimation]:
                measurements = gtsam.Point2Vector(
                    [gtsam.Point2(keypoints_i1.coordinates[corr[0]]), gtsam.Point2(keypoints_i2.coordinates[corr[1]])]
                )
                try:
                    point = gtsam.triangulatePoint3(
                        camposes, new_intrin_i1, measurements, rank_tol=1e-4, optimize=False
                    )
                except RuntimeError:
                    continue

                # Point is in world frame, same as i2.
                point /= point[2]
                reproj_p2 = new_intrin_i2.K() @ point
                reproj_p2 = reproj_p2[:2]
                this_errors.append(np.linalg.norm(reproj_p2 - keypoints_i2.coordinates[corr[1]]))
            if len(this_errors) > 0:
                errors.append(np.nanmean(this_errors))
                valid_candidates.append((f1, f2))

        if len(valid_candidates) <= 0.3 * len(focal_candidate_pairs):
            # This is likely not a good camera pair to do focal length estimation.
            return (None, None, None, None)

        best_idx = np.argmin(errors)
        best_f1, best_f2 = valid_candidates[best_idx]
        if (
            best_f1 == self.min_focal_length_ratio * intrinsics_i1.fx()
            or best_f2 == self.min_focal_length_ratio * intrinsics_i2.fx()
            or best_f1 == self.max_focal_length_ratio * intrinsics_i1.fx()
            or best_f2 == self.max_focal_length_ratio * intrinsics_i2.fx()
        ):
            # the minima was at the bounds, so likely not a good pair for estimation.
            return (None, None, None, None)
        return (best_f1, best_f2, errors[best_idx])

    def compute_intrinsics(
        self,
        intrinsics_i1,
        intrinsics_i2,
        keypoints_i1,
        keypoints_i2,
        putative_corr_idxs,
    ):
        f_range = np.linspace(
            start=self.min_focal_length_ratio, stop=self.max_focal_length_ratio, num=self.num_focal_length_steps
        )
        i2Fi1, mask = self._F_verifier.estimate_F(keypoints_i1, keypoints_i2, putative_corr_idxs)

        verified_corr = putative_corr_idxs[mask.squeeze()]
        errors = []
        f_vals = []

        for f1 in f_range:
            new_f_i1 = intrinsics_i1.fx() * f1
            new_f_i2 = intrinsics_i2.fx() * f1
            f_vals.append(new_f_i1)
            new_intrin_i1 = gtsam.Cal3Bundler(
                new_f_i1,
                intrinsics_i1.k1(),
                intrinsics_i1.k2(),
                intrinsics_i1.px(),
                intrinsics_i1.py(),
            )
            new_intrin_i2 = gtsam.Cal3Bundler(
                new_f_i2,
                intrinsics_i2.k1(),
                intrinsics_i2.k2(),
                intrinsics_i2.px(),
                intrinsics_i2.py(),
            )
            i2Ei1 = verify_utils.fundamental_to_essential_matrix(i2Fi1, new_intrin_i1, new_intrin_i2)
            # i2Ri1, i2Ui1 = verify_utils.recover_relative_pose_from_essential_matrix(
            #     i2Ei1,
            #     keypoints_i1.coordinates[verified_corr[:, 0]],
            #     keypoints_i2.coordinates[verified_corr[:, 1]],
            #     new_intrin_i1,
            #     new_intrin_i2,
            # )
            match_kp1 = keypoints_i1.coordinates[verified_corr[:, 0]]
            match_kp2 = keypoints_i2.coordinates[verified_corr[:, 1]]
            normd_kp1 = np.concatenate([match_kp1, np.ones_like(match_kp1[:, :1])], axis=-1)
            normd_kp2 = np.concatenate([match_kp2, np.ones_like(match_kp2[:, :1])], axis=-1)
            normd_kp1 = (np.linalg.inv(new_intrin_i1.K()) @ normd_kp1.transpose()).transpose()
            normd_kp2 = (np.linalg.inv(new_intrin_i2.K()) @ normd_kp2.transpose()).transpose()
            error = (
                verify_utils.compute_epipolar_distances_sq_sampson_essential(normd_kp1, normd_kp2, i2Ei1)
                + verify_utils.compute_epipolar_distances_sq_sampson_essential(normd_kp2, normd_kp1, i2Ei1.T)
            ).mean() / 2.0

            # i2Fi1_recon = verify_utils.essential_to_fundamental_matrix(
            #     gtsam.EssentialMatrix(i2Ri1, i2Ui1), new_intrin_i1, new_intrin_i2
            # )

            # error = np.abs(
            #     verify_utils.compute_epipolar_distances_sq_sampson(
            #         keypoints_i1.coordinates[verified_corr[:, 0]],
            #         keypoints_i2.coordinates[verified_corr[:, 1]],
            #         i2Fi1_recon,
            #     )
            # ).mean()
            errors.append(error)

        best_fx = f_range[np.argmin(errors)]
        return (
            gtsam.Cal3Bundler(
                best_fx * intrinsics_i1.fx(),
                intrinsics_i1.k1(),
                intrinsics_i1.k2(),
                intrinsics_i1.px(),
                intrinsics_i1.py(),
            ),
            f_vals,
            errors,
        )

    def create_computation_graph_for_pair(
        self,
        intrinsics_i1,
        intrinsics_i2,
        keypoints_i1,
        keypoints_i2,
        putative_corr_idxs,
    ):
        return dask.delayed(self.compute_intrinsics_triang, nout=3)(
            intrinsics_i1, intrinsics_i2, keypoints_i1, keypoints_i2, putative_corr_idxs
        )

    def average_focals(self, focals, errors):
        valid_mask = errors < self.max_reprojection_error_px
        if valid_mask.sum() == 0:
            return None
        return np.average(focals[valid_mask], weights=1.0 / errors[valid_mask])

    def get_updated_focals(self, cam_idx, orig_intrinsics, focal_estimates, estimate_errors):
        valid_errors = []
        valid_focals = []
        for i in range(len(estimate_errors)):
            if estimate_errors[i] is None or estimate_errors[i] is np.nan or focal_estimates[i] is None:
                continue
            valid_errors.append(estimate_errors[i])
            valid_focals.append(focal_estimates[i])
        if len(valid_errors) == 0:
            logging.info("Focal length estimation failed for camera {cam_idx}, using original intrinsics.")
            return orig_intrinsics

        errors = np.array(valid_errors)
        focals = np.array(valid_focals)

        averaged_focal = self.average_focals(focals, errors)
        if averaged_focal is None:
            logging.info("Focal length estimation failed for camera {cam_idx}, using original intrinsics.")
            return orig_intrinsics
        return Cal3Bundler(
            averaged_focal, orig_intrinsics.k1(), orig_intrinsics.k2(), orig_intrinsics.px(), orig_intrinsics.py()
        )

    def create_computation_graph(
        self,
        keypoints_list,
        putative_corr_idxs_dict,
        image_pair_indices,
        all_intrinsics,
    ):
        """The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times."""
        focals_for_camera = defaultdict(list)
        errors_for_camera = defaultdict(list)

        for i1, i2 in image_pair_indices:
            focal_i1, focal_i2, error = self.create_computation_graph_for_pair(
                all_intrinsics[i1],
                all_intrinsics[i2],
                keypoints_list[i1],
                keypoints_list[i2],
                putative_corr_idxs_dict[(i1, i2)],
            )
            focals_for_camera[i1].append(focal_i1)
            focals_for_camera[i2].append(focal_i2)
            errors_for_camera[i1].append(error)
            errors_for_camera[i2].append(error)

        if self.per_camera_intrinsics:
            raise ValueError("yet to be implemented")

        all_focals = []
        all_errors = []
        for focals in focals_for_camera.values():
            all_focals.extend(focals)
        for errors in errors_for_camera.values():
            all_errors.extend(errors)
        if len(all_errors) == 0:
            return all_intrinsics

        ref_updated_intrinsics = dask.delayed(self.get_updated_focals)(0, all_intrinsics[0], all_focals, all_errors)
        result_intrinsics = [ref_updated_intrinsics for i in range(len(all_intrinsics))]

        return result_intrinsics

    def estimate_intrinsics(
        self, keypoints_i1, keypoints_i2, putative_corr_idxs, camera_intrinsics_i1, camera_intrinsics_i2
    ):
        cx = camera_intrinsics_i1.px()
        cy = camera_intrinsics_i1.py()
        keypoints_i1.coordinates = keypoints_i1.coordinates - np.array([[cx, cy]])
        keypoints_i2.coordinates = keypoints_i2.coordinates - np.array([[cx, cy]])
        i2Fi1, mask = self._F_verifier.estimate_F(keypoints_i1, keypoints_i2, putative_corr_idxs)
        try:
            focal = verify_utils.shared_focal_lengths_from_fundamental_matrix(i2Fi1)
        except:
            return None
        return gtsam.Cal3Bundler(
            focal,
            camera_intrinsics_i1.k1(),
            camera_intrinsics_i1.k2(),
            camera_intrinsics_i1.px(),
            camera_intrinsics_i1.py(),
        )

    def verify_with_coarse_focal_estimation(
        self,
        intrinsics_i1,
        intrinsics_i2,
        keypoints_i1,
        keypoints_i2,
        putative_corr_idxs,
    ):
        f_range = np.linspace(start=0.7, stop=1.5, num=20)
        best_pre_ba_inlier_ratio_wrt_estimate = 0.0
        best_pre_ba_i2Ri1 = None
        best_pre_ba_i2Ui1 = None
        best_pre_ba_v_corr_idxs = None
        best_intrin_i1 = None
        best_intrin_i2 = None
        all_inlier_ratios = []
        for f1 in f_range:
            new_f_i1 = intrinsics_i1.fx() * f1
            new_f_i2 = intrinsics_i2.fx() * f1
            new_intrin_i1 = gtsam.Cal3Bundler(
                new_f_i1,
                intrinsics_i1.k1(),
                intrinsics_i1.k2(),
                intrinsics_i1.px(),
                intrinsics_i1.py(),
            )
            new_intrin_i2 = gtsam.Cal3Bundler(
                new_f_i2,
                intrinsics_i2.k1(),
                intrinsics_i2.k2(),
                intrinsics_i2.px(),
                intrinsics_i2.py(),
            )
            (
                pre_ba_i2Ri1,
                pre_ba_i2Ui1,
                pre_ba_v_corr_idxs,
                pre_ba_inlier_ratio_wrt_estimate,
            ) = self._verifier.verify(
                keypoints_i1,
                keypoints_i2,
                putative_corr_idxs,
                new_intrin_i1,
                new_intrin_i2,
            )
            all_inlier_ratios.append(pre_ba_inlier_ratio_wrt_estimate)
            if pre_ba_inlier_ratio_wrt_estimate > best_pre_ba_inlier_ratio_wrt_estimate:
                best_pre_ba_inlier_ratio_wrt_estimate = pre_ba_inlier_ratio_wrt_estimate
                best_pre_ba_i2Ri1 = pre_ba_i2Ri1
                best_pre_ba_i2Ui1 = pre_ba_i2Ui1
                best_pre_ba_v_corr_idxs = pre_ba_v_corr_idxs
                best_intrin_i1 = new_intrin_i1
                best_intrin_i2 = new_intrin_i2
        return (
            best_pre_ba_i2Ri1,
            best_pre_ba_i2Ui1,
            best_pre_ba_v_corr_idxs,
            best_pre_ba_inlier_ratio_wrt_estimate,
            best_intrin_i1,
            best_intrin_i2,
        )


def average_all_intrinsics(estimates, triang_angles, pose_params):
    valid_focals = []
    valid_triang_angles = []
    reference = estimates[0]

    for i in range(len(estimates)):
        if pose_params[i][0] is None or pose_params[i][1] is None:
            continue
        valid_focals.append(estimates[i].fx())
        valid_triang_angles.append(triang_angles[i])

    valid_triang_angles = np.array(valid_triang_angles)
    idx = np.flip(np.argsort(valid_triang_angles))
    focals = np.take_along_axis(np.array(valid_focals), idx, axis=0)[:20]
    new_intrinsics = Cal3Bundler(np.mean(focals), reference.k1(), reference.k2(), reference.px(), reference.py())
    return new_intrinsics
