"""Augment correspondences on two-view estimates.

Authors: Ayush Baid
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dask.distributed import Client, Future
from gtsam import Pose3, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
import gtsfm.frontend.two_view_ba_utils as two_view_ba_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.bundle.two_view_ba import TwoViewBundleAdjustment
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior, PosePriorType
from gtsfm.data_association.point3d_initializer import TriangulationOptions
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TWO_VIEW_REPORT_TAG, generate_two_view_report_from_result

RELATIVE_POSE_PRIOR_SIGMA = np.ones((6,)) * 3e-2


class CorrespondenceAugmenter:
    """Augment correspondences on two-view estimates."""

    def __init__(
        self,
        correspondence_generator: CorrespondenceGeneratorBase,
        eval_threshold_px: float,
        triangulation_options: TriangulationOptions,
        bundle_adjust_2view_maxiters: int = 10,
        ba_reproj_error_thresholds: List[Optional[float]] = [3],
    ) -> None:
        self._correspondence_generator = correspondence_generator
        self._corr_metric_dist_threshold = eval_threshold_px
        self._triangulation_options = triangulation_options
        self._ba_optimizer = TwoViewBundleAdjustment(
            reproj_error_thresholds=ba_reproj_error_thresholds,
            robust_measurement_noise=True,
            max_iterations=bundle_adjust_2view_maxiters,
        )

    def _augment_keypoints(self, keypoints_original: Keypoints, keypoints_new: Keypoints) -> Tuple[Keypoints, int]:
        if len(keypoints_original) == 0:
            return keypoints_new, 0
        if len(keypoints_new) == 0:
            return keypoints_original, 0

        num_original_keypoints = len(keypoints_original)

        # Note(Ayush): we avoid augmenting scales and responses and they do not make sense across different detectors
        return (
            Keypoints(coordinates=np.append(keypoints_original.coordinates, keypoints_new.coordinates, axis=0)),
            num_original_keypoints,
        )

    def augment_correspondences(
        self,
        client: Client,
        images: List[Future],
        keypoints_list: List[Keypoints],
        camera_intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
        two_view_outputs: Dict[Tuple[int, int], TWO_VIEW_OUTPUT],
        gt_cameras: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_scene_mesh: Optional[Any],
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], TWO_VIEW_OUTPUT]]:
        (
            keypoints_to_augment_list,
            putative_corr_idxs_to_augment,
        ) = self._correspondence_generator.generate_correspondences(
            client=client, images=images, image_pairs=list(two_view_outputs.keys())
        )

        def verify_putative_correspondences(
            keypoints_i1: Keypoints,
            Keypoints_i2: Keypoints,
            putative_corr_idxs: np.ndarray,
            camera_intrinsics_i1: gtsfm_types.CALIBRATION_TYPE,
            camera_intrinsics_i2: gtsfm_types.CALIBRATION_TYPE,
            i2Ri1: Rot3,
            i2Ui1: Unit3,
            triangulation_options: TriangulationOptions,
            ba_optimizer: BundleAdjustmentOptimizer,
        ) -> np.ndarray:
            if i2Ri1 is None or i2Ui1 is None:
                return np.array([], dtype=np.uint64)

            i2Ti1_prior = PosePrior(
                value=Pose3(i2Ri1, i2Ui1.point3()),
                covariance=RELATIVE_POSE_PRIOR_SIGMA,
                type=PosePriorType.SOFT_CONSTRAINT,
            )

            _, _, verified_corr_idxs = two_view_ba_utils.bundle_adjust(
                keypoints_i1=keypoints_i1,
                keypoints_i2=Keypoints_i2,
                verified_corr_idxs=putative_corr_idxs,
                camera_intrinsics_i1=camera_intrinsics_i1,
                camera_intrinsics_i2=camera_intrinsics_i2,
                i2Ri1_initial=i2Ri1,
                i2Ui1_initial=i2Ui1,
                i2Ti1_prior=i2Ti1_prior,
                triangulation_options=triangulation_options,
                ba_optimizer=ba_optimizer,
            )

            return verified_corr_idxs

        verified_corr_idxs_futures = {
            (i1, i2): client.submit(
                verify_putative_correspondences,
                keypoints_to_augment_list[i1],
                keypoints_to_augment_list[i2],
                putative_corr_idxs_to_augment[(i1, i2)],
                camera_intrinsics[i1],
                camera_intrinsics[i2],
                two_view_output[0],
                two_view_output[1],
                self._triangulation_options,
                self._ba_optimizer,
            )
            for (i1, i2), two_view_output in two_view_outputs.items()
        }
        verified_corr_idxs_dict = client.gather(verified_corr_idxs_futures)

        augmented_keypoints_list: List[Keypoints] = []
        keypoint_idx_offsets: List[int] = []
        for keypoints_original, keypoints_new in zip(keypoints_list, keypoints_to_augment_list):
            keypoints_augmented, offset = self._augment_keypoints(
                keypoints_original=keypoints_original, keypoints_new=keypoints_new
            )
            augmented_keypoints_list.append(keypoints_augmented)
            keypoint_idx_offsets.append(offset)

        # Merge the corr_idxs
        results: Dict[Tuple[int, int], TWO_VIEW_OUTPUT] = {}
        for (i1, i2), two_view_output in two_view_outputs.items():
            new_corr_idxs = verified_corr_idxs_dict[(i1, i2)]
            if len(new_corr_idxs) > 0:
                new_corr_idxs[:, 0] += keypoint_idx_offsets[i1]
                new_corr_idxs[:, 1] += keypoint_idx_offsets[i2]

                augmented_v_corr_idxs = np.append(two_view_output[2], new_corr_idxs, axis=0)
            else:
                augmented_v_corr_idxs = two_view_output[2]

            two_view_reports = two_view_output[3]
            two_view_report_before_augmentation = two_view_reports[TWO_VIEW_REPORT_TAG.POST_ISP]
            num_putative_corrs_before_augmentation = (
                int(
                    two_view_report_before_augmentation.num_inliers_est_model
                    / two_view_report_before_augmentation.inlier_ratio_est_model
                )
                if two_view_report_before_augmentation.inlier_ratio_est_model is not None
                else 0
            )
            num_putative_corrs_post_augmentation = (
                num_putative_corrs_before_augmentation + putative_corr_idxs_to_augment[(i1, i2)].shape[0]
            )

            two_view_report_post_augmentation = generate_two_view_report_from_result(
                i2Ri1_computed=two_view_output[0],
                i2Ui1_computed=two_view_output[1],
                keypoints_i1=augmented_keypoints_list[i1],
                keypoints_i2=augmented_keypoints_list[i2],
                verified_corr_idxs=augmented_v_corr_idxs,
                inlier_ratio_wrt_estimate=augmented_v_corr_idxs.shape[0] / num_putative_corrs_post_augmentation,
                gt_camera_i1=gt_cameras[i1],
                gt_camera_i2=gt_cameras[i2],
                gt_scene_mesh=gt_scene_mesh,
                corr_metric_dist_threshold=self._corr_metric_dist_threshold,
            )

            two_view_reports[TWO_VIEW_REPORT_TAG.CORRESPONDENCE_AUGMENTED] = two_view_report_post_augmentation

            results[(i1, i2)] = (two_view_output[0], two_view_output[1], augmented_v_corr_idxs, two_view_reports)

        return augmented_keypoints_list, results
