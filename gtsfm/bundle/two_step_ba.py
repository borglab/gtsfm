"""2-step bundle adjustment with outlier filtering in the middle.

Authors: Ayush Baid
"""
from typing import Dict, List, Optional, Tuple

import numpy as np

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.pose_prior import PosePrior
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer

DEFAULT_INTERMEDIATE_REPROJ_ERROR_THRESH = 10
FIRST_BA_NUM_ITERATIONS = 5


class TwoStepBA(BundleAdjustmentOptimizer):
    def __init__(
        self,
        intermediate_reproj_error_thresh: float = DEFAULT_INTERMEDIATE_REPROJ_ERROR_THRESH,
        output_reproj_error_thresh: Optional[float] = None,
        robust_measurement_noise: bool = False,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
    ) -> None:
        self._ba1 = BundleAdjustmentOptimizer(
            output_reproj_error_thresh=intermediate_reproj_error_thresh,
            robust_measurement_noise=robust_measurement_noise,
            shared_calib=shared_calib,
            max_iterations=FIRST_BA_NUM_ITERATIONS,
        )

        super().__init__(
            output_reproj_error_thresh=output_reproj_error_thresh,
            robust_measurement_noise=robust_measurement_noise,
            shared_calib=shared_calib,
            max_iterations=None if max_iterations is None else max(1, max_iterations - FIRST_BA_NUM_ITERATIONS),
        )

    def run(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        verbose: bool = True,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool]]:
        _, intermediate_filtered_result, mask = self._ba1.run(
            initial_data=initial_data,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
            verbose=verbose,
        )
        valid_tracks_for_2nd_step = np.where(mask)[0]

        final_result_unfiltered, final_result_filtered, mask = super().run(
            initial_data=intermediate_filtered_result,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
            verbose=verbose,
        )

        valid_tracks_final = valid_tracks_for_2nd_step[np.where(mask)[0]]
        final_mask = [False] * initial_data.number_tracks()
        for track_idx in valid_tracks_final:
            final_mask[track_idx] = True

        return final_result_unfiltered, final_result_filtered, final_mask
