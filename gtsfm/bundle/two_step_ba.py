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
    """Bundle adjustment optimizer which performs bundle adjustment twice: runs for a few iterations, filters out the
    outlier tracks, and then runs for the remaining iterations with the inlier tracks."""

    def __init__(
        self,
        intermediate_reproj_error_thresh: float = DEFAULT_INTERMEDIATE_REPROJ_ERROR_THRESH,
        output_reproj_error_thresh: Optional[float] = None,
        robust_measurement_noise: bool = False,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
    ) -> None:
        """Initialize the optimizer.

        Args:
            intermediate_reproj_error_thresh (optional): The reprojection error threshold used to filter out tracks
                                                         after the 1st round of BA. Defaults to
                                                         DEFAULT_INTERMEDIATE_REPROJ_ERROR_THRESH.
            output_reproj_error_thresh (optional): Reprojection error threshold for the final result. Defaults to None.
            robust_measurement_noise (optional): Flag to enable use of robust noise model for measurement noise.
                                                 Defaults to False.
            shared_calib (optional): Flag to enable shared calibration across all cameras. Defaults to False.
            max_iterations (optional): Max number of iterations when optimizing the factor graph. None means no cap.
                                       Defaults to None.
        """
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
        """Run the bundle adjustment by forming factor graph and optimizing using Levenberg–Marquardt optimization.

        Args:
            initial_data: initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: priors to be used on cameras.
            relative_pose_priors: priors on the pose between two cameras.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics.
            Optimized camera poses after filtering landmarks (and cameras with no remaining landmarks).
            Valid mask as a list of booleans, indicating for each input track whether it was below the re-projection
                threshold.
        """
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
