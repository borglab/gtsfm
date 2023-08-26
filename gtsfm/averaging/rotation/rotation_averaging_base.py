"""Base class for the rotation averaging component of the GTSFM pipeline.

Authors: Jing Wu, Ayush Baid
"""
import abc
import time
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Pose3, Rot3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.common.pose_prior import PosePrior
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class RotationAveragingBase(GTSFMProcess):
    """Base class for rotation averaging.

    This class generates global rotation estimates from the pairwise relative
    rotations.
    """

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Rotation Averaging",
            input_products=("View-Graph Relative Rotations", "Relative Pose Priors"),
            output_products="Global Rotations",
            parent_plate="Sparse Reconstruction",
        )

    # ignored-abstractmethod
    @abc.abstractmethod
    def run_rotation_averaging(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging.

        Args:
            num_images: number of poses.
            i2Ri1_dict: relative rotations as dictionary (i1, i2): i2Ri1.
            i1Ti2_priors: priors on relative poses as dictionary(i1, i2): PosePrior on i1Ti2.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system).
        """

    def _run_rotation_averaging_base(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
        wTi_gt: List[Optional[Pose3]],
    ) -> Tuple[List[Optional[Rot3]], GtsfmMetricsGroup]:
        """Runs rotation averaging and computes metrics.

        Args:
            num_images: Number of poses.
            i2Ri1_dict: Relative rotations as dictionary (i1, i2): i2Ri1.
            i1Ti2_priors: Priors on relative poses as dictionary(i1, i2): PosePrior on i1Ti2.
            wTi_gt: Ground truth global rotations to compare against.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system).
            Metrics on global rotations.
        """
        start_time = time.time()
        wRis = self.run_rotation_averaging(num_images, i2Ri1_dict, i1Ti2_priors)
        run_time = time.time() - start_time

        metrics = self.evaluate(wRis, wTi_gt)
        metrics.add_metric(GtsfmMetric("total_duration_sec", run_time))

        return wRis, metrics

    def evaluate(self, wRi_computed: List[Optional[Rot3]], wTi_gt: List[Optional[Pose3]]) -> GtsfmMetricsGroup:
        """Evaluates the global rotations computed by the rotation averaging implementation.

        Args:
            wRi_computed: List of global rotations computed.
            wTi_gt: Ground truth global rotations to compare against.

        Raises:
            ValueError: If the length of the computed and GT list differ.

        Returns:
            Metrics on global rotations.
        """
        wRi_gt = [wTi.rotation() if wTi is not None else None for wTi in wTi_gt]

        if len(wRi_computed) != len(wRi_gt):
            raise ValueError("Lengths of wRi_list and gt_wRi_list should be the same.")

        wRi_aligned = comp_utils.align_rotations(wRi_gt, wRi_computed)

        metrics = []
        metrics.append(GtsfmMetric(name="num_rotations_computed", data=len([x for x in wRi_computed if x is not None])))
        metrics.append(metric_utils.compute_rotation_angle_metric(wRi_aligned, wRi_gt))
        return GtsfmMetricsGroup(name="rotation_averaging_metrics", metrics=metrics)

    def create_computation_graph(
        self,
        num_images: int,
        i2Ri1_graph: Delayed,
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
        gt_wTi_list: List[Optional[Pose3]],
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing rotation averaging.

        Args:
            num_images: number of poses.
            i2Ri1_graph: dictionary of relative rotations as a delayed task.
            i1Ti2_priors: priors on relative poses as (i1, i2): PosePrior on i1Ti2.
            gt_wTi_list: ground truth poses, to be used for evaluation.

        Returns:
            global rotations wrapped using dask.delayed.
        """

        wRis, metrics = dask.delayed(self._run_rotation_averaging_base, nout=2)(
            num_images, i2Ri1_dict=i2Ri1_graph, i1Ti2_priors=i1Ti2_priors, wTi_gt=gt_wTi_list
        )

        return wRis, metrics
