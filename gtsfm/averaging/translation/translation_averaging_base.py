"""Base class for the translation averaging component of the GTSFM pipeline.

Authors: Ayush Baid, Akshay Krishnan
"""
import abc
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Pose3, Rot3, Unit3

import gtsfm.common.types as gtsfm_types
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class TranslationAveragingBase(GTSFMProcess):
    """Base class for translation averaging.

    This class generates global unit translation estimates from pairwise relative unit translation and global rotations.
    """

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Translation Averaging",
            input_products=(
                "View-Graph Relative Translations",
                "Global Rotations",
                "Absolute Pose Priors",
                "Relative Pose Priors",
            ),
            output_products=("Global Translations",),
            parent_plate="Sparse Reconstruction",
        )

    def __init__(self, robust_measurement_noise: bool = True) -> None:
        """Initializes the translation averaging.

        Args:
            robust_measurement_noise: Whether to use a robust noise model for the measurements, defaults to true.
        """
        self._robust_measurement_noise = robust_measurement_noise

    # ignored-abstractmethod
    @abc.abstractmethod
    def run_translation_averaging(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        tracks_2d: List[SfmTrack2d] = [],
        intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]] = [],
        absolute_pose_priors: List[Optional[PosePrior]] = [],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior] = {},
        scale_factor: float = 1.0,
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[List[Optional[Pose3]], Optional[GtsfmMetricsGroup]]:
        """Run the translation averaging, and combine the estimated global translations with global rotations.

        Args:
            num_images: number of camera poses.
            i2Ui1_dict: relative unit-trans as dictionary (i1, i2): i2Ui1.
            wRi_list: global rotations for each camera pose in the world coordinates.
            absolute_pose_priors: priors on the camera poses (not delayed).
            i2Ti1_priors: priors on the pose between camera pairs (not delayed) as (i1, i2): i2Ti1.
            scale_factor: non-negative global scaling factor.
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.

        Returns:
            Global camera poses wTi. The number of entries in the list is `num_images`. The list
                may contain `None` where the global translations could not be computed (either underconstrained system
                or ill-constrained system).
        """

    def create_computation_graph(
        self,
        num_images: int,
        i2Ui1_graph: Delayed,
        wRi_graph: Delayed,
        tracks_2d: Delayed,
        intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]] = [],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior] = {},
        scale_factor: float = 1.0,
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing translation averaging.

        Args:
            num_images: number of camera poses.
            i2Ui1_graph: dictionary of relative unit translations as a delayed task.
            wRi_graph: list of global rotations wrapped up in Delayed.
            absolute_pose_priors: priors on the camera poses (not delayed).
            i2Ti1_priors: priors on the pose between camera pairs (not delayed) as (i1, i2): i2Ti1.
            scale_factor: non-negative global scaling factor.
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.

        Returns:
            Global poses wrapped as Delayed.
            A GtsfmMetricsGroup with translation averaging metrics wrapped as Delayed.
        """
        return dask.delayed(self.run_translation_averaging, nout=2)(
            num_images,
            i2Ui1_graph,
            wRi_graph,
            tracks_2d,
            intrinsics,
            absolute_pose_priors,
            i2Ti1_priors,
            scale_factor,
            gt_wTi_list,
        )
