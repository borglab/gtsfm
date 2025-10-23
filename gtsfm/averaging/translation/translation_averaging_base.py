"""Base class for the translation averaging component of the GTSFM pipeline.

Authors: Ayush Baid, Akshay Krishnan
"""

import abc
from typing import Dict, List, Optional, Sequence, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Pose3, Rot3, Unit3  # type: ignore

import gtsfm.common.types as gtsfm_types
from gtsfm.common.outputs import Outputs
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.products.visibility_graph import ImageIndexPairs
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class TranslationAveragingBase(GTSFMProcess):
    """Base class for translation averaging.

    This class generates global unit translation estimates from pairwise relative unit translation and global rotations.
    """

    @staticmethod
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
        tracks_2d: Optional[List[SfmTrack2d]] = None,
        intrinsics: Optional[List[Optional[gtsfm_types.CALIBRATION_TYPE]]] = None,
        absolute_pose_priors: List[Optional[PosePrior]] = [],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior] = {},
        scale_factor: float = 1.0,
        gt_wTi_list: List[Optional[Pose3]] = [],
    ) -> Tuple[List[Optional[Pose3]], Optional[GtsfmMetricsGroup], Optional[ImageIndexPairs]]:
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

    def _run_translation_averaging_with_sink(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        tracks_2d: Optional[List[SfmTrack2d]],
        intrinsics: Sequence[List[Optional[gtsfm_types.CALIBRATION_TYPE]]],
        absolute_pose_priors: List[Optional[PosePrior]],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior],
        scale_factor: float,
        gt_wTi_list: List[Optional[Pose3]],
        outputs: Optional[Outputs],
    ) -> Tuple[List[Optional[Pose3]], Optional[ImageIndexPairs]]:
        wTi_list, metrics, inlier_indices = self.run_translation_averaging(
            num_images=num_images,
            i2Ui1_dict=i2Ui1_dict,
            wRi_list=wRi_list,
            tracks_2d=tracks_2d,
            intrinsics=intrinsics,
            absolute_pose_priors=absolute_pose_priors,
            i2Ti1_priors=i2Ti1_priors,
            scale_factor=scale_factor,
            gt_wTi_list=gt_wTi_list,
        )

        sink = outputs.metrics_sink if outputs is not None else None
        if sink is not None and metrics is not None:
            sink.record(metrics)

        return wTi_list, metrics, inlier_indices

    def create_computation_graph(
        self,
        num_images: int,
        i2Ui1_graph: Delayed,
        wRi_graph: Delayed,
        tracks_2d: Optional[Delayed] = None,
        intrinsics: Sequence[List[Optional[gtsfm_types.CALIBRATION_TYPE]]] = None,
        absolute_pose_priors: List[Optional[PosePrior]] = [],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior] = {},
        scale_factor: float = 1.0,
        gt_wTi_list: List[Optional[Pose3]] = [],
        outputs: Optional[Outputs] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing translation averaging.

        Args:
            num_images: Number of camera poses.
            i2Ui1_graph: Dictionary of relative unit translations as a delayed task.
            wRi_graph: List of global rotations wrapped up in Delayed.
            tracks_2d: 2d tracks wrapped up in Delayed.
            intrinsics: List of camera intrinsics.
            absolute_pose_priors: Priors on the camera poses (not delayed).
            i2Ti1_priors: Priors on the pose between camera pairs (not delayed) as (i1, i2): i2Ti1.
            scale_factor: Non-negative global scaling factor.
            gt_wTi_list: List of ground truth poses (wTi) for computing metrics.
            outputs: Optional collection of paths and sinks for persistence.

        Returns:
            Delayed global poses, delayed metrics group, and delayed inlier indices.
        """
        return dask.delayed(self._run_translation_averaging_with_sink, nout=3)(
            num_images=num_images,
            i2Ui1_dict=i2Ui1_graph,
            wRi_list=wRi_graph,
            tracks_2d=tracks_2d,
            intrinsics=intrinsics,
            absolute_pose_priors=absolute_pose_priors,
            i2Ti1_priors=i2Ti1_priors,
            scale_factor=scale_factor,
            gt_wTi_list=gt_wTi_list,
            outputs=outputs,
        )
