"""Base class for the rotation averaging component of the GTSFM pipeline.

Authors: Jing Wu, Ayush Baid
"""
import abc
from typing import Dict, List, Optional, Tuple

from gtsam import Pose3, Rot3


from gtsfm.common.pose_prior import PosePrior
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class RotationAveragingBase(GTSFMProcess):
    """Base class for rotation averaging.

    This class generates global rotation estimates from the pairwise relative
    rotations.
    """

    @staticmethod
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
    def apply(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i1Ti2_priors: Dict[Tuple[int, int], PosePrior],
        wTi_gt: List[Optional[Pose3]],
    ) -> Tuple[List[Optional[Rot3]], GtsfmMetricsGroup]:
        """Run the rotation averaging.

        Args:
            num_images: number of poses.
            i2Ri1_dict: relative rotations as dictionary (i1, i2): i2Ri1.
            i1Ti2_priors: priors on relative poses as dictionary(i1, i2): PosePrior on i1Ti2.
            wTi_gt: ground truth global rotations to compare against.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system).
            Metrics on global rotations.
        """
