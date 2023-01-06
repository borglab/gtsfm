"""A ViewGraphEstimator implementation which passes all the input graph edges
without any filtering.

Authors: Hayk Stepanyan
"""

from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase
from typing import Dict, List, Set, Tuple

import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.keypoints import Keypoints
from gtsfm.two_view_estimator import TwoViewEstimationReport


class ViewGraphWithoutFiltering(ViewGraphEstimatorBase):
    
    def run(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
    ) -> Set[Tuple[int, int]]:
        """
        Return graph input edges with no filtering.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2 (unused).
            calibrations: list of calibrations for each image (unused).
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2 (unused).
            keypoints: keypoints for each images (unused).
            two_view_reports: Dict from (i1, i2) to the TwoViewEstimationReport of the edge.

        Returns:
            Edges of the view-graph.
        """

        return set(i2Ri1_dict.keys())
