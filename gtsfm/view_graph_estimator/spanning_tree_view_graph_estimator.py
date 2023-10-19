""" """
from typing import Dict, Tuple

import os
import time
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.two_view_estimator import TwoViewEstimationReport
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase

logger = logger_utils.get_logger()


class SpanningTreeViewGraphEstimator(ViewGraphEstimatorBase):
    """
    https://www.maths.lth.se/matematiklth/vision/publdb/reports/pdf/enqvist-kahl-etal-wovcnnc-11.pdf

    Compute a maximum spanning tree, T, with weights wij .
    Set Ec = T.
    for each e ∈ E \ T
        Let C be the cycle formed by e and T.
        if the error in C is less than sqrt(|C| * eps):

            E_c = E_c \cup e
    Estimate absolute orientations from Ec (see Section 4.1).

    Let Eoutlier be the set of outlier edges after the for-loop of Algorithm 1.

    Run search heuristics.
     For each e ∈ Eoutlier, estimate absolute rotations for
    Ec ∪ e and check for consistency.

    - For each e ∈ Eoutlier, create a new spanning tree that
    contains e and repeat steps 2 to 7 of Algorithm 1.
    """

    def __init__(self) -> None:
        """ """
        pass


    def run(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
    ) -> Set[Tuple[int, int]]:
        """Estimates the view graph, needs to be implemented by the derived class.

        The input rotation and unit translation dicts are guaranteed to be valid, i.e., i1 < i2 and
        neither i2Ri1 nor i2Ui1 are None.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: List of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints: Keypoints for each image.
            two_view_reports: two-view reports between image pairs from the TwoViewEstimator.

        Returns:
            Edges of the view-graph, which are the subset of the image pairs in the input args.
        """
        total_num_inliers = 1000

        G = nx.Graph()
        for (i1, i2) in i2Ri1_dict:
            num_inliers = two_view_reports[(i1,i2)].num_inliers_est_model
            weight = num_inliers / total_num_inliers
            G.add_edge(i1, i2, weight=weight)

        import pdb; pdb.set_trace()
        # Plot the graph.
        graph_utils.draw_view_graph_topology(
            edges=list(i2Ri1_dict.keys()),
            two_view_reports=two_view_reports,
            title="Title",
            save_fpath="",
            cameras_gt=None,
        )

        # Find max weight spanning tree.
        T = nx.maximum_spanning_tree(G)

        unused_edges = set(G) - set(T)

        # Add an edge. Compute cycle error.
        cycle_path = list(nx.find_cycle(G, orientation="original"))

        # Check average rotation consistency error, by running Shonan on this.

        i0Ri0_from_cycle = i2Ri0.inverse().compose(i2Ri1).compose(i1Ri0)
        comp_utils.compute_relative_rotation_angle(Rot3(), i0Ri0_from_cycle)


        return T.edges()


def main():

    fpath = "/Users/johnlambert/Downloads/gtsfm_2023_07_08/gtsfm/door_results_2023_10_18/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    data = io_utils.read_json_file(fpath)

    i2Ri1_dict = {}
    two_view_reports_dict = {}

    for entry in data:
        i1 = entry["i1"]
        i2 = entry["i2"]
        rot_dict = entry['rotation']
        qw = rot_dict['qw']
        qx = rot_dict['qx']
        qy = rot_dict['qy']
        qz = rot_dict['qz']

        i2Ri1 = Rot3(float(qw), float(qx), float(qy), float(qz))
        i2Ri1_dict[(i1,i2)] = i2Ri1

        # import pdb; pdb.set_trace()
        # print("here")

        two_view_reports_dict[(i1,i2)] = TwoViewEstimationReport(
            v_corr_idxs=None,
            num_inliers_est_model=entry["num_inliers_est_model"],
            R_error_deg = entry["rotation_angular_error"],
            U_error_deg = entry["translation_angular_error"],
        )

    estimator = SpanningTreeViewGraphEstimator()
    estimator.run(
        i2Ri1_dict=i2Ri1_dict,
        i2Ui1_dict={},
        calibrations=[],
        corr_idxs_i1i2={},
        keypoints=[],
        two_view_reports=two_view_reports_dict,
    )


if __name__ == "__main__":
    main()



