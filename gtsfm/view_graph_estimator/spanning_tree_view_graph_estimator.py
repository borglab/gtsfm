""" """
from typing import Dict, Tuple

import copy
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
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
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
        self._rot_avg_module = ShonanRotationAveraging()

    def run(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        output_dir: Optional[Path] = None,
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
            output_dir: Path to directory where outputs for debugging will be saved.

        Returns:
            Edges of the view-graph, which are the subset of the image pairs in the input args.
        """
        total_num_inliers = 1000

        G = nx.Graph()
        for (i1, i2) in i2Ri1_dict:
            num_inliers = two_view_reports[(i1, i2)].num_inliers_est_model
            weight = num_inliers / total_num_inliers
            #print("Edge weight: ", weight)
            G.add_edge(i1, i2, weight=weight)

        avg_error = np.mean([two_view_reports[(i1, i2)].R_error_deg for (i1, i2) in i2Ri1_dict.keys()])
        print(f"Avg Full Graph error: {avg_error}")
        # import pdb; pdb.set_trace()

        # Plot the graph.
        # graph_utils.draw_view_graph_topology(
        #     edges=list(i2Ri1_dict.keys()),
        #     two_view_reports=two_view_reports,
        #     title="Title",
        #     save_fpath="",
        #     cameras_gt=None,
        # )

        # Sample spanning trees.
        # for ...

        # Find max weight spanning tree.
        #
        for sample_idx in range(100):
            print(f"Sample index={sample_idx}")
            #T = nx.random_spanning_tree(G, weight="weight", multiplicative=True) #, seed=42)
            T = nx.maximum_spanning_tree(G)

            T_edges = T.edges()
            sorted_T_edges = [tuple(sorted([i1, i2])) for (i1, i2) in T_edges]
            avg_error = np.mean([two_view_reports[(i1, i2)].R_error_deg for (i1, i2) in sorted_T_edges])
            print(f"Sample {sample_idx} Avg ST error: {avg_error}")

            # graph_utils.draw_view_graph_topology(
            #     edges=sorted_T_edges,
            #     two_view_reports=two_view_reports,
            #     title="Title",
            #     save_fpath="",
            #     cameras_gt=None,
            # )

            unused_edges = set(i2Ri1_dict.keys()) - set(sorted_T_edges)

            clean_edges_to_add = []

            cycle_errors = []
            gt_errors = []

            # Add an edge. Compute cycle error.
            for unused_edge_idx, unused_edge in enumerate(unused_edges):
                #print(f"Unused edge index={unused_edge_idx}")
                # Networkx returns them in unsorted order.
                i1, i2 = sorted(unused_edge)
                T_augmented = copy.deepcopy(T)
                num_inliers = two_view_reports[(i1, i2)].num_inliers_est_model
                weight = num_inliers / total_num_inliers
                T_augmented.add_edge(i1, i2, weight=weight)
                try:
                    cycle_path = list(nx.find_cycle(T_augmented, orientation="original"))
                except:
                    import pdb; pdb.set_trace()

                R = Rot3()
                for (i1, i2, direction) in cycle_path:
                    if i1 < i2:
                        R = R.compose(i2Ri1_dict[(i1, i2)])
                        gt_error = two_view_reports[(i1, i2)].R_error_deg
                    else:
                        R = R.compose(i2Ri1_dict[(i2, i1)].inverse())
                        gt_error = two_view_reports[(i2, i1)].R_error_deg

                cycle_error = comp_utils.compute_relative_rotation_angle(Rot3(), R)

                error_epsilon = 0.75
                acceptance_threshold_deg = np.sqrt(len(cycle_path)) * error_epsilon
                # print(
                #     f"Cycle error: {cycle_error:.2f} vs acceptance thresh {acceptance_threshold_deg:.2f} for length {len(cycle_path)}"
                # )
                if cycle_error < acceptance_threshold_deg:
                    clean_edges_to_add.append(unused_edge)

                cycle_errors.append(cycle_error)
                gt_errors.append(gt_error)

            # plt.figure(figsize=(10, 10))
            # plt.ylabel("GT error (deg)")
            # plt.xlabel("Cycle error (deg)")
            # plt.scatter(cycle_errors, gt_errors, 10, color="r", marker=".")
            # #plt.savefig(output_dir / "n_length_cycle_error_vs_R_error_deg.jpg", dpi=500)
            # plt.show()

            # Check average rotation consistency error, by running Shonan on this.
            filtered_edges = set(clean_edges_to_add).union(set(T.edges()))
            sorted_filtered_edges = [tuple(sorted([i1, i2])) for (i1, i2) in filtered_edges]
            i2Ri1_dict_filtered = {(i1, i2): i2Ri1_dict[(i1, i2)] for (i1, i2) in sorted_filtered_edges}
            num_images = len(keypoints)
            wRi_list = self._rot_avg_module.run_rotation_averaging(
                num_images=num_images,
                i2Ri1_dict=i2Ri1_dict_filtered,
                i1Ti2_priors={},
            )
            metrics_group = self._rot_avg_module.evaluate(
                wRi_computed=wRi_list,
                wTi_gt=[None] * num_images,
                i2Ri1_dict=i2Ri1_dict_filtered,
            )
            rot_avg_metrics_dict = metrics_group.get_metrics_as_dict()["rotation_averaging_metrics"]

            # Check consistency statistics.
            consistency_avg_error = rot_avg_metrics_dict["relative_rotation_angle_consistency_error_deg"]["summary"]["mean"]
            consistency_max_error = rot_avg_metrics_dict["relative_rotation_angle_consistency_error_deg"]["summary"]["max"]
            
            print(f"Consistency avg error: {consistency_avg_error:.2f}")
            print(f"Consistency max error: {consistency_max_error:.2f}")

        return sorted_filtered_edges




def main():

    fpath = "/Users/johnlambert/Documents/2023_10_19_0013_spanning_tree_view_graph_estimator/20231019_041341/20231019_041341__gerrard-hall-100__results__num_matched5__maxframelookahead10__760p__unified_sift/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    # fpath = "/Users/johnlambert/Downloads/gtsfm_2023_07_08/gtsfm/door_results_2023_10_18/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    data = io_utils.read_json_file(fpath)

    i2Ri1_dict = {}
    two_view_reports_dict = {}

    for entry in data:
        i1 = entry["i1"]
        i2 = entry["i2"]
        rot_dict = entry["rotation"]
        qw = rot_dict["qw"]
        qx = rot_dict["qx"]
        qy = rot_dict["qy"]
        qz = rot_dict["qz"]

        i2Ri1 = Rot3(float(qw), float(qx), float(qy), float(qz))
        i2Ri1_dict[(i1, i2)] = i2Ri1

        # import pdb; pdb.set_trace()
        # print("here")

        two_view_reports_dict[(i1, i2)] = TwoViewEstimationReport(
            v_corr_idxs=None,
            num_inliers_est_model=entry["num_inliers_est_model"],
            R_error_deg=entry["rotation_angular_error"],
            U_error_deg=entry["translation_angular_error"],
        )

    estimator = SpanningTreeViewGraphEstimator()
    estimator.run(
        i2Ri1_dict=i2Ri1_dict,
        i2Ui1_dict={},
        calibrations=[],
        corr_idxs_i1i2={},
        keypoints=[Keypoints(coordinates=np.zeros((0, 2)))] * 100,
        two_view_reports=two_view_reports_dict,
    )


if __name__ == "__main__":
    main()
