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
from gtsfm.averaging.rotation.spanning_tree import SpanningTreeRotationEstimator

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
        #self._rot_avg_module = ShonanRotationAveraging()
        self._rot_avg_module = SpanningTreeRotationEstimator()


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
        T = nx.maximum_spanning_tree(G)

        T_edges = T.edges()
        sorted_T_edges = [tuple(sorted([i1, i2])) for (i1, i2) in T_edges]
        avg_error = np.mean([two_view_reports[(i1, i2)].R_error_deg for (i1, i2) in sorted_T_edges])
        max_error = np.amax([two_view_reports[(i1, i2)].R_error_deg for (i1, i2) in sorted_T_edges])
        for (i1,i2) in sorted_T_edges:
            print(f"({i1},{i2}) -> {two_view_reports[(i1, i2)].R_error_deg:.2f} GT error")
        print(f"Avg MaxST error: {avg_error}")
        print(f"Max MaxST error: {max_error}")

        

        # Sample a triplet with the highest number of inliers.
        triplets = graph_utils.extract_cyclic_triplets_from_edges(list(i2Ri1_dict.keys()))

        # Sort them by #inliers.
        num_inliers_all_cycles = []
        for (i0, i1, i2) in triplets:
            num_inliers_cycle = (
                two_view_reports[(i0, i1)].num_inliers_est_model
                 + two_view_reports[(i1, i2)].num_inliers_est_model
                 + two_view_reports[(i0, i2)].num_inliers_est_model
            )
            num_inliers_all_cycles.append(num_inliers_cycle)

        sorted_idxs = np.argsort(-1 * np.array(num_inliers_all_cycles))
        sorted_cycles = [triplets[idx] for idx in sorted_idxs]
        
        triplet_is_used = np.zeros(len(triplets), dtype=bool)

        clean_G = nx.Graph()

        for triplet_cycle_error_threshold in [1.5, 2.5, 5.0]:

            for epsilon in [0.5, 4]: # 0.75, 1.0, 1.25, 1.5, 2, 2.5, 3, 3.5,

                for triplet_idx, (i0, i1, i2) in enumerate(sorted_cycles):
                    
                    if triplet_is_used[triplet_idx]:
                        continue

                    triplet_cycle_error = comp_utils.compute_cyclic_rotation_error(
                        i1Ri0=i2Ri1_dict[(i0, i1)], i2Ri1=i2Ri1_dict[(i1, i2)], i2Ri0=i2Ri1_dict[(i0, i2)]
                    )
                    gt_errors = [
                        two_view_reports[(i0,i1)].R_error_deg,
                        two_view_reports[(i1,i2)].R_error_deg,
                        two_view_reports[(i0,i2)].R_error_deg
                    ]
                    print(f"cycle_error={triplet_cycle_error:.2f} vs. GT errors={np.round(gt_errors,2)}")
                    if triplet_cycle_error < triplet_cycle_error_threshold:
                        print(f"\tClean graph now has {len(clean_G)} nodes, on Triplet={triplet_idx}.")
                        augmented_clean_G = copy.deepcopy(clean_G)
                        detected_cycles = sorted(nx.simple_cycles(augmented_clean_G, length_bound=5))
                        print(f"Found {len(detected_cycles)} cycles in the augmented graph")

                        derived_cycle_errors = []
                        cycle_lengths = []
                        for detected_cycle_idx, ordered_cycle_nodes in enumerate(detected_cycles):
                            derived_cycle_error = compute_cycle_error(ordered_cycle_nodes, i2Ri1_dict)
                            cycle_len = len(ordered_cycle_nodes)
                            
                            derived_cycle_errors.append(derived_cycle_error)
                            cycle_lengths.append(cycle_len)

                        success = True
                        #epsilon = 0.5
                        for detected_cycle_idx, (cycle_len, derived_cycle_error) in enumerate(zip(cycle_lengths, derived_cycle_errors)):
                            if derived_cycle_error > np.sqrt(cycle_len) * epsilon:
                                print(f"\t\tReject cycle {detected_cycle_idx}: len-{cycle_len} derived_cycle_error {derived_cycle_error:.2f} > {np.sqrt(cycle_len) * epsilon:.2f} thresh")
                                success = False
                            else:
                                print(f"\tAcceptable derived cycle {detected_cycle_idx} error for len-{cycle_len}: {derived_cycle_error:.2f}")

                        if success:
                            triplet_is_used[triplet_idx] = True
                            clean_G.add_edge(i0, i1)
                            clean_G.add_edge(i1, i2)
                            clean_G.add_edge(i0, i2)
                            

                    else:
                        print("\tSkip this triplet.")
                    #cycle_path = list(nx.find_cycle(T_augmented, orientation="original"))
                
                percent_triplets_used = triplet_is_used.mean() * 100
                print(f"% triplets used: {percent_triplets_used:.2f} % w/ epsilon={epsilon}")
                print(f"\tClean graph now has {len(clean_G)} nodes")
                import time
                time.sleep(10)
                #import pdb; pdb.set_trace()
            #     
        #     cycle_gt_errors = []
        #     R = Rot3() # think of as i2Ri2.
        #     for (i1, i2, direction) in cycle_path:
        #         if i1 < i2:
        #             R = R.compose(i2Ri1_dict[(i1, i2)])
        #             cycle_gt_errors.append(two_view_reports[(i1, i2)].R_error_deg)
        #         else:
        #             R = R.compose(i2Ri1_dict[(i2, i1)].inverse())
        #             cycle_gt_errors.append(two_view_reports[(i2, i1)].R_error_deg)

        #     cycle_error = comp_utils.compute_relative_rotation_angle(Rot3(), R)


        #, length_bound=3)) #length_bound=None)
        #import pdb; pdb.set_trace()


        # consistency_avg_error, consistency_max_error = self._compute_rotation_consistency_stats(
        #     num_images = len(keypoints),
        #     i2Ri1_dict_filtered={(i1,i2): i2Ri1_dict[(i1,i2)] for (i1,i2) in sorted_T_edges},
        #     i2Ri1_dict=i2Ri1_dict,
        # )
        # print(f"Max ST Consistency avg error: {consistency_avg_error:.2f}")
        # print(f"Max ST Consistency max error: {consistency_max_error:.2f}")

        # unused_edges = set(i2Ri1_dict.keys()) - set(sorted_T_edges)

        # clean_edges_to_add = []

        # cycle_errors = []
        # gt_errors = []

        # outlier_edges = []
        # outlier_edge_weights = []

        # print("\n\nClassify")
        # Add an edge. Compute cycle error.
        # for unused_edge_idx, unused_edge in enumerate(unused_edges):
        #     #print(f"Unused edge index={unused_edge_idx}")
        #     # Networkx returns them in unsorted order.
        #     i1, i2 = sorted(unused_edge)
        #     T_augmented = copy.deepcopy(T)
        #     num_inliers = two_view_reports[(i1, i2)].num_inliers_est_model
        #     weight = num_inliers / total_num_inliers
        #     T_augmented.add_edge(i1, i2, weight=weight)
        #     try:
        #         cycle_path = list(nx.find_cycle(T_augmented, orientation="original"))
        #     except:
        #         import pdb; pdb.set_trace()

        #     gt_error = two_view_reports[unused_edge].R_error_deg
        #     cycle_gt_errors = []


        #     error_epsilon = 0.75
        #     acceptance_threshold_deg = np.sqrt(len(cycle_path)) * error_epsilon
            
        #     print_str = f"GT error {gt_error:.2f}, Cycle error: {cycle_error:.2f} vs acceptance thresh {acceptance_threshold_deg:.2f} for length {len(cycle_path)}"

        #     if cycle_error < acceptance_threshold_deg:
        #         clean_edges_to_add.append(unused_edge)
        #         #print(f"Clean classified edge: {unused_edge} ", print_str)
        #     else:

        #         # print(f"Noisy classified edge: {unused_edge} ", print_str)
        #         # print(f"{unused_edge}: cycle path: {cycle_path}", "cycle gt errors: ",  cycle_gt_errors)
        #         outlier_edges.append(unused_edge)
        #         outlier_edge_weights.append(cycle_error)
                
        #     cycle_errors.append(cycle_error)
        #     gt_errors.append(gt_error)

        
        # sorted_idxs = np.argsort(-1 * np.array(outlier_edge_weights))
        # outlier_edges_sorted = [outlier_edges[idx] for idx in sorted_idxs]

        # # plt.figure(figsize=(10, 10))
        # # plt.ylabel("GT error (deg)")
        # # plt.xlabel("Cycle error (deg)")
        # # plt.scatter(cycle_errors, gt_errors, 10, color="r", marker=".")
        # # #plt.savefig(output_dir / "n_length_cycle_error_vs_R_error_deg.jpg", dpi=500)
        # # plt.show()

        # for outlier_edge in outlier_edges_sorted:

        #     G_new = nx.Graph()
        #     G_new.add_edge(outlier_edge[0], outlier_edge[1], weight=1.0)

        #     print("Add outlier edge: ", outlier_edge)

        #     for (i1,i2) in sorted_T_edges:
        #         new_weight = G.get_edge_data(i1, i2)["weight"] / 2
        #         G_new.add_edge(i1, i2, weight=new_weight)
            
        #     new_T = nx.maximum_spanning_tree(G_new)

        #     new_T_edges = new_T.edges()
        #     sorted_new_T_edges = [tuple(sorted([i1, i2])) for (i1, i2) in new_T_edges]

        #     i2Ri1_dict_filtered = {(i1, i2): i2Ri1_dict[(i1, i2)] for (i1, i2) in sorted_new_T_edges}

        #     consistency_avg_error, consistency_max_error = self._compute_rotation_consistency_stats(
        #         num_images = len(keypoints),
        #         i2Ri1_dict_filtered=i2Ri1_dict_filtered,
        #         i2Ri1_dict=i2Ri1_dict,
        #     )
        #     print(f"\tConsistency avg error: {consistency_avg_error:.2f}")
        #     print(f"\tConsistency max error: {consistency_max_error:.2f}")

        #     avg_error = np.mean([two_view_reports[(i1, i2)].R_error_deg for (i1, i2) in sorted_new_T_edges])
        #     max_error = np.amax([two_view_reports[(i1, i2)].R_error_deg for (i1, i2) in sorted_new_T_edges])
        #     print(f"\tAvg New ST error: {avg_error}")
        #     print(f"\tMax New ST error: {max_error}")

        #     # graph_utils.draw_view_graph_topology(
        #     #     edges=sorted_new_T_edges,
        #     #     two_view_reports=two_view_reports,
        #     #     title="Title",
        #     #     save_fpath="",
        #     #     cameras_gt=None,
        #     # )
            

        #     # Remove edge that creates cycle.

        #     # Add supposed "outlier edge" to the spanning tree.

        #     # Run Shonan, see if max consistency error goes down.



        # # Check average rotation consistency error, by running Shonan on this.
        # filtered_edges = set(clean_edges_to_add).union(set(T.edges()))
        # sorted_filtered_edges = [tuple(sorted([i1, i2])) for (i1, i2) in filtered_edges]
        # i2Ri1_dict_filtered = {(i1, i2): i2Ri1_dict[(i1, i2)] for (i1, i2) in sorted_filtered_edges}

        # consistency_avg_error, consistency_max_error = self._compute_rotation_consistency_stats(
        #     num_images = len(keypoints),
        #     i2Ri1_dict_filtered=i2Ri1_dict_filtered,
        #     i2Ri1_dict=i2Ri1_dict,
        # )
        # print(f"Consistency avg error: {consistency_avg_error:.2f}")
        # print(f"Consistency max error: {consistency_max_error:.2f}")

        return sorted_filtered_edges


    def _compute_rotation_consistency_stats(
        self,
        num_images: int,
        i2Ri1_dict_filtered: Dict[Tuple[int,int], Rot3],
        i2Ri1_dict,
    ) -> Tuple[float, float]:
        """Runs rotation averaging and checks consistency of measurements vs. global rotation estimate"""
        wRi_list = self._rot_avg_module.run_rotation_averaging(
            num_images=num_images,
            i2Ri1_dict=i2Ri1_dict_filtered,
            i1Ti2_priors={},
        )
        metrics_group = self._rot_avg_module.evaluate(
            wRi_computed=wRi_list,
            wTi_gt=[None] * len(wRi_list),
            i2Ri1_dict=i2Ri1_dict, # i2Ri1_dict_filtered,
        )
        print("Estimated ", sum([1 if wRi is not None else 0 for wRi in wRi_list]))
        rot_avg_metrics_dict = metrics_group.get_metrics_as_dict()["rotation_averaging_metrics"]

        # Check consistency statistics.
        consistency_avg_error = rot_avg_metrics_dict["relative_rotation_angle_consistency_error_deg"]["summary"]["mean"]
        consistency_max_error = rot_avg_metrics_dict["relative_rotation_angle_consistency_error_deg"]["summary"]["max"]
        return consistency_avg_error, consistency_max_error



def compute_cycle_error(ordered_cycle_nodes, i2Ri1_dict) -> float:
    """Compute cycle error for arbitrary-length cycle."""

    # Edges in the cycle path.
    edges = list(zip(ordered_cycle_nodes,(ordered_cycle_nodes[1:]+ordered_cycle_nodes[:1])))

    R = Rot3() # think of as i2Ri2.
    for (i1, i2) in edges:
        if i1 < i2:
            R = R.compose(i2Ri1_dict[(i1, i2)])
        else:
            R = R.compose(i2Ri1_dict[(i2, i1)].inverse())

    cycle_error = comp_utils.compute_relative_rotation_angle(Rot3(), R)
    return cycle_error


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
