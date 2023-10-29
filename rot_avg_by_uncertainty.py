""" """

import dask
from gtsam import Rot3, Unit3
import numpy as np
from pathlib import Path

from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
import gtsfm.utils.io as io_utils
from gtsfm.loader.colmap_loader import ColmapLoader

from gtsfm.view_graph_estimator.cycle_consistent_rotation_estimator import CycleConsistentRotationViewGraphEstimator, EdgeErrorAggregationCriterion

import networkx as nx

def main() -> None:

    images_dir = "/Users/johnlambert/Downloads/gerrard-hall-100/images"
    colmap_files_dirpath = "/Users/johnlambert/Downloads/gerrard-hall-100/colmap-3.7-sparse-txt-2023-07-27"

    loader = ColmapLoader(
        colmap_files_dirpath=colmap_files_dirpath,
        images_dir=images_dir,
        max_resolution=760,
    )
    gt_wTi_list = loader.get_gt_poses()

    #num_images = 128
    #fpath = "/Users/johnlambert/Documents/2023_10_19_0013_spanning_tree_view_graph_estimator/20231019_041341/20231019_041341__south-building-128__results__num_matched5__maxframelookahead10__760p__unified_superglue/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"

    num_images = 98 # 100
    fpath = "/Users/johnlambert/Documents/2023_10_19_0013_spanning_tree_view_graph_estimator/20231019_041341/20231019_041341__gerrard-hall-100__results__num_matched5__maxframelookahead10__760p__unified_sift/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    # fpath = "/Users/johnlambert/Downloads/gtsfm_2023_07_08/gtsfm/door_results_2023_10_18/result_metrics/two_view_report_POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT.json"
    data = io_utils.read_json_file(fpath)

    i2Ri1_dict = {}
    two_view_reports_dict = {}
    frontend_uncertainty_dict = {}

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

        two_view_reports_dict[(i1, i2)] = TwoViewEstimationReport(
            v_corr_idxs=None,
            num_inliers_est_model=entry["num_inliers_est_model"],
            R_error_deg=entry["rotation_angular_error"],
            U_error_deg=entry["translation_angular_error"],
        )

        #uncertainty = 1 / two_view_reports_dict[(i1, i2)].num_inliers_est_model
        uncertainty = np.exp(1 / (two_view_reports_dict[(i1, i2)].num_inliers_est_model / 50))
        # 
        # uncertainty = two_view_reports_dict[(i1, i2)].R_error_deg ** 3
        # # 
        # print(f"Inliers {two_view_reports_dict[(i1, i2)].num_inliers_est_model} Uncertainty: ", uncertainty)
        # if uncertainty < 0.05:

        #     uncertainty = max(uncertainty, 0.05)
        #     print(f"\tClamped to {uncertainty}")

        assert uncertainty > 0.01
        # assert uncertainty < 200

        frontend_uncertainty_dict[(i1,i2)] = 1.0# uncertainty
        

    (
        viewgraph_i2Ri1_graph,
        viewgraph_i2Ui1_graph,
        viewgraph_v_corr_idxs_graph,
        viewgraph_two_view_reports_graph,
        viewgraph_estimation_metrics,
        frontend_uncertainty_dict,
    ) = CycleConsistentRotationViewGraphEstimator(
        edge_error_aggregation_criterion=EdgeErrorAggregationCriterion.MEDIAN_EDGE_ERROR
    ).create_computation_graph(
        i2Ri1_dict,
        i2Ui1_dict={(i1,i2): Unit3() for (i1,i2) in i2Ri1_dict.keys() },
        calibrations=[],
        corr_idxs_i1i2={(i1,i2): np.zeros((0,2)) for (i1,i2) in i2Ri1_dict.keys() },
        keypoints=[],
        two_view_reports=two_view_reports_dict,
        debug_output_dir=None,
    )
    i2Ri1_dict, frontend_uncertainty_dict = dask.compute(viewgraph_i2Ri1_graph, frontend_uncertainty_dict)

    # frontend_uncertainty_dict = {k: 1 if v > 500 else 10 for k,v in frontend_uncertainty_dict.items()}

    # import matplotlib.pyplot as plt
    # for (i1,i2) in i2Ri1_dict.keys():

    #     uncertainty = frontend_uncertainty_dict[(i1,i2)]

    #     print(f"Inliers {two_view_reports_dict[(i1, i2)].num_inliers_est_model} Uncertainty: ", uncertainty)

    #     plt.scatter(
            
    #         two_view_reports_dict[(i1, i2)].R_error_deg,
    #         two_view_reports_dict[(i1, i2)].num_inliers_est_model,
    #         #uncertainty,
    #         10,
    #         color='r',
    #         marker='.'
    #     )
    # plt.xlabel("R error deg")
    # plt.ylabel('uncertainty')
    # plt.show()

    # frontend_uncertainty_dict = {
    #     (i1,i2): 1 if two_view_reports_dict[(i1, i2)].num_inliers_est_model > 500 else 10
    #     for (i1,i2) in i2Ri1_dict.keys()
    # }

    # i2Ri1_dict_final = {}
    # for (i1,i2) in i2Ri1_dict.keys():
    #     if two_view_reports_dict[(i1, i2)].num_inliers_est_model > 500:
    #         i2Ri1_dict_final[(i1,i2)] = i2Ri1_dict[(i1,i2)]

    total_num_inliers = 1000

    G = nx.Graph()
    for (i1, i2) in i2Ri1_dict:
        num_inliers = two_view_reports_dict[(i1, i2)].num_inliers_est_model
        weight = num_inliers / total_num_inliers
        #print("Edge weight: ", weight)
        G.add_edge(i1, i2, weight=weight)
    T = nx.maximum_spanning_tree(G)

    T_edges = T.edges()
    sorted_T_edges = [tuple(sorted([i1, i2])) for (i1, i2) in T_edges]

    i2Ri1_dict_final = {(i1,i2): i2Ri1_dict[(i1,i2)] for (i1,i2) in sorted_T_edges}
    frontend_uncertainty_dict = {(i1,i2): 1 for (i1,i2) in i2Ri1_dict_final}

    rot_avg_module = ShonanRotationAveraging()
    wRis, metrics = rot_avg_module._run_rotation_averaging_base(
        num_images,
        i2Ri1_dict=i2Ri1_dict_final,
        i1Ti2_priors={},
        frontend_uncertainty_dict=frontend_uncertainty_dict,
        wTi_gt=gt_wTi_list,
    )
    metrics_dict = metrics.get_metrics_as_dict()
    #print("Metrics:", )

    print(metrics_dict['rotation_averaging_metrics']['num_rotations_computed'])
    print(metrics_dict['rotation_averaging_metrics']['rotation_angle_error_deg']['summary'])



if __name__ == "__main__":
    main()