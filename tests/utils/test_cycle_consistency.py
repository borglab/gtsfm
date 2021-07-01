
"""Unit tests to ensure correctness of cycle triplet extraction and cycle error computation.

Author: John Lambert
"""

import numpy as np
from gtsam import Rot3
from scipy.spatial.transform import Rotation

import gtsfm.utils.cycle_consistency as cycle_utils
from gtsfm.two_view_estimator import TwoViewEstimationReport


def test_extract_triplets_adjacency_list_intersection1() -> None:
    """Ensure triplets are recovered accurately.

    Consider the following undirected graph with 1 cycle:

    0 ---- 1
          /|
         / |
        /  |
      2 -- 3
           |
           |
           4
    """
    i2Ri1_dict = {
        (0, 1): Rot3(),
        (1, 2): Rot3(),
        (2, 3): Rot3(),
        (1, 3): Rot3(),
        (3, 4): Rot3(),
    }

    for extraction_fn in [cycle_utils.extract_triplets_adjacency_list_intersection, cycle_utils.extract_triplets_n3]:

        triplets = extraction_fn(i2Ri1_dict)
        assert len(triplets) == 1
        assert triplets[0] == (1, 2, 3)
        assert isinstance(triplets, list)


def test_extract_triplets_adjacency_list_intersection2() -> None:
    """Ensure triplets are recovered accurately.

	Consider the following undirected graph with 2 cycles:

	0 ---- 1
	      /|\
	     / | \
	    /  |  \
	  2 -- 3 -- 5
	       |
	       |
	       4
	"""
    i2Ri1_dict = {
        (0, 1): Rot3(),
        (1, 2): Rot3(),
        (2, 3): Rot3(),
        (1, 3): Rot3(),
        (3, 4): Rot3(),
        (1, 5): Rot3(),
        (3, 5): Rot3(),
    }

    for extraction_fn in [cycle_utils.extract_triplets_adjacency_list_intersection, cycle_utils.extract_triplets_n3]:

        triplets = extraction_fn(i2Ri1_dict)
        assert len(triplets) == 2
        assert triplets[0] == (1, 2, 3)
        assert triplets[1] == (1, 3, 5)

        assert isinstance(triplets, list)


def test_compute_cycle_error_known_GT() -> None:
    """Ensure cycle error is computed correctly within a triplet, when ground truth is known.

    Imagine 3 poses, all centered at the origin, at different orientations.

    Ground truth poses:
       Let i0 face along +x axis (0 degrees in yaw)
       Let i2 have a 30 degree rotation from the +x axis.
       Let i4 have a 90 degree rotation from the +x axis.

    However, suppose one edge measurement is corrupted (from i0 -> i4) by 5 degrees.
    """
    i2Ri0 = Rot3(Rotation.from_euler("y", 30, degrees=True).as_matrix())
    i4Ri2 = Rot3(Rotation.from_euler("y", 60, degrees=True).as_matrix())
    i4Ri0 = Rot3(Rotation.from_euler("y", 95, degrees=True).as_matrix())

    cycle_nodes = [0, 2, 4]
    i2Ri1_dict = {
        (0,2): i2Ri0, # edge i
        (2,4): i4Ri2, # edge j
        (0,4): i4Ri0, # edge k
    }

    two_view_reports_dict = {}
    # rest of attributes will default to None
    two_view_reports_dict[(0,4)] = TwoViewEstimationReport(
        v_corr_idxs = np.array([]), # dummy array
        num_inliers_est_model = 10, # dummy value
        num_H_inliers=0,
        H_inlier_ratio=0,
        R_error_deg=5,
        U_error_deg=0
    )

    two_view_reports_dict[(0,2)] = TwoViewEstimationReport(
        v_corr_idxs = np.array([]), # dummy array
        num_inliers_est_model = 10, # dummy value
        num_H_inliers=0,
        H_inlier_ratio=0,
        R_error_deg=0,
        U_error_deg=0
    )

    two_view_reports_dict[(2,4)] = TwoViewEstimationReport(
        v_corr_idxs = np.array([]), # dummy array
        num_inliers_est_model = 10, # dummy value
        num_H_inliers=0,
        H_inlier_ratio=0,
        R_error_deg=0,
        U_error_deg=0
    )

    cycle_error, max_rot_error, max_trans_error = cycle_utils.compute_cycle_error(
        i2Ri1_dict,
        cycle_nodes,
        two_view_reports_dict
    )
    
    assert np.isclose(cycle_error, 5)
    assert np.isclose(max_rot_error, 5)
    assert max_trans_error == 0


# def main():

#     # edges = read_json_file("/Users/johnlambert/Downloads/gtsfm-skynet-2021-06-05/gtsfm/result_metrics/frontend_full.json")
#     edges = read_json_file("/Users/johnlambert/Documents/gtsfm/result_metrics/frontend_full.json")
#     num_images = 12

#     i2Ui1_dict = {}
#     i2Ri1_dict = {}

#     two_view_reports_dict = {}

#     for e_info in edges:

#         i1 = e_info["i1"]
#         i2 = e_info["i2"]

#         if e_info["i2Ri1"]:
#             coeffs_dict = e_info["i2Ri1"]
#             qx, qy, qz, qw = coeffs_dict["qx"], coeffs_dict["qy"], coeffs_dict["qz"], coeffs_dict["qw"]
#             i2Ri1 = Rot3(Rotation.from_quat([qx, qy, qz, qw]).as_matrix())
#         else:
#             i2Ri1 = None

#         i2Ri1_dict[(i1, i2)] = i2Ri1

#         if e_info["i2Ui1"]:
#             i2Ui1 = Unit3(np.array(e_info["i2Ui1"]))
#         else:
#             i2Ui1 = None
#         i2Ui1_dict[(i1, i2)] = i2Ui1

#         report_dict = {
#             "R_error_deg": e_info["rotation_angular_error"],
#             "U_error_deg": e_info["translation_angular_error"],
#         }
#         from types import SimpleNamespace

#         two_view_reports_dict[(i1, i2)] = SimpleNamespace(**report_dict)

#     filter_to_cycle_consistent_edges(i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, visualize=True)




if __name__ == "__main__":

    # test_extract_triplets_adjacency_list_intersection1()
    # test_extract_triplets_adjacency_list_intersection2()

    test_compute_cycle_error_known_GT()

