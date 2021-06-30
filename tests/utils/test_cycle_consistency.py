
"""Unit tests to ensure correctness of cycle triplet extraction and cycle error computation.

Author: John Lambert
"""

from gtsam import Rot3

import gtsfm.utils.cycle_consistency as cycle_utils


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

        import pdb; pdb.set_trace()
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



def get_quaternion_coeff_dict(R: np.ndarray):
    """ """
    qx, qy, qz, qw = Rotation.from_matrix(R.matrix()).as_quat().tolist()

    coeffs_dict = {"qx": qx, "qy": qy, "qz": qz, "qw": qw}
    return coeffs_dict


def test_compute_cycle_error():
    """
    0 -> 4
    2 -> 4
    0 -> 2
    """

    wTi0 = Pose3()
    wTi2 = Pose3()
    wTi4 = Pose3()

    i4Ri0 = wTi4.between(wTi0).rotation()
    i4Ri2 = wTi4.between(wTi2).rotation()
    i2Ri0 = wTi2.between(wTi0).rotation()

    i4Ri0 = Rot3(Rotation.from_euler("y", 95, degrees=True).as_matrix())
    i4Ri2 = Rot3(Rotation.from_euler("y", 60, degrees=True).as_matrix())
    i2Ri0 = Rot3(Rotation.from_euler("y", 30, degrees=True).as_matrix())

    cycle_nodes = [4, 2, 0]
    edge_i_info = {
        "i1": 0,
        "i2": 4,
        "rotation_angular_error": 0,
        "translation_angular_error": 0,
        "i2Ri1": get_quaternion_coeff_dict(i4Ri0),
    }
    edge_j_info = {
        "i1": 2,
        "i2": 4,
        "rotation_angular_error": 0,
        "translation_angular_error": 0,
        "i2Ri1": get_quaternion_coeff_dict(i4Ri2),
    }
    edge_k_info = {
        "i1": 0,
        "i2": 2,
        "rotation_angular_error": 0,
        "translation_angular_error": 0,
        "i2Ri1": get_quaternion_coeff_dict(i2Ri0),
    }

    cycle_error, average_rot_error, average_trans_error = compute_cycle_error(
        cycle_nodes, edge_i_info, edge_j_info, edge_k_info
    )
    assert np.isclose(cycle_error, 5)


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

    test_extract_triplets_adjacency_list_intersection1()
    test_extract_triplets_adjacency_list_intersection2()

    test_compute_cycle_error()

