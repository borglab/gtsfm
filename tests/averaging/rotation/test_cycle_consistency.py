"""Unit tests to ensure correctness of cycle triplet extraction and cycle error computation.

Author: John Lambert
"""
import time
from typing import Dict, List, Tuple

import numpy as np
from gtsam import Rot3, Unit3

import gtsfm.averaging.rotation.cycle_consistency as cycle_utils
from gtsfm.two_view_estimator import TwoViewEstimationReport


def test_extract_triplets_1() -> None:
    """Ensure triplets are recovered accurately via intersection of adjacency lists.

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

    for extraction_fn in [cycle_utils.extract_triplets, extract_triplets_brute_force]:

        triplets = extraction_fn(i2Ri1_dict)
        assert len(triplets) == 1
        assert triplets[0] == (1, 2, 3)
        assert isinstance(triplets, list)


def test_extract_triplets_2() -> None:
    """Ensure triplets are recovered accurately via intersection of adjacency lists.

	Consider the following undirected graph with 2 cycles. The cycles share an edge:

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

    for extraction_fn in [cycle_utils.extract_triplets, extract_triplets_brute_force]:

        triplets = extraction_fn(i2Ri1_dict)
        assert len(triplets) == 2
        assert triplets[0] == (1, 2, 3)
        assert triplets[1] == (1, 3, 5)

        assert isinstance(triplets, list)


def test_extract_triplets_3() -> None:
    """Ensure triplets are recovered accurately via intersection of adjacency lists.

    Consider the following undirected graph with 2 cycles. The cycles share a node:

    0 ---- 1
          /|
         / |
        /  |
      2 -- 3
           |\
           | \
           |  \
           4 -- 5
    """
    i2Ri1_dict = {
        (0, 1): Rot3(),
        (1, 2): Rot3(),
        (2, 3): Rot3(),
        (1, 3): Rot3(),
        (3, 4): Rot3(),
        (3, 5): Rot3(),
        (4, 5): Rot3(),
    }

    for extraction_fn in [cycle_utils.extract_triplets, extract_triplets_brute_force]:
        triplets = extraction_fn(i2Ri1_dict)
        assert len(triplets) == 2

        assert triplets[0] == (3, 4, 5)
        assert triplets[1] == (1, 2, 3)

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
    i2Ri0 = Rot3.Ry(np.deg2rad(30))
    i4Ri2 = Rot3.Ry(np.deg2rad(60))
    i4Ri0 = Rot3.Ry(np.deg2rad(95))

    cycle_nodes = [0, 2, 4]
    i2Ri1_dict = {
        (0, 2): i2Ri0,  # edge i
        (2, 4): i4Ri2,  # edge j
        (0, 4): i4Ri0,  # edge k
    }

    def make_dummy_report(R_error_deg: float, U_error_deg: float) -> TwoViewEstimationReport:
        """Create a dummy report about a two-view verification result."""
        # rest of attributes will default to None
        return TwoViewEstimationReport(
            v_corr_idxs=np.array([]),  # dummy array
            num_inliers_est_model=10,  # dummy value
            num_H_inliers=0,
            H_inlier_ratio=0,
            R_error_deg=R_error_deg,
            U_error_deg=U_error_deg,
        )

    two_view_reports_dict = {}
    two_view_reports_dict[(0, 4)] = make_dummy_report(R_error_deg=5, U_error_deg=0)
    two_view_reports_dict[(0, 2)] = make_dummy_report(R_error_deg=0, U_error_deg=0)
    two_view_reports_dict[(2, 4)] = make_dummy_report(R_error_deg=0, U_error_deg=0)

    cycle_error, max_rot_error, max_trans_error = cycle_utils.compute_cycle_error(
        i2Ri1_dict, cycle_nodes, two_view_reports_dict
    )

    assert np.isclose(cycle_error, 5)
    assert np.isclose(max_rot_error, 5)
    assert max_trans_error == 0


def test_compute_cycle_error_unknown_GT() -> None:
    """Ensure cycle error is computed correctly within a triplet, when ground truth is known.

    Imagine 3 poses, all centered at the origin, at different orientations.

    Ground truth poses:
       Let i0 face along +x axis (0 degrees in yaw)
       Let i2 have a 30 degree rotation from the +x axis.
       Let i4 have a 90 degree rotation from the +x axis.

    However, suppose one edge measurement is corrupted (from i0 -> i4) by 5 degrees.
    """
    i2Ri0 = Rot3.Ry(np.deg2rad(30))
    i4Ri2 = Rot3.Ry(np.deg2rad(60))
    i4Ri0 = Rot3.Ry(np.deg2rad(95))

    cycle_nodes = [0, 2, 4]
    i2Ri1_dict = {
        (0, 2): i2Ri0,  # edge i
        (2, 4): i4Ri2,  # edge j
        (0, 4): i4Ri0,  # edge k
    }

    def make_dummy_report() -> TwoViewEstimationReport:
        """Create a dummy report about a two-view verification result."""
        # rest of attributes will default to None
        return TwoViewEstimationReport(
            v_corr_idxs=np.array([]),  # dummy array
            num_inliers_est_model=10,  # dummy value
            num_H_inliers=0,
            H_inlier_ratio=0,
        )

    two_view_reports_dict = {}
    two_view_reports_dict[(0, 4)] = make_dummy_report()
    two_view_reports_dict[(0, 2)] = make_dummy_report()
    two_view_reports_dict[(2, 4)] = make_dummy_report()

    cycle_error, max_rot_error, max_trans_error = cycle_utils.compute_cycle_error(
        i2Ri1_dict, cycle_nodes, two_view_reports_dict
    )

    assert np.isclose(cycle_error, 5)
    assert max_rot_error is None
    assert max_trans_error is None


def test_filter_to_cycle_consistent_edges() -> None:
    """Ensure correct edges are kept in a 2-triplet scenario.

    Scenario Ground Truth: consider 5 camera poses in a line, connected as follows, all with identity rotations:

    Spatial layout:
      _________    ________
     /         \\ /        \
    i4 -- i3 -- i2 -- i1 -- i0

    Topological layout:

     i4          i0
           i2
     i3          i1

    In the measurements, suppose, the measurement for (i2,i4) was corrupted by 15 degrees.
    """
    i2Ri1_dict = {
        (0, 1): Rot3(),
        (1, 2): Rot3(),
        (0, 2): Rot3(),
        (2, 3): Rot3(),
        (3, 4): Rot3(),
        (2, 4): Rot3.Ry(np.deg2rad(15)),
    }
    i2Ui1_dict = {
        (0, 1): Unit3(np.array([1, 0, 0])),
        (1, 2): Unit3(np.array([1, 0, 0])),
        (0, 2): Unit3(np.array([1, 0, 0])),
        (2, 3): Unit3(np.array([1, 0, 0])),
        (3, 4): Unit3(np.array([1, 0, 0])),
        (2, 4): Unit3(np.array([1, 0, 0])),
    }
    # assume no ground truth information available at runtime
    two_view_reports_dict = {}

    for (i1, i2) in i2Ri1_dict.keys():
        two_view_reports_dict[(i1, i2)] = TwoViewEstimationReport(
            v_corr_idxs=np.array([]),  # dummy array
            num_inliers_est_model=10,  # dummy value
            num_H_inliers=0,
            H_inlier_ratio=0,
        )

    i2Ri1_dict_consistent, i2Ui1_dict_consistent = cycle_utils.filter_to_cycle_consistent_edges(
        i2Ri1_dict, i2Ui1_dict, two_view_reports_dict, visualize=True
    )
    # non-self-consistent triplet should have been removed
    expected_keys = {(0, 1), (1, 2), (0, 2)}
    assert set(i2Ri1_dict_consistent.keys()) == expected_keys


def extract_triplets_brute_force(i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> List[Tuple[int, int, int]]:
    """Use triple for-loop to find triplets from a graph G=(V,E) in O(n^3) time.

    **Much** slower implementation for large graphs, when compared to `extract_triplets()` that uses intersection of adjacency lists.
    Used to check correctness inside the unit test.

    Args:
        i2Ri1_dict: mapping from image pair indices to relative rotation.

    Returns:
        triplets: 3-tuples of nodes that form a cycle. Nodes of each triplet are provided in sorted order.
    """
    triplets = set()

    for (i1, i2), i2Ri1 in i2Ri1_dict.items():
        if i2Ri1 is None:
            continue

        for (j1, j2), j2Rj1 in i2Ri1_dict.items():
            if j2Rj1 is None:
                continue

            for (k1, k2), k2Rk1 in i2Ri1_dict.items():
                if k2Rk1 is None:
                    continue

                # check how many nodes are spanned by these 3 edges
                cycle_nodes = set([i1, i2]).union(set([j1, j2])).union(set([k1, k2]))
                # sort them in increasing order
                cycle_nodes = tuple(sorted(cycle_nodes))

                # nodes cannot be repeated
                unique_edges = set([(i1, i2), (j1, j2), (k1, k2)])
                edges_are_unique = len(unique_edges) == 3

                if len(cycle_nodes) == 3 and edges_are_unique:
                    triplets.add(cycle_nodes)

    return list(triplets)


def test_triplet_extraction_correctness_runtime() -> None:
    """Ensure that for large graphs, the adjacency-list-based algorithm is faster and still correct,
    when compared with the brute-force O(n^3) implementation.
    """
    num_pairs = 100
    # suppose we have 200 images for a scene
    pairs = np.random.randint(low=0, high=200, size=(num_pairs, 2))
    # i1 < i2 by construction inside loader classes
    pairs = np.sort(pairs, axis=1)

    # remove edges that would represent self-loops, i.e. (i1,i1) is not valid for a measurement
    invalid = pairs[:, 0] == pairs[:, 1]
    pairs = pairs[~invalid]
    num_valid_pairs = pairs.shape[0]

    i2Ri1_dict = {(pairs[i, 0], pairs[i, 1]): Rot3() for i in range(num_valid_pairs)}

    start = time.time()
    triplets = cycle_utils.extract_triplets(i2Ri1_dict)
    end = time.time()
    duration = end - start

    # Now, compare with the brute force method
    start = time.time()
    triplets_bf = extract_triplets_brute_force(i2Ri1_dict)
    end = time.time()
    duration_bf = end - start

    assert duration < duration_bf
    assert set(triplets) == set(triplets_bf)
