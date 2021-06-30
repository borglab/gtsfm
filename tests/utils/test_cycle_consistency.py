
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
