"""Unit tests to ensure correctness of cycle error computation.

Author: John Lambert, Akshay Krishnan
"""
import numpy as np
from gtsam import Cal3Bundler, Rot3, Unit3

from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.view_graph_estimator.cycle_consistent_rotation_estimator import (
    CycleConsistentRotationViewGraphEstimator,
    EdgeErrorAggregationCriterion,
)


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
    # The ViewGraphEstimator assumes these dicts contain the keys corresponding to the the rotation edges.
    calibrations = {k: Cal3Bundler() for k in range(0, 5)}
    corr_idxs_i1i2 = {i1i2: np.array([]) for i1i2 in i2Ri1_dict.keys()}
    keypoints = {k: np.array([]) for k in range(0, 5)}

    # populate dummy 2-view reports
    two_view_reports = {}
    for (i1, i2) in i2Ri1_dict.keys():
        two_view_reports[(i1, i2)] = TwoViewEstimationReport(
            v_corr_idxs=np.array([]),  # dummy array
            num_inliers_est_model=10,  # dummy value
        )

    rcc_estimator = CycleConsistentRotationViewGraphEstimator(EdgeErrorAggregationCriterion.MEDIAN_EDGE_ERROR)
    viewgraph_edges = rcc_estimator.get_viewgraph_edges(
        i2Ri1_dict=i2Ri1_dict,
        i2Ui1_dict=i2Ui1_dict,
        calibrations=calibrations,
        corr_idxs_i1i2=corr_idxs_i1i2,
        keypoints=keypoints,
        two_view_reports=two_view_reports,
    )

    # non-self-consistent triplet should have been removed
    expected_edges = {(0, 1), (1, 2), (0, 2)}
    assert set(viewgraph_edges) == expected_edges
