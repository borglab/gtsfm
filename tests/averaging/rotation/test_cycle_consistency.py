"""Unit tests to ensure correctness of cycle error computation.

Author: John Lambert
"""
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pytest
from gtsam import Rot3, Unit3

import gtsfm.averaging.rotation.cycle_consistency as cycle_utils
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport


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
            R_error_deg=R_error_deg,
            U_error_deg=U_error_deg,
            inlier_ratio_H=0.5,  # dummy value
            num_inliers_H=100,  # dummy value
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
            num_inliers_est_model=10,  # dummy value,
            inlier_ratio_H=0.5,  # dummy value
            num_inliers_H=100,  # dummy value
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
    v_corr_idxs_dict = {}

    for (i1, i2) in i2Ri1_dict.keys():
        two_view_reports_dict[(i1, i2)] = TwoViewEstimationReport(
            v_corr_idxs=np.array([]),  # dummy array
            num_inliers_est_model=10,  # dummy value
            inlier_ratio_H=0.5,  # dummy value
            num_inliers_H=100,  # dummy value
        )
        v_corr_idxs_dict[(i1, i2)] = None  # fill with dummy value

    (
        i2Ri1_dict_consistent,
        i2Ui1_dict_consistent,
        v_corr_idxs_dict_consistent,
        rcc_metrics_group,
    ) = cycle_utils.filter_to_cycle_consistent_edges(
        i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict, two_view_reports_dict, visualize=True
    )
    assert isinstance(rcc_metrics_group, GtsfmMetricsGroup)
    # non-self-consistent triplet should have been removed
    expected_keys = {(0, 1), (1, 2), (0, 2)}
    assert set(i2Ri1_dict_consistent.keys()) == expected_keys
