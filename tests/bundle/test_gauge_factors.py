"""Tests for sparse gauge regularization in bundle adjustment."""

import unittest

import gtsam
import numpy as np
from gtsam.symbol_shorthand import P, X

from gtsfm.bundle.gauge_factors import add_gauge_factors


class TestGaugeFactors(unittest.TestCase):
    def test_sparse_factors_and_cost(self) -> None:
        """The only binary factor measures the initial diameter in world coordinates."""
        point_keys = [P(4), P(7)]
        camera_keys = [X(8), X(3), X(6)]
        points = np.array([[1.0, 2, 2], [-1, 0, 2]])
        centers = np.array([[1.0, 3, 4], [0, 2, 3], [4, 2, 3]])
        rotation = gtsam.Rot3.RzRyRx(0.2, 0.4, -0.5)
        values = gtsam.Values()
        for key, point in zip(point_keys, points):
            values.insert(key, point)
        for key, center in zip(camera_keys, centers):
            values.insert(key, gtsam.Pose3(rotation, center))

        graph = gtsam.NonlinearFactorGraph()
        add_gauge_factors(
            graph,
            values,
            point_keys,
            camera_keys,
            point_sigma=2.0,
            camera_y_sigma=1.0,
            camera_z_sigma=0.5,
            baseline_sigma=0.25,
        )
        scopes = [tuple(graph.at(k).keys()) for k in range(graph.size())]
        self.assertEqual({keys[0] for keys in scopes if len(keys) == 1}, set(point_keys + camera_keys))
        self.assertEqual([set(keys) for keys in scopes if len(keys) != 1], [{X(3), X(6)}])

        diameter = 4.0
        expected = (
            np.mean(np.sum(points**2, axis=1)) / 2.0**2
            + np.mean(centers[:, 1] ** 2)
            + np.mean(centers[:, 2] ** 2) / 0.5**2
        ) / (2 * diameter**2)
        self.assertAlmostEqual(graph.error(values), expected)

        # Moving the farthest endpoint along world X changes only the baseline residual.
        values.update(X(6), gtsam.Pose3(rotation, np.array([5.0, 2, 3])))
        self.assertAlmostEqual(graph.error(values), expected + 0.5 / (diameter * 0.25) ** 2)

    def test_invalid_inputs_leave_graph_untouched(self) -> None:
        """Reject an undefined baseline, missing point set, or invalid noise scales before adding factors."""
        values = gtsam.Values()
        values.insert(P(0), np.ones(3))
        values.insert(X(0), gtsam.Pose3())
        values.insert(X(1), gtsam.Pose3())
        values.insert(X(2), gtsam.Pose3(gtsam.Rot3(), np.ones(3)))
        points, cameras = [P(0)], [X(0), X(2)]
        graph = gtsam.NonlinearFactorGraph()
        for point_keys, camera_keys, options in [
            ([], cameras, {}),
            (points, [X(0)], {}),
            (points, [X(0), X(1)], {}),
            (points, cameras, {"point_sigma": 0.0}),
            (points, cameras, {"baseline_sigma": float("nan")}),
            (points, cameras, {"camera_y_sigma": 10.0, "camera_z_sigma": 10.0}),
        ]:
            with self.subTest(point_keys=point_keys, camera_keys=camera_keys, options=options):
                with self.assertRaises(ValueError):
                    add_gauge_factors(graph, values, point_keys, camera_keys, **options)
                self.assertEqual(graph.size(), 0)
