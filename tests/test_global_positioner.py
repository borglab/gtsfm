"""Exercise global positioning against the installed GTSAM Python bindings."""

import unittest

import numpy as np
from gtsam.symbol_shorthand import A as C
from gtsam.symbol_shorthand import B as L

from gtsfm.global_positioner.global_positioner import build_and_solve, build_and_solve_with_trace


class TestGlobalPositioner(unittest.TestCase):
    def test_solver_paths_accept_camera_and_landmark_keys(self):
        """Both paths must cross the native binding boundary and produce finite positions."""
        cameras = {2: np.array([0, 0, 0]), 5: np.array([1, 0, 0]), 9: np.array([0, 1, 0])}
        points = np.random.default_rng(42).uniform([-1, -1, 3], [2, 2, 6], (12, 3))
        observations = [
            (j, i, (point - center) / np.linalg.norm(point - center))
            for i, center in cameras.items()
            for j, point in enumerate(points)
        ]
        for solve in (build_and_solve, build_and_solve_with_trace):
            with self.subTest(solver=solve.__name__):
                result = solve(observations, noise_sigma=0.01, huber_loss_scale=0.1, max_iterations=50)
                values, camera_indices, track_indices = result[:3]
                self.assertEqual(camera_indices, set(cameras))
                self.assertEqual(track_indices, set(range(len(points))))
                # The optimizer also returns auxiliary measurement-scale variables.
                self.assertGreaterEqual(values.size(), len(cameras) + len(points))
                for key in [*(C(i) for i in cameras), *(L(j) for j in range(len(points)))]:
                    self.assertTrue(np.isfinite(values.atPoint3(key)).all())
                if solve is build_and_solve_with_trace:
                    trace, errors = result[3:]
                    self.assertGreater(len(trace), 1)
                    self.assertEqual(len(trace), len(errors))
                    self.assertTrue(np.isfinite(errors).all())
                    self.assertLess(errors[-1], errors[0])
                    for frame in trace:
                        for i in cameras:
                            self.assertTrue(np.isfinite(frame.atPoint3(C(i))).all())
                        for j in range(len(points)):
                            self.assertTrue(np.isfinite(frame.atPoint3(L(j))).all())
