"""Unit tests for bundle adjustment.

Authors: Ayush Baid
"""

import unittest
from unittest.mock import MagicMock, patch

import dask
import gtsam  # type: ignore
import numpy as np

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer, BundleAdjustmentOptions
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = GtsfmData.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))


class TestBundleAdjustmentOptimizer(unittest.TestCase):
    """Unit tests for BundleAdjustmentOptimizer class."""

    def setUp(self):
        super().setUp()

        reproj_error_thresholds = [100.0]
        self.ba = BundleAdjustmentOptimizer(reproj_error_thresholds=reproj_error_thresholds, min_tracks_per_camera=5)

        self.test_data = EXAMPLE_DATA

    def _clone_with_shared_calibration(self, data: GtsfmData) -> GtsfmData:
        """Return a copy of ``data`` where all cameras share the first camera's calibration."""
        shared_clone = GtsfmData(data.number_images())
        valid_camera_indices = data.get_valid_camera_indices()
        assert len(valid_camera_indices) > 0
        first_cam = data.get_camera(valid_camera_indices[0])
        assert first_cam is not None
        camera_type = type(first_cam)
        shared_calibration = first_cam.calibration()

        for idx in valid_camera_indices:
            cam = data.get_camera(idx)
            assert cam is not None
            shared_clone.add_camera(idx, camera_type(cam.pose(), shared_calibration))

        for track_idx in range(data.number_tracks()):
            shared_clone.add_track(data.get_track(track_idx))

        return shared_clone

    # def test_simple_scene(self):
    #     """Test the simple scene using the `run_ba` API."""

    #     computed_result = self.ba.run_ba(self.test_data)

    #     expected_error = 0.046137573704557046

    #     self.assertTrue(np.isclose(expected_error, computed_result.total_reproj_error, atol=1e-2, rtol=1e-2))

    def test_create_computation_graph(self):
        """Test the simple scene as dask computation graph."""
        sfm_data_graph = dask.delayed(self.test_data)

        absolute_pose_priors = [None] * EXAMPLE_DATA.number_images()
        relative_pose_priors = {}

        expected_result, _, _, _ = self.ba.run_ba(
            self.test_data, absolute_pose_priors=absolute_pose_priors, relative_pose_priors=relative_pose_priors
        )

        computed_result, _ = self.ba.create_computation_graph(
            sfm_data_graph,
            absolute_pose_priors,
            relative_pose_priors,
            cameras_gt=[None] * self.test_data.number_images(),
        )

        with dask.config.set(scheduler="single-threaded"):
            result = dask.compute(computed_result)[0]

        self.assertEqual(result, expected_result)

    def test_values_roundtrip_without_shared_calib(self):
        """Ensure GtsfmData <-> gtsam.Values conversion preserves scene when calibrations are independent."""
        values = self.test_data.to_values(shared_calib=False)
        reconstructed = GtsfmData.from_values(values, initial_data=self.test_data, shared_calib=False)
        self.assertEqual(reconstructed, self.test_data)

    def test_values_roundtrip_with_shared_calib(self):
        """Ensure GtsfmData <-> gtsam.Values conversion preserves scene when calibration is shared."""
        shared_calib_data = self._clone_with_shared_calibration(self.test_data)
        values = shared_calib_data.to_values(shared_calib=True)
        reconstructed = GtsfmData.from_values(values, initial_data=shared_calib_data, shared_calib=True)
        self.assertEqual(reconstructed, shared_calib_data)

    def test_from_values_without_initial_data(self):
        """Ensure from_values succeeds without access to the original initial_data."""
        values = self.test_data.to_values(shared_calib=False)
        reconstructed = GtsfmData.from_values(values)

        self.assertEqual(reconstructed.number_images(), self.test_data.number_images())
        self.assertEqual(reconstructed.get_valid_camera_indices(), self.test_data.get_valid_camera_indices())
        self.assertEqual(reconstructed.number_tracks(), self.test_data.number_tracks())

        for camera_idx in self.test_data.get_valid_camera_indices():
            original = self.test_data.get_camera(camera_idx)
            rebuilt = reconstructed.get_camera(camera_idx)
            self.assertIsNotNone(original)
            self.assertIsNotNone(rebuilt)
            assert original is not None and rebuilt is not None
            self.assertTrue(original.pose().equals(rebuilt.pose(), 1e-9))
            self.assertTrue(original.calibration().equals(rebuilt.calibration(), 1e-9))

        for track_idx in range(self.test_data.number_tracks()):
            original_point = self.test_data.get_track(track_idx).point3()
            rebuilt_point = reconstructed.get_track(track_idx).point3()
            self.assertTrue(np.allclose(rebuilt_point, original_point))

    def test_run_simple_ba(self):
        """Test run_simple_ba on simple scene."""
        computed_result, error = self.ba.run_simple_ba(self.test_data)

        self.assertEqual(computed_result.number_images(), self.test_data.number_images())
        self.assertAlmostEqual(error, 0.3675, places=2)

    def test_multistage_ba_uses_previous_filtered_result(self):
        """Ensure each BA stage consumes the previous stage's filtered output."""
        ba = BundleAdjustmentOptimizer(reproj_error_thresholds=[10.0, 5.0, 3.0])

        input_data = MagicMock(spec=GtsfmData)
        input_data.number_tracks.return_value = 3

        stage1_filtered = MagicMock(spec=GtsfmData)
        stage1_filtered.number_tracks.return_value = 2

        stage2_filtered = MagicMock(spec=GtsfmData)
        stage2_filtered.number_tracks.return_value = 1

        stage3_filtered = MagicMock(spec=GtsfmData)
        stage3_filtered.number_tracks.return_value = 1

        stage_outputs = [
            (MagicMock(spec=GtsfmData), stage1_filtered, [True, False, True], 1.0),
            (MagicMock(spec=GtsfmData), stage2_filtered, [False, True], 0.5),
            (MagicMock(spec=GtsfmData), stage3_filtered, [True], 0.1),
        ]

        call_inputs = []

        def capture_stage_input(stage_input, *args, **kwargs):
            call_inputs.append(stage_input)
            return stage_outputs[len(call_inputs) - 1]

        with patch.object(ba, "run_ba_stage_with_filtering", side_effect=capture_stage_input):
            _, final_filtered, valid_mask, _ = ba.run_ba(
                input_data,
                absolute_pose_priors=[],
                relative_pose_priors={},
                verbose=False,
            )

        self.assertEqual(call_inputs, [input_data, stage1_filtered, stage2_filtered])
        self.assertIs(final_filtered, stage3_filtered)
        self.assertEqual(valid_mask, [False, False, True])

    def test_stage_mask_maps_gnc_filtered_tracks_to_input_tracks(self):
        """Ensure GNC-pruned tracks are represented in the stage validity mask."""
        # min_tracks_per_camera=0 disables the insufficient-tracks check, so the BA
        # path doesn't try to mutate optimized_data._cameras (which the MagicMock
        # spec=GtsfmData wouldn't expose since it's set in __init__ rather than at
        # class level).
        ba = BundleAdjustmentOptimizer(reproj_error_thresholds=[5.0], min_tracks_per_camera=0)

        initial_data = MagicMock(spec=GtsfmData)
        initial_data.number_tracks.return_value = 4
        initial_data.get_valid_camera_indices.return_value = [0, 1, 2]

        optimized_data = MagicMock(spec=GtsfmData)
        optimized_data.number_tracks.return_value = 3

        filtered_result = MagicMock(spec=GtsfmData)
        optimized_data.filter_landmarks.return_value = (filtered_result, [True, False, True])

        with patch.object(
            ba,
            "_BundleAdjustmentOptimizer__construct_factor_graph",
            return_value=MagicMock(),
        ), patch.object(
            ba,
            "_BundleAdjustmentOptimizer__optimize_and_recover",
            return_value=(optimized_data, MagicMock(), 1.0, [True, False, True, True]),
        ):
            _, _, valid_mask, _ = ba.run_ba_stage_with_filtering(
                initial_data,
                absolute_pose_priors=[],
                relative_pose_priors={},
                reproj_error_thresh=5.0,
                verbose=False,
            )

        self.assertEqual(valid_mask, [True, False, False, True])

    def test_bundle_adjustment_options_cuda(self):
        """Ensure BundleAdjustmentOptions defaults and propagation for CUDA LM."""
        options = BundleAdjustmentOptions()
        self.assertFalse(options.use_cuda)
        self.assertEqual(options.cuda_linear_solver, "PCG")
        self.assertIsNone(options.cuda_pcg_max_iterations)
        self.assertIsNone(options.cuda_pcg_relative_tolerance)
        self.assertFalse(options.cuda_pcg_warm_start)
        self.assertIsNone(options.cuda_pcg_convergence_check_interval)
        self.assertTrue(options.cuda_fallback_on_unsupported)
        self.assertFalse(options.cuda_collect_timing)

        # Custom options
        custom_options = BundleAdjustmentOptions(
            use_cuda=True,
            cuda_linear_solver="CUDSS",
            cuda_pcg_max_iterations=150,
            cuda_pcg_relative_tolerance=1e-8,
            cuda_pcg_warm_start=True,
            cuda_pcg_convergence_check_interval=5,
            cuda_fallback_on_unsupported=False,
            cuda_collect_timing=True,
        )
        optimizer = custom_options.to_optimizer(min_tracks_per_camera=0)
        self.assertTrue(optimizer._use_cuda)
        self.assertEqual(optimizer._cuda_linear_solver, "CUDSS")
        self.assertEqual(optimizer._cuda_pcg_max_iterations, 150)
        self.assertEqual(optimizer._cuda_pcg_relative_tolerance, 1e-8)
        self.assertTrue(optimizer._cuda_pcg_warm_start)
        self.assertEqual(optimizer._cuda_pcg_convergence_check_interval, 5)
        self.assertFalse(optimizer._cuda_fallback_on_unsupported)
        self.assertTrue(optimizer._cuda_collect_timing)

    def test_bundle_adjustment_optimizer_init_cuda(self):
        """Ensure BundleAdjustmentOptimizer stores CUDA parameters."""
        ba = BundleAdjustmentOptimizer(
            use_cuda=True,
            cuda_linear_solver="PCG",
            cuda_pcg_max_iterations=300,
            cuda_pcg_relative_tolerance=1e-7,
            cuda_pcg_warm_start=True,
            cuda_pcg_convergence_check_interval=2,
            cuda_fallback_on_unsupported=True,
            cuda_collect_timing=True,
        )
        self.assertTrue(ba._use_cuda)
        self.assertEqual(ba._cuda_linear_solver, "PCG")
        self.assertEqual(ba._cuda_pcg_max_iterations, 300)
        self.assertEqual(ba._cuda_pcg_relative_tolerance, 1e-7)
        self.assertTrue(ba._cuda_pcg_warm_start)
        self.assertEqual(ba._cuda_pcg_convergence_check_interval, 2)
        self.assertTrue(ba._cuda_fallback_on_unsupported)
        self.assertTrue(ba._cuda_collect_timing)
        self.assertIsNone(ba._last_cuda_result)

    def test_cuda_fallback_when_gtsam_cuda_absent(self):
        """Ensure smooth CPU fallback when gtsam.cuda is absent and fallback is enabled."""
        ba = BundleAdjustmentOptimizer(
            reproj_error_thresholds=[100.0],
            min_tracks_per_camera=5,
            use_cuda=True,
            cuda_fallback_on_unsupported=True,
        )
        # Ensure gtsam has no cuda attribute during this call
        with patch.object(gtsam, "cuda", None, create=True):
            computed_result, error = ba.run_simple_ba(self.test_data)
            self.assertEqual(computed_result.number_images(), self.test_data.number_images())
            self.assertAlmostEqual(error, 0.3675, places=2)
            self.assertIsNone(ba._last_cuda_result)

    def test_cuda_error_when_gtsam_cuda_absent_and_no_fallback(self):
        """Ensure RuntimeError is raised when gtsam.cuda is absent and fallback is disabled."""
        ba = BundleAdjustmentOptimizer(
            reproj_error_thresholds=[100.0],
            min_tracks_per_camera=5,
            use_cuda=True,
            cuda_fallback_on_unsupported=False,
        )
        with patch.object(gtsam, "cuda", None, create=True):
            with self.assertRaises(RuntimeError) as context:
                ba.run_simple_ba(self.test_data)
            self.assertIn("gtsam.cuda is missing", str(context.exception))

    def test_cuda_optimization_mocked_execution(self):
        """Ensure CUDA Sparse LM is properly configured and called when gtsam.cuda is available."""
        ba = BundleAdjustmentOptimizer(
            reproj_error_thresholds=[100.0],
            min_tracks_per_camera=5,
            use_cuda=True,
            cuda_linear_solver="PCG",
            cuda_pcg_max_iterations=80,
            cuda_pcg_relative_tolerance=1e-9,
            cuda_pcg_warm_start=True,
            cuda_pcg_convergence_check_interval=4,
            cuda_fallback_on_unsupported=True,
            cuda_collect_timing=True,
        )

        mock_cuda = MagicMock()
        mock_cuda.LinearSolverType.Pcg = "Pcg"
        mock_cuda.LinearSolverType.Cudss = "Cudss"

        # Mock options and params
        mock_linear_opts = MagicMock()
        mock_cuda.LinearSolverOptions.return_value = mock_linear_opts

        mock_pcg_opts = MagicMock()
        mock_cuda.PcgOptions.return_value = mock_pcg_opts

        mock_params = MagicMock()
        mock_cuda.SparseLevenbergMarquardtParams.return_value = mock_params

        # Mock result and optimizer
        mock_result = MagicMock()
        mock_result.backend = "Device"
        mock_result.termination = "Converged"
        mock_result.iterations = 7
        mock_result.initialError = 50.0
        mock_result.finalError = 0.5

        # Return dummy values from optimize()
        mock_values = self.test_data.to_values(shared_calib=False)
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = mock_values
        mock_optimizer.result.return_value = mock_result
        mock_cuda.SparseLevenbergMarquardtOptimizer.return_value = mock_optimizer

        with patch.object(gtsam, "cuda", mock_cuda, create=True):
            computed_result, error = ba.run_simple_ba(self.test_data)

        # Verify options were configured as requested
        self.assertEqual(mock_linear_opts.backend, "Pcg")
        self.assertEqual(mock_pcg_opts.maxIterations, 80)
        self.assertEqual(mock_pcg_opts.relativeTolerance, 1e-9)
        self.assertTrue(mock_pcg_opts.warmStart)
        self.assertEqual(mock_pcg_opts.convergenceCheckInterval, 4)

        self.assertEqual(mock_params.linear, mock_linear_opts)
        self.assertEqual(mock_params.pcg, mock_pcg_opts)
        self.assertTrue(mock_params.fallbackOnUnsupported)
        self.assertTrue(mock_params.collectTiming)

        # Verify optimizer was called and result stored
        mock_optimizer.optimize.assert_called_once()
        self.assertIs(ba._last_cuda_result, mock_result)
        self.assertEqual(ba._last_cuda_result.iterations, 7)

    def test_cuda_optimization_cpu_fallback_diagnostic_logged(self):
        """Ensure CpuFallback diagnostic from CUDA optimizer is reported."""
        ba = BundleAdjustmentOptimizer(
            reproj_error_thresholds=[100.0],
            min_tracks_per_camera=5,
            use_cuda=True,
            cuda_linear_solver="CUDSS",
        )

        mock_cuda = MagicMock()
        mock_cuda.LinearSolverType.Cudss = "Cudss"
        mock_cuda.SparseLevenbergMarquardtBackend.CpuFallback = "CpuFallback"

        mock_result = MagicMock()
        mock_result.backend = mock_cuda.SparseLevenbergMarquardtBackend.CpuFallback
        mock_result.fallbackReason = "PlanIncompatible"
        mock_result.fallbackDetail = "Unsupported factor type"
        mock_result.termination = "MaxIterations"
        mock_result.iterations = 3
        mock_result.initialError = 25.0
        mock_result.finalError = 1.2

        mock_values = self.test_data.to_values(shared_calib=False)
        mock_optimizer = MagicMock()
        mock_optimizer.optimize.return_value = mock_values
        mock_optimizer.result.return_value = mock_result
        mock_cuda.SparseLevenbergMarquardtOptimizer.return_value = mock_optimizer

        with patch.object(gtsam, "cuda", mock_cuda, create=True):
            computed_result, error = ba.run_simple_ba(self.test_data)

        self.assertEqual(ba._last_cuda_result.fallbackReason, "PlanIncompatible")
        self.assertEqual(ba._last_cuda_result.fallbackDetail, "Unsupported factor type")

    def test_cuda_gnc_incompatibility_handling(self):
        """Ensure GNC with use_cuda logs warning or raises error depending on fallback setting."""
        # Fallback allowed -> warns and runs CPU GNC
        ba_fallback = BundleAdjustmentOptimizer(
            reproj_error_thresholds=[100.0],
            min_tracks_per_camera=5,
            use_cuda=True,
            use_gnc=True,
            cuda_fallback_on_unsupported=True,
        )
        computed_result, error = ba_fallback.run_simple_ba(self.test_data)
        self.assertIsNotNone(computed_result)

        # Fallback disallowed -> raises ValueError
        ba_no_fallback = BundleAdjustmentOptimizer(
            reproj_error_thresholds=[100.0],
            min_tracks_per_camera=5,
            use_cuda=True,
            use_gnc=True,
            cuda_fallback_on_unsupported=False,
        )
        with self.assertRaises(ValueError) as context:
            ba_no_fallback.run_simple_ba(self.test_data)
        self.assertIn("GNC bundle adjustment is not supported by the CUDA Sparse LM optimizer", str(context.exception))

    def test_cuda_invalid_solver_backend_raises(self):
        """Ensure unsupported backend strings raise a ValueError."""
        ba = BundleAdjustmentOptimizer(
            reproj_error_thresholds=[100.0],
            min_tracks_per_camera=5,
            use_cuda=True,
            cuda_linear_solver="UNKNOWN_BACKEND",
        )
        mock_cuda = MagicMock()
        with patch.object(gtsam, "cuda", mock_cuda, create=True):
            with self.assertRaises(ValueError) as context:
                ba.run_simple_ba(self.test_data)
            self.assertIn("Unsupported CUDA linear solver type: 'UNKNOWN_BACKEND'", str(context.exception))


if __name__ == "__main__":
    unittest.main()
