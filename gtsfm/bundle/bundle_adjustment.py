"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import time
from collections import Counter
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dask
import gtsam  # type: ignore
import numpy as np
from dask.delayed import Delayed
from gtsam import BetweenFactorPose3, NonlinearFactorGraph, PriorFactorPoint3, PriorFactorPose3, Values
from gtsam.noiseModel import Diagonal, Isotropic, Robust, mEstimator  # type: ignore
from gtsam.symbol_shorthand import K, P, X  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common import gtsfm_data
from gtsfm.common.pose_prior import PosePrior
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

METRICS_GROUP = "bundle_adjustment_metrics"

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"
"""In this file, we use the GTSAM's GeneralSFMFactor2 instead of GeneralSFMFactor because Factor2 enables decoupling
of the camera pose and the camera intrinsics, and hence gives an option to share the intrinsics between cameras.
"""


CAM_POSE3_DOF = 6  # 6 dof for pose of camera
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logger = logger_utils.get_logger()


class RobustBAMode(Enum):
    """Robust BA modes."""

    NONE = "NONE"
    HUBER = "HUBER"
    GMC = "GMC"


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation.

    Due to the process graph requiring separate classes for separate graph objects, this class is a superclass for
    TwoViewBundleAdjustment and GlobalBundleAdjustment (defined in gtsfm/bundle/).
    """

    def __init__(
        self,
        reproj_error_thresholds: Sequence[Optional[float]] = [None],
        robust_ba_mode: RobustBAMode = RobustBAMode.NONE,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
        cam_pose3_prior_noise_sigma: float = 0.1,
        calibration_prior_noise_sigma: float = 0.05,
        measurement_noise_sigma: float = 2.0,
        allow_indeterminate_linear_system: bool = True,
        print_summary: bool = False,
        ordering_type: str = "METIS",
        save_iteration_visualization: bool = False,
        robust_noise_basin: float = 1.345,
        use_karcher_mean_factor: bool = True,
        use_calibration_prior: bool = True,
        use_first_point_prior: bool = False,
    ) -> None:
        """Initializes the parameters for bundle adjustment module.

        Args:
            reproj_error_thresholds (optional): List of reprojection error thresholds used to perform filtering after
                each global bundle adjustment step. Implicitly defines the number of global BA steps, e.g., if
                len(reproj_error_thresholds) == 1, only one step will be performed. If the threshold is None, no
                filtering on output data is performed. Defaults to None.
            robust_ba_mode (optional): Robust BA mode to use, defaults to NONE.
            shared_calib (optional): Flag to enable shared calibration across all cameras. Defaults to False.
            max_iterations (optional): Max number of iterations when optimizing the factor graph. None means no cap.
                Defaults to None.
            cam_pose3_prior_noise_sigma (optional): Camera Pose3 prior noise sigma.
            calibration_prior_noise_sigma (optional): Calibration prior noise sigma. Default to 1e-5, which is
                essentially fixed.
            measurement_noise_sigma (optional): Measurement noise sigma in pixel units.
            allow_indeterminate_linear_system: Reject a two-view measurement if an indeterminate linear system is
                encountered during marginal covariance computation after bundle adjustment.
            ordering_type (optional): The ordering algorithm to use for variable elimination.
            save_iteration_visualization (optional): Save a Plotly animation showing optimization progress.
            robust_noise_basin (optional): Basin to use for the robust noise model.
            use_karcher_mean_factor (optional): Use Karcher mean factor to constrain the camera poses.
            use_calibration_prior (optional): Use calibration prior to constrain the camera intrinsics.
            use_first_point_prior (optional): Use first point prior to constrain the scale of the reconstruction.
        """
        self._reproj_error_thresholds = reproj_error_thresholds
        self._robust_ba_mode = robust_ba_mode
        self._shared_calib = shared_calib
        self._max_iterations = max_iterations
        self._cam_pose3_prior_noise_sigma = cam_pose3_prior_noise_sigma
        self._calibration_prior_noise_sigma = calibration_prior_noise_sigma
        self._measurement_noise_sigma = measurement_noise_sigma
        self._allow_indeterminate_linear_system = allow_indeterminate_linear_system
        self._ordering_type = ordering_type
        self._print_summary = print_summary
        self._save_iteration_visualization = save_iteration_visualization
        self._robust_noise_basin = robust_noise_basin
        self._use_karcher_mean_factor = use_karcher_mean_factor
        self._use_calibration_prior = use_calibration_prior
        self._use_first_point_prior = use_first_point_prior

    def __map_to_calibration_variable(self, camera_idx: int) -> int:
        return 0 if self._shared_calib else camera_idx

    def __reprojection_factors(
        self, initial_data: GtsfmData, cameras_to_model: List[int], robust_noise_basin: float | None = None
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Generate reprojection factors using the tracks."""
        graph = NonlinearFactorGraph()

        # noise model for measurements -- one pixel in u and v
        measurement_noise = Isotropic.Sigma(IMG_MEASUREMENT_DIM, self._measurement_noise_sigma)
        noise_basin = robust_noise_basin if robust_noise_basin is not None else self._robust_noise_basin
        if self._robust_ba_mode == RobustBAMode.HUBER:
            measurement_noise = Robust(mEstimator.Huber(noise_basin), measurement_noise)
        elif self._robust_ba_mode == RobustBAMode.GMC:
            measurement_noise = Robust(mEstimator.GemanMcClure(noise_basin), measurement_noise)

        # Note: Assumes all calibration types are the same.
        first_camera = initial_data.get_camera(cameras_to_model[0])
        assert first_camera is not None, "First camera in initial data is None"
        sfm_factor_class = gtsfm_types.get_sfm_factor_for_calibration(first_camera.calibration())

        cameras_with_tracks = set()
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            # Retrieve the SfmMeasurement objects.
            for m_idx in range(track.numberMeasurements()):
                # `i` represents the camera index, and `uv` is the 2d measurement
                i, uv = track.measurement(m_idx)
                cameras_with_tracks.add(i)
                # Note use of shorthand symbols `X` and `P`.
                if i not in cameras_to_model:
                    continue
                graph.push_back(
                    sfm_factor_class(
                        uv,
                        measurement_noise,
                        X(i),
                        P(j),
                        K(self.__map_to_calibration_variable(i)),
                    )  # type: ignore
                )

        cameras_without_tracks = {}
        for i in cameras_to_model:
            if i not in cameras_with_tracks:
                cameras_without_tracks[i] = initial_data.get_camera(i)
        if len(cameras_without_tracks) > 0:
            logger.info(f"Cameras without tracks: {cameras_without_tracks.keys()}")

        return graph, cameras_without_tracks

    def _between_factors(
        self, relative_pose_priors: Dict[Tuple[int, int], PosePrior], cameras_to_model: List[int]
    ) -> NonlinearFactorGraph:
        """Generate BetweenFactors on relative poses for pose variables."""
        graph = NonlinearFactorGraph()

        for (i1, i2), i2Ti1_prior in relative_pose_priors.items():
            if i1 not in cameras_to_model or i2 not in cameras_to_model:
                continue

            graph.push_back(
                BetweenFactorPose3(
                    X(i1),
                    X(i2),
                    i2Ti1_prior.value.inverse(),
                    Diagonal.Sigmas(i2Ti1_prior.covariance),
                )
            )

        return graph

    def __pose_priors(
        self,
        absolute_pose_priors: List[Optional[PosePrior]],
        initial_data: GtsfmData,
        first_valid_camera_idx: int,
    ) -> NonlinearFactorGraph:
        """Generate prior factors (in the world frame) on pose variables."""
        graph = NonlinearFactorGraph()

        # TODO(Ayush): start using absolute prior factors.
        num_priors_added = 0

        if num_priors_added == 0:
            # Adding a prior to fix origin as no absolute prior exists.
            first_camera = initial_data.get_camera(first_valid_camera_idx)
            assert first_camera is not None, "First camera in initial data is None"
            graph.push_back(
                PriorFactorPose3(
                    X(first_valid_camera_idx),
                    first_camera.pose(),
                    Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
                )
            )

        return graph

    def __calibration_priors(self, initial_data: GtsfmData, cameras_to_model: list[int]) -> NonlinearFactorGraph:
        """Generate prior factors on calibration parameters of the cameras."""
        graph = NonlinearFactorGraph()

        # Note: Assumes all calibration types are the same.
        first_valid_camera_idx = cameras_to_model[0]
        first_camera = initial_data.get_camera(first_valid_camera_idx)
        assert first_camera is not None, "First camera in initial data is None"
        calibration_prior_factor_class = gtsfm_types.get_prior_factor_for_calibration(first_camera.calibration())
        calibration_dim = first_camera.calibration().dim()
        noise_model = gtsfm_types.get_noise_model_for_calibration(
            first_camera.calibration(), focal_sigma=self._calibration_prior_noise_sigma, pp_sigma=1e-5
        )
        if self._shared_calib:
            graph.push_back(
                calibration_prior_factor_class(
                    K(self.__map_to_calibration_variable(first_valid_camera_idx)),
                    gtsfm_data.get_average_calibration(initial_data, cameras_to_model),  # type: ignore
                    noise_model,
                )
            )
        else:
            for i in cameras_to_model:
                camera_i = initial_data.get_camera(i)
                assert camera_i is not None, f"Camera {i} in initial data is None"
                if camera_i.calibration().dim() != calibration_dim:
                    raise ValueError(
                        "BundleAdjustmentOptimizer: Assumption that all calibration types are the same is violated"
                    )
                graph.push_back(
                    calibration_prior_factor_class(
                        K(self.__map_to_calibration_variable(i)), camera_i.calibration(), noise_model  # type: ignore
                    )
                )

        return graph

    def __construct_simple_factor_graph(
        self, cameras_to_model: List[int], initial_data: GtsfmData, robust_noise_basin: float | None = None
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Construct the factor graph with just reprojection factors and calibration priors."""

        graph = NonlinearFactorGraph()
        if not cameras_to_model:
            return graph

        reprojection_graph, cameras_without_tracks = self.__reprojection_factors(
            initial_data=initial_data,
            cameras_to_model=cameras_to_model,
            robust_noise_basin=robust_noise_basin,
        )
        graph.push_back(reprojection_graph)

        if self._use_karcher_mean_factor:
            camera_keys = [X(i) for i in cameras_to_model]
            graph.push_back(gtsam.KarcherMeanFactorPose3(camera_keys, 6, 1000))
        else:
            first_camera = initial_data.get_camera(cameras_to_model[0])
            assert first_camera is not None, "First camera in initial data is None"
            graph.push_back(
                self.__pose_priors(
                    absolute_pose_priors=[],
                    initial_data=initial_data,
                    first_valid_camera_idx=cameras_to_model[0],
                )
            )

        if self._use_first_point_prior:
            graph.push_back(
                PriorFactorPoint3(P(0), initial_data.get_track(0).point3(), Isotropic.Sigma(POINT3_DOF, 0.1))
            )
        if self._use_calibration_prior:
            graph.push_back(self.__calibration_priors(initial_data, cameras_to_model))

        return graph, cameras_without_tracks

    def __construct_factor_graph(
        self,
        cameras_to_model: List[int],
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        robust_noise_basin: float | None = None,
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Construct the factor graph with reprojection factors, BetweenFactors, and prior factors."""
        # Create a factor graph.
        graph, cameras_without_tracks = self.__construct_simple_factor_graph(
            cameras_to_model, initial_data, robust_noise_basin
        )

        # Add priors
        graph.push_back(
            self._between_factors(relative_pose_priors=relative_pose_priors, cameras_to_model=cameras_to_model)
        )
        graph.push_back(
            self.__pose_priors(
                absolute_pose_priors=absolute_pose_priors,
                initial_data=initial_data,
                first_valid_camera_idx=cameras_to_model[0],
            )
        )

        return graph, cameras_without_tracks

    def __optimize_factor_graph(
        self, graph: NonlinearFactorGraph, initial_values: Values, ordering_type: str
    ) -> Tuple[Values, Optional[List[Values]]]:
        """Optimize the factor graph, optionally capturing per-iteration values."""
        start_time = time.time()

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR" if not self._print_summary else "SUMMARY")
        params.setOrderingType(ordering_type)
        if self._max_iterations:
            params.setMaxIterations(self._max_iterations)

        lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)

        if not self._save_iteration_visualization:
            result_values = lm.optimize()
            values_trace = None
        else:
            values_trace = [initial_values]
            max_iters = self._max_iterations
            if max_iters is None:
                try:
                    max_iters = int(params.getMaxIterations())
                except Exception:
                    max_iters = 100

            abs_tol = float(params.getAbsoluteErrorTol())
            rel_tol = float(params.getRelativeErrorTol())
            prev_error = float(lm.error())

            for _ in range(max_iters):
                lm.iterate()
                values_trace.append(lm.values())
                curr_error = float(lm.error())
                error_delta = prev_error - curr_error
                if abs(error_delta) < abs_tol:
                    logger.info("🚀 Absolute error tolerance reached.")
                    break
                if prev_error > 0.0 and abs(error_delta) / prev_error < rel_tol:
                    logger.info("🚀 Relative error tolerance reached.")
                    break
                prev_error = curr_error
            result_values = lm.values()

        elapsed_time = time.time() - start_time
        logger.info(f"🚀 Factor graph optimization completed in {elapsed_time:.2f} seconds.")

        return result_values, values_trace

    def get_two_view_ba_pose_graph_keys(self, initial_data: GtsfmData):
        """Retrieves GTSAM keys for camera poses in a 2-view BA problem."""
        valid_camera_indices = initial_data.get_valid_camera_indices()
        return [X(valid_camera_indices[0]), X(valid_camera_indices[1])]

    def is_two_view_ba(self, initial_data: GtsfmData) -> bool:
        """Determines whether two-view bundle adjustment is being executed."""
        return len(initial_data.get_valid_camera_indices()) == 2

    def __optimize_and_recover(
        self, initial_data: GtsfmData, graph: NonlinearFactorGraph, ordering_type: str
    ) -> Tuple[GtsfmData, Values, float]:
        """Optimize the graph, report errors, and convert `Values` back to `GtsfmData`."""
        initial_values = initial_data.to_values(shared_calib=self._shared_calib)
        result_values, _ = self.__optimize_factor_graph(graph, initial_values, ordering_type)
        final_error = graph.error(result_values)
        optimized_data = GtsfmData.from_values(result_values, initial_data, self._shared_calib)
        return optimized_data, result_values, final_error

    def run_simple_ba(
        self, initial_data: GtsfmData, robust_noise_basin: float | None = None
    ) -> Tuple[GtsfmData, float]:
        """Runs bundle adjustment and optionally filters the resulting tracks by reprojection error.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            robust_noise_basin: Robust noise basin to use for the BA, not used if self._robust_ba_mode is NONE.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Final error value of the optimization problem.
        """
        cameras_to_model = sorted(initial_data.get_valid_camera_indices())

        graph, cameras_without_tracks = self.__construct_simple_factor_graph(
            cameras_to_model, initial_data, robust_noise_basin
        )
        optimized_data, _, final_error = self.__optimize_and_recover(
            initial_data,
            graph,
            self._ordering_type if not cameras_without_tracks else "COLAMD",
        )
        return optimized_data, final_error

    def run_iterative_robust_ba(
        self, initial_data: GtsfmData, robust_noise_basins: List[float]
    ) -> Tuple[GtsfmData, float]:
        """Runs iterative robust bundle adjustment, using different robust noise basins for each iteration."""
        optimized_data = initial_data
        final_error = float("nan")
        for robust_noise_basin in robust_noise_basins:
            optimized_data, final_error = self.run_simple_ba(optimized_data, robust_noise_basin)
        return optimized_data, final_error

    def run_ba_stage_with_filtering(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        reproj_error_thresh: Optional[float],
        verbose: bool = True,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool], float]:
        """Runs bundle adjustment and optionally filters the resulting tracks by reprojection error.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: Priors to be used on cameras.
            relative_pose_priors: Priors on the pose between two cameras.
            reproj_error_thresh: Maximum 3D track reprojection error, for filtering tracks after BA.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Optimized camera poses after filtering landmarks (and cameras with no remaining landmarks).
            Valid mask as a list of booleans, indicating for each input track whether it was below the re-projection
                threshold.
            Final error value of the optimization problem.
        """
        logger.info(
            "Input: %d tracks on %d cameras", initial_data.number_tracks(), len(initial_data.get_valid_camera_indices())
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # No cameras or tracks to optimize, so bundle adjustment is not possible, return invalid result.
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return initial_data, initial_data, [False] * initial_data.number_tracks(), 0.0

        cameras_to_model = sorted(initial_data.get_valid_camera_indices())
        graph, cameras_without_tracks = self.__construct_factor_graph(
            cameras_to_model, initial_data, absolute_pose_priors, relative_pose_priors
        )
        optimized_data, result_values, final_error = self.__optimize_and_recover(
            initial_data, graph, self._ordering_type if not cameras_without_tracks else "COLAMD", verbose
        )

        if self.is_two_view_ba(initial_data):
            try:
                # Calculate marginal covariances for all two pose variables.
                marginals = gtsam.Marginals(graph, result_values)
                graph_keys = self.get_two_view_ba_pose_graph_keys(initial_data)
                for key in graph_keys:
                    _ = marginals.marginalCovariance(key)

            except RuntimeError:
                if not self._allow_indeterminate_linear_system:
                    logger.error(
                        "BA result discarded due to Indeterminate Linear System (ILS) when computing marginals."
                    )
                    return None, None, None, None

        # Convert the `Values` results to a `GtsfmData` instance.
        # Filter landmarks by reprojection error.
        if reproj_error_thresh is not None:
            if verbose:
                logger.info("[Result] Number of tracks before filtering: %d", optimized_data.number_tracks())
            filtered_result, valid_mask = optimized_data.filter_landmarks(reproj_error_thresh)
            if verbose:
                logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())

        else:
            valid_mask = [True] * optimized_data.number_tracks()
            filtered_result = optimized_data

        return optimized_data, filtered_result, valid_mask, final_error

    def run_ba(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        verbose: bool = True,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool]]:
        """Runs bundle adjustment by forming a factor graph and optimizing it using Levenberg–Marquardt optimization.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: Priors to be used on cameras.
            relative_pose_priors: Priors on the pose between two cameras.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Optimized camera poses after filtering landmarks (and cameras with no remaining landmarks).
            Valid mask as a list of booleans, indicating for each input track whether it was below the re-projection
                threshold.
        """
        num_ba_steps = len(self._reproj_error_thresholds)
        for step, reproj_error_thresh in enumerate(self._reproj_error_thresholds):
            # Use intermediate result as initial condition for next step.
            (optimized_data, filtered_result, valid_mask, final_error) = self.run_ba_stage_with_filtering(
                initial_data,
                absolute_pose_priors,
                relative_pose_priors,
                reproj_error_thresh,
                verbose,
            )
            # Print intermediate results.
            if num_ba_steps > 1:
                logger.info(
                    "[BA Step %d/%d] Error: %.2f, Number of tracks: %d"
                    % (step + 1, num_ba_steps, final_error, filtered_result.number_tracks())
                )

        return optimized_data, filtered_result, valid_mask  # type: ignore

    def _run_ba_and_evaluate(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
        verbose: bool = True,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool], GtsfmMetricsGroup]:
        """Runs the equivalent of `run_ba()` and `evaluate()` in a single function, to enable time profiling."""
        logger.info(
            "Input: %d tracks on %d cameras", initial_data.number_tracks(), len(initial_data.get_valid_camera_indices())
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # No cameras or tracks to optimize, so bundle adjustment is not possible.
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return (
                initial_data,
                initial_data,
                [False] * initial_data.number_tracks(),
                GtsfmMetricsGroup(METRICS_GROUP, []),
            )
        step_times = []
        start_time = time.time()

        num_ba_steps = len(self._reproj_error_thresholds)
        assert num_ba_steps > 0, "No BA steps to perform"

        for step, reproj_error_thresh in enumerate(self._reproj_error_thresholds):
            step_start_time = time.time()
            (optimized_data, filtered_result, valid_mask, final_error) = self.run_ba_stage_with_filtering(
                initial_data=initial_data,
                absolute_pose_priors=absolute_pose_priors,
                relative_pose_priors=relative_pose_priors,
                reproj_error_thresh=reproj_error_thresh,
                verbose=verbose,
            )
            step_times.append(time.time() - step_start_time)

            # Print intermediate results.
            if num_ba_steps > 1:
                logger.info(
                    "[BA Stage @ thresh=%.2f px %d/%d] Error: %.2f, Number of tracks: %d"
                    % (
                        reproj_error_thresh if reproj_error_thresh is not None else float("nan"),
                        step + 1,
                        num_ba_steps,
                        final_error,
                        filtered_result.number_tracks(),
                    )
                )
        total_time = time.time() - start_time

        metrics = self.evaluate(optimized_data, filtered_result, cameras_gt, save_dir)  # type: ignore
        for i, step_time in enumerate(step_times):
            metrics.add_metric(GtsfmMetric(f"step_{i}_run_duration_sec", step_time))
        metrics.add_metric(GtsfmMetric("total_run_duration_sec", total_time))

        return optimized_data, filtered_result, valid_mask, metrics  # type: ignore

    def evaluate(
        self,
        unfiltered_data: GtsfmData,
        filtered_data: GtsfmData,
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
    ) -> GtsfmMetricsGroup:
        """Computes metrics on the bundle adjustment result, and packages them in a GtsfmMetricsGroup object.

        Args:
            unfiltered_data: Optimized BA result, before filtering landmarks by reprojection error.
            filtered_data: Optimized BA result, after filtering landmarks and cameras.
            cameras_gt: Cameras with GT intrinsics and GT extrinsics.

        Returns:
            Metrics group containing metrics for both filtered and unfiltered BA results.
        """
        ba_metrics = GtsfmMetricsGroup(name=METRICS_GROUP, metrics=unfiltered_data.get_metrics(suffix="_unfiltered"))

        input_image_idxs = list(unfiltered_data._image_info.keys())
        poses_gt = {
            i: cameras_gt[i].pose() for i in input_image_idxs if i < len(cameras_gt) and cameras_gt[i] is not None
        }
        if not poses_gt:
            return ba_metrics

        # Align the sparse multi-view estimate after BA to the ground truth pose graph.
        aligned_filtered_data = filtered_data.align_via_sim3_and_transform(poses_gt)
        ba_pose_error_metrics = metrics_utils.compute_ba_pose_metrics(
            gt_wTi=poses_gt, computed_wTi=aligned_filtered_data.get_camera_poses(), save_dir=save_dir
        )
        ba_metrics.extend(metrics_group=ba_pose_error_metrics)

        output_tracks_exit_codes = track_utils.classify_tracks3d_with_gt_cameras(
            tracks=aligned_filtered_data.get_tracks(), cameras_gt=cameras_gt
        )
        output_tracks_exit_codes_distribution = Counter(output_tracks_exit_codes)

        for exit_code, count in output_tracks_exit_codes_distribution.items():
            metric_name = "Filtered tracks triangulated with GT cams: {}".format(exit_code.name)
            ba_metrics.add_metric(GtsfmMetric(name=metric_name, data=count))

        ba_metrics.add_metrics(aligned_filtered_data.get_metrics(suffix="_filtered"))

        logger.info("[Result] Mean track length %.3f", np.mean(aligned_filtered_data.get_track_lengths()))
        logger.info("[Result] Median track length %.3f", np.median(aligned_filtered_data.get_track_lengths()))
        aligned_filtered_data.log_scene_reprojection_error_stats()

        return ba_metrics

    def create_computation_graph(
        self,
        sfm_data_graph: Delayed,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: An GtsfmData object wrapped up using dask.delayed.
            absolute_pose_priors: Priors on the poses of the cameras (not delayed).
            relative_pose_priors: Priors on poses between cameras (not delayed).
            cameras_gt: Ground truth camera calibration & pose for each image/camera.
            save_dir: Directory where artifacts and plots should be saved to disk.

        Returns:
            GtsfmData aligned to GT (if provided), wrapped up using dask.delayed
            Metrics group for BA results, wrapped up using dask.delayed
        """

        _, filtered_sfm_data, _, metrics_graph = dask.delayed(self._run_ba_and_evaluate, nout=4)(
            sfm_data_graph,
            absolute_pose_priors,
            relative_pose_priors,
            cameras_gt,
            save_dir=save_dir,
        )
        return filtered_sfm_data, metrics_graph
