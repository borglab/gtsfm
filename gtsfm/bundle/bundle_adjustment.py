"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import os
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import dask
import gtsam  # type: ignore
import numpy as np
from dask.delayed import Delayed
from gtsam import (
    BetweenFactorPose3,
    NonlinearFactorGraph,
    PinholeCameraCal3Fisheye,
    PriorFactorPose3,
    PriorFactorPoint3,
    Values,
)
from gtsam.noiseModel import Diagonal, Isotropic, Robust, mEstimator  # type: ignore
from gtsam.symbol_shorthand import K, P, X  # type: ignore

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.pose_prior import PosePrior
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

METRICS_GROUP = "bundle_adjustment_metrics"

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"
FACTOR_ERROR_LOG_PATH_INITIAL = METRICS_PATH / "ba_factor_errors_initial.txt"
FACTOR_ERROR_LOG_PATH_FINAL = METRICS_PATH / "ba_factor_errors_final.txt"
FACTOR_ERROR_STATS_PATH_INITIAL = METRICS_PATH / "ba_factor_error_stats_initial.txt"
FACTOR_ERROR_STATS_PATH_FINAL = METRICS_PATH / "ba_factor_error_stats_final.txt"
FACTOR_ERROR_CAMERA_VIZ_INITIAL = METRICS_PATH / "ba_camera_error_initial.html"
FACTOR_ERROR_CAMERA_VIZ_FINAL = METRICS_PATH / "ba_camera_error_final.html"
FACTOR_ERROR_POINTS_VIZ_INITIAL = METRICS_PATH / "ba_point_error_colormap_initial.html"
FACTOR_ERROR_POINTS_VIZ_FINAL = METRICS_PATH / "ba_point_error_colormap_final.html"
FACTOR_ERROR_OPTIMIZATION_VIZ = METRICS_PATH / "ba_optimization_progress.html"

"""In this file, we use the GTSAM's GeneralSFMFactor2 instead of GeneralSFMFactor because Factor2 enables decoupling
of the camera pose and the camera intrinsics, and hence gives an option to share the intrinsics between cameras.
"""


CAM_POSE3_DOF = 6  # 6 dof for pose of camera
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logger = logger_utils.get_logger()


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
        robust_measurement_noise: bool = True,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
        cam_pose3_prior_noise_sigma: float = 0.1,
        calibration_prior_noise_sigma: float = 0.05,
        calibration_prior_dist_sigma: float = 1e-5,
        measurement_noise_sigma: float = 2.0,
        allow_indeterminate_linear_system: bool = True,
        print_summary: bool = False,
        ordering_type: str = "METIS",
        save_iteration_visualization: bool = False,
        robust_noise_basin: float = 0.2,
    ) -> None:
        """Initializes the parameters for bundle adjustment module.

        Args:
            reproj_error_thresholds (optional): List of reprojection error thresholds used to perform filtering after
                each global bundle adjustment step. Implicitly defines the number of global BA steps, e.g., if
                len(reproj_error_thresholds) == 1, only one step will be performed. If the threshold is None, no
                filtering on output data is performed. Defaults to None.
            robust_measurement_noise (optional): Flag to enable use of robust noise model for measurement noise.
                Defaults to False.
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
        """
        self._reproj_error_thresholds = reproj_error_thresholds
        self._robust_measurement_noise = robust_measurement_noise
        self._shared_calib = shared_calib
        self._max_iterations = max_iterations
        self._cam_pose3_prior_noise_sigma = cam_pose3_prior_noise_sigma
        self._calibration_prior_noise_sigma = calibration_prior_noise_sigma
        self._calibration_prior_dist_sigma = calibration_prior_dist_sigma
        self._measurement_noise_sigma = measurement_noise_sigma
        self._allow_indeterminate_linear_system = allow_indeterminate_linear_system
        self._ordering_type = ordering_type
        self._print_summary = print_summary
        self._save_iteration_visualization = save_iteration_visualization
        self._robust_noise_basin = robust_noise_basin

    def __map_to_calibration_variable(self, camera_idx: int) -> int:
        return 0 if self._shared_calib else camera_idx

    def __reprojection_factors(
        self, initial_data: GtsfmData, cameras_to_model: List[int], is_fisheye_calibration: bool
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Generate reprojection factors using the tracks."""
        graph = NonlinearFactorGraph()

        # noise model for measurements -- one pixel in u and v
        measurement_noise = Isotropic.Sigma(IMG_MEASUREMENT_DIM, self._measurement_noise_sigma)
        if self._robust_measurement_noise:
            # measurement_noise = Robust(mEstimator.Huber(0.5), measurement_noise)
            measurement_noise = Robust(mEstimator.GemanMcClure(self._robust_noise_basin), measurement_noise)

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
            logger.info(f"cameras without tracks: {cameras_without_tracks.keys()}")

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

    def __calibration_priors(
        self, initial_data: GtsfmData, cameras_to_model: List[int], is_fisheye_calibration: bool
    ) -> NonlinearFactorGraph:
        """Generate prior factors on calibration parameters of the cameras."""
        graph = NonlinearFactorGraph()

        # Note: Assumes all calibration types are the same.
        first_valid_camera_idx = cameras_to_model[0]
        first_camera = initial_data.get_camera(first_valid_camera_idx)
        assert first_camera is not None, "First camera in initial data is None"
        calibration_prior_factor_class = gtsfm_types.get_prior_factor_for_calibration(first_camera.calibration())
        calibration_dim = first_camera.calibration().dim()
        # noise_model = Isotropic.Sigma(calibration_dim, self._calibration_prior_noise_sigma)
        noise_model = gtsfm_types.get_noise_model_for_calibration(
            first_camera.calibration(),
            self._calibration_prior_noise_sigma,
            pp_sigma=1e-5,
            dist_sigma=self._calibration_prior_dist_sigma,
        )
        if self._shared_calib:
            graph.push_back(
                calibration_prior_factor_class(
                    K(self.__map_to_calibration_variable(first_valid_camera_idx)),
                    first_camera.calibration(),
                    noise_model,
                )  # type: ignore
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
                        K(self.__map_to_calibration_variable(i)),
                        camera_i.calibration(),
                        noise_model,
                    )  # type: ignore
                )

        return graph

    def __construct_simple_factor_graph(
        self, cameras_to_model: List[int], initial_data: GtsfmData
    ) -> tuple[NonlinearFactorGraph, Dict[int, gtsfm_types.CAMERA_TYPE]]:
        """Construct the factor graph with just reprojection factors and calibration priors."""
        is_fisheye_calibration = isinstance(initial_data.get_camera(cameras_to_model[0]), PinholeCameraCal3Fisheye)

        graph = NonlinearFactorGraph()

        reprojection_graph, cameras_without_tracks = self.__reprojection_factors(
            initial_data=initial_data,
            cameras_to_model=cameras_to_model,
            is_fisheye_calibration=is_fisheye_calibration,
        )
        # Create a factor graph.
        graph.push_back(reprojection_graph)
        if graph.size() == 0:
            raise ValueError("BundleAdjustmentOptimizer: No reprojection factors available.")

        if not cameras_to_model:
            return graph, {}

        first_camera = initial_data.get_camera(cameras_to_model[0])
        assert first_camera is not None, "First camera in initial data is None"
        # graph.push_back(
        #     gtsam.NonlinearEqualityPose3(
        #         X(cameras_to_model[0]),
        #         first_camera.pose(),
        #     )
        # )
        # graph.push_back(
        #     PriorFactorPose3(
        #         X(cameras_to_model[0]),
        #         first_camera.pose(),
        #         Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
        #     )
        # )
        if len(cameras_to_model) > 1:
            camera_keys = [X(i) for i in cameras_to_model]
            graph.push_back(gtsam.KarcherMeanFactorPose3(camera_keys, 6, 1000))

        # if initial_data.number_tracks() > 0:
        #     graph.push_back(
        #         PriorFactorPoint3(P(0), initial_data.get_track(0).point3(), Isotropic.Sigma(POINT3_DOF, 0.1))
        #     )
        # graph.push_back(self.__calibration_priors(initial_data, cameras_to_model, is_fisheye_calibration))

        return graph, cameras_without_tracks

    def __construct_factor_graph(
        self,
        cameras_to_model: List[int],
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
    ) -> NonlinearFactorGraph:
        """Construct the factor graph with reprojection factors, BetweenFactors, and prior factors."""
        # Create a factor graph.
        graph, _ = self.__construct_simple_factor_graph(cameras_to_model, initial_data)

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
        # Also add a prior on the position of the first landmark to fix the scale
        graph.push_back(
            gtsam.PriorFactorPoint3(P(0), initial_data.get_track(0).point3(), Isotropic.Sigma(POINT3_DOF, 0.1))
        )

        return graph

    def __optimize_factor_graph(
        self, graph: NonlinearFactorGraph, initial_values: Values, ordering_type: str
    ) -> Tuple[Values, Optional[List[Values]]]:
        """Optimize the factor graph, optionally capturing per-iteration values."""
        start_time = time.time()

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR" if not self._print_summary else "SUMMARY")
        params.setOrderingType(ordering_type)
        # params.setAbsoluteErrorTol(1.0)
        if self._max_iterations:
            params.setMaxIterations(self._max_iterations)

        # gnc_params = gtsam.GncLMParams()
        # gnc_params.setLossType(gtsam.GncLossType.GM)

        lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
        # lm = gtsam.GncLMOptimizer(graph, initial_values, gnc_params)

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
        self, initial_data: GtsfmData, graph: NonlinearFactorGraph, ordering_type: str, verbose: bool
    ) -> Tuple[GtsfmData, Values, float]:
        """Optimize the graph, report errors, and convert `Values` back to `GtsfmData`."""
        initial_values = initial_data.to_values(shared_calib=self._shared_calib)
        if verbose:
            self.__write_factor_errors(graph, initial_values, FACTOR_ERROR_LOG_PATH_INITIAL)
            self.__write_factor_error_stats(graph, initial_values, FACTOR_ERROR_STATS_PATH_INITIAL)
        # filtered_graph, filtered_values = self.__filter_graph_and_values(
        #     graph,
        #     initial_values,
        #     max_factor_error=2000.0,
        #     max_point_mean_error=1000.0,
        #     min_factors_per_point=2,
        # )
        if verbose:
            self.__write_factor_error_visualizations(
                graph,
                initial_values,
                FACTOR_ERROR_CAMERA_VIZ_INITIAL,
                FACTOR_ERROR_POINTS_VIZ_INITIAL,
            )
        result_values, values_trace = self.__optimize_factor_graph(graph, initial_values, ordering_type)
        final_error = graph.error(result_values)
        if verbose:
            logger.info("initial error: %.2f", graph.error(initial_values))
            logger.info("final error: %.2f", final_error)
            self.__write_factor_errors(graph, result_values, FACTOR_ERROR_LOG_PATH_FINAL)
            self.__write_factor_error_stats(graph, result_values, FACTOR_ERROR_STATS_PATH_FINAL)
            self.__write_factor_error_visualizations(
                graph,
                result_values,
                FACTOR_ERROR_CAMERA_VIZ_FINAL,
                FACTOR_ERROR_POINTS_VIZ_FINAL,
            )
            if self._save_iteration_visualization and values_trace is not None:
                self.__write_iteration_visualization(
                    values_trace,
                    initial_values,
                    FACTOR_ERROR_OPTIMIZATION_VIZ,
                )
        optimized_data = GtsfmData.from_values(result_values, initial_data, self._shared_calib)
        return optimized_data, result_values, final_error

    def __write_factor_errors(self, graph: NonlinearFactorGraph, values: Values, output_path: Path) -> None:
        """Write per-factor errors to a text file using GTSAM's printErrors()."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as output_file:
            # GTSAM prints to C++ stdout, so redirect the file descriptor.
            output_fd = output_file.fileno()
            stdout_fd = os.dup(1)
            try:
                os.dup2(output_fd, 1)
                graph.printErrors(values)
            finally:
                os.dup2(stdout_fd, 1)
                os.close(stdout_fd)

    def __write_factor_graph(self, graph: NonlinearFactorGraph, output_path: Path) -> None:
        """Write factor graph structure to a text file using GTSAM's print()."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as output_file:
            output_fd = output_file.fileno()
            stdout_fd = os.dup(1)
            try:
                os.dup2(output_fd, 1)
                graph.print("NonlinearFactorGraph")
            finally:
                os.dup2(stdout_fd, 1)
                os.close(stdout_fd)

    def __filter_graph_and_values(
        self,
        graph: NonlinearFactorGraph,
        values: Values,
        max_factor_error: float,
        max_point_mean_error: float,
        min_factors_per_point: int,
    ) -> Tuple[NonlinearFactorGraph, Values]:
        """Filter factors and points using factor errors computed from provided values."""
        total_factors = int(graph.size())
        filtered_graph = NonlinearFactorGraph()

        def _factor_at(index: int):
            try:
                return graph.at(index)
            except AttributeError:
                return graph[index]

        def _key_type(key: int) -> str:
            try:
                return gtsam.Symbol(key).string()[0]
            except Exception:
                return "?"

        def _point_keys(factor) -> list[int]:
            try:
                return [int(k) for k in factor.keys() if _key_type(int(k)) == "p"]
            except Exception:
                return []

        retained_factors = []
        for i in range(total_factors):
            factor = _factor_at(i)
            if factor is None:
                continue
            try:
                error = float(factor.error(values))
            except RuntimeError:
                continue
            if error <= max_factor_error:
                retained_factors.append((factor, error))

        point_error_sums: dict[int, float] = {}
        point_factor_counts: dict[int, int] = {}
        for factor, error in retained_factors:
            for key in _point_keys(factor):
                point_error_sums[key] = point_error_sums.get(key, 0.0) + error
                point_factor_counts[key] = point_factor_counts.get(key, 0) + 1

        points_to_remove = {
            key
            for key, count in point_factor_counts.items()
            if count < min_factors_per_point or (point_error_sums.get(key, 0.0) / max(count, 1)) > max_point_mean_error
        }

        for factor, _ in retained_factors:
            point_keys = _point_keys(factor)
            if any(key in points_to_remove for key in point_keys):
                continue
            filtered_graph.push_back(factor)

        if points_to_remove:
            for key in points_to_remove:
                try:
                    if values.exists(key):
                        values.erase(key)
                except Exception:
                    continue

        removed_factor_count = total_factors - int(filtered_graph.size())
        if removed_factor_count > 0 or points_to_remove:
            logger.info(
                "Filtered factors: removed %d/%d, removed %d point keys",
                removed_factor_count,
                total_factors,
                len(points_to_remove),
            )

        return filtered_graph, values

    def __write_factor_error_stats(self, graph: NonlinearFactorGraph, values: Values, output_path: Path) -> None:
        """Write statistics for per-factor errors and key concentration."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        thresholds = [1e3, 1e4]
        total_factors = int(graph.size())
        errors: list[tuple[int, float, list[str]]] = []
        key_counts_by_threshold: dict[float, dict[str, int]] = {t: {} for t in thresholds}
        key_type_counts_by_threshold: dict[float, dict[str, int]] = {t: {} for t in thresholds}
        key_error_sums: dict[str, float] = {}

        def _factor_at(index: int):
            try:
                return graph.at(index)
            except AttributeError:
                return graph[index]

        def _format_key(key: int) -> str:
            try:
                return gtsam.Symbol(key).string()
            except Exception:
                return str(key)

        for i in range(total_factors):
            factor = _factor_at(i)
            if factor is None:
                continue
            try:
                error = float(factor.error(values))
            except RuntimeError:
                continue
            try:
                keys = [int(k) for k in factor.keys()]
            except Exception:
                keys = []
            key_strs = [_format_key(k) for k in keys]
            errors.append((i, error, key_strs))
            for key_str in key_strs:
                key_error_sums[key_str] = key_error_sums.get(key_str, 0.0) + error
            for threshold in thresholds:
                if error >= threshold:
                    for key_str in key_strs:
                        key_counts_by_threshold[threshold][key_str] = (
                            key_counts_by_threshold[threshold].get(key_str, 0) + 1
                        )
                        key_type = key_str[0] if key_str else "?"
                        key_type_counts_by_threshold[threshold][key_type] = (
                            key_type_counts_by_threshold[threshold].get(key_type, 0) + 1
                        )

        errors_sorted = sorted(errors, key=lambda item: item[1], reverse=True)

        with output_path.open("w") as output_file:
            output_file.write(f"total_factors: {total_factors}\n")
            if not errors_sorted:
                output_file.write("no_errors_recorded\n")
                return
            all_errors = [e for _, e, _ in errors_sorted]
            output_file.write(f"min_error: {min(all_errors):.6f}\n")
            output_file.write(f"max_error: {max(all_errors):.6f}\n")
            output_file.write(f"median_error: {np.median(all_errors):.6f}\n")
            output_file.write(f"mean_error: {np.mean(all_errors):.6f}\n")
            output_file.write(f"p95_error: {np.percentile(all_errors, 95):.6f}\n")
            output_file.write(f"p99_error: {np.percentile(all_errors, 99):.6f}\n")

            for threshold in thresholds:
                count = sum(1 for _, e, _ in errors_sorted if e >= threshold)
                fraction = count / total_factors if total_factors else 0.0
                output_file.write(f"\nerrors_ge_{int(threshold)}: {count} ({fraction:.4%})\n")
                key_counts = key_counts_by_threshold[threshold]
                key_type_counts = key_type_counts_by_threshold[threshold]
                if key_type_counts:
                    output_file.write("key_type_counts:\n")
                    for key_type, kt_count in sorted(key_type_counts.items(), key=lambda item: item[1], reverse=True)[
                        :10
                    ]:
                        output_file.write(f"  {key_type}: {kt_count}\n")
                if key_counts:
                    output_file.write("top_keys_by_category:\n")
                    for category in ["x", "p", "k", "?"]:
                        category_keys = {
                            key_str: key_count
                            for key_str, key_count in key_counts.items()
                            if (key_str[0] if key_str else "?") == category
                        }
                        if not category_keys:
                            continue
                        output_file.write(f"  category {category}:\n")
                        for key_str, key_count in sorted(category_keys.items(), key=lambda item: item[1], reverse=True)[
                            :20
                        ]:
                            output_file.write(f"    {key_str}: {key_count}\n")

            if key_error_sums:
                output_file.write("\nkey_error_sums_by_category:\n")
                for category in ["x", "p", "k", "?"]:
                    category_sums = {
                        key_str: total_error
                        for key_str, total_error in key_error_sums.items()
                        if (key_str[0] if key_str else "?") == category
                    }
                    if not category_sums:
                        continue
                    output_file.write(f"  category {category}:\n")
                    for key_str, total_error in sorted(category_sums.items(), key=lambda item: item[1], reverse=True)[
                        :20
                    ]:
                        output_file.write(f"    {key_str}: {total_error:.6f}\n")

            output_file.write("\nworst_factors:\n")
            for factor_index, error, key_strs in errors_sorted[:50]:
                key_list = ", ".join(key_strs)
                output_file.write(f"  factor {factor_index}: error={error:.6f} keys={{ {key_list} }}\n")

    def __write_factor_error_visualizations(
        self, graph: NonlinearFactorGraph, values: Values, camera_output_path: Path, point_output_path: Path
    ) -> None:
        """Write Plotly HTML visualizations for camera and point error distributions."""
        try:
            import plotly.graph_objects as go  # type: ignore
            import plotly.io as pio  # type: ignore
        except Exception:
            logger.warning("Plotly not available, skipping factor error visualizations.")
            return

        camera_output_path.parent.mkdir(parents=True, exist_ok=True)
        point_output_path.parent.mkdir(parents=True, exist_ok=True)

        def _factor_at(index: int):
            try:
                return graph.at(index)
            except AttributeError:
                return graph[index]

        def _format_key(key: int) -> str:
            try:
                return gtsam.Symbol(key).string()
            except Exception:
                return str(key)

        camera_error_lists: dict[int, list[float]] = {}
        point_error_lists: dict[int, list[float]] = {}

        for i in range(int(graph.size())):
            factor = _factor_at(i)
            if factor is None:
                continue
            try:
                error = float(factor.error(values))
            except RuntimeError:
                continue
            try:
                keys = [int(k) for k in factor.keys()]
            except Exception:
                keys = []
            for key in keys:
                key_str = _format_key(key)
                key_type = key_str[0] if key_str else "?"
                if key_type == "x":
                    camera_error_lists.setdefault(key, []).append(error)
                elif key_type == "p":
                    point_error_lists.setdefault(key, []).append(error)

        if camera_error_lists:
            camera_means = {key: float(np.mean(errors)) for key, errors in camera_error_lists.items()}
            cam_keys = [_format_key(key) for key in camera_means.keys()]
            cam_errors = [camera_means[key] for key in camera_means.keys()]
            fig = go.Figure(data=[go.Bar(x=cam_keys, y=cam_errors)])
            fig.update_layout(
                title="Camera error means by key",
                xaxis_title="Camera key",
                yaxis_title="Mean factor error",
                xaxis_tickangle=45,
                margin=dict(l=40, r=20, t=40, b=120),
            )
            pio.write_html(fig, file=str(camera_output_path), auto_open=False)

        if point_error_lists:
            points_xyz = []
            point_errors = []
            point_labels = []
            point_means = {key: float(np.mean(errors)) for key, errors in point_error_lists.items()}
            for key, error_mean in point_means.items():
                try:
                    point = values.atPoint3(key)
                except Exception:
                    continue
                points_xyz.append(point)
                point_errors.append(error_mean)
                point_labels.append(_format_key(key))

            if points_xyz:
                xyz = np.asarray(points_xyz)
                errors_np = np.asarray(point_errors, dtype=float)
                percentiles = [50, 60, 70, 80, 90, 95, 98, 99]
                clip_values = {p: float(np.percentile(errors_np, p)) for p in percentiles}

                def _clipped_colors(clip_max: float) -> np.ndarray:
                    return np.minimum(errors_np, clip_max)

                initial_percentile = 80
                initial_clip = clip_values[initial_percentile]
                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=xyz[:, 0],
                            y=xyz[:, 1],
                            z=xyz[:, 2],
                            mode="markers",
                            marker=dict(
                                size=3,
                                color=_clipped_colors(initial_clip),
                                colorscale="Viridis",
                                opacity=0.8,
                                colorbar=dict(title="Mean factor error"),
                            ),
                            text=point_labels,
                        )
                    ]
                )

                steps = []
                for p in percentiles:
                    clip_max = clip_values[p]
                    steps.append(
                        dict(
                            method="update",
                            label=f"p{p}",
                            args=[
                                {"marker.color": [_clipped_colors(clip_max)]},
                                {"title": f"Point error colormap (clipped at p{p}={clip_max:.3f})"},
                            ],
                        )
                    )

                fig.update_layout(
                    title=f"Point error colormap (mean errors, clipped at p{initial_percentile}={initial_clip:.3f})",
                    scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
                    margin=dict(l=0, r=0, t=40, b=0),
                    sliders=[
                        dict(
                            active=percentiles.index(initial_percentile),
                            currentvalue={"prefix": "Clip: "},
                            pad={"t": 30},
                            steps=steps,
                        )
                    ],
                )
                pio.write_html(fig, file=str(point_output_path), auto_open=False)

    def __write_iteration_visualization(
        self, values_trace: List[Values], initial_values: Values, output_path: Path
    ) -> None:
        """Write a Plotly animation showing optimization progress over iterations."""
        try:
            import plotly.graph_objects as go  # type: ignore
            import plotly.io as pio  # type: ignore
            import visu3d as v3d  # type: ignore
        except Exception:
            logger.warning("Plotly not available, skipping optimization visualization.")
            return

        output_path.parent.mkdir(parents=True, exist_ok=True)

        def _symbol_char(key: int) -> str:
            try:
                return chr(gtsam.Symbol(key).chr())
            except Exception:
                return "?"

        def _symbol_index(key: int) -> int:
            try:
                return int(gtsam.Symbol(key).index())
            except Exception:
                return -1

        def _point_keys(values: Values) -> list[int]:
            return [int(k) for k in values.keys() if _symbol_char(int(k)) == "p"]

        def _camera_keys(values: Values) -> list[int]:
            keys = [int(k) for k in values.keys() if _symbol_char(int(k)) == "x"]
            return sorted(keys, key=_symbol_index)

        point_keys = _point_keys(initial_values)
        camera_keys = _camera_keys(initial_values)
        if not camera_keys:
            logger.warning("No camera keys found for optimization visualization.")
            return

        max_points = 20000
        if len(point_keys) > max_points:
            indices = np.linspace(0, len(point_keys) - 1, max_points, dtype=int)
            point_keys = [point_keys[i] for i in indices]

        def _pose_to_v3d_matrix(pose: gtsam.Pose3) -> np.ndarray:
            matrix = np.concatenate([0.1 * pose.rotation().matrix(), pose.translation()[:, None]], axis=-1)
            matrix = np.concatenate([matrix, np.array([[0, 0, 0, 1]], dtype=np.float64)], axis=0)
            return matrix.astype(np.float32)

        def _extract_frame(values: Values) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            points = []
            for key in point_keys:
                try:
                    points.append(values.atPoint3(key))
                except Exception:
                    points.append([np.nan, np.nan, np.nan])
            camera_matrices = []
            for key in camera_keys:
                try:
                    pose = values.atPose3(key)
                    camera_matrices.append(_pose_to_v3d_matrix(pose))
                except Exception:
                    camera_matrices.append(np.full((4, 4), np.nan, dtype=np.float32))
            first_camera_matrix = camera_matrices[0:1]
            return (
                np.asarray(points, dtype=float),
                np.asarray(camera_matrices, dtype=np.float32),
                np.asarray(first_camera_matrix, dtype=np.float32),
            )

        initial_points, initial_camera_matrices, initial_first_camera = _extract_frame(values_trace[0])
        camera_traces = v3d.make_fig(
            [
                v3d.Transform.from_matrix(initial_camera_matrices),
                v3d.Transform.from_matrix(initial_first_camera),
            ]
        )
        fig = go.Figure(data=list(camera_traces.data))
        fig.add_trace(
            go.Scatter3d(
                x=initial_points[:, 0],
                y=initial_points[:, 1],
                z=initial_points[:, 2],
                mode="markers",
                marker=dict(size=2, color="rgba(31, 119, 180, 0.35)"),
                name="Points",
            )
        )

        frames = []
        for idx, values in enumerate(values_trace):
            points, camera_matrices, first_camera_matrix = _extract_frame(values)
            camera_frame = v3d.make_fig(
                [
                    v3d.Transform.from_matrix(camera_matrices),
                    v3d.Transform.from_matrix(first_camera_matrix),
                ]
            )
            frame_traces = list(camera_frame.data)
            frame_traces = frame_traces + [
                go.Scatter3d(
                    x=points[:, 0],
                    y=points[:, 1],
                    z=points[:, 2],
                    mode="markers",
                )
            ]
            frames.append(
                go.Frame(
                    data=frame_traces,
                    name=str(idx),
                )
            )

        fig.frames = frames
        fig.update_layout(
            title="BA optimization progress (points + cameras)",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, t=40, b=0),
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    y=1.05,
                    x=1.0,
                    xanchor="right",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="Play",
                            method="animate",
                            args=[None, {"frame": {"duration": 200, "redraw": False}, "fromcurrent": True}],
                        ),
                        dict(
                            label="Pause",
                            method="animate",
                            args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "Iter: "},
                    pad={"t": 30},
                    steps=[
                        dict(method="animate", label=str(i), args=[[str(i)], {"mode": "immediate"}])
                        for i in range(len(values_trace))
                    ],
                )
            ],
        )
        pio.write_html(fig, file=str(output_path), auto_open=False)

    def run_simple_ba(
        self, initial_data: GtsfmData, verbose: bool = True, factor_graph_output_path: Optional[str] = None
    ) -> Tuple[GtsfmData, float]:
        """Runs bundle adjustment and optionally filters the resulting tracks by reprojection error.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Final error value of the optimization problem.
        """
        cameras_to_model = sorted(initial_data.get_valid_camera_indices())

        graph, cameras_without_tracks = self.__construct_simple_factor_graph(cameras_to_model, initial_data)

        if factor_graph_output_path is not None:
            self.__write_factor_graph(graph, Path(factor_graph_output_path))
        optimized_data, result_values, final_error = self.__optimize_and_recover(
            initial_data, graph, self._ordering_type if not cameras_without_tracks else "COLAMD", verbose
        )
        # print("final error is ", final_error)
        final_T_i0 = result_values.atPose3(X(cameras_to_model[0]))
        init_T_i0 = initial_data.get_camera(cameras_to_model[0]).pose()
        init_T_final = init_T_i0.compose(final_T_i0.inverse())
        transformed_values = Values()

        for c in cameras_to_model:
            transformed_values.insert(X(c), init_T_final.compose(optimized_data.get_camera(c).pose()))
        for t in range(optimized_data.number_tracks()):
            transformed_values.insert(P(t), init_T_final.transformFrom(optimized_data.get_track(t).point3()))
        camera_ids = [cameras_to_model[0]] if self._shared_calib else cameras_to_model
        for c in camera_ids:
            transformed_values.insert(
                K(self.__map_to_calibration_variable(c)), optimized_data.get_camera(c).calibration()
            )
        logger.info("composed new error is %f", graph.error(transformed_values))
        for i, cam in cameras_without_tracks.items():
            optimized_data.add_camera(i, cam)
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
        graph = self.__construct_factor_graph(
            cameras_to_model, initial_data, absolute_pose_priors, relative_pose_priors
        )
        optimized_data, result_values, final_error = self.__optimize_and_recover(initial_data, graph, verbose)

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

        return optimized_data, filtered_result, valid_mask

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

        metrics = self.evaluate(optimized_data, filtered_result, cameras_gt, save_dir)
        for i, step_time in enumerate(step_times):
            metrics.add_metric(GtsfmMetric(f"step_{i}_run_duration_sec", step_time))
        metrics.add_metric(GtsfmMetric("total_run_duration_sec", total_time))

        return optimized_data, filtered_result, valid_mask, metrics

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

        input_image_idxs = unfiltered_data.get_valid_camera_indices()
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
