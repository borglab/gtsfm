"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import numpy as np
from pathlib import Path
from typing import List, NamedTuple, Tuple

import dask
from dask.delayed import Delayed
from gtsam import (
    GeneralSFMFactor2Cal3Bundler,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    NonlinearFactorGraph,
    PinholeCameraCal3Bundler,
    PriorFactorCal3Bundler,
    PriorFactorPose3,
    SfmTrack,
    Values,
    symbol_shorthand,
)
from gtsam.noiseModel import Isotropic, mEstimator, Robust

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"

# TODO: any way this goes away?
P = symbol_shorthand.P  # 3d point
X = symbol_shorthand.X  # camera pose
K = symbol_shorthand.K  # calibration

CAM_POSE3_DOF = 6  # 6 dof for pose of camera
CAM_CAL3BUNDLER_DOF = 3  # 3 dof for f, k1, k2 for intrinsics of camera
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logger = logger_utils.get_logger()


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation."""

    def __init__(
        self, output_reproj_error_thresh: float, shared_calib: bool = False, robust_measurement_noise: bool = False
    ):
        """Initializes the parameters for bundle adjustment module.

        Args:
            output_reproj_error_thresh: the max reprojection error allowed in output.
            shared_calib (optional): Flag to enable shared calibration across all cameras. Defaults to False.
            robust_measurement_noise (optional): Flag to enable use of robust noise model for measurement noise.
                                                 Defaults to False.
            
        """
        self._output_reproj_error_thresh = output_reproj_error_thresh
        self._shared_calib = shared_calib
        self._robust_measurement_noise = robust_measurement_noise

    def __add_camera_prior_and_initial_value(
        self, graph: NonlinearFactorGraph, initial_values: Values, camera: PinholeCameraCal3Bundler, camera_idx: int,
    ) -> None:
        """Add a prior factor in the factor graph and initial values for the
        camera parameters.

        Args:
            graph: factor graph for the problem.
            initial_values: object holding the initial values for camera
                            parameters and 3d points.
            camera: the camera object to add to graph and to initial values.
            camera_idx: index of the camera.
        """

        # add prior factor for pose
        graph.push_back(PriorFactorPose3(X(camera_idx), camera.pose(), Isotropic.Sigma(CAM_POSE3_DOF, 0.1),))

        # add initial value for pose
        initial_values.insert(X(camera_idx), camera.pose())

        # process just one camera if shared calibration; otherwise process all
        if camera_idx == 0 or not self._shared_calib:
            # add prior factor for calibration
            graph.push_back(
                PriorFactorCal3Bundler(K(camera_idx), camera.calibration(), Isotropic.Sigma(CAM_CAL3BUNDLER_DOF, 0.1),)
            )

            # add initial value for calibration
            initial_values.insert(K(camera_idx), camera.calibration())

    def __add_measurement_factors_and_initial_values(
        self,
        graph: NonlinearFactorGraph,
        initial_values: Values,
        track: SfmTrack,
        track_idx: int,
        measurement_noise: Isotropic,  # TODO: change it
    ) -> None:
        """Add prior factor for each 2D measurement and initial values for each
        3d point.

        Args:
            graph: factor graph for the problem.
            initial_values: object holding the initial values for camera
                            parameters and 3d points.
            track: [description]
            measurement_noise (Isotropic): [description]
        """

        # Add measurements to the factor graph
        for m_idx in range(track.number_measurements()):
            # i represents the camera index, and uv is the 2d measurement
            i, uv = track.measurement(m_idx)
            graph.add(
                GeneralSFMFactor2Cal3Bundler(
                    uv, measurement_noise, X(i), P(track_idx), K(0 if self._shared_calib else i),
                )
            )

        # add initial value for 3d point
        initial_values.insert(P(track_idx), track.point3())

    def run(self, initial_data: GtsfmData) -> Tuple[GtsfmData, GtsfmMetricsGroup]:
        """Run the bundle adjustment by forming factor graph and optimizing using Levenberg–Marquardt optimization.

        Args:
            initial_data: initialized cameras, tracks w/ their 3d landmark from triangulation.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics.
            Metrics group containing metrics for both filtered and unfiltered BA results.
        """
        logger.info(
            f"Input: {initial_data.number_tracks()} tracks on {len(initial_data.get_valid_camera_indices())} cameras\n"
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # no cameras or tracks to optimize, so bundle adjustment is not possible
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return initial_data

        # noise model for measurements -- one pixel in u and v
        if self._robust_measurement_noise:
            measurement_noise = Robust(mEstimator.Huber(1.35), Isotropic.Sigma(IMG_MEASUREMENT_DIM, 1.0),)
        else:
            measurement_noise = Isotropic.Sigma(IMG_MEASUREMENT_DIM, 1.0)

        # Create a factor graph
        graph = NonlinearFactorGraph()

        # Create initial estimate
        initial_values = Values()

        # adding factors for measurements and 3D points's initial values
        for j in range(initial_data.number_tracks()):
            self.__add_measurement_factors_and_initial_values(
                graph, initial_values, initial_data.get_track(j), j, measurement_noise,
            )
        )
        # Also add a prior on the position of the first landmark to fix the scale
        graph.push_back(
            gtsam.PriorFactorPoint3(
                P(0), initial_data.get_track(0).point3(), gtsam.noiseModel.Isotropic.Sigma(POINT3_DOF, 0.1)
            )
        )

        # add prior on all camera poses
        for i in initial_data.get_valid_camera_indices():
            initialized_cam = initial_data.get_camera(i)

            self.__add_camera_prior_and_initial_value(graph, initial_values, initialized_cam, i)

        # Optimize the graph and print results
        try:
            params = LevenbergMarquardtParams()
            params.setVerbosityLM("ERROR")
            lm = LevenbergMarquardtOptimizer(graph, initial_values, params)
            result_values = lm.optimize()
        except Exception:
            logger.exception("LM Optimization failed")
            # as we did not perform the bundle adjustment, we skip computing the total reprojection error
            return GtsfmData(initial_data.number_images())

        initial_error = graph.error(initial_values)
        final_error = graph.error(result_values)

        # Error drops from ~2764.22 to ~0.046
        logger.info(f"initial error: {initial_error:.2f}")
        logger.info(f"final error: {final_error:.2f}")

        # construct the results
        optimized_data = values_to_gtsfm_data(result_values, initial_data, self._shared_calib)

        def get_metrics_from_sfm_data(sfm_data: GtsfmData, suffix: str) -> List[GtsfmMetric]:
            """Helper to get bundle adjustment metrics from a GtsfmData object with a suffix for metric names."""
            metrics = []
            metrics.append(GtsfmMetric("number_tracks" + suffix, sfm_data.number_tracks()))
            metrics.append(GtsfmMetric("3d_track_lengths" + suffix, sfm_data.get_track_lengths()))
            metrics.append(GtsfmMetric("reprojection_errors" + suffix, sfm_data.get_scene_reprojection_errors()))
            return metrics

        ba_metrics = GtsfmMetricsGroup(
            "bundle_adjustment_metrics", get_metrics_from_sfm_data(optimized_data, suffix="_unfiltered")
        )
        logger.info("[Result] Number of tracks before filtering: %d", optimized_data.number_tracks())

        # filter the largest errors
        filtered_result = optimized_data.filter_landmarks(self._output_reproj_error_thresh)

        ba_metrics.add_metrics(get_metrics_from_sfm_data(filtered_result, suffix="_filtered"))
        # ba_metrics.save_to_json(os.path.join(METRICS_PATH, "bundle_adjustment_metrics.json"))

        logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())
        logger.info("[Result] Mean track length %.3f", np.mean(filtered_result.get_track_lengths()))
        logger.info("[Result] Median track length %.3f", np.median(filtered_result.get_track_lengths()))
        filtered_result.log_scene_reprojection_error_stats()

        return filtered_result, ba_metrics

    def create_computation_graph(self, sfm_data_graph: Delayed) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: an GtsfmData object wrapped up using dask.delayed

        Returns:
            GtsfmData wrapped up using dask.delayed
            Metrics group for BA results, wrapped up using dask.delayed
        """
        data_metrics_graph = dask.delayed(self.run)(sfm_data_graph)
        return data_metrics_graph[0], data_metrics_graph[1]


def values_to_gtsfm_data(values: Values, initial_data: GtsfmData, shared_calib: bool) -> GtsfmData:
    """Cast results from the optimization to GtsfmData object.

    Args:
        values: results of factor graph optimization.
        initial_data: data used to generate the factor graph; used to extract information about poses and 3d points in
                      the graph.

    Returns:
        optimized poses and landmarks.
    """
    result = GtsfmData(initial_data.number_images())

    # add cameras
    for i in initial_data.get_valid_camera_indices():
        result.add_camera(
            i, PinholeCameraCal3Bundler(values.atPose3(X(i)), values.atCal3Bundler(K(0 if shared_calib else i)),)
        )

    # add tracks
    for j in range(initial_data.number_tracks()):
        input_track = initial_data.get_track(j)

        # populate the result with optimized 3D point
        result_track = SfmTrack(values.atPoint3(P(j)))

        for measurement_idx in range(input_track.number_measurements()):
            i, uv = input_track.measurement(measurement_idx)
            result_track.add_measurement(i, uv)

        result.add_track(result_track)

    return result
