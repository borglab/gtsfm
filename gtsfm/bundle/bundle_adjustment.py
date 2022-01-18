"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import dask
import gtsam
import numpy as np
from dask.delayed import Delayed
from gtsam import (
    GeneralSFMFactor2Cal3Bundler,
    NonlinearFactorGraph,
    PinholeCameraCal3Bundler,
    PriorFactorCal3Bundler,
    PriorFactorPose3,
    SfmTrack,
    Values,
    symbol_shorthand,
)

import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

METRICS_GROUP = "bundle_adjustment_metrics"

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"

"""In this file, we use the GTSAM's GeneralSFMFactor2 instead of GeneralSFMFactor because Factor2 enables decoupling
of the camera pose and the camera intrinsics, and hence gives an option to share the intrinsics between cameras.
"""

P = symbol_shorthand.P  # 3d point
X = symbol_shorthand.X  # camera pose
K = symbol_shorthand.K  # calibration

CAM_POSE3_DOF = 6  # 6 dof for pose of camera
CAM_CAL3BUNDLER_DOF = 3  # 3 dof for f, k1, k2 for intrinsics of camera
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof


# noise model params
CAM_POSE3_PRIOR_NOISE_SIGMA = 0.1
CAM_CAL3BUNDLER_PRIOR_NOISE_SIGMA = 1e-5  # essentially fixed
MEASUREMENT_NOISE_SIGMA = 1.0  # in pixels

logger = logger_utils.get_logger()


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation."""

    def __init__(
        self,
        output_reproj_error_thresh: Optional[float] = None,
        robust_measurement_noise: bool = False,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
    ):
        """Initializes the parameters for bundle adjustment module.

        Args:
            output_reproj_error_thresh (optional): the max reprojection error allowed in output. If the threshold is
                                                   none, no filtering on output data is performed. Defaults to None.
            robust_measurement_noise (optional): Flag to enable use of robust noise model for measurement noise.
                                                 Defaults to False.
            shared_calib (optional): Flag to enable shared calibration across all cameras. Defaults to False.
            max_iterations (optional): Max number of iterations when optimizing the factor graph. None means no cap.
                                       Defaults to None.

        """
        self._output_reproj_error_thresh = output_reproj_error_thresh
        self._robust_measurement_noise = robust_measurement_noise
        self._shared_calib = shared_calib
        self._max_iterations = max_iterations

    def __map_to_calibration_variable(self, camera_idx: int) -> int:
        return 0 if self._shared_calib else camera_idx

    def __construct_factor_graph(self, initial_data: GtsfmData) -> NonlinearFactorGraph:
        graph = NonlinearFactorGraph()

        # noise model for measurements -- one pixel in u and v
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(IMG_MEASUREMENT_DIM, MEASUREMENT_NOISE_SIGMA)
        if self._robust_measurement_noise:
            measurement_noise = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.345), measurement_noise)

        # Create a factor graph

        # Add measurements to the factor graph
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            # retrieve the SfmMeasurement objects
            for m_idx in range(track.number_measurements()):
                # i represents the camera index, and uv is the 2d measurement
                i, uv = track.measurement(m_idx)
                # note use of shorthand symbols C and P
                graph.add(
                    GeneralSFMFactor2Cal3Bundler(
                        uv,
                        measurement_noise,
                        X(i),
                        P(j),
                        K(self.__map_to_calibration_variable(i)),
                    )
                )

        # get all the valid camera indices, which need to be added to the graph.
        valid_camera_indices = initial_data.get_valid_camera_indices()

        # Add a prior on first pose. This indirectly specifies where the origin is.
        graph.push_back(
            PriorFactorPose3(
                X(valid_camera_indices[0]),
                initial_data.get_camera(valid_camera_indices[0]).pose(),
                gtsam.noiseModel.Isotropic.Sigma(CAM_POSE3_DOF, CAM_POSE3_PRIOR_NOISE_SIGMA),
            )
        )

        # add prior on all calibrations
        for i in valid_camera_indices[: 1 if self._shared_calib else len(valid_camera_indices)]:
            graph.push_back(
                PriorFactorCal3Bundler(
                    K(self.__map_to_calibration_variable(i)),
                    initial_data.get_camera(i).calibration(),
                    gtsam.noiseModel.Isotropic.Sigma(CAM_CAL3BUNDLER_DOF, CAM_CAL3BUNDLER_PRIOR_NOISE_SIGMA),
                )
            )

        # Also add a prior on the position of the first landmark to fix the scale
        graph.push_back(
            gtsam.PriorFactorPoint3(
                P(0), initial_data.get_track(0).point3(), gtsam.noiseModel.Isotropic.Sigma(POINT3_DOF, 0.1)
            )
        )

        return graph

    def __construct_initial_values(self, initial_data: GtsfmData) -> Values:
        # Create initial estimate
        initial_values = gtsam.Values()

        # add each camera
        for loop_idx, i in enumerate(initial_data.get_valid_camera_indices()):
            camera = initial_data.get_camera(i)
            initial_values.insert(X(i), camera.pose())
            if not self._shared_calib or loop_idx == 0:
                # add only one value if calibrations are shared
                initial_values.insert(K(self.__map_to_calibration_variable(i)), camera.calibration())

        # add each SfmTrack
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)
            initial_values.insert(P(j), track.point3())

        return initial_values

    def __optimize_factor_graph(self, graph: NonlinearFactorGraph, initial_values: Values) -> Values:
        # Configure optimizer.
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        if self._max_iterations:
            params.setMaxIterations(self._max_iterations)
        lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)

        result_values = lm.optimize()
        return result_values

    def run(
        self,
        initial_data: GtsfmData,
        verbose: bool = True,
    ) -> Tuple[GtsfmData, GtsfmMetricsGroup]:
        """Run the bundle adjustment by forming factor graph and optimizing using Levenberg–Marquardt optimization.

        Args:
            initial_data: initialized cameras, tracks w/ their 3d landmark from triangulation.
            cameras_gt: list of GT cameras, ordered by camera index.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
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
            return initial_data, GtsfmMetricsGroup(name=METRICS_GROUP, metrics=[])

        graph = self.__construct_factor_graph(initial_data=initial_data)
        initial_values = self.__construct_initial_values(initial_data=initial_data)
        result_values = self.__optimize_factor_graph(graph, initial_values)

        final_error = graph.error(result_values)

        # Error drops from ~2764.22 to ~0.046
        if verbose:
            logger.info(f"initial error: {graph.error(initial_values):.2f}")
            logger.info(f"final error: {final_error:.2f}")

        # construct the results
        optimized_data = values_to_gtsfm_data(result_values, initial_data, self._shared_calib)

        if verbose:
            logger.info("[Result] Number of tracks before filtering: %d", optimized_data.number_tracks())

        # filter the largest errors
        if self._output_reproj_error_thresh:
            filtered_result = optimized_data.filter_landmarks(self._output_reproj_error_thresh)
        else:
            filtered_result = optimized_data

        logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())

        return optimized_data, filtered_result

    def evaluate(
        self, unfiltered_data: GtsfmData, filtered_data: GtsfmData, cameras_gt: List[PinholeCameraCal3Bundler] = None
    ) -> GtsfmMetricsGroup:
        ba_metrics = GtsfmMetricsGroup(
            name=METRICS_GROUP, metrics=metrics_utils.get_stats_for_sfmdata(unfiltered_data, suffix="_unfiltered")
        )

        if cameras_gt is not None:
            poses_gt = [cam.pose() for cam in cameras_gt]

            # align the sparse multi-view estimate after BA to the ground truth pose graph.
            aligned_filtered_data = filtered_data.align_via_Sim3_to_poses(wTi_list_ref=poses_gt)
            ba_pose_error_metrics = metrics_utils.compute_ba_pose_metrics(
                gt_wTi_list=poses_gt, ba_output=aligned_filtered_data
            )
            ba_metrics.extend(metrics_group=ba_pose_error_metrics)

            output_tracks_exit_codes = track_utils.classify_tracks3d_with_gt_cameras(
                tracks=aligned_filtered_data.get_tracks(), cameras_gt=cameras_gt
            )
            output_tracks_exit_codes_distribution = Counter(output_tracks_exit_codes)

            for exit_code, count in output_tracks_exit_codes_distribution.items():
                metric_name = "Filtered tracks triangulated with GT cams: {}".format(exit_code.name)
                ba_metrics.add_metric(GtsfmMetric(name=metric_name, data=count))

        ba_metrics.add_metrics(metrics_utils.get_stats_for_sfmdata(aligned_filtered_data, suffix="_filtered"))
        # ba_metrics.save_to_json(os.path.join(METRICS_PATH, "bundle_adjustment_metrics.json"))

        logger.info("[Result] Mean track length %.3f", np.mean(aligned_filtered_data.get_track_lengths()))
        logger.info("[Result] Median track length %.3f", np.median(aligned_filtered_data.get_track_lengths()))
        aligned_filtered_data.log_scene_reprojection_error_stats()

        return ba_metrics

    def create_computation_graph(
        self,
        sfm_data_graph: Delayed,
        gt_cameras_graph: Optional[List[Delayed]] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: an GtsfmData object wrapped up using dask.delayed
            gt_cameras_graph: list of GT cameras, ordered by camera index, each object wrapped up as Delayed.

        Returns:
            GtsfmData aligned to GT (if provided), wrapped up using dask.delayed
            Metrics group for BA results, wrapped up using dask.delayed
        """
        optimized_sfm_data, filtered_sfm_data = dask.delayed(self.run, nout=2)(sfm_data_graph)
        metrics_graph = dask.delayed(self.evaluate)(optimized_sfm_data, filtered_sfm_data, gt_cameras_graph)
        return filtered_sfm_data, metrics_graph


def values_to_gtsfm_data(values: Values, initial_data: GtsfmData, shared_calib: bool) -> GtsfmData:
    """Cast results from the optimization to GtsfmData object.

    Args:
        values: results of factor graph optimization.
        initial_data: data used to generate the factor graph; used to extract information about poses and 3d points in
                      the graph.
        shared_calib: flag indicating if calibrations were shared between the cameras.

    Returns:
        optimized poses and landmarks.
    """
    result = GtsfmData(initial_data.number_images())

    # add cameras
    for i in initial_data.get_valid_camera_indices():
        result.add_camera(
            i,
            PinholeCameraCal3Bundler(
                values.atPose3(X(i)),
                values.atCal3Bundler(K(0 if shared_calib else i)),
            ),
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
