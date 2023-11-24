"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import dask
import gtsam
import numpy as np
from dask.delayed import Delayed
from gtsam import (
    BetweenFactorPose3,
    GeneralSFMFactor2Cal3Bundler,
    GeneralSFMFactor2Cal3Fisheye,
    NonlinearFactorGraph,
    PinholeCameraCal3Bundler,
    PinholeCameraCal3Fisheye,
    PriorFactorCal3Bundler,
    PriorFactorCal3Fisheye,
    PriorFactorPose3,
    SfmTrack,
    Values,
    symbol_shorthand,
)

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.pose_prior import PosePrior
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
CAM_CAL3FISHEYE_DOF = 9
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logger = logging.getLogger(__name__)


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation.

    Due to the process graph requiring separate classes for separate graph objects, this class is a superclass for
    TwoViewBundleAdjustment and GlobalBundleAdjustment (defined in gtsfm/bundle/).
    """

    def __init__(
        self,
        reproj_error_thresholds: List[Optional[float]] = [None],
        robust_measurement_noise: bool = False,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
        cam_pose3_prior_noise_sigma: float = 0.1,
        calibration_prior_noise_sigma: float = 1e-5,
        measurement_noise_sigma: float = 1.0,
        allow_indeterminate_linear_system: bool = True
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
        """
        self._reproj_error_thresholds = reproj_error_thresholds
        self._robust_measurement_noise = robust_measurement_noise
        self._shared_calib = shared_calib
        self._max_iterations = max_iterations
        self._cam_pose3_prior_noise_sigma = cam_pose3_prior_noise_sigma
        self._calibration_prior_noise_sigma = calibration_prior_noise_sigma
        self._measurement_noise_sigma = measurement_noise_sigma
        self._allow_indeterminate_linear_system = allow_indeterminate_linear_system

    def __map_to_calibration_variable(self, camera_idx: int) -> int:
        return 0 if self._shared_calib else camera_idx

    def __reprojection_factors(self, initial_data: GtsfmData, is_fisheye_calibration: bool) -> NonlinearFactorGraph:
        """Generate reprojection factors using the tracks."""
        graph = NonlinearFactorGraph()

        # noise model for measurements -- one pixel in u and v
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(IMG_MEASUREMENT_DIM, self._measurement_noise_sigma)
        if self._robust_measurement_noise:
            measurement_noise = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.345), measurement_noise)

        sfm_factor_class = GeneralSFMFactor2Cal3Fisheye if is_fisheye_calibration else GeneralSFMFactor2Cal3Bundler
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            # Retrieve the SfmMeasurement objects.
            for m_idx in range(track.numberMeasurements()):
                # `i` represents the camera index, and `uv` is the 2d measurement
                i, uv = track.measurement(m_idx)
                # Note use of shorthand symbols `X` and `P`.
                graph.push_back(
                    sfm_factor_class(
                        uv,
                        measurement_noise,
                        X(i),
                        P(j),
                        K(self.__map_to_calibration_variable(i)),
                    )
                )

        return graph

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
                    gtsam.noiseModel.Diagonal.Sigmas(i2Ti1_prior.covariance),
                )
            )

        return graph

    def __pose_priors(
        self,
        absolute_pose_priors: List[Optional[PosePrior]],
        initial_data: GtsfmData,
        camera_for_origin: gtsfm_types.CAMERA_TYPE,
    ) -> NonlinearFactorGraph:
        """Generate prior factors (in the world frame) on pose variables."""
        graph = NonlinearFactorGraph()

        # TODO(Ayush): start using absolute prior factors.
        num_priors_added = 0

        if num_priors_added == 0:
            # Adding a prior to fix origin as no absolute prior exists.
            graph.push_back(
                PriorFactorPose3(
                    X(camera_for_origin),
                    initial_data.get_camera(camera_for_origin).pose(),
                    gtsam.noiseModel.Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
                )
            )

        return graph

    def __calibration_priors(
        self, initial_data: GtsfmData, cameras_to_model: List[int], is_fisheye_calibration: bool
    ) -> NonlinearFactorGraph:
        """Generate prior factors on calibration parameters of the cameras."""
        graph = NonlinearFactorGraph()

        calibration_prior_factor_class = PriorFactorCal3Fisheye if is_fisheye_calibration else PriorFactorCal3Bundler
        calibration_prior_factor_dof = CAM_CAL3FISHEYE_DOF if is_fisheye_calibration else CAM_CAL3BUNDLER_DOF
        if self._shared_calib:
            graph.push_back(
                calibration_prior_factor_class(
                    K(self.__map_to_calibration_variable(cameras_to_model[0])),
                    initial_data.get_camera(cameras_to_model[0]).calibration(),
                    gtsam.noiseModel.Isotropic.Sigma(calibration_prior_factor_dof, self._calibration_prior_noise_sigma),
                )
            )
        else:
            for i in cameras_to_model:
                graph.push_back(
                    calibration_prior_factor_class(
                        K(self.__map_to_calibration_variable(i)),
                        initial_data.get_camera(i).calibration(),
                        gtsam.noiseModel.Isotropic.Sigma(
                            calibration_prior_factor_dof, self._calibration_prior_noise_sigma
                        ),
                    )
                )

        return graph

    def __construct_factor_graph(
        self,
        cameras_to_model: List[int],
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
    ) -> NonlinearFactorGraph:
        """Construct the factor graph with reprojection factors, BetweenFactors, and prior factors."""
        is_fisheye_calibration = isinstance(initial_data.get_camera(cameras_to_model[0]), PinholeCameraCal3Fisheye)

        graph = NonlinearFactorGraph()

        # Create a factor graph.
        graph.push_back(
            self.__reprojection_factors(
                initial_data=initial_data,
                is_fisheye_calibration=is_fisheye_calibration,
            )
        )
        graph.push_back(
            self._between_factors(relative_pose_priors=relative_pose_priors, cameras_to_model=cameras_to_model)
        )
        graph.push_back(
            self.__pose_priors(
                absolute_pose_priors=absolute_pose_priors,
                initial_data=initial_data,
                camera_for_origin=cameras_to_model[0],
            )
        )
        graph.push_back(self.__calibration_priors(initial_data, cameras_to_model, is_fisheye_calibration))

        # Also add a prior on the position of the first landmark to fix the scale
        graph.push_back(
            gtsam.PriorFactorPoint3(
                P(0), initial_data.get_track(0).point3(), gtsam.noiseModel.Isotropic.Sigma(POINT3_DOF, 0.1)
            )
        )

        return graph

    def __initial_values(self, initial_data: GtsfmData) -> Values:
        """Initialize all the variables in the factor graph."""
        initial_values = gtsam.Values()

        # Add each camera.
        for loop_idx, i in enumerate(initial_data.get_valid_camera_indices()):
            camera = initial_data.get_camera(i)
            initial_values.insert(X(i), camera.pose())
            if not self._shared_calib or loop_idx == 0:
                # add only one value if calibrations are shared
                initial_values.insert(K(self.__map_to_calibration_variable(i)), camera.calibration())

        # Add each SfmTrack.
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)
            initial_values.insert(P(j), track.point3())

        return initial_values

    def __optimize_factor_graph(self, graph: NonlinearFactorGraph, initial_values: Values) -> Values:
        """Optimize the factor graph."""
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR")
        if self._max_iterations:
            params.setMaxIterations(self._max_iterations)
        lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)

        result_values = lm.optimize()
        return result_values

    def __cameras_to_model(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
    ) -> List[int]:
        """Get the cameras which are to be modeled in the factor graph. We are using ability to add initial values as
        proxy for this function."""
        cameras: Set[int] = set(initial_data.get_valid_camera_indices())

        return sorted(list(cameras))

    def get_two_view_ba_pose_graph_keys(self, initial_data: GtsfmData):
        """Retrieves GTSAM keys for camera poses in a 2-view BA problem."""
        return [ X(0), X(1) ]

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

        cameras_to_model = self.__cameras_to_model(initial_data, absolute_pose_priors, relative_pose_priors)
        graph = self.__construct_factor_graph(
            cameras_to_model=cameras_to_model,
            initial_data=initial_data,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
        )
        initial_values = self.__initial_values(initial_data=initial_data)
        result_values = self.__optimize_factor_graph(graph, initial_values)

        # Print error.
        final_error = graph.error(result_values)
        if verbose:
            logger.info("initial error: %.2f", graph.error(initial_values))
            logger.info("final error: %.2f", final_error)

        if self.is_two_view_ba():
            try:
                # Calculate marginal covariances for all two pose variables.
                marginals = gtsam.Marginals(graph, result_values)
                graph_keys = self.get_two_view_ba_pose_graph_keys(initial_data)
                for key in graph_keys:
                    _ = marginals.marginalCovariance(key)

            except RuntimeError:
                if not self._allow_indeterminate_linear_system:
                    logger.error("BA result discarded due to Indeterminate Linear System (ILS) when computing marginals.")
                    return None, None, None, None

        # Convert the `Values` results to a `GtsfmData` instance.
        optimized_data = values_to_gtsfm_data(result_values, initial_data, self._shared_calib)

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
                        reproj_error_thresh,
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
        ba_metrics = GtsfmMetricsGroup(
            name=METRICS_GROUP, metrics=metrics_utils.get_metrics_for_sfmdata(unfiltered_data, suffix="_unfiltered")
        )

        poses_gt = [cam.pose() if cam is not None else None for cam in cameras_gt]

        valid_poses_gt_count = len(poses_gt) - poses_gt.count(None)
        if valid_poses_gt_count == 0:
            return ba_metrics

        # Align the sparse multi-view estimate after BA to the ground truth pose graph.
        aligned_filtered_data = filtered_data.align_via_Sim3_to_poses(wTi_list_ref=poses_gt)
        ba_pose_error_metrics = metrics_utils.compute_ba_pose_metrics(
            gt_wTi_list=poses_gt, ba_output=aligned_filtered_data, save_dir=save_dir
        )
        ba_metrics.extend(metrics_group=ba_pose_error_metrics)

        output_tracks_exit_codes = track_utils.classify_tracks3d_with_gt_cameras(
            tracks=aligned_filtered_data.get_tracks(), cameras_gt=cameras_gt
        )
        output_tracks_exit_codes_distribution = Counter(output_tracks_exit_codes)

        for exit_code, count in output_tracks_exit_codes_distribution.items():
            metric_name = "Filtered tracks triangulated with GT cams: {}".format(exit_code.name)
            ba_metrics.add_metric(GtsfmMetric(name=metric_name, data=count))

        ba_metrics.add_metrics(metrics_utils.get_metrics_for_sfmdata(aligned_filtered_data, suffix="_filtered"))

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


def values_to_gtsfm_data(values: Values, initial_data: GtsfmData, shared_calib: bool) -> GtsfmData:
    """Cast results from the optimization to GtsfmData object.

    Args:
        values: Results of factor graph optimization.
        initial_data: Data used to generate the factor graph; used to extract information about poses and 3d points in
                      the graph.
        shared_calib: Flag indicating if calibrations were shared between the cameras.

    Returns:
        Optimized poses and landmarks.
    """
    result = GtsfmData(initial_data.number_images())

    is_fisheye_calibration = isinstance(initial_data.get_camera(0), PinholeCameraCal3Fisheye)
    if is_fisheye_calibration:
        cal3_value_extraction_lambda = lambda i: values.atCal3Fisheye(K(0 if shared_calib else i))
    else:
        cal3_value_extraction_lambda = lambda i: values.atCal3Bundler(K(0 if shared_calib else i))
    camera_class = PinholeCameraCal3Fisheye if is_fisheye_calibration else PinholeCameraCal3Bundler

    # Add cameras.
    for i in initial_data.get_valid_camera_indices():
        result.add_camera(
            i,
            camera_class(values.atPose3(X(i)), cal3_value_extraction_lambda(i)),
        )

    # Add tracks.
    for j in range(initial_data.number_tracks()):
        input_track = initial_data.get_track(j)

        # Populate the result with optimized 3D point.
        result_track = SfmTrack(values.atPoint3(P(j)))

        for measurement_idx in range(input_track.numberMeasurements()):
            i, uv = input_track.measurement(measurement_idx)
            result_track.addMeasurement(i, uv)

        result.add_track(result_track)

    return result
