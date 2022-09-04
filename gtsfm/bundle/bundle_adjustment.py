"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""
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
import gtsfm.utils.logger as logger_utils
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


# noise model params
CAM_POSE3_PRIOR_NOISE_SIGMA = 0.1
CAM_CAL3BUNDLER_PRIOR_NOISE_SIGMA = 1e-5  # essentially fixed
CAM_CAL3FISHEYE_PRIOR_NOISE_SIGMA = 1e-5  # essentially fixed
MEASUREMENT_NOISE_SIGMA = 1.0  # in pixels

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
        output_reproj_error_thresh: Optional[float] = None,
        robust_measurement_noise: bool = False,
        shared_calib: bool = False,
        max_iterations: Optional[int] = None,
    ) -> None:
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

    def __reprojection_factors(self, initial_data: GtsfmData, is_fisheye_calibration: bool) -> NonlinearFactorGraph:
        """Generate reprojection factors using the tracks."""
        graph = NonlinearFactorGraph()

        # noise model for measurements -- one pixel in u and v
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(IMG_MEASUREMENT_DIM, MEASUREMENT_NOISE_SIGMA)
        if self._robust_measurement_noise:
            measurement_noise = gtsam.noiseModel.Robust(gtsam.noiseModel.mEstimator.Huber(1.345), measurement_noise)

        sfm_factor_class = GeneralSFMFactor2Cal3Fisheye if is_fisheye_calibration else GeneralSFMFactor2Cal3Bundler
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            # retrieve the SfmMeasurement objects
            for m_idx in range(track.numberMeasurements()):
                # i represents the camera index, and uv is the 2d measurement
                i, uv = track.measurement(m_idx)
                # note use of shorthand symbols C and P
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
                    gtsam.noiseModel.Isotropic.Sigma(CAM_POSE3_DOF, CAM_POSE3_PRIOR_NOISE_SIGMA),
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
        calibration_prior_noise_sigma = (
            CAM_CAL3FISHEYE_PRIOR_NOISE_SIGMA if is_fisheye_calibration else CAM_CAL3BUNDLER_PRIOR_NOISE_SIGMA
        )
        if self._shared_calib:
            graph.push_back(
                calibration_prior_factor_class(
                    K(self.__map_to_calibration_variable(cameras_to_model[0])),
                    initial_data.get_camera(cameras_to_model[0]).calibration(),
                    gtsam.noiseModel.Isotropic.Sigma(calibration_prior_factor_dof, calibration_prior_noise_sigma),
                )
            )
        else:
            for i in cameras_to_model:
                graph.push_back(
                    calibration_prior_factor_class(
                        K(self.__map_to_calibration_variable(i)),
                        initial_data.get_camera(i).calibration(),
                        gtsam.noiseModel.Isotropic.Sigma(calibration_prior_factor_dof, calibration_prior_noise_sigma),
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

        # Create a factor graph
        graph.push_back(
            self.__reprojection_factors(initial_data=initial_data, is_fisheye_calibration=is_fisheye_calibration)
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

    def run_ba(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        verbose: bool = True,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool]]:
        """Run the bundle adjustment by forming factor graph and optimizing using Levenberg–Marquardt optimization.

        Args:
            initial_data: initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: priors to be used on cameras.
            relative_pose_priors: priors on the pose between two cameras.
            verbose: Boolean flag to print out additional info for debugging.

        Results:
            Optimized camera poses, 3D point w/ tracks, and error metrics, aligned to GT (if provided).
            Optimized camera poses after filtering landmarks (and cameras with no remaining landmarks).
            Valid mask as a list of booleans, indicating for each input track whether it was below the re-projection
                threshold.
        """
        logger.info(
            f"Input: {initial_data.number_tracks()} tracks on {len(initial_data.get_valid_camera_indices())} cameras\n"
        )
        if initial_data.number_tracks() == 0 or len(initial_data.get_valid_camera_indices()) == 0:
            # no cameras or tracks to optimize, so bundle adjustment is not possible
            logger.error(
                "Bundle adjustment aborting, optimization cannot be performed without any tracks or any cameras."
            )
            return initial_data, initial_data, [False] * initial_data.number_tracks()

        cameras_to_model = self.__cameras_to_model(initial_data, absolute_pose_priors, relative_pose_priors)

        graph = self.__construct_factor_graph(
            cameras_to_model=cameras_to_model,
            initial_data=initial_data,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
        )
        initial_values = self.__initial_values(initial_data=initial_data)
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
            filtered_result, valid_mask = optimized_data.filter_landmarks(self._output_reproj_error_thresh)
        else:
            valid_mask = [True] * optimized_data.number_tracks()
            filtered_result = optimized_data

        logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())

        return optimized_data, filtered_result, valid_mask

    def evaluate(
        self, unfiltered_data: GtsfmData, filtered_data: GtsfmData, cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]]
    ) -> GtsfmMetricsGroup:
        """
        Args:
            unfiltered_data: optimized BA result, before filtering landmarks by reprojection error.
            filtered_data: optimized BA result, after filtering landmarks and cameras.
            cameras_gt: cameras with GT intrinsics and GT extrinsics.

        Returns:
            Metrics group containing metrics for both filtered and unfiltered BA results.
        """
        ba_metrics = GtsfmMetricsGroup(
            name=METRICS_GROUP, metrics=metrics_utils.get_stats_for_sfmdata(unfiltered_data, suffix="_unfiltered")
        )

        poses_gt = [cam.pose() if cam is not None else None for cam in cameras_gt]

        valid_poses_gt_count = len(poses_gt) - poses_gt.count(None)
        if valid_poses_gt_count == 0:
            return ba_metrics

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
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: an GtsfmData object wrapped up using dask.delayed
            absolute_pose_priors: priors on the poses of the cameras (not delayed).
            relative_pose_priors: priors on poses between cameras (not delayed).

        Returns:
            GtsfmData aligned to GT (if provided), wrapped up using dask.delayed
            Metrics group for BA results, wrapped up using dask.delayed
        """
        optimized_sfm_data, filtered_sfm_data, _ = dask.delayed(self.run_ba, nout=3)(
            sfm_data_graph, absolute_pose_priors, relative_pose_priors
        )
        metrics_graph = dask.delayed(self.evaluate)(optimized_sfm_data, filtered_sfm_data, cameras_gt)
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

    is_fisheye_calibration = isinstance(initial_data.get_camera(0), PinholeCameraCal3Fisheye)
    if is_fisheye_calibration:
        cal3_value_extraction_lambda = lambda i: values.atCal3Fisheye(K(0 if shared_calib else i))
    else:
        cal3_value_extraction_lambda = lambda i: values.atCal3Bundler(K(0 if shared_calib else i))
    camera_class = PinholeCameraCal3Fisheye if is_fisheye_calibration else PinholeCameraCal3Bundler

    # add cameras
    for i in initial_data.get_valid_camera_indices():
        result.add_camera(
            i,
            camera_class(values.atPose3(X(i)), cal3_value_extraction_lambda(i)),
        )

    # add tracks
    for j in range(initial_data.number_tracks()):
        input_track = initial_data.get_track(j)

        # populate the result with optimized 3D point
        result_track = SfmTrack(values.atPoint3(P(j)))

        for measurement_idx in range(input_track.numberMeasurements()):
            i, uv = input_track.measurement(measurement_idx)
            result_track.addMeasurement(i, uv)

        result.add_track(result_track)

    return result
