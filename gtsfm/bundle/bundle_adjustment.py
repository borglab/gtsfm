"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import dask
import gtsam  # type: ignore
import numpy as np
from dask.delayed import Delayed
from gtsam import BetweenFactorPose3, NonlinearFactorGraph, PriorFactorPoint3, PriorFactorPose3, Values  # type: ignore
from gtsam.noiseModel import Diagonal, Isotropic, Robust, mEstimator  # type: ignore
from gtsam.symbol_shorthand import K, P, X  # type: ignore
from numpy.typing import NDArray

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common import gtsfm_data
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationOptions,
    TriangulationSamplingMode,
)
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
    TLS = "TLS"


def multi_view_retriangulate_from_2d_tracks(
    gtsfm_data: GtsfmData,
    tracks_2d: List["SfmTrack2d"],
    triangulation_options: Optional[TriangulationOptions] = None,
    min_track_length: int = 3,
) -> GtsfmData:
    """Re-triangulate the union-find 2D track set against post-BA cameras.

    Each input track is solved by `Point3dInitializer.triangulate`: RANSAC samples
    camera pairs, triangulates a candidate 3D point via 2-view DLT, scores by
    counting inliers (measurements within `reproj_error_threshold`) across the
    whole track, and picks the best sample. The final 3D point is then computed
    by a multi-view DLT over the inlier measurements.

    Recovers tracks that were dropped earlier in the pipeline when cameras
    weren't yet converged — those triangulations can now succeed against the
    refined cameras.

    2-view tracks are excluded by default (`min_track_length=3`). They are weak
    (no inlier consensus across views, single-pair baseline) and don't help BA.

    Args:
        gtsfm_data: Scene with optimized cameras (existing tracks ignored; cameras re-used).
        tracks_2d: Full 2D track set from `CppDsfTracksEstimator.run()`.
        triangulation_options: Per-track solve config. Defaults to RANSAC_SAMPLE_UNIFORM,
            reproj_error_threshold=10 px, max_num_hypotheses=100. For tracks with
            N measurements there are N*(N-1)/2 possible 2-view samples, so for
            typical short tracks the cap is rarely binding; set lower if RANSAC
            wall-time matters on long tracks.
        min_track_length: Drop tracks with fewer measurements than this.

    Returns:
        New GtsfmData with same cameras and an augmented track set.
    """
    if triangulation_options is None:
        triangulation_options = TriangulationOptions(
            reproj_error_threshold=10.0,
            mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
            max_num_hypotheses=100,
        )

    camera_dict = {i: cam for i, cam in gtsfm_data.cameras().items() if cam is not None}
    initializer = Point3dInitializer(camera_dict, triangulation_options)

    out = GtsfmData(gtsfm_data.number_images())
    for cam_idx, camera in camera_dict.items():
        out.add_camera(cam_idx, camera)
        info = gtsfm_data.get_image_info(cam_idx)
        out.set_image_info(cam_idx, name=info.name, shape=info.shape)

    n_in = len(tracks_2d)
    n_kept = 0
    n_short_input = 0
    n_no_cams = 0
    n_short_output = 0
    n_triangulation_failed = 0
    for track_2d in tracks_2d:
        if track_2d.number_measurements() < min_track_length:
            n_short_input += 1
            continue
        valid_measurements = [m for m in track_2d.measurements if m.i in camera_dict]
        if len(valid_measurements) < min_track_length:
            n_no_cams += 1
            continue
        track_2d_filtered = SfmTrack2d(measurements=valid_measurements)

        new_track, _, _ = initializer.triangulate(track_2d_filtered)
        if new_track is None:
            n_triangulation_failed += 1
            continue
        if new_track.numberMeasurements() < min_track_length:
            n_short_output += 1
            continue
        out.add_track(new_track)
        n_kept += 1

    logger.info(
        "Multi-view retriangulation: %d kept / %d input (min_len=%d): "
        "%d short_in, %d no_cams, %d tri_failed, %d short_out.",
        n_kept, n_in, min_track_length, n_short_input, n_no_cams,
        n_triangulation_failed, n_short_output,
    )
    return out


@dataclass
class BundleAdjustmentOptions:
    """Shared configuration for bundle adjustment across leaf BA, merging BA, etc.

    This dataclass captures the commonly-configured subset of BA parameters
    that vary between call sites (leaf vs. merging). Parameters with fixed
    defaults (e.g. ordering_type, allow_indeterminate_linear_system) are left
    to the ``BundleAdjustmentOptimizer`` constructor.
    """

    robust_ba_mode: Union[RobustBAMode, str] = RobustBAMode.GMC
    shared_calib: bool = False
    use_calibration_prior: bool = False
    use_pose_prior_all_cameras: bool = False
    use_pose_prior_first_camera: bool = False
    # Sigma for the PriorFactorPose3 added by the use_pose_prior_* flags, anchored at the
    # initialization poses. Near-zero (e.g. 1e-3, in scene units) freezes poses so the BA
    # refines only structure + calibration ("self-calibration against the init poses").
    cam_pose3_prior_noise_sigma: float = 0.1
    use_gnc: bool = False
    gnc_loss: Union[RobustBAMode, str] = RobustBAMode.GMC
    factor_weight_outlier_threshold: float = 0.0
    min_track_length: int = 2
    calibration_prior_focal_sigma: float = 20.0
    calibration_prior_dist_sigma: float | Sequence[float] = 0.1
    calibration_prior_pp_sigma: float = 1e-5
    robust_noise_basin: float = 1.345
    min_tracks_per_camera: int = 15
    compute_pose_covariances: bool = False
    optimizer_relative_cost_tol: float = 1e-5
    use_multi_view_retriangulation: bool = False
    mv_retri_min_track_length: int = 3
    mv_retri_reproj_error_thresh: float = 10.0
    mv_retri_max_num_hypotheses: int = 100

    def to_optimizer(self, **overrides) -> "BundleAdjustmentOptimizer":
        """Construct a :class:`BundleAdjustmentOptimizer` from these options.

        Args:
            **overrides: Keyword arguments that override dataclass fields or add
                extra constructor params (e.g. ``min_track_length``,
                ``reproj_error_thresholds``).
        """
        kwargs = dict(
            robust_ba_mode=self.robust_ba_mode,
            shared_calib=self.shared_calib,
            use_calibration_prior=self.use_calibration_prior,
            use_pose_prior_all_cameras=self.use_pose_prior_all_cameras,
            use_pose_prior_first_camera=self.use_pose_prior_first_camera,
            cam_pose3_prior_noise_sigma=self.cam_pose3_prior_noise_sigma,
            use_gnc=self.use_gnc,
            gnc_loss=self.gnc_loss,
            factor_weight_outlier_threshold=self.factor_weight_outlier_threshold,
            min_track_length=self.min_track_length,
            calibration_prior_focal_sigma=self.calibration_prior_focal_sigma,
            calibration_prior_dist_sigma=self.calibration_prior_dist_sigma,
            calibration_prior_pp_sigma=self.calibration_prior_pp_sigma,
            robust_noise_basin=self.robust_noise_basin,
            min_tracks_per_camera=self.min_tracks_per_camera,
            compute_pose_covariances=self.compute_pose_covariances,
            optimizer_relative_cost_tol=self.optimizer_relative_cost_tol,
            use_multi_view_retriangulation=self.use_multi_view_retriangulation,
            mv_retri_min_track_length=self.mv_retri_min_track_length,
            mv_retri_reproj_error_thresh=self.mv_retri_reproj_error_thresh,
            mv_retri_max_num_hypotheses=self.mv_retri_max_num_hypotheses,
        )
        kwargs.update(overrides)
        return BundleAdjustmentOptimizer(**kwargs)


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
        calibration_prior_focal_sigma: float = 20.0,
        calibration_prior_dist_sigma: float | Sequence[float] = 0.1,
        calibration_prior_pp_sigma: float = 1e-5,
        measurement_noise_sigma: float = 2.0,
        allow_indeterminate_linear_system: bool = True,
        print_summary: bool = False,
        ordering_type: str = "METIS",
        save_iteration_visualization: bool = False,
        robust_noise_basin: float = 1.345,
        use_karcher_mean_factor: bool = True,
        use_pose_prior_all_cameras: bool = False,
        use_pose_prior_first_camera: bool = False,
        use_calibration_prior: bool = True,
        use_first_point_prior: bool = False,
        use_gnc: bool = False,
        gnc_loss: RobustBAMode | str = RobustBAMode.GMC,
        factor_weight_outlier_threshold: float = 0.0,
        min_track_length: int = 2,
        min_tracks_per_camera: int = 15,
        compute_pose_covariances: bool = False,
        optimizer_relative_cost_tol: float = 1e-5,
        use_multi_view_retriangulation: bool = False,
        mv_retri_min_track_length: int = 3,
        mv_retri_reproj_error_thresh: float = 10.0,
        mv_retri_max_num_hypotheses: int = 100,
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
            calibration_prior_focal_sigma (optional): Sigma for the prior on the focal length, defaults to 20.0.
            calibration_prior_dist_sigma (optional): Sigma for the prior on the distortion parameters, defaults to 0.1.
            measurement_noise_sigma (optional): Measurement noise sigma in pixel units.
            allow_indeterminate_linear_system: Reject a two-view measurement if an indeterminate linear system is
                encountered during marginal covariance computation after bundle adjustment.
            ordering_type (optional): The ordering algorithm to use for variable elimination.
            save_iteration_visualization (optional): Save a Plotly animation showing optimization progress.
            robust_noise_basin (optional): Basin to use for the robust noise model.
            use_karcher_mean_factor (optional): Use Karcher mean factor to constrain the camera poses.
            use_pose_prior (optional): Use pose prior to constrain the camera poses. (only used if we use karcher mean)
            use_calibration_prior (optional): Use calibration prior to constrain the camera intrinsics.
            use_first_point_prior (optional): Use first point prior to constrain the scale of the reconstruction.
            use_gnc (optional): Use the GNC optimizer for bundle adjustment.
            gnc_loss (optional): GNC loss to use. Defaults to GMC.
            factor_weight_outlier_threshold (optional): Threshold weight for a reprojection factor to be kept.
            min_track_length: min number of measurements required to keep a track after weight filtering.
            compute_pose_covariances: If true, compute marginal covariance for all camera pose variables and return it.
        """
        self._reproj_error_thresholds = reproj_error_thresholds
        if isinstance(robust_ba_mode, str):
            self._robust_ba_mode = RobustBAMode[robust_ba_mode]
        else:
            self._robust_ba_mode = robust_ba_mode
        self._shared_calib = shared_calib
        self._max_iterations = max_iterations
        self._cam_pose3_prior_noise_sigma = cam_pose3_prior_noise_sigma
        self._use_calibration_prior = use_calibration_prior
        self._calibration_prior_focal_sigma = calibration_prior_focal_sigma
        self._calibration_prior_dist_sigma = calibration_prior_dist_sigma
        self._calibration_prior_pp_sigma = calibration_prior_pp_sigma
        self._measurement_noise_sigma = measurement_noise_sigma
        self._allow_indeterminate_linear_system = allow_indeterminate_linear_system
        self._ordering_type = ordering_type
        self._print_summary = print_summary
        self._save_iteration_visualization = save_iteration_visualization
        self._robust_noise_basin = robust_noise_basin
        self._use_karcher_mean_factor = use_karcher_mean_factor
        self._use_pose_prior_all_cameras = use_pose_prior_all_cameras
        self._use_pose_prior_first_camera = use_pose_prior_first_camera
        self._use_first_point_prior = use_first_point_prior
        self._use_gnc = use_gnc
        if isinstance(gnc_loss, str):
            self._gnc_loss = RobustBAMode[gnc_loss]
        else:
            self._gnc_loss = gnc_loss
        self._factor_weight_outlier_threshold = factor_weight_outlier_threshold
        self._min_track_length = min_track_length
        self._min_tracks_per_camera = min_tracks_per_camera
        self._compute_pose_covariances = compute_pose_covariances
        self._optimizer_relative_cost_tol = optimizer_relative_cost_tol

        # Post-BA multi-view retriangulation (opt-in). See `__init__` docstring above.
        self._use_multi_view_retriangulation = use_multi_view_retriangulation
        self._mv_retri_min_track_length = mv_retri_min_track_length
        self._mv_retri_reproj_error_thresh = mv_retri_reproj_error_thresh
        self._mv_retri_max_num_hypotheses = mv_retri_max_num_hypotheses

    def __map_to_calibration_variable(self, camera_idx: int) -> int:
        return 0 if self._shared_calib else camera_idx

    def __get_cameras_with_insufficient_tracks(self, initial_data: GtsfmData) -> set[int]:
        """Get the cameras with insufficient fewer tracks that self._min_tracks_per_camera."""
        cameras_with_insufficient_tracks = set()
        for i in initial_data.get_valid_camera_indices():
            if len(initial_data.get_measurements_for_camera(i)) < self._min_tracks_per_camera:
                cameras_with_insufficient_tracks.add(i)
        return cameras_with_insufficient_tracks

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

        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            valid_measurements = [
                m_idx for m_idx in range(track.numberMeasurements()) if track.measurement(m_idx)[0] in cameras_to_model
            ]
            if len(valid_measurements) < self._min_track_length:
                continue
            # Retrieve the SfmMeasurement objects.
            for m_idx in valid_measurements:
                # `i` represents the camera index, and `uv` is the 2d measurement
                i, uv = track.measurement(m_idx)
                graph.push_back(
                    sfm_factor_class(
                        uv,
                        measurement_noise,
                        X(i),
                        P(j),
                        K(self.__map_to_calibration_variable(i)),
                    )  # type: ignore
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
                    Diagonal.Sigmas(i2Ti1_prior.covariance),
                )
            )

        return graph

    def __pose_priors(
        self,
        absolute_pose_priors: List[Optional[PosePrior]],
        initial_data: GtsfmData,
        cameras_to_model: List[int],
    ) -> NonlinearFactorGraph:
        """Generate prior factors (in the world frame) on pose variables."""
        graph = NonlinearFactorGraph()

        # TODO(Ayush): start using absolute prior factors.

        if self._use_karcher_mean_factor:
            camera_keys = [X(i) for i in cameras_to_model]
            graph.push_back(gtsam.KarcherMeanFactorPose3(camera_keys, 6, 1000))

        if self._use_pose_prior_all_cameras:
            for camera_idx in cameras_to_model:
                camera_i = initial_data.get_camera(camera_idx)
                assert camera_i is not None, f"Camera {camera_idx} in initial data is None"
                graph.push_back(
                    PriorFactorPose3(
                        X(camera_idx),
                        camera_i.pose(),
                        Isotropic.Sigma(CAM_POSE3_DOF, self._cam_pose3_prior_noise_sigma),
                    )
                )
        elif self._use_pose_prior_first_camera:
            first_camera = initial_data.get_camera(cameras_to_model[0])
            assert first_camera is not None, "First camera in initial data is None"
            graph.push_back(
                PriorFactorPose3(
                    X(cameras_to_model[0]),
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
            first_camera.calibration(),
            focal_sigma=self._calibration_prior_focal_sigma,
            dist_sigma=self._calibration_prior_dist_sigma,
            pp_sigma=self._calibration_prior_pp_sigma,
            skew_sigma=1e-6,
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
    ) -> NonlinearFactorGraph:
        """Construct the factor graph with just reprojection factors and calibration priors."""

        graph = NonlinearFactorGraph()
        if not cameras_to_model:
            return graph

        reprojection_graph = self.__reprojection_factors(
            initial_data=initial_data,
            cameras_to_model=cameras_to_model,
            robust_noise_basin=robust_noise_basin,
        )
        graph.push_back(reprojection_graph)

        graph.push_back(
            self.__pose_priors(absolute_pose_priors=[], initial_data=initial_data, cameras_to_model=cameras_to_model)
        )

        if self._use_first_point_prior and initial_data.number_tracks() > 0:
            graph.push_back(
                PriorFactorPoint3(P(0), initial_data.get_track(0).point3(), Isotropic.Sigma(POINT3_DOF, 0.1))
            )
        if self._use_calibration_prior:
            graph.push_back(self.__calibration_priors(initial_data, cameras_to_model))

        return graph

    def __construct_factor_graph(
        self,
        cameras_to_model: List[int],
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        robust_noise_basin: float | None = None,
    ) -> NonlinearFactorGraph:
        """Construct the factor graph with reprojection factors, BetweenFactors, and prior factors."""
        # Create a factor graph.
        graph = self.__construct_simple_factor_graph(cameras_to_model, initial_data, robust_noise_basin)

        # Add priors
        graph.push_back(
            self._between_factors(relative_pose_priors=relative_pose_priors, cameras_to_model=cameras_to_model)
        )

        return graph

    def __optimize_factor_graph(
        self, graph: NonlinearFactorGraph, initial_values: Values, ordering_type: str
    ) -> Tuple[Values, Optional[List[Values]], Optional[NDArray[np.float64]]]:
        """Optimize the factor graph, optionally capturing per-iteration values."""
        start_time = time.time()

        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosityLM("ERROR" if not self._print_summary else "SUMMARY")
        params.setOrderingType(ordering_type)
        if self._max_iterations:
            params.setMaxIterations(self._max_iterations)

        if not self._use_gnc:
            if self._optimizer_relative_cost_tol is not None:
                params.setRelativeErrorTol(self._optimizer_relative_cost_tol)
            lm = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
        else:
            gnc_params = gtsam.GncLMParams(params)
            if self._gnc_loss == RobustBAMode.GMC:
                gnc_params.setLossType(gtsam.GncLossType.GM)
            elif self._gnc_loss == RobustBAMode.TLS:
                gnc_params.setLossType(gtsam.GncLossType.TLS)
            else:
                raise ValueError(f"Unsupported GNC loss type: {self._gnc_loss}.")
            gnc_params.setVerbosityGNC(gtsam.GncLMParams.Verbosity.SUMMARY)
            gnc_params.setAllowNonNoiseModelFactors(True)
            if self._optimizer_relative_cost_tol is not None:
                gnc_params.setRelativeCostTol(self._optimizer_relative_cost_tol)
            lm = gtsam.GncLMOptimizer(graph, initial_values, gnc_params)

        # gnc does not support getAbsoluteErrorTol and getRelativeErrorTol params
        if not self._save_iteration_visualization or self._use_gnc:
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
        if self._use_gnc:
            weights = lm.getWeights()
            return result_values, values_trace, weights
        return result_values, values_trace, None

    def get_two_view_ba_pose_graph_keys(self, initial_data: GtsfmData):
        """Retrieves GTSAM keys for camera poses in a 2-view BA problem."""
        valid_camera_indices = initial_data.get_valid_camera_indices()
        return [X(valid_camera_indices[0]), X(valid_camera_indices[1])]

    def is_two_view_ba(self, initial_data: GtsfmData) -> bool:
        """Determines whether two-view bundle adjustment is being executed."""
        return len(initial_data.get_valid_camera_indices()) == 2

    def __compute_camera_pose_covariances(
        self, marginals: gtsam.Marginals, cameras_to_model: List[int]
    ) -> Dict[int, NDArray[np.float64]]:
        """Compute marginal covariance for all camera pose variables."""
        pose_covariances: Dict[int, NDArray[np.float64]] = {}
        for camera_idx in cameras_to_model:
            try:
                full_cov = np.asarray(marginals.marginalCovariance(X(camera_idx)))
                pose_covariances[camera_idx] = full_cov[:CAM_POSE3_DOF, :CAM_POSE3_DOF]
            except Exception:
                logger.error(
                    f"Error computing covariance for camera {camera_idx}, likely due to indeterminate linear system."
                )
                continue
        return pose_covariances

    def __optimize_and_recover(
        self, initial_data: GtsfmData, graph: NonlinearFactorGraph, ordering_type: str
    ) -> Tuple[GtsfmData, Values, float, List[bool]]:
        """Optimize the graph, report errors, and convert `Values` back to `GtsfmData`."""
        initial_values = initial_data.to_values(shared_calib=self._shared_calib)
        result_values, _, weights = self.__optimize_factor_graph(graph, initial_values, ordering_type)
        final_error = graph.error(result_values)
        optimized_data = GtsfmData.from_values(result_values, initial_data, self._shared_calib)
        gnc_valid_mask = [True] * initial_data.number_tracks()
        if self._use_gnc and weights is not None and self._factor_weight_outlier_threshold > 0:
            optimized_data, gnc_valid_mask = self.__filter_tracks_by_factor_weights(graph, optimized_data, weights)
        return optimized_data, result_values, final_error, gnc_valid_mask

    def __filter_tracks_by_factor_weights(
        self, graph: NonlinearFactorGraph, optimized_data: GtsfmData, weights: NDArray[np.float64]
    ) -> Tuple[GtsfmData, List[bool]]:
        """Filter tracks based on the weights of the reprojection factors, if GNC optimization is used."""
        gnc_valid_mask = [True] * optimized_data.number_tracks()
        if weights is None:
            logger.error("Weights array is None, cannot filter tracks by factor weights.")
            return optimized_data, gnc_valid_mask
        cameras_to_model = sorted(optimized_data.get_valid_camera_indices())
        first_camera = optimized_data.get_camera(cameras_to_model[0])
        assert first_camera is not None, "First camera in optimized factor graph is None"
        sfm_factor_class = gtsfm_types.get_sfm_factor_for_calibration(first_camera.calibration())
        cams_to_remove_per_track: dict[int, set[int]] = defaultdict(set)
        for i in range(graph.nrFactors()):
            if weights[i] >= self._factor_weight_outlier_threshold:
                continue
            factor = graph.at(i)
            if isinstance(factor, sfm_factor_class):
                camera_key, track_key, _ = factor.keys()
                camera_id = int(gtsam.symbolIndex(camera_key))
                track_id = int(gtsam.symbolIndex(track_key))
                cams_to_remove_per_track[track_id].add(camera_id)

        if not cams_to_remove_per_track:
            return optimized_data, gnc_valid_mask

        for track_id, camera_ids in cams_to_remove_per_track.items():
            track = optimized_data.get_track(track_id)
            new_measurements = []
            for m_idx in range(track.numberMeasurements()):
                cam_id, _ = track.measurement(m_idx)
                if cam_id not in camera_ids:
                    new_measurements.append(track.measurement(m_idx))
            track.measurements = new_measurements

        length_filtered_tracks = []
        for track_idx, track in enumerate(optimized_data.get_tracks()):
            if track.numberMeasurements() >= self._min_track_length:
                length_filtered_tracks.append(track)
            else:
                gnc_valid_mask[track_idx] = False

        if len(length_filtered_tracks) == optimized_data.number_tracks():
            optimized_data._camera_to_measurement_map = None
            return optimized_data, gnc_valid_mask

        image_info = {
            image_id: optimized_data.get_image_info(image_id) for image_id in optimized_data.get_all_image_ids()
        }

        filtered_data = GtsfmData.from_cameras_and_tracks(
            cameras=optimized_data.cameras(),
            tracks=length_filtered_tracks,
            number_images=optimized_data.number_images(),
            image_info=image_info,
            gaussian_splats=optimized_data.get_gaussian_splats(),
        )

        return filtered_data, gnc_valid_mask

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
        cameras_with_insufficient_tracks = self.__get_cameras_with_insufficient_tracks(initial_data)
        cameras_to_model = [i for i in cameras_to_model if i not in cameras_with_insufficient_tracks]
        logger.info(
            "Cameras with insufficient tracks (fewer than %d): %s, will be excluded from BA.",
            self._min_tracks_per_camera,
            cameras_with_insufficient_tracks,
        )
        graph = self.__construct_simple_factor_graph(cameras_to_model, initial_data, robust_noise_basin)
        if len(cameras_with_insufficient_tracks) == len(initial_data.cameras()):
            logger.warning("Skipping bundle adjustment because all cameras are without tracks.")
            return initial_data, 0.0
        optimized_data, result_values, final_error, _ = self.__optimize_and_recover(
            initial_data, graph, self._ordering_type if not cameras_with_insufficient_tracks else "COLAMD"
        )
        if self._compute_pose_covariances:
            try:
                marginals = gtsam.Marginals(graph, result_values)
                optimized_data.set_camera_pose_covariances(
                    self.__compute_camera_pose_covariances(marginals, cameras_to_model)
                )
            except Exception:
                logger.info("Error computing marginals, likely due to indeterminate linear system.")

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
    ) -> Tuple[Optional[GtsfmData], Optional[GtsfmData], Optional[List[bool]], Optional[float]]:
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

        running_two_view_ba = self.is_two_view_ba(initial_data)

        cameras_to_model = sorted(initial_data.get_valid_camera_indices())
        cameras_with_insufficient_tracks = None
        if not running_two_view_ba:
            cameras_with_insufficient_tracks = self.__get_cameras_with_insufficient_tracks(initial_data)
            cameras_to_model = [i for i in cameras_to_model if i not in cameras_with_insufficient_tracks]
            logger.info(
                "Cameras with insufficient tracks (fewer than %d): %s, will be excluded from BA.",
                self._min_track_length,
                cameras_with_insufficient_tracks,
            )

        graph = self.__construct_factor_graph(
            cameras_to_model, initial_data, absolute_pose_priors, relative_pose_priors
        )
        optimized_data, result_values, final_error, gnc_valid_mask = self.__optimize_and_recover(
            initial_data, graph, self._ordering_type if not cameras_with_insufficient_tracks else "COLAMD"
        )
        if not running_two_view_ba:
            # Add the non-BA cameras from initial_data back.
            for camera_idx in cameras_with_insufficient_tracks:
                optimized_data._cameras[camera_idx] = initial_data.get_camera(camera_idx)

        if running_two_view_ba or self._compute_pose_covariances:
            try:
                marginals = gtsam.Marginals(graph, result_values)
                if running_two_view_ba:
                    # Calculate marginal covariances for all two pose variables.
                    graph_keys = self.get_two_view_ba_pose_graph_keys(initial_data)
                    for key in graph_keys:
                        _ = marginals.marginalCovariance(key)
                if self._compute_pose_covariances:
                    optimized_data.set_camera_pose_covariances(
                        self.__compute_camera_pose_covariances(marginals, cameras_to_model)
                    )

            except RuntimeError:
                if running_two_view_ba and not self._allow_indeterminate_linear_system:
                    logger.error(
                        "BA result discarded due to Indeterminate Linear System (ILS) when computing marginals."
                    )
                    return None, None, None, None
                elif not running_two_view_ba:
                    logger.info("Error computing marginals, likely due to indeterminate linear system.")

        # Convert the `Values` results to a `GtsfmData` instance.
        # Filter landmarks by reprojection error.
        if reproj_error_thresh is not None:
            if verbose:
                logger.info("[Result] Number of tracks before filtering: %d", optimized_data.number_tracks())
            filtered_result, postfilter_valid_mask = optimized_data.filter_landmarks(reproj_error_thresh)
            if verbose:
                logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())

        else:
            postfilter_valid_mask = [True] * optimized_data.number_tracks()
            filtered_result = optimized_data

        valid_mask = [False] * initial_data.number_tracks()
        kept_track_indices = [track_idx for track_idx, is_valid in enumerate(gnc_valid_mask) if is_valid]
        for track_idx, is_valid in zip(kept_track_indices, postfilter_valid_mask):
            valid_mask[track_idx] = is_valid
        if self._compute_pose_covariances:
            filtered_pose_covariances = {
                i: cov
                for i, cov in optimized_data.get_camera_pose_covariances().items()
                if i in filtered_result.get_valid_camera_indices()
            }
            filtered_result.set_camera_pose_covariances(filtered_pose_covariances)
        return optimized_data, filtered_result, valid_mask, final_error

    def run_ba(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        verbose: bool = True,
    ) -> Tuple[Optional[GtsfmData], Optional[GtsfmData], Optional[List[bool]], List[float]]:
        """Runs bundle adjustment by forming a factor graph and optimizing it using Levenberg–Marquardt optimization.

        Args:
            initial_data: Initialized cameras, tracks w/ their 3d landmark from triangulation.
            absolute_pose_priors: Priors to be used on cameras.
            relative_pose_priors: Priors on the pose between two cameras.
            verbose: Boolean flag to print out additional info for debugging.

        Returns:
            Optimized cameras and tracks with error metrics, aligned to GT if provided.
            Filtered result after removing high-reprojection-error landmarks and cameras with no remaining tracks.
            Boolean mask over input tracks: True if the track survived all BA filtering steps.
            Per-step wall-clock times for each BA iteration.
        """
        num_ba_steps = len(self._reproj_error_thresholds)
        assert num_ba_steps > 0, "No BA steps to perform"

        current_input = initial_data
        active_track_indices = list(range(initial_data.number_tracks()))
        cumulative_valid_mask = [True] * initial_data.number_tracks()
        step_times: List[float] = []

        for step, reproj_error_thresh in enumerate(self._reproj_error_thresholds):
            step_start_time = time.time()
            optimized_data, filtered_result, stage_valid_mask, final_error = self.run_ba_stage_with_filtering(
                current_input,
                absolute_pose_priors,
                relative_pose_priors,
                reproj_error_thresh,
                verbose,
            )
            step_times.append(time.time() - step_start_time)

            if optimized_data is None or filtered_result is None or stage_valid_mask is None:
                return optimized_data, filtered_result, stage_valid_mask, step_times  # type: ignore

            if len(stage_valid_mask) != len(active_track_indices):
                raise ValueError(
                    "Bundle adjustment stage returned a validity mask whose length does not match the input tracks."
                )

            next_active_track_indices = []
            for original_track_idx, is_valid in zip(active_track_indices, stage_valid_mask):
                cumulative_valid_mask[original_track_idx] = is_valid
                if is_valid:
                    next_active_track_indices.append(original_track_idx)
            active_track_indices = next_active_track_indices
            current_input = filtered_result

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

        return optimized_data, filtered_result, cumulative_valid_mask, step_times  # type: ignore

    def _run_ba_and_evaluate(
        self,
        initial_data: GtsfmData,
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        save_dir: Optional[str] = None,
        verbose: bool = True,
        tracks_2d: Optional[List["SfmTrack2d"]] = None,
    ) -> Tuple[GtsfmData, GtsfmData, List[bool], GtsfmMetricsGroup]:
        """Runs the equivalent of `run_ba()` and `evaluate()` in a single function, to enable time profiling.

        Args:
            tracks_2d: (optional) Union-find 2D track set from `CppDsfTracksEstimator.run()`.
                Required when `use_multi_view_retriangulation` is enabled — used by
                the post-BA retriangulation stage.
        """
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
        start_time = time.time()
        optimized_data, filtered_result, valid_mask, step_times = self.run_ba(
            initial_data=initial_data,
            absolute_pose_priors=absolute_pose_priors,
            relative_pose_priors=relative_pose_priors,
            verbose=verbose,
        )
        total_time = time.time() - start_time

        if self._use_multi_view_retriangulation:
            if tracks_2d is None:
                logger.warning(
                    "use_multi_view_retriangulation is True but tracks_2d was not "
                    "passed to _run_ba_and_evaluate — skipping retri stage."
                )
            else:
                retri_start = time.time()
                retri_options = TriangulationOptions(
                    reproj_error_threshold=self._mv_retri_reproj_error_thresh,
                    mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
                    max_num_hypotheses=self._mv_retri_max_num_hypotheses,
                )
                logger.info(
                    "[Retri] Multi-view retriangulation on %d 2D tracks "
                    "(min_track_len=%d, reproj_error_thresh=%.1fpx)",
                    len(tracks_2d), self._mv_retri_min_track_length,
                    self._mv_retri_reproj_error_thresh,
                )
                retri_data = multi_view_retriangulate_from_2d_tracks(
                    gtsfm_data=filtered_result,
                    tracks_2d=tracks_2d,
                    triangulation_options=retri_options,
                    min_track_length=self._mv_retri_min_track_length,
                )
                if retri_data.number_tracks() > 0:
                    (optimized_data, filtered_result, valid_mask, _) = self.run_ba_stage_with_filtering(
                        initial_data=retri_data,
                        absolute_pose_priors=absolute_pose_priors,
                        relative_pose_priors=relative_pose_priors,
                        reproj_error_thresh=None,
                        verbose=verbose,
                    )
                    logger.info(
                        "[Retri] Stage complete: %d tracks, %.1fs",
                        filtered_result.number_tracks(), time.time() - retri_start,
                    )
                step_times.append(time.time() - retri_start)

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
            gt_wTi=poses_gt,
            computed_wTi=aligned_filtered_data.get_camera_poses(),
            save_dir=save_dir,
            metric_constructed_only=True,
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
        tracks_2d: Optional[Delayed] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: An GtsfmData object wrapped up using dask.delayed.
            absolute_pose_priors: Priors on the poses of the cameras (not delayed).
            relative_pose_priors: Priors on poses between cameras (not delayed).
            cameras_gt: Ground truth camera calibration & pose for each image/camera.
            save_dir: Directory where artifacts and plots should be saved to disk.
            tracks_2d: (optional) Delayed list of 2D tracks from `CppDsfTracksEstimator`.
                Required when `use_multi_view_retriangulation` is enabled.

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
            tracks_2d=tracks_2d,
        )
        return filtered_sfm_data, metrics_graph
