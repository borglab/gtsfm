"""Translation averaging using 1DSFM.

This algorithm was proposed in `Robust Global Translations with 1DSFM' and is implemented by wrapping GTSAM's classes.

References:
- https://research.cs.cornell.edu/1dsfm/
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/MFAS.h
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/TranslationRecovery.h
- https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/TranslationAveragingExample.py

Authors: Jing Wu, Ayush Baid, Akshay Krishnan
"""

import time
import timeit
from collections import defaultdict
from enum import Enum
from typing import Any, DefaultDict, Dict, List, Optional, Set, Tuple

import dask
import gtsam
import numpy as np
from distributed.worker import get_client
from gtsam import (
    MFAS,
    BinaryMeasurementPoint3,
    BinaryMeasurementsPoint3,
    BinaryMeasurementsUnit3,
    BinaryMeasurementUnit3,
    Point3,
    Pose3,
    Rot3,
    TranslationRecovery,
    Unit3,
    symbol_shorthand,
)

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.alignment as alignment_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
import gtsfm.utils.sampling as sampling_utils
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.products.visibility_graph import ImageIndexPairs

# Hyperparameters for 1D-SFM
# maximum number of times 1dsfm will project the Unit3's to a 1d subspace for outlier rejection
MAX_PROJECTION_DIRECTIONS = 2000
OUTLIER_WEIGHT_THRESHOLD = 0.125

NOISE_MODEL_DIMENSION = 3  # chordal distances on Unit3
NOISE_MODEL_SIGMA = 0.01
HUBER_LOSS_K = 1.3  # default value from GTSAM

MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0

# Minimum number of measurements required for a track to be used for averaging.
MIN_TRACK_MEASUREMENTS_FOR_AVERAGING = 3

# Number of track measurements to be added for each camera. Can be reduced to 8 for speed at the cost of some accuracy.
TRACKS_MEASUREMENTS_PER_CAMERA = 12

# Heuristically set to limit the number of delayed tasks, as recommended by Dask:
# https://docs.dask.org/en/stable/delayed-best-practices.html#avoid-too-many-tasks
MAX_DELAYED_CALLS = 16

logger = logger_utils.get_logger()

C = symbol_shorthand.A  # for camera translation variables
L = symbol_shorthand.B  # for track (landmark) translation variables

RelativeDirectionsDict = Dict[Tuple[int, int], Unit3]
DUMMY_NOISE_MODEL = gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)  # MFAS does not use this.


class TranslationAveraging1DSFM(TranslationAveragingBase):
    """1D-SFM translation averaging with outlier rejection."""

    class ProjectionSamplingMethod(str, Enum):
        """Used to select how the projection directions in 1DSfM are sampled."""

        # The string values for enums enable using them in the config.

        # Randomly choose projection directions from input measurements.
        SAMPLE_INPUT_MEASUREMENTS = "SAMPLE_INPUT_MEASUREMENTS"
        # Fit a Gaussian density to input measurements and sample from it.
        SAMPLE_WITH_INPUT_DENSITY = "SAMPLE_WITH_INPUT_DENSITY"
        # Uniformly sample 3D directions at random.
        SAMPLE_WITH_UNIFORM_DENSITY = "SAMPLE_WITH_UNIFORM_DENSITY"

    def __init__(
        self,
        robust_measurement_noise: bool = True,
        use_tracks_for_averaging: bool = True,
        reject_outliers: bool = True,
        projection_sampling_method: ProjectionSamplingMethod = ProjectionSamplingMethod.SAMPLE_WITH_UNIFORM_DENSITY,
        max_delayed_calls: int = MAX_DELAYED_CALLS,
        use_all_tracks_for_averaging: bool = False,
        use_relative_camera_poses: bool = True,
    ) -> None:
        """Initializes the 1DSFM averaging instance.

        Args:
            robust_measurement_noise: Whether to use a robust noise model for the measurements, defaults to true.
            use_tracks_for_averaging:
            reject_outliers: whether to perform outlier rejection with MFAS algorithm (default True).
            projection_sampling_method: ProjectionSamplingMethod to be used for directions to run 1DSfM.
            max_delayed_calls: Maximum number of concurrent delayed tasks to create.
            use_all_tracks_for_averaging: Use
        """
        super().__init__(robust_measurement_noise)

        self._max_1dsfm_projection_directions = MAX_PROJECTION_DIRECTIONS
        self._outlier_weight_threshold = OUTLIER_WEIGHT_THRESHOLD
        self._reject_outliers = reject_outliers
        self._projection_sampling_method = projection_sampling_method
        self._use_tracks_for_averaging = use_tracks_for_averaging
        self._max_delayed_calls = max_delayed_calls
        self._use_relative_camera_poses = use_relative_camera_poses
        self._use_all_tracks_for_averaging = use_all_tracks_for_averaging

        np.random.seed(0)

    def __sample_projection_directions(
        self,
        w_i2Ui1_list: List[Unit3],
    ) -> List[Unit3]:
        """Samples projection directions for 1DSfM based on the provided sampling method.

        Args:
            w_i2Ui1_list: List of unit translations to be used for biasing sampling.
            Used only if the sampling method is SAMPLE_INPUT_MEASUREMENTS or SAMPLE_WITH_INPUT_DENSITY.

        Returns:
            List of sampled Unit3 projection directions.
        """
        num_measurements = len(w_i2Ui1_list)

        if self._projection_sampling_method == self.ProjectionSamplingMethod.SAMPLE_INPUT_MEASUREMENTS:
            num_samples = min(num_measurements, self._max_1dsfm_projection_directions)
            sampled_indices = np.random.choice(num_measurements, num_samples, replace=False)
            projections = [w_i2Ui1_list[idx] for idx in sampled_indices]
        elif self._projection_sampling_method == self.ProjectionSamplingMethod.SAMPLE_WITH_INPUT_DENSITY:
            projections = sampling_utils.sample_kde_directions(
                w_i2Ui1_list, num_samples=self._max_1dsfm_projection_directions
            )
        elif self._projection_sampling_method == self.ProjectionSamplingMethod.SAMPLE_WITH_UNIFORM_DENSITY:
            projections = sampling_utils.sample_random_directions(num_samples=self._max_1dsfm_projection_directions)
        else:
            raise ValueError("Unsupported sampling method!")

        return projections

    @staticmethod
    def _binary_measurements_from_dict(
        w_i2Ui1_dict: RelativeDirectionsDict,
        w_iUj_dict_tracks: RelativeDirectionsDict,
        noise_model: gtsam.noiseModel,
    ) -> BinaryMeasurementsUnit3:
        """Gets a list of BinaryMeasurementUnit3 by combining measurements in w_i2Ui1_dict and w_i2Ui1_dict_tracks.

        Args:
            w_i2Ui1_dict: Dictionary of Unit3 relative translations between cameras.
            w_iUj_dict_tracks: Dictionary of Unit3 relative translations between cameras and landmarks.
            noise_model: Noise model to use for the measurements.

        Returns:
            List of binary measurements.
        """
        w_i1Ui2_measurements = BinaryMeasurementsUnit3()
        for (i1, i2), w_i2Ui1 in w_i2Ui1_dict.items():
            w_i1Ui2_measurements.append(BinaryMeasurementUnit3(C(i2), C(i1), w_i2Ui1, noise_model))
        for (j, i), w_iUj in w_iUj_dict_tracks.items():
            w_i1Ui2_measurements.append(BinaryMeasurementUnit3(C(i), L(j), w_iUj, noise_model))

        return w_i1Ui2_measurements

    def _binary_measurements_from_priors(
        self, i2Ti1_priors: Dict[Tuple[int, int], PosePrior], wRi_list: List[Rot3]
    ) -> BinaryMeasurementsPoint3:
        """Converts the priors from relative Pose3 priors to relative Point3 measurements in world frame.

        Args:
            i2Ti1_priors: Relative pose priors between cameras, could be a hard or soft prior.
            wRi_list: Absolute rotation estimates from rotation averaging.

        Returns:
            BinaryMeasurementsPoint3 containing Point3 priors in world frame.
        """

        def get_prior_in_world_frame(i2, i2Ti1_prior):
            return wRi_list[i2].rotate(i2Ti1_prior.value.translation())

        w_i1ti2_prior_measurements = BinaryMeasurementsPoint3()
        if len(i2Ti1_priors) == 0:
            return w_i1ti2_prior_measurements

        # TODO(akshay-krishnan): Use the translation covariance, transform to world frame.
        # noise_model = gtsam.noiseModel.Gaussian.Covariance(i2Ti1_prior.covariance)
        # TODO(akshay-krishnan): use robust noise model for priors?
        noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)
        for (i1, i2), i2Ti1_prior in i2Ti1_priors.items():
            w_i1ti2_prior_measurements.append(
                BinaryMeasurementPoint3(
                    C(i2),
                    C(i1),
                    get_prior_in_world_frame(i2, i2Ti1_prior),
                    noise_model,
                )
            )
        return w_i1ti2_prior_measurements

    @staticmethod
    def run_mfas(
        w_i2Ui1_dict: RelativeDirectionsDict,
        w_iUj_dict_tracks: RelativeDirectionsDict,
        directions: List[Unit3],
    ) -> Dict[Tuple[int, int], float]:
        """Runs MFAS on a batch of directions."""
        w_i1Ui2_measurements = TranslationAveraging1DSFM._binary_measurements_from_dict(
            w_i2Ui1_dict, w_iUj_dict_tracks, DUMMY_NOISE_MODEL
        )
        results = []
        for dir in directions:
            # Note: Have to convert output of MFAS::computeOutlierWeights to Dict, as Dask has no instructions to pickle
            #   KeyPairDoubleMap objects.
            results.append(dict(MFAS(w_i1Ui2_measurements, dir).computeOutlierWeights()))

        return results

    def compute_inliers(
        self,
        w_i2Ui1_dict: RelativeDirectionsDict,
        w_iUj_dict_tracks: RelativeDirectionsDict,
    ) -> Tuple[RelativeDirectionsDict, RelativeDirectionsDict, Set[int]]:
        """Perform inlier detection for the relative direction measurements.

        Args:
            w_i2Ui1_dict: Dictionary of Unit3 relative translations between cameras.
            w_i2Ui1_dict_tracks: Dictionary of Unit3 relative translations between cameras and landmarks.

        Returns:
            Tuple of:
            inlier_w_i2Ui1_dict: Dictionary of inlier Unit3 relative translations between cameras.
            w_iUj_dict_tracks: Dictionary of inlier Unit3 relative translations between cameras and landmarks.
            inlier_cameras: Set of inlier cameras.
        """

        # Sample directions for projection
        combined_measurements = list(w_i2Ui1_dict.values()) + list(w_iUj_dict_tracks.values())
        projection_directions = self.__sample_projection_directions(combined_measurements)

        # Convert to measurements: map indexes to symbols.
        w_i1Ui2_measurements = self._binary_measurements_from_dict(w_i2Ui1_dict, w_iUj_dict_tracks, DUMMY_NOISE_MODEL)

        # Scatter data to all workers if client available.
        try:
            client = get_client()
            future_w_i2Ui1_dict = client.scatter(w_i2Ui1_dict, broadcast=True)
            future_w_iUj_dict_tracks = client.scatter(w_iUj_dict_tracks, broadcast=True)
        except ValueError:  # allows use without initializing client.
            logger.info("No Dask client found... Running without scattering.")
            future_w_i2Ui1_dict = w_i2Ui1_dict
            future_w_iUj_dict_tracks = w_iUj_dict_tracks

        # Loop through tracks and and generate delayed MFAS tasks.
        batch_size = int(np.ceil(len(projection_directions) / self._max_delayed_calls))
        batched_outlier_weights: List[Any] = []
        for j in range(0, len(projection_directions), batch_size):
            batched_outlier_weights.append(
                dask.delayed(self.run_mfas)(
                    future_w_i2Ui1_dict,
                    future_w_iUj_dict_tracks,
                    projection_directions[j : j + batch_size],
                )
            )

        # Compute outlier weights in parallel.
        _t2 = timeit.default_timer()
        batched_outlier_weights = dask.compute(*batched_outlier_weights)
        logger.info("⏱️ Computed outlier weights using MFAS in %.2f seconds." % (timeit.default_timer() - _t2))

        # Compute average outlier weight.
        outlier_weights_sum: DefaultDict[Tuple[int, int], float] = defaultdict(float)
        inliers = set()
        for batch_outlier_weights in batched_outlier_weights:
            for outlier_weight_dict in batch_outlier_weights:
                for w_i1Ui2 in w_i1Ui2_measurements:
                    i1, i2 = w_i1Ui2.key1(), w_i1Ui2.key2()
                    outlier_weights_sum[(i1, i2)] += outlier_weight_dict[(i1, i2)]
        for (i1, i2), weight_sum in outlier_weights_sum.items():
            if weight_sum / len(projection_directions) < OUTLIER_WEIGHT_THRESHOLD:
                inliers.add((i1, i2))

        # Filter outliers, index back from symbol to int.
        # `inliers` contains both camera-camera and camera-landmark inliers. We separate them here.
        inlier_w_i2Ui1_dict = {}
        inlier_w_iUj_dict_tracks = {}
        inlier_cameras: Set[int] = set()
        for i1, i2 in w_i2Ui1_dict:
            if (C(i2), C(i1)) in inliers:  # there is a flip in indices from w_i2Ui1_dict to inliers.
                inlier_w_i2Ui1_dict[(i1, i2)] = w_i2Ui1_dict[(i1, i2)]
                inlier_cameras.add(i1)
                inlier_cameras.add(i2)

        for j, i in w_iUj_dict_tracks:
            # Same as above, `inliers` contains symbols that are flipped - C(i), L(j).
            # Only add an inlier camera-track measurements if the camera has other camera-camera inliers.
            if (C(i), L(j)) in inliers and i in inlier_cameras:
                inlier_w_iUj_dict_tracks[(j, i)] = w_iUj_dict_tracks[(j, i)]

        return inlier_w_i2Ui1_dict, inlier_w_iUj_dict_tracks, inlier_cameras

    def __get_initial_values(self, wTi_initial: List[Optional[PosePrior]]) -> gtsam.Values:
        """Converts translations from a list of absolute poses to gtsam.Values for initialization.

        Args:
            wTi_initial: List of absolute poses.

        Returns:
            gtsam.Values containing initial translations (uses symbols for camera index).
        """
        initial = gtsam.Values()
        for i, wTi in enumerate(wTi_initial):
            if wTi is not None:
                initial.insertPoint3(C(i), wTi.value.translation())
        return initial

    def _select_tracks_for_averaging(
        self,
        tracks: List[SfmTrack2d],
        valid_cameras: Set[int],
        intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        measurements_per_camera=TRACKS_MEASUREMENTS_PER_CAMERA,
    ) -> List[SfmTrack2d]:
        """Removes bad tracks and selects the longest ones until all cameras see `measurements_per_camera` tracks.

        Bad tracks are those that have fewer than 3 measurements from valid_cameras.
        Selects tracks based on the number of measurements contributed until all cameras see at least
         `measurements_per_camera` tracks.
        This is based on 1dsfm's implementation here:
        https://github.com/wilsonkl/SfM_Init/blob/fd012ef93462b8623e8d65fa0c6fa95b32270a3c/sfminit/transproblem.py#L235

        Args:
            tracks: List of all input tracks.
            valid_cameras: Set of valid camera indices (these have direction measurements and valid rotations).
            intrinsics: List of camera intrinsics.
            measurements_per_camera: Number of track direction measurements that need to be observed by each camera.

        Returns:
            List of tracks to use for averaging.
        """
        filtered_tracks = []
        for track in tracks:
            valid_cameras_track = track.select_for_cameras(camera_idxs=valid_cameras)
            if valid_cameras_track.number_measurements() < MIN_TRACK_MEASUREMENTS_FOR_AVERAGING:
                continue
            filtered_tracks.append(valid_cameras_track)

        if self._use_all_tracks_for_averaging:
            return filtered_tracks

        tracks_subset = []

        # Number of measurements per camera that we still need to add.
        num_remaining_measurements = {c: measurements_per_camera for c in valid_cameras}
        # Number of measurements added by each track.
        improvement = [t.number_measurements() for t in filtered_tracks]  # how much cover each track would add

        # preparation: make a lookup from camera to tracks in the camera
        camera_track_lookup = {c: [] for c in valid_cameras}

        for track_id, track in enumerate(filtered_tracks):
            for j in range(track.number_measurements()):
                measurement = track.measurement(j)
                camera_track_lookup[measurement.i].append(track_id)

        for count in range(len(filtered_tracks)):  # artificial limit to avoid infinite loop
            # choose track that maximizes the heuristic
            best_track_id = np.argmax(improvement)
            if improvement[best_track_id] <= 0:
                break
            tracks_subset.append(filtered_tracks[best_track_id])

            # update the state variables: num_remaining_measurements and improvement
            improvement[best_track_id] = 0
            for measurement in filtered_tracks[best_track_id].measurements:
                if num_remaining_measurements[measurement.i] == 0:
                    continue
                num_remaining_measurements[measurement.i] = num_remaining_measurements[measurement.i] - 1

                # If this image just got covered the k'th time, it's done.
                # Lower the improvement for all tracks that see it.
                if num_remaining_measurements[measurement.i] == 0:
                    for t in camera_track_lookup[measurement.i]:
                        improvement[t] = improvement[t] - 1 if improvement[t] > 0 else 0

        return tracks_subset

    def _get_landmark_directions(
        self,
        tracks_2d: List[SfmTrack2d],
        intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        wRi_list: List[Optional[Rot3]],
    ) -> RelativeDirectionsDict:
        """Computes the camera to landmark directions for each track, in world frame.

        Args:
            tracks_2d: 2d tracks in each image, assuming that all measurements in all tracks are for valid cameras.
            intrinsics: camera intrinsics for each camera.
            wRi_list: camera rotations in world frame.

        Returns:
            Dictionary of unit directions from camera to track in world frame indexed by (track_id, camera_id).
        """
        landmark_directions = {}
        for track_id, track in enumerate(tracks_2d):
            for j in range(track.number_measurements()):
                measurement = track.measurement(j)
                cam_idx = measurement.i

                if intrinsics[cam_idx] is None or wRi_list[cam_idx] is None:
                    raise ValueError("Camera intrinsics or rotation cannot be None for input track measurements")

                measurement_xy = intrinsics[cam_idx].calibrate(measurement.uv)
                measurement_homog = Point3(measurement_xy[0], measurement_xy[1], 1.0)
                w_cam_U_track = Unit3(wRi_list[cam_idx].rotate(Unit3(measurement_homog).point3()))

                # Direction starts at camera, but first index is track_id.
                landmark_directions[(track_id, cam_idx)] = w_cam_U_track
        return landmark_directions

    def __run_averaging(
        self,
        num_images: int,
        w_i2Ui1_dict: RelativeDirectionsDict,
        w_i2Ui1_dict_tracks: RelativeDirectionsDict,
        wRi_list: List[Optional[Rot3]],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior],
        absolute_pose_priors: List[Optional[PosePrior]],
        scale_factor: float,
    ) -> List[Optional[Point3]]:
        """Runs the averaging optimization.

        Args:
            num_images: number of images.
            w_i2Ui1_dict: Unit directions from i2 to i1 in world frame indexed by (i1, i2).
            w_i2Ui1_dict_tracks: Directions from camera to track in world frame indexed by (track_id, camera_id).
            wRi_list: camera rotations in world frame.
            i2Ti1_priors: relative pose priors.
            absolute_pose_priors: absolute pose priors.
            scale_factor: scale factor for the esimated translations.

        Returns:
            List of camera translations in world frame, with as many entries as the number of images.
        """
        logger.info(
            "Using %d track measurements and %d camera measurements", len(w_i2Ui1_dict_tracks), len(w_i2Ui1_dict)
        )

        noise_model = gtsam.noiseModel.Isotropic.Sigma(NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)
        if self._robust_measurement_noise:
            huber_loss = gtsam.noiseModel.mEstimator.Huber.Create(HUBER_LOSS_K)
            noise_model = gtsam.noiseModel.Robust.Create(huber_loss, noise_model)

        w_i1Ui2_measurements = self._binary_measurements_from_dict(w_i2Ui1_dict, w_i2Ui1_dict_tracks, noise_model)

        # Run the optimizer.
        try:
            algorithm = TranslationRecovery()
            if len(i2Ti1_priors) > 0:
                # scale is ignored here.
                w_i1ti2_priors = self._binary_measurements_from_priors(i2Ti1_priors, wRi_list)
                wti_initial = self.__get_initial_values(absolute_pose_priors)
                wti_values = algorithm.run(w_i1Ui2_measurements, 0.0, w_i1ti2_priors, wti_initial)
            else:
                wti_values = algorithm.run(w_i1Ui2_measurements, scale_factor)
        except TypeError as e:
            # TODO(akshay-krishnan): remove when no longer supporting gtsam versions before 4.2a7.
            logger.error("TypeError: {}".format(str(e)))
            algorithm = TranslationRecovery(w_i1Ui2_measurements)
            wti_values = algorithm.run(scale_factor)

        # Transforms the result to a list of Point3 objects.
        wti_list: List[Optional[Point3]] = [None] * num_images
        for i in range(num_images):
            if wRi_list[i] is not None and wti_values.exists(C(i)):
                wti_list[i] = wti_values.atPoint3(C(i))
        return wti_list

    # TODO(ayushbaid): Change wTi_initial to Pose3.
    def run_translation_averaging(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        tracks_2d: Optional[List[SfmTrack2d]] = None,
        intrinsics: Optional[List[Optional[gtsfm_types.CALIBRATION_TYPE]]] = None,
        absolute_pose_priors: List[Optional[PosePrior]] = [],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior] = {},
        scale_factor: float = 1.0,
        gt_wTi_list: List[Optional[Pose3]] = [],
    ) -> Tuple[List[Optional[Pose3]], Optional[GtsfmMetricsGroup], Optional[ImageIndexPairs]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i2Ui1_dict: relative unit-translation as dictionary (i1, i2): i2Ui1
            wRi_list: global rotations for each camera pose in the world coordinates.
            absolute_pose_priors: priors on the camera poses (not delayed).
            i2Ti1_priors: priors on the pose between camera pairs (not delayed) as (i1, i2): i2Ti1.
            scale_factor: non-negative global scaling factor.
            gt_wTi_list: ground truth poses for computing metrics.

        Returns:
            Global translation wti for each camera pose. The number of entries in the list is `num_images`. The list
                may contain `None` where the global translations could not be computed (either underconstrained system
                or ill-constrained system).
            A GtsfmMetricsGroup of 1DSfM metrics.
            List of camera pair indices that are classified as inliers by 1dsfm.
        """
        logger.info("Running translation averaging on %d unit translations", len(i2Ui1_dict))

        w_i2Ui1_dict, valid_cameras = get_valid_measurements_in_world_frame(i2Ui1_dict, wRi_list)
        if not self._use_relative_camera_poses:
            w_i2Ui1_dict = {}

        start_time = time.time()
        if self._use_tracks_for_averaging:
            if tracks_2d is None:
                raise ValueError("Tracks must be provided when they are to be used in translation averaging.")
            if intrinsics is None or len(intrinsics) != len(wRi_list):
                raise ValueError("Number of intrinsics must match number of cameras when tracks are provided.")
            selected_tracks = self._select_tracks_for_averaging(tracks_2d, valid_cameras, intrinsics)
            w_i2Ui1_dict_tracks = self._get_landmark_directions(selected_tracks, intrinsics, wRi_list)
        else:
            w_i2Ui1_dict_tracks = {}

        inlier_computation_start_time = time.time()
        if self._reject_outliers:
            w_i2Ui1_dict_inliers, w_i2Ui1_dict_tracks_inliers, inlier_cameras = self.compute_inliers(
                w_i2Ui1_dict, w_i2Ui1_dict_tracks
            )
        else:
            w_i2Ui1_dict_inliers = w_i2Ui1_dict
            w_i2Ui1_dict_tracks_inliers = w_i2Ui1_dict_tracks

        inlier_computation_time = time.time() - inlier_computation_start_time

        averaging_start_time = time.time()
        wti_list = self.__run_averaging(
            num_images=num_images,
            w_i2Ui1_dict=w_i2Ui1_dict_inliers,
            w_i2Ui1_dict_tracks=w_i2Ui1_dict_tracks_inliers,
            wRi_list=wRi_list,
            i2Ti1_priors=i2Ti1_priors,
            absolute_pose_priors=absolute_pose_priors,
            scale_factor=scale_factor,
        )
        averaging_time = time.time() - averaging_start_time

        # Compute the metrics.
        ta_metrics = compute_metrics(
            set(w_i2Ui1_dict_inliers.keys()), i2Ui1_dict, w_i2Ui1_dict_tracks_inliers, wRi_list, wti_list, gt_wTi_list
        )

        num_translations = sum([1 for wti in wti_list if wti is not None])
        logger.info("Estimated %d translations out of %d images.", num_translations, num_images)

        # Combine estimated global rotations and global translations to Pose(3) objects.
        wTi_list = [
            Pose3(wRi, wti) if wRi is not None and wti is not None else None for wRi, wti in zip(wRi_list, wti_list)
        ]
        total_time = time.time() - start_time
        logger.info("⏱️ Translation averaging took %.4f seconds.", total_time)
        ta_metrics.add_metric(GtsfmMetric("total_duration_sec", total_time))
        ta_metrics.add_metric(GtsfmMetric("outlier_rejection_duration_sec", inlier_computation_time))
        ta_metrics.add_metric(GtsfmMetric("optimization_duration_sec", averaging_time))

        return wTi_list, ta_metrics, list(w_i2Ui1_dict_inliers.keys())


def compute_metrics(
    inlier_i1_i2_pairs: Set[Tuple[int, int]],
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
    w_i2Ui1_dict_tracks: Dict[Tuple[int, int], Optional[Unit3]],
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
) -> GtsfmMetricsGroup:
    """Computes the translation averaging metrics as a metrics group.
    Args:
        inlier_i1_i2_pairs: List of inlier camera pair indices.
        i2Ui1_dict: Translation directions between camera pairs (inputs to translation averaging).
        wRi_list: Estimated camera rotations from rotation averaging.
        wti_list: Estimated camera translations from translation averaging.
        gt_wTi_list: List of ground truth camera poses.
    Returns:
        Translation averaging metrics as a metrics group. Includes the following metrics:
        - Number of inlier, outlier and total measurements.
        - Distribution of translation direction angles for inlier measurements.
        - Distribution of translation direction angle for outlier measurements.
    """

    num_camera_measurements = len([k for k, val in i2Ui1_dict.items() if val is not None])
    num_track_measurements = len([k for k, val in w_i2Ui1_dict_tracks.items() if val is not None])
    ta_metrics = [
        GtsfmMetric("num_total_1dsfm_measurements", num_camera_measurements + num_track_measurements),
        GtsfmMetric("num_camera_1dsfm_measurements", num_camera_measurements),
        GtsfmMetric("num_track_1dsfm_measurements", num_track_measurements),
        GtsfmMetric("num_translations_estimated", len([wti for wti in wti_list if wti is not None])),
    ]

    # Remaining metrics require ground truth, so return if GT is not available.
    gt_available = np.array([gt_wTi is not None for gt_wTi in gt_wTi_list]).any()
    if not gt_available:
        return GtsfmMetricsGroup("translation_averaging_metrics", ta_metrics)

    # Get ground truth translation directions for the measurements.
    _, gt_i2Ui1_dict = metrics_utils.get_all_relative_rotations_translations(gt_wTi_list)

    if len(inlier_i1_i2_pairs) > 0:
        threshold_suffix = str(int(MAX_INLIER_MEASUREMENT_ERROR_DEG)) + "_deg"
        outlier_i1_i2_pairs = (
            set([pair_idx for pair_idx, val in i2Ui1_dict.items() if val is not None]) - inlier_i1_i2_pairs
        )
        # Angle between i2Ui1 measurement and GT i2Ui1 measurement for inliers and outliers.
        inlier_angular_errors = metrics_utils.get_measurement_angle_errors(
            inlier_i1_i2_pairs, i2Ui1_dict, gt_i2Ui1_dict
        )
        outlier_angular_errors = metrics_utils.get_measurement_angle_errors(
            outlier_i1_i2_pairs, i2Ui1_dict, gt_i2Ui1_dict
        )
        precision, recall = metrics_utils.get_precision_recall_from_errors(
            inlier_angular_errors, outlier_angular_errors, MAX_INLIER_MEASUREMENT_ERROR_DEG
        )

        measured_gt_i2Ui1_dict = {}
        for i1, i2 in set.union(inlier_i1_i2_pairs, outlier_i1_i2_pairs):
            measured_gt_i2Ui1_dict[(i1, i2)] = gt_i2Ui1_dict[(i1, i2)]

        ta_metrics.extend(
            [
                metrics_utils.compute_relative_translation_angle_metric(
                    measured_gt_i2Ui1_dict, gt_wTi_list, prefix="inlier_"
                ),
                GtsfmMetric("num_inlier_1dsfm_measurements", len(inlier_i1_i2_pairs)),
                GtsfmMetric("num_outlier_1dsfm_measurements", len(outlier_i1_i2_pairs)),
                GtsfmMetric("1dsfm_precision_" + threshold_suffix, precision),
                GtsfmMetric("1dsfm_recall_" + threshold_suffix, recall),
                GtsfmMetric("1dsfm_inlier_angular_errors_deg", inlier_angular_errors),
                GtsfmMetric("1dsfm_outlier_angular_errors_deg", outlier_angular_errors),
            ]
        )

    # Compute estimated poses after the averaging step and align them to ground truth.
    wTi_list: List[Optional[Pose3]] = []
    for wRi, wti in zip(wRi_list, wti_list):
        if wRi is None or wti is None:
            wTi_list.append(None)
        else:
            wTi_list.append(Pose3(wRi, wti))
    wTi_aligned_list, _ = alignment_utils.align_poses_sim3_ignore_missing(gt_wTi_list, wTi_list)
    wti_aligned_list = [wTi.translation() if wTi is not None else None for wTi in wTi_aligned_list]
    gt_wti_list = [gt_wTi.translation() if gt_wTi is not None else None for gt_wTi in gt_wTi_list]
    _, gt_i2Ui1_dict = metrics_utils.get_all_relative_rotations_translations(gt_wTi_list)

    ta_metrics.extend(
        [
            metrics_utils.compute_relative_translation_angle_metric(gt_i2Ui1_dict, wTi_aligned_list),
            metrics_utils.compute_translation_distance_metric(wti_aligned_list, gt_wti_list),
            metrics_utils.compute_translation_angle_metric(gt_wTi_list, wTi_aligned_list),
        ]
    )

    return GtsfmMetricsGroup("translation_averaging_metrics", ta_metrics)


def get_valid_measurements_in_world_frame(
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]], wRi_list: List[Optional[Rot3]]
) -> Tuple[RelativeDirectionsDict, Set[int]]:
    """Returns measurements for which both cameras have valid rotations, transformed to the world frame.

    Args:
        i2Ui1_dict: Relative translation directions between camera pairs.
        wRi_list: List of estimated camera rotations.

    Returns:
        Tuple of:
            Relative translation directions between camera pairs, in world frame.
            Set of camera indices for which we have valid rotations and measurements.
    """

    w_i2Ui1_dict = {}
    valid_cameras: Set[int] = set()
    for (i1, i2), i2Ui1 in i2Ui1_dict.items():
        wRi2 = wRi_list[i2]
        if i2Ui1 is not None and wRi2 is not None:
            w_i2Ui1_dict[(i1, i2)] = Unit3(wRi2.rotate(i2Ui1.point3()))
            valid_cameras.add(i1)
            valid_cameras.add(i2)
    return w_i2Ui1_dict, valid_cameras
