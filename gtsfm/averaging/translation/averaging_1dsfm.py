"""Translation averaging using 1DSFM.

This algorithm was proposed in 'Robust Global Translations with 1DSFM' and is implemented by wrapping GTSAM's classes.

References:
- https://research.cs.cornell.edu/1dsfm/
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/MFAS.h
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/TranslationRecovery.h
- https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/TranslationAveragingExample.py

Authors: Jing Wu, Ayush Baid, Akshay Krishnan
"""
from collections import defaultdict
from enum import Enum
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

import gtsam
import numpy as np
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
)
from scipy import stats

import gtsfm.utils.coordinate_conversions as conversion_utils
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.common.pose_prior import PosePrior
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

# Hyperparameters for 1D-SFM
# maximum number of times 1dsfm will project the Unit3's to a 1d subspace for outlier rejection
MAX_PROJECTION_DIRECTIONS = 2000
OUTLIER_WEIGHT_THRESHOLD = 0.125
MAX_KDE_SAMPLES = 2000

NOISE_MODEL_DIMENSION = 3  # chordal distances on Unit3
NOISE_MODEL_SIGMA = 0.01
HUBER_LOSS_K = 1.345  # default value from GTSAM

MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0

logger = logger_utils.get_logger()


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
        reject_outliers: bool = True,
        projection_sampling_method: ProjectionSamplingMethod = ProjectionSamplingMethod.SAMPLE_WITH_UNIFORM_DENSITY,
        
    ) -> None:
        """Initializes the 1DSFM averaging instance.

        Args:
            robust_measurement_noise: Whether to use a robust noise model for the measurements, defaults to true.
            projection_sampling_method: ProjectionSamplingMethod to be used for directions to run 1DSfM.
            MFAS_outlier_rejection: whether to perform outlier rejection with MFAS algorithm (default True).
        """
        super().__init__(robust_measurement_noise)

        self._max_1dsfm_projection_directions = MAX_PROJECTION_DIRECTIONS
        self._outlier_weight_threshold = OUTLIER_WEIGHT_THRESHOLD
        self._reject_outliers = reject_outliers
        self._projection_sampling_method = projection_sampling_method

    def __sample_projection_directions(
        self,
        w_i2Ui1_measurements: BinaryMeasurementsUnit3,
    ) -> List[Unit3]:
        """Samples projection directions for 1DSfM based on the provided sampling method.

        Args:
            w_i2Ui1_measurements: Unit translation measurements which are input to 1DSfM.

        Returns:
            List of sampled Unit3 projection directions.
        """
        num_measurements = len(w_i2Ui1_measurements)

        if self._projection_sampling_method == self.ProjectionSamplingMethod.SAMPLE_INPUT_MEASUREMENTS:
            num_samples = min(num_measurements, self._max_1dsfm_projection_directions)
            sampled_indices = np.random.choice(num_measurements, num_samples, replace=False)
            projections = [w_i2Ui1_measurements[idx].measured() for idx in sampled_indices]
        elif self._projection_sampling_method == self.ProjectionSamplingMethod.SAMPLE_WITH_INPUT_DENSITY:
            projections = _sample_kde_directions(
                w_i2Ui1_measurements, num_samples=self._max_1dsfm_projection_directions
            )
        elif self._projection_sampling_method == self.ProjectionSamplingMethod.SAMPLE_WITH_UNIFORM_DENSITY:
            projections = _sample_random_directions(num_samples=self._max_1dsfm_projection_directions)
        else:
            raise ValueError("Unsupported sampling method!")

        return projections

    def __compute_inlier_mask(
        self,
        w_i2Ui1_measurements: BinaryMeasurementsUnit3,
    ) -> Set[Tuple[int, int]]:
        """Perform inlier detection for the relative direction measurements.

        Args:
            w_i2Ui1_measurements: relative directions in the common world coordinate frame.

        Returns:
            Set of indices (i1, i2) which are inliers.
        """
        # Sample projection directions for 1DSfM.
        projection_directions = self.__sample_projection_directions(w_i2Ui1_measurements)

        # Compute outlier weights using MFAS.
        outlier_weights: List[Dict[Tuple[int, int], float]] = []
        # TODO(ayush): parallelize this step.
        for direction in projection_directions:
            mfas_instance = MFAS(w_i2Ui1_measurements, direction)
            outlier_weights.append(mfas_instance.computeOutlierWeights())
        logger.debug("Computed outlier weights using MFAS.")

        # Compute average outlier weight.
        outlier_weights_sum: DefaultDict[Tuple[int, int], float] = defaultdict(float)
        inliers = set()
        for outlier_weight_dict in outlier_weights:
            # TODO(akshay-krishnan): use keys from outlier weight dict once we can iterate over it (gtsam fix).
            for idx in range(len(w_i2Ui1_measurements)):
                w_i2Ui1 = w_i2Ui1_measurements[idx]
                i2, i1 = w_i2Ui1.key1(), w_i2Ui1.key2()
                outlier_weights_sum[(i2, i1)] += outlier_weight_dict[(i2, i1)]

        for (i2, i1) in outlier_weights_sum:
            if outlier_weights_sum[(i2, i1)] / len(projection_directions) < OUTLIER_WEIGHT_THRESHOLD:
                inliers.add((i1, i2))

        return inliers

    def _get_prior_measurements_in_world_frame(
        self,
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior],
        wRi_list: List[Optional[Rot3]],
    ) -> BinaryMeasurementsPoint3:
        """Converts the priors from relative Pose3 priors to relative Point3 priors in world frame.

        Args:
            i2Ti1_priors: Relative pose priors between cameras, could be a hard or soft prior.
            wRi_list: Absolute rotation estimates from rotation averaging.

        Returns:
            BinaryMeasurementsPoint3 containing Point3 priors in world frame.
        """
        if len(i2Ti1_priors) == 0:
            return gtsam.BinaryMeasurementsPoint3()

        def get_prior_in_world_frame(i2, i2Ti1_prior):
            return wRi_list[i2].rotate(i2Ti1_prior.value.translation())

        w_i2ti1_priors = gtsam.BinaryMeasurementsPoint3()

        for (i1, i2), i2Ti1_prior in i2Ti1_priors.items():
            # TODO(akshay-krishnan): Use the translation covariance, transform to world frame.
            # noise_model = gtsam.noiseModel.Gaussian.Covariance(i2Ti1_prior.covariance)
            noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 1e-2)
            w_i2ti1_priors.append(
                BinaryMeasurementPoint3(
                    i2,
                    i1,
                    get_prior_in_world_frame(i2, i2Ti1_prior),
                    noise_model,
                )
            )
        return w_i2ti1_priors

    def __get_initial_values(self, wTi_initial: List[Optional[PosePrior]]):
        """Converts translations from a list of absolute poses to gtsam.Values for initialization.

        Args:
            wTi_initial: List of
        """
        initial = gtsam.Values()
        for i, wTi in enumerate(wTi_initial):
            if wTi is not None:
                initial.insertPoint3(i, wTi.value.translation())
        return initial

    # TODO(ayushbaid): Change wTi_initial to Pose3.
    def run_translation_averaging(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        absolute_pose_priors: List[Optional[PosePrior]] = [],
        i2Ti1_priors: Dict[Tuple[int, int], PosePrior] = {},
        scale_factor: float = 1.0,
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[List[Optional[Pose3]], Optional[GtsfmMetricsGroup]]:
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
        """
        logger.info("Running translation averaging on %s unit translations", len(i2Ui1_dict))
        noise_model = gtsam.noiseModel.Isotropic.Sigma(NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)
        if self._robust_measurement_noise:
            huber_loss = gtsam.noiseModel.mEstimator.Huber.Create(HUBER_LOSS_K)
            noise_model = gtsam.noiseModel.Robust.Create(huber_loss, noise_model)

        w_i2Ui1_measurements = cast_to_measurements_variable_in_global_coordinate_frame(
            i2Ui1_dict, wRi_list, noise_model
        )
        logger.debug("Created measurements in global frame.")
        # Possibly perform (slow!) outlier rejection with Minimum Feedback Arc Set algorithm.
        if self._reject_outliers:
            inliers: Set[Tuple[int, int]] = self.__compute_inlier_mask(w_i2Ui1_measurements)
    
            w_i2Ui1_inlier_measurements = BinaryMeasurementsUnit3()
            for idx in range(len(w_i2Ui1_measurements)):
                # key1 is i2 and key2 is i1 above.
                w_i2Ui1 = w_i2Ui1_measurements[idx]
                i1 = w_i2Ui1.key2()
                i2 = w_i2Ui1.key1()
                if (i1, i2) in inliers:
                    w_i2Ui1_inlier_measurements.append(w_i2Ui1)
        else:
            inliers = set(i2Ui1_dict.keys())

        # Run the optimizer
        # TODO(akshay-krishnan): remove once latest gtsam pip wheels updated.
        try:
            algorithm = TranslationRecovery()
            w_i2ti1_priors = self._get_prior_measurements_in_world_frame(i2Ti1_priors, wRi_list)
            wti_initial = self.__get_initial_values(absolute_pose_priors)
            if len(w_i2ti1_priors) > 0:
                # scale is ignored here.
                wti_values = algorithm.run(w_i2Ui1_inlier_measurements, 0.0, w_i2ti1_priors, wti_initial)
            else:
                wti_values = algorithm.run(w_i2Ui1_inlier_measurements, scale_factor)
        except TypeError as e:
            logger.error("TypeError: {}".format(str(e)))
            recovery = TranslationRecovery(w_i2Ui1_inlier_measurements)
            wti_values = recovery.run(scale_factor)

        # Transform the result to a list of Point3.
        wti_list = [
            wti_values.atPoint3(i) if wRi_list[i] is not None and wti_values.exists(i) else None
            for i in range(num_images)
        ]
        # Combine estimated global rotations and global translations to Pose(3) objects.
        wTi_list = [
            Pose3(wRi, wti) if wRi is not None and wti is not None else None for wRi, wti in zip(wRi_list, wti_list)
        ]

        # Compute the metrics.
        if gt_wTi_list is not None:
            ta_metrics = _compute_metrics(inliers, i2Ui1_dict, wRi_list, wti_list, gt_wTi_list)
        else:
            ta_metrics = None

        num_translations = sum([1 for wti in wti_list if wti is not None])
        logger.info("Estimated %s translations out of %s images.", num_translations, num_images)
        return wTi_list, ta_metrics


def _sample_kde_directions(w_i2Ui1_measurements: BinaryMeasurementsUnit3, num_samples: int) -> List[Unit3]:
    """Fits a Gaussian density kernel to the provided measurements, and then samples num_samples from this kernel.

    Args:
        w_i2Ui1_measurements: List of BinaryMeasurementUnit3 direction measurements.
        num_samples: Number of samples to be sampled from the kernel.

    Returns:
        List of sampled Unit3 directions.
    """
    w_i2Ui1_list = [w_i2Ui1.measured() for w_i2Ui1 in w_i2Ui1_measurements]
    if len(w_i2Ui1_list) > MAX_KDE_SAMPLES:
        w_i2Ui1_subset_indices = np.random.choice(range(len(w_i2Ui1_list)), MAX_KDE_SAMPLES, replace=False).tolist()
        w_i2Ui1_list = [w_i2Ui1_list[i] for i in w_i2Ui1_subset_indices]

    w_i2Ui1_spherical = conversion_utils.cartesian_to_spherical_directions(w_i2Ui1_list)

    # gaussian_kde expects each sample to be a column, hence transpose.
    kde = stats.gaussian_kde(w_i2Ui1_spherical.T)
    sampled_directions_spherical = kde.resample(size=num_samples).T
    return conversion_utils.spherical_to_cartesian_directions(sampled_directions_spherical)


def _sample_random_directions(num_samples: int) -> List[Unit3]:
    """Samples num_samples Unit3 3D directions.
    The sampling is done in 2D spherical coordinates (azimuth, elevation), and then converted to Cartesian coordinates.
    Sampling in spherical coordinates was found to have precision-recall for 1dsfm outlier rejection.

    Args:
        num_samples: Number of samples required.

    Returns:
        List of sampled Unit3 directions.
    """
    sampled_azimuth = np.random.uniform(low=-np.pi, high=np.pi, size=(num_samples, 1))
    sampled_elevation = np.random.uniform(low=0.0, high=np.pi, size=(num_samples, 1))
    sampled_azimuth_elevation = np.concatenate((sampled_azimuth, sampled_elevation), axis=1)

    return conversion_utils.spherical_to_cartesian_directions(sampled_azimuth_elevation)


def _get_measurement_angle_errors(
    i1_i2_pairs: Set[Tuple[int, int]],
    i2Ui1_measurements: Dict[Tuple[int, int], Unit3],
    gt_i2Ui1_measurements: Dict[Tuple[int, int], Unit3],
) -> List[Optional[float]]:
    """Returns a list of the angle between i2Ui1_measurements and gt_i2Ui1_measurements for every
    (i1, i2) in i1_i2_pairs.

    Args:
        i1_i2_pairs: List of (i1, i2) tuples for which the angles must be computed.
        i2Ui1_measurements: Measured translation direction of i1 WRT i2.
        gt_i2Ui1_measurements: Ground truth translation direction of i1 WRT i2.

    Returns:
        List of angles between the measured and ground truth translation directions.
    """
    errors: List[Optional[float]] = []
    for (i1, i2) in i1_i2_pairs:
        if (i1, i2) in i2Ui1_measurements and (i1, i2) in gt_i2Ui1_measurements:
            errors.append(
                comp_utils.compute_relative_unit_translation_angle(
                    i2Ui1_measurements[(i1, i2)], gt_i2Ui1_measurements[(i1, i2)]
                )
            )
    return errors


def _compute_metrics(
    inlier_i1_i2_pairs: Set[Tuple[int, int]],
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
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
    # Get ground truth translation directions for the measurements.
    gt_i2Ui1_dict = metrics_utils.get_twoview_translation_directions(gt_wTi_list)
    outlier_i1_i2_pairs = (
        set([pair_idx for pair_idx, val in i2Ui1_dict.items() if val is not None]) - inlier_i1_i2_pairs
    )

    # Angle between i2Ui1 measurement and GT i2Ui1 measurement for inliers and outliers.
    inlier_angular_errors = _get_measurement_angle_errors(inlier_i1_i2_pairs, i2Ui1_dict, gt_i2Ui1_dict)
    outlier_angular_errors = _get_measurement_angle_errors(outlier_i1_i2_pairs, i2Ui1_dict, gt_i2Ui1_dict)
    precision, recall = metrics_utils.get_precision_recall_from_errors(
        inlier_angular_errors, outlier_angular_errors, MAX_INLIER_MEASUREMENT_ERROR_DEG
    )

    measured_gt_i2Ui1_dict = {}
    for (i1, i2) in set.union(inlier_i1_i2_pairs, outlier_i1_i2_pairs):
        measured_gt_i2Ui1_dict[(i1, i2)] = gt_i2Ui1_dict[(i1, i2)]

    # Compute estimated poses after the averaging step and align them to ground truth.
    wTi_list: List[Optional[Pose3]] = []
    for (wRi, wti) in zip(wRi_list, wti_list):
        if wRi is None or wti is None:
            wTi_list.append(None)
        else:
            wTi_list.append(Pose3(wRi, wti))
    wTi_aligned_list, _ = comp_utils.align_poses_sim3_ignore_missing(gt_wTi_list, wTi_list)
    wti_aligned_list = [wTi.translation() if wTi is not None else None for wTi in wTi_aligned_list]
    gt_wti_list = [gt_wTi.translation() if gt_wTi is not None else None for gt_wTi in gt_wTi_list]

    num_total_measurements = len(inlier_i1_i2_pairs) + len(outlier_i1_i2_pairs)
    threshold_suffix = str(int(MAX_INLIER_MEASUREMENT_ERROR_DEG)) + "_deg"
    ta_metrics = [
        GtsfmMetric("num_total_1dsfm_measurements", num_total_measurements),
        GtsfmMetric("num_inlier_1dsfm_measurements", len(inlier_i1_i2_pairs)),
        GtsfmMetric("num_outlier_1dsfm_measurements", len(outlier_i1_i2_pairs)),
        GtsfmMetric("1dsfm_precision_" + threshold_suffix, precision),
        GtsfmMetric("1dsfm_recall_" + threshold_suffix, recall),
        GtsfmMetric("num_translations_estimated", len([wti for wti in wti_list if wti is not None])),
        GtsfmMetric("1dsfm_inlier_angular_errors_deg", inlier_angular_errors),
        GtsfmMetric("1dsfm_outlier_angular_errors_deg", outlier_angular_errors),
        metrics_utils.compute_translation_angle_metric(measured_gt_i2Ui1_dict, wTi_aligned_list),
        metrics_utils.compute_translation_distance_metric(wti_aligned_list, gt_wti_list),
    ]

    return GtsfmMetricsGroup("translation_averaging_metrics", ta_metrics)


def cast_to_measurements_variable_in_global_coordinate_frame(
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]], wRi_list: List[Optional[Rot3]], noise_model: gtsam.noiseModel
) -> BinaryMeasurementsUnit3:

    w_i2Ui1_measurements = BinaryMeasurementsUnit3()
    for (i1, i2), i2Ui1 in i2Ui1_dict.items():
        wRi2 = wRi_list[i2]
        if i2Ui1 is not None and wRi2 is not None:
            # TODO: what if wRi2 is None, but wRi1 is not? Can we still transform.
            w_i2Ui1_measurements.append(BinaryMeasurementUnit3(i2, i1, Unit3(wRi2.rotate(i2Ui1.point3())), noise_model))

    return w_i2Ui1_measurements
