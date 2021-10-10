"""Translation averaging using 1DSFM.

This algorithm was proposed in 'Robust Global Translations with 1DSFM' and is implemented by wrapping GTSAM's classes.

References:
- https://research.cs.cornell.edu/1dsfm/
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/MFAS.h
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/TranslationRecovery.h
- https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/TranslationAveragingExample.py

Authors: Jing Wu, Ayush Baid, Akshay Krishnan
"""
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np
from gtsam import MFAS, BinaryMeasurementsUnit3, BinaryMeasurementUnit3, Point3, Pose3, Rot3, TranslationRecovery, Unit3

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

# Hyperparameters for 1D-SFM
# maximum number of times 1dsfm will project the Unit3's to a 1d subspace for outlier rejection
MAX_PROJECTION_DIRECTIONS = 50
OUTLIER_WEIGHT_THRESHOLD = 0.1

NOISE_MODEL_DIMENSION = 3  # chordal distances on Unit3
NOISE_MODEL_SIGMA = 0.01
HUBER_LOSS_K = 1.345  # default value from GTSAM

MAX_INLIER_MEASUREMENT_ERROR_DEG = 5.0


class TranslationAveraging1DSFM(TranslationAveragingBase):
    """1D-SFM translation averaging with outlier rejection."""

    def __init__(self, robust_measurement_noise: bool = True) -> None:
        """Initializes the 1DSFM averaging instance.

        Args:
            robust_measurement_noise: Whether to use a robust noise model for the measurements, defaults to true.
        """
        super().__init__(robust_measurement_noise)

        self._max_1dsfm_projection_directions = MAX_PROJECTION_DIRECTIONS
        self._outlier_weight_threshold = OUTLIER_WEIGHT_THRESHOLD

    def run(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        scale_factor: float = 1.0,
        gt_wTi_list: Optional[List[Optional[Pose3]]] = None,
    ) -> Tuple[List[Optional[Point3]], Optional[GtsfmMetricsGroup]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i2Ui1_dict: relative unit-translation as dictionary (i1, i2): i2Ui1
            wRi_list: global rotations for each camera pose in the world coordinates.
            scale_factor: non-negative global scaling factor.
            gt_wTi_list: ground truth poses for computing metrics.

        Returns:
            Global translation wti for each camera pose. The number of entries in the list is `num_images`. The list
                may contain `None` where the global translations could not be computed (either underconstrained system
                or ill-constrained system).
            A GtsfmMetricsGroup of 1DSfM metrics.
        """
        noise_model = gtsam.noiseModel.Isotropic.Sigma(NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)
        if self._robust_measurement_noise:
            huber_loss = gtsam.noiseModel.mEstimator.Huber.Create(HUBER_LOSS_K)
            noise_model = gtsam.noiseModel.Robust.Create(huber_loss, noise_model)

        # Note: all measurements are relative translation directions in the
        # world frame.

        # convert translation direction in global frame using rotations.
        w_i2Ui1_measurements = BinaryMeasurementsUnit3()
        for (i1, i2), i2Ui1 in i2Ui1_dict.items():
            if i2Ui1 is not None and wRi_list[i2] is not None:
                w_i2Ui1_measurements.append(
                    BinaryMeasurementUnit3(i2, i1, Unit3(wRi_list[i2].rotate(i2Ui1.point3())), noise_model)
                )

        # sample indices to be used as projection directions
        num_valid_measurements = len(w_i2Ui1_measurements)
        indices = np.random.choice(
            num_valid_measurements,
            min(self._max_1dsfm_projection_directions, num_valid_measurements),
            replace=False,
        )

        projection_directions = [w_i2Ui1_measurements[idx].measured() for idx in indices]

        # compute outlier weights using MFAS
        outlier_weights = []

        # TODO(ayush): parallelize this step.
        for direction in projection_directions:
            algorithm = MFAS(w_i2Ui1_measurements, direction)
            outlier_weights.append(algorithm.computeOutlierWeights())

        # compute average outlier weight
        avg_outlier_weights = {}
        for outlier_weight_dict in outlier_weights:
            for index_pair, weight in outlier_weight_dict.items():
                if index_pair in avg_outlier_weights:
                    avg_outlier_weights[index_pair] += weight / len(outlier_weights)
                else:
                    avg_outlier_weights[index_pair] = weight / len(outlier_weights)

        # filter out outlier measurements
        w_i2Ui1_inlier_measurements = BinaryMeasurementsUnit3()
        inliers = []
        outliers = []
        for w_i2Ui1 in w_i2Ui1_measurements:
            # key1 is i2 and key2 is i1 above.
            i1 = w_i2Ui1.key2()
            i2 = w_i2Ui1.key1()
            if avg_outlier_weights[(i2, i1)] < self._outlier_weight_threshold:
                w_i2Ui1_inlier_measurements.append(w_i2Ui1)
                inliers.append((i1, i2))
            else:
                outliers.append((i1, i2))

        # Run the optimizer
        wti_values = TranslationRecovery(w_i2Ui1_inlier_measurements).run(scale_factor)

        # transforming the result to the list of Point3
        wti_list = [None] * num_images
        for i in range(num_images):
            if wRi_list[i] is not None and wti_values.exists(i):
                wti_list[i] = wti_values.atPoint3(i)

        # Compute the metrics.
        if gt_wTi_list is not None:
            ta_metrics = _compute_metrics(inliers, outliers, i2Ui1_dict, wRi_list, wti_list, gt_wTi_list)
        else:
            ta_metrics = None

        return wti_list, ta_metrics


def _get_measurement_angle_errors(
    i1_i2_pairs: Tuple[int, int],
    i2Ui1_measurements: Dict[Tuple[int, int], Unit3],
    gt_i2Ui1_measurements: Dict[Tuple[int, int], Unit3],
) -> List[float]:
    """Returns a list of the angle between i2Ui1_measurements and gt_i2Ui1_measurements for every
    (i1, i2) in i1_i2_pairs.

    Args:
        i1_i2_pairs: List of (i1, i2) tuples for which the angles must be computed.
        i2Ui1_measurements: Measured translation direction of i1 WRT i2.
        gt_i2Ui1_measurements: Ground truth translation direction of i1 WRT i2.

    Returns:
        List of angles between the measured and ground truth translation directions.
    """
    errors = []
    for (i1, i2) in i1_i2_pairs:
        if (i1, i2) in i2Ui1_measurements and (i1, i2) in gt_i2Ui1_measurements:
            errors.append(
                comp_utils.compute_relative_unit_translation_angle(
                    i2Ui1_measurements[(i1, i2)], gt_i2Ui1_measurements[(i1, i2)]
                )
            )
    return errors


def _compute_metrics(
    inlier_i1_i2_pairs: List[Tuple[int, int]],
    outlier_i1_i2_pairs: List[Tuple[int, int]],
    i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
    wRi_list: List[Optional[Rot3]],
    wti_list: List[Optional[Point3]],
    gt_wTi_list: List[Optional[Pose3]],
) -> GtsfmMetricsGroup:
    """Computes the translation averaging metrics as a metrics group.

    Args:
        inlier_i1_i2_pairs: List of inlier camera pair indices.
        outlier_i1_i2_pairs: List of outlier camera pair indices.
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

    # Angle between i2Ui1 measurement and GT i2Ui1 measurement for inliers and outliers.
    inlier_angular_errors = _get_measurement_angle_errors(inlier_i1_i2_pairs, i2Ui1_dict, gt_i2Ui1_dict)
    outlier_angular_errors = _get_measurement_angle_errors(outlier_i1_i2_pairs, i2Ui1_dict, gt_i2Ui1_dict)
    precision, recall = metrics_utils.get_precision_recall_from_errors(
        inlier_angular_errors, outlier_angular_errors, MAX_INLIER_MEASUREMENT_ERROR_DEG
    )

    measured_gt_i2Ui1_dict = {}
    for (i1, i2) in inlier_i1_i2_pairs + outlier_i1_i2_pairs:
        measured_gt_i2Ui1_dict[(i1, i2)] = gt_i2Ui1_dict[(i1, i2)]

    # Compute estimated poses after the averaging step and align them to ground truth.
    wTi_list = []
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
