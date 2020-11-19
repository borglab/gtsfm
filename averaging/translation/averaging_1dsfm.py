"""Translation averaging using 1D-SFM.

Authors: Akshay Krishnan, Jing Wu, Ayush Baid.
"""
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np
from gtsam import MFAS, BinaryMeasurementUnit3, Point3, Rot3, Unit3

from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase

# hyperparamters for 1D-SFM
MAX_PROJECTION_DISTANCE = 50
OUTLIER_WEIGHT_THRESHOLD = 0.1

NOISE_MODEL_DIMENSION = 3  # should be fixed
NOISE_MODEL_SIGMA = 0.01


class TranslationAveraging1DSFM(TranslationAveragingBase):
    """1D-SFM translation averaging with outlier rejection."""

    def __init__(self) -> None:
        super().__init__()

        self._max_1dsfm_projection_direction = MAX_PROJECTION_DISTANCE
        self._outlier_weight_threshold = OUTLIER_WEIGHT_THRESHOLD

    def run(self,
            num_images: int,
            i1_t_i2_dict: Dict[Tuple[int, int], Optional[Unit3]],
            w_R_i_list: List[Optional[Rot3]],
            global_gauge_ambiguity: float = 1.0
            ) -> List[Optional[Point3]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i1_t_i2_dict: relative unit translations between pairs of camera
                          poses (direction of translation of i2^th pose in
                          i1^th frame for various pairs of (i1, i2). The pairs
                          serve as keys of the dictionary).
            w_R_i_list: global rotations for each camera pose in the world
                        coordinates.
            global_gauge_ambiguity: non-negative global scaling factor.

        Returns:
            global translation for each camera pose.
        """

        noise_model = gtsam.noiseModel.Isotropic.Sigma(
            NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)

        # convert translation direction in global frame using rotations.
        translation_measurements = gtsam.BinaryMeasurementsUnit3()
        for (i1, i2), translation_direction in i1_t_i2_dict.items():
            if translation_direction is not None:
                translation_measurements.append(BinaryMeasurementUnit3(
                    i1,
                    i2,
                    Unit3(w_R_i_list[i1].rotate(
                        translation_direction.point3())),
                    noise_model))

        # sample indices to be used as projection directions
        num_measurements = len(i1_t_i2_dict)
        indices = np.random.choice(
            num_measurements,
            min(self._max_1dsfm_projection_direction, num_measurements),
            replace=False)

        projection_directions = [
            translation_measurements[idx].measured() for idx in indices]

        # compute outlier weights using MFAS
        outlier_weights = []
        for direction in projection_directions:
            algorithm = MFAS(translation_measurements, direction)
            outlier_weights.append(algorithm.computeOutlierWeights())

        # compute average outlier weight
        avg_outlier_weights = {}
        for outlier_weight_dict in outlier_weights:
            for k, v in outlier_weight_dict.items():
                if k in avg_outlier_weights:
                    avg_outlier_weights[k] += v/len(outlier_weights)
                else:
                    avg_outlier_weights[k] = v/len(outlier_weights)

        # filter out oulier measumenets
        inlier_translation_measurements = gtsam.BinaryMeasurementsUnit3()
        for i in translation_measurements:
            if avg_outlier_weights[(i.key1(), i.key2())] < \
                    self._outlier_weight_threshold:
                inlier_translation_measurements.append(i)

        # Run the optimizer
        global_translations = gtsam.TranslationRecovery(
            inlier_translation_measurements).run(global_gauge_ambiguity)

        # transforming the result to the list of Point3
        results = [None]*num_images
        for i in range(num_images):
            if w_R_i_list[i] is not None:
                results[i] = global_translations.atPoint3(i)

        return results
