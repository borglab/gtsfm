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
            scale_factor: float = 1.0
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
            scale_factor: non-negative global scaling factor.

        Returns:
            global translation for each camera pose.
        """

        noise_model = gtsam.noiseModel.Isotropic.Sigma(
            NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)

        # convert translation direction in global frame using rotations.
        i1_Z_i2_list = gtsam.BinaryMeasurementsUnit3()
        for (i1, i2), i1_t_i2 in i1_t_i2_dict.items():
            if i1_t_i2 is not None:
                i1_Z_i2_list.append(BinaryMeasurementUnit3(
                    i1,
                    i2,
                    Unit3(w_R_i_list[i1].rotate(
                        i1_t_i2.point3())),
                    noise_model))

        # sample indices to be used as projection directions
        num_measurements = len(i1_t_i2_dict)
        indices = np.random.choice(
            num_measurements,
            min(self._max_1dsfm_projection_direction, num_measurements),
            replace=False)

        projection_directions = [
            i1_Z_i2_list[idx].measured() for idx in indices]

        # compute outlier weights using MFAS
        outlier_weights = []
        for direction in projection_directions:
            algorithm = MFAS(i1_Z_i2_list, direction)
            outlier_weights.append(algorithm.computeOutlierWeights())

        # compute average outlier weight
        avg_outlier_weights = {}
        for outlier_weight_dict in outlier_weights:
            for index_pair, weight in outlier_weight_dict.items():
                if index_pair in avg_outlier_weights:
                    avg_outlier_weights[index_pair] += weight / \
                        len(outlier_weights)
                else:
                    avg_outlier_weights[index_pair] = weight / \
                        len(outlier_weights)

        # filter out oulier measumenets
        inlier_i1_Z_i2_list = gtsam.BinaryMeasurementsUnit3()
        for measurement in i1_Z_i2_list:
            if avg_outlier_weights[(measurement.key1(), measurement.key2())] < \
                    self._outlier_weight_threshold:
                inlier_i1_Z_i2_list.append(measurement)

        # Run the optimizer
        w_t_i_values = gtsam.TranslationRecovery(
            inlier_i1_Z_i2_list).run(scale_factor)

        # transforming the result to the list of Point3
        results = [None]*num_images
        for i in range(num_images):
            if w_R_i_list[i] is not None:
                results[i] = w_t_i_values.atPoint3(i)

        return results
