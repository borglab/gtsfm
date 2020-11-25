"""Translation averaging using 1DSFM.

This algorithm was proposed in 'Robust Global Translations with 1DSFM' and is
build by wrapping GTSAM's classes.

References:
- https://research.cs.cornell.edu/1dsfm/
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/MFAS.h
- https://github.com/borglab/gtsam/blob/develop/gtsam/sfm/TranslationRecovery.h
- https://github.com/borglab/gtsam/blob/develop/python/gtsam/examples/TranslationAveragingExample.py

Authors: Jing Wu, Ayush Baid.
"""
from typing import Dict, List, Optional, Tuple

import gtsam
import numpy as np
from gtsam import (MFAS, BinaryMeasurementsUnit3, BinaryMeasurementUnit3,
                   Point3, Rot3, TranslationRecovery, Unit3)

from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase

# hyperparamters for 1D-SFM
MAX_PROJECTION_DISTANCE = 50
OUTLIER_WEIGHT_THRESHOLD = 0.1

NOISE_MODEL_DIMENSION = 3  # chordal distances on Unit3
NOISE_MODEL_SIGMA = 0.01


class TranslationAveraging1DSFM(TranslationAveragingBase):
    """1D-SFM translation averaging with outlier rejection."""

    def __init__(self) -> None:
        super().__init__()

        self._max_1dsfm_projection_direction = MAX_PROJECTION_DISTANCE
        self._outlier_weight_threshold = OUTLIER_WEIGHT_THRESHOLD

    def run(self,
            num_images: int,
            i1Ti2_dict: Dict[Tuple[int, int], Optional[Unit3]],
            wRi_list: List[Optional[Rot3]],
            scale_factor: float = 1.0
            ) -> List[Optional[Point3]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i1Ti2_dict: relative unit translations between pairs of camera
                        poses (direction of translation of i2^th pose in
                        i1^th frame for various pairs of (i1, i2). The pairs
                        serve as keys of the dictionary).
            wRi_list: global rotations for each camera pose in the world
                      coordinates.
            scale_factor: non-negative global scaling factor.

        Returns:
            global translation for each camera pose.
        """

        noise_model = gtsam.noiseModel.Isotropic.Sigma(
            NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)

        # Note: all measurements are relative translation directions in the
        # world frame.

        # convert translation direction in global frame using rotations.
        w_i1Ui2_measurements = BinaryMeasurementsUnit3()
        for (i1, i2), i1Ti2 in i1Ti2_dict.items():
            if i1Ti2 is not None and wRi_list[i1] is not None:
                w_i1Ui2_measurements.append(BinaryMeasurementUnit3(
                    i1,
                    i2,
                    Unit3(wRi_list[i1].rotate(i1Ti2.point3())),
                    noise_model))

        # sample indices to be used as projection directions
        num_measurements = len(i1Ti2_dict)
        indices = np.random.choice(
            num_measurements,
            min(self._max_1dsfm_projection_direction, num_measurements),
            replace=False)

        projection_directions = [
            w_i1Ui2_measurements[idx].measured() for idx in indices]

        # compute outlier weights using MFAS
        outlier_weights = []

        # TODO(ayush): parallelize this step.
        for direction in projection_directions:
            algorithm = MFAS(w_i1Ui2_measurements, direction)
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
        w_i1Ui2_inlier_measurements = BinaryMeasurementsUnit3()
        for w_i1Ui2 in w_i1Ui2_measurements:
            if avg_outlier_weights[(w_i1Ui2.key1(), w_i1Ui2.key2())] < \
                    self._outlier_weight_threshold:
                w_i1Ui2_inlier_measurements.append(w_i1Ui2)

        # Run the optimizer
        wTi_values = TranslationRecovery(
            w_i1Ui2_inlier_measurements).run(scale_factor)

        # transforming the result to the list of Point3
        wTi_list = [None]*num_images
        for i in range(num_images):
            if wRi_list[i] is not None:
                wTi_list[i] = wTi_values.atPoint3(i)

        return wTi_list
