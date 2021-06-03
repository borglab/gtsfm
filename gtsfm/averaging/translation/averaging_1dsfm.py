"""Translation averaging using 1DSFM.

This algorithm was proposed in 'Robust Global Translations with 1DSFM' and is implemented by wrapping GTSAM's classes.

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
from gtsam import (
    MFAS,
    BinaryMeasurementsUnit3,
    BinaryMeasurementUnit3,
    Point3,
    Rot3,
    TranslationRecovery,
    Unit3,
)

from gtsfm.averaging.translation.translation_averaging_base import (
    TranslationAveragingBase,
)

# Hyperparameters for 1D-SFM
# maximum number of times 1dsfm will project the Unit3's to a 1d subspace for outlier rejection
MAX_PROJECTION_DIRECTIONS = 50
OUTLIER_WEIGHT_THRESHOLD = 0.1

NOISE_MODEL_DIMENSION = 3  # chordal distances on Unit3
NOISE_MODEL_SIGMA = 0.01


class TranslationAveraging1DSFM(TranslationAveragingBase):
    """1D-SFM translation averaging with outlier rejection."""

    def __init__(self) -> None:
        super().__init__()

        self._max_1dsfm_projection_directions = MAX_PROJECTION_DIRECTIONS
        self._outlier_weight_threshold = OUTLIER_WEIGHT_THRESHOLD

    def run(
        self,
        num_images: int,
        i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
        wRi_list: List[Optional[Rot3]],
        scale_factor: float = 1.0,
    ) -> List[Optional[Point3]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i2Ui1_dict: relative unit-translation as dictionary (i1, i2): i2Ui1
            wRi_list: global rotations for each camera pose in the world coordinates.
            scale_factor: non-negative global scaling factor.

        Returns:
            Global translation wti for each camera pose. The number of entries in the list is `num_images`. The list
                may contain `None` where the global translations could not be computed (either underconstrained system
                or ill-constrained system).
        """
        noise_model = gtsam.noiseModel.Isotropic.Sigma(NOISE_MODEL_DIMENSION, NOISE_MODEL_SIGMA)

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
        # note: if some unit translations are None, then we can only use valid values
        num_measurements = len(w_i2Ui1_measurements)
        indices = np.random.choice(
            num_measurements,
            size=min(self._max_1dsfm_projection_directions, num_measurements),
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
        for w_i2Ui1 in w_i2Ui1_measurements:
            if avg_outlier_weights[(w_i2Ui1.key1(), w_i2Ui1.key2())] < self._outlier_weight_threshold:
                w_i2Ui1_inlier_measurements.append(w_i2Ui1)

        # Run the optimizer
        wti_values = TranslationRecovery(w_i2Ui1_inlier_measurements).run(scale_factor)

        # transforming the result to the list of Point3
        wti_list = [None] * num_images
        for i in range(num_images):
            if wRi_list[i] is not None and wti_values.exists(i):
                wti_list[i] = wti_values.atPoint3(i)
        return wti_list
