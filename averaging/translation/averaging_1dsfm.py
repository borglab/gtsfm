"""Translation averaging using 1D-SFM.

Authors: Akshay Krishnan, Jing Wu, Ayush Baid.
"""
from typing import Dict, List, Tuple, Union

import gtsam
import numpy as np
from gtsam import BinaryMeasurementUnit3, MFAS, Rot3, Unit3

from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase


class TranslationAveraging1DSFM(TranslationAveragingBase):
    """1D-SFM translation averaging with outlier rejection."""

    def __init__(self) -> None:
        super().__init__()

        self._max_1dsfm_projection_direction = 50
        self._outlier_weight_threshold = 0.1

    def run(self,
            num_poses: int,
            i1ti2_dict: Dict[Tuple[int, int], Union[Unit3, None]],
            wRi_list: List[Rot3]
            ) -> List[Unit3]:
        """Run the translation averaging.

        Args:
            num_poses: number of poses.
            i1ti2_dict: relative unit translation between camera poses (
                        translation direction of i2^th pose in i1^th frame).
            wRi_list: global rotations.
        Returns:
            List[Unit3]: global unit translation for each camera pose.
        """

        noise_model = gtsam.noiseModel.Isotropic.Sigma(3, 0.01)

        # Convert translation direction in global frame by using rotations
        translation_measurements = gtsam.BinaryMeasurementsUnit3()
        for pose_index_i1i2, translation_direction in i1ti2_dict.items():
            if translation_direction is not None:
                translation_measurements.append(BinaryMeasurementUnit3(
                    pose_index_i1i2[0],
                    pose_index_i1i2[1],
                    Unit3(
                        wRi_list[pose_index_i1i2[0]].rotate(
                            translation_direction.point3())),
                    noise_model))

        # Indices of relative translations to be used as projection directions
        num_measurements = len(i1ti2_dict)
        indices = np.random.choice(num_measurements,
                                   min(self._max_1dsfm_projection_direction,
                                       num_measurements),
                                   replace=False)
        projection_directions = [
            translation_measurements[idx].measured() for idx in indices]

        # Compute outlier weights and calculate the average
        outlier_weights = []
        for direction in projection_directions:
            algorithm = MFAS(translation_measurements, direction)
            outlier_weights.append(algorithm.computeOutlierWeights())
        avg_outlier_weights = {}
        for outlier_weight_dict in outlier_weights:
            for k, v in outlier_weight_dict.items():
                if k in avg_outlier_weights:
                    avg_outlier_weights[k] += v/len(outlier_weights)
                else:
                    avg_outlier_weights[k] = v/len(outlier_weights)

        inlier_translation_measurements = gtsam.BinaryMeasurementsUnit3()
        for i in translation_measurements:
            if avg_outlier_weights[(i.key1(), i.key2())] < \
                    self._outlier_weight_threshold:
                inlier_translation_measurements.append(i)

        # Run the optimizer
        global_translations = gtsam.TranslationRecovery(
            inlier_translation_measurements).run()

        # transforming the result to the list of Unit3
        results = [None]*num_poses
        for i in range(num_poses):
            if wRi_list[i] is not None:
                results[i] = Unit3(global_translations.atPoint3(i))

        return results
