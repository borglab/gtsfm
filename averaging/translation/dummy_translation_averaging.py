"""A dummy translation averaging class which is used for testing.

Authors: Ayush Baid
"""

from typing import Dict, List, Tuple, Union

import numpy as np
from gtsam import Rot3, Unit3

from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase


class DummyTranslationAveraging(TranslationAveragingBase):
    """Assigns random unit-translations to each pose."""

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

        # create the random seed using relative rotations
        seed_rotation = Rot3()
        for rotation in wRi_list:
            seed_rotation = seed_rotation.compose(rotation)

        np.random.seed(
            int(1000*np.sum(seed_rotation.xyz(), axis=None) % (2 ^ 32))
        )

        # generate dummy rotations
        results = []
        for _ in range(num_poses):
            random_vector = np.random.rand(3)
            results.append(Unit3(random_vector))

        return results
