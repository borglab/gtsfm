"""A dummy translation averaging class which is used for testing.

Authors: Ayush Baid
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
from gtsam import Point3, Rot3, Unit3

from gtsfm.averaging.translation.translation_averaging_base import (
    TranslationAveragingBase,
)


class DummyTranslationAveraging(TranslationAveragingBase):
    """Assigns random unit-translations to each pose."""

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
            i2Ui1_dict: relative unit-trans as dictionary (i1, i2): i2Ui1.
            wRi_list: global rotations for each camera pose in the world coordinates.
            scale_factor: non-negative global scaling factor.

        Returns:
            Global translation wti for each camera pose. The number of entries in the list is `num_images`. The list
                may contain `None` where the global translations could not be computed (either underconstrained system
                or ill-constrained system).
        """

        if len(wRi_list) == 0:
            return []

        # create the random seed using relative rotations
        seed_rotation = wRi_list[0]

        np.random.seed(int(1000 * seed_rotation.xyz()[0]) % (2 ^ 32))

        # generate dummy output
        results = [None] * num_images
        for idx in range(num_images):
            if wRi_list[idx] is not None:
                random_vector = np.random.rand(3)
                results[idx] = scale_factor * Unit3(random_vector).point3()

        return results
