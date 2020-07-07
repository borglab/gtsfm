"""
Base class for translation averaging.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import gtsam


class TranslationAveragingBase(metaclass=abc.ABCMeta):
    def run(self, relative_translations: Dict[Tuple[int, int], gtsam.Point3], global_rotations: List[gtsam.Rot3]) -> List[gtsam.Point3]:
        """
        Runs the translation averaging to generate global translations for all camera poses.

        Args:
            relative_translations (Dict[Tuple[int, int], gtsam.Point3]): [description]
            global_rotations (List[gtsam.Rot3]): [description]

        Returns:
            List[gtsam.Point3]: [description]
        """
