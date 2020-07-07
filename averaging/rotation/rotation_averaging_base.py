"""
Base class for rotation averaging

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import gtsam


class RotationAvergagingBase(metaclass=abc.ABCMeta):

    @abc.abstractmethod
    def run(self, relative_rotations: Dict[Tuple[int, int], gtsam.Rot3]) -> List[gtsam.Rot3]:
        """
        Run the rotation averaging to generate global rotation for all the poses.

        Args:
            relative_rotations (Dict[Tuple[int, int], gtsam.Rot3]): [description]

        Returns:
            List[gtsam.Rot3]: [description]
        """
