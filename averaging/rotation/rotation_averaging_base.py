"""
Base class for rotation averaging module.

A rotation averaging algorithm processes relative rotation between poses as
inputs and estimates global rotation for all poses.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import gtsam


class RotationAveragingBase(metaclass=abc.ABCMeta):
    """Base class for all rotation averaging algorithms."""

    @abc.abstractmethod
    def run(self, relative_rotations: Dict[Tuple[int, int], gtsam.Rot3]) -> List[gtsam.Rot3]:
        """
        Run the rotation averaging to generate global rotation for all the
        poses.

        Args: 
            relative_rotations (Dict[Tuple[int, int], gtsam.Rot3]): pairwise
                relative rotation between poses. The dictionary contains the pairs
                of pose indices as keys and the relative rotation as values.

        Returns: 
            List[gtsam.Rot3]: computed global rotation for every pose.
        """
        # TODO: we need an anchor?
