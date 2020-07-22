"""
Base class for translation averaging module.

A translation averaging algorithm processes estimates global translation between
poses using relative translation and global rotations. We essentially have the
global pose for every input.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import gtsam


class TranslationAveragingBase(metaclass=abc.ABCMeta):
    """Base class for translation averaging algorithm."""

    def run(self,
            relative_translations: Dict[Tuple[int, int], gtsam.Point3],
            global_rotations: List[gtsam.Rot3]) -> List[gtsam.Pose3]:
        """
        Runs the translation averaging to generate global translations for all camera poses.

        Args:
            relative_translations (Dict[Tuple[int, int], gtsam.Point3]): pairwise
                relative rotation between poses. The dictionary contains the pairs
                of pose indices as keys and the relative rotation as values.
            global_rotations (List[gtsam.Rot3]): global rotation for every pose.
                list index corresponds to image index.

        Returns:
            List[gtsam.Pose3]: computed global pose.
        """
        # TODO: how to take in the anchor?
        # TODO: some algorithms take feature points as input.
