"""
Base class for rotation averaging module.

A rotation averaging algorithm processes relative rotation between poses as
inputs and estimates global rotation for all poses.

Authors: Ayush Baid, John Lambert
"""
import abc
from typing import Dict, List, Tuple

import gtsam


class RotationAveragingBase(metaclass=abc.ABCMeta):
    """Base class for all rotation averaging algorithms."""

    @abc.abstractmethod
    def run(
        self,
        relative_rotations: Dict[Tuple[int, int], gtsam.Rot3]
        ) -> List[gtsam.Rot3]:
        """
        Run the rotation averaging to generate global rotation for all the
        poses.
        Based off of
        https://github.com/borglab/gtsam/blob/develop/cython/gtsam/examples/Pose3SLAMExample_initializePose3Chordal.py

        Args: 
            relative_rotations (Dict[Tuple[int, int], gtsam.Rot3]): pairwise
                relative rotation between poses. The dictionary contains the pairs
                of pose indices as keys and the relative rotation as values.

        Returns: 
            List[gtsam.Rot3]: computed global rotation for every pose.
                image index is the list index.
        """
        # TODO: we need an anchor?

        # is3D = True
        # graph, initial = gtsam.readG2o(g2oFile, is3D)

        graph = None
        initial = None

        # Add prior on the first key. TODO: assumes first key ios z
        priorModel = gtsam.noiseModel_Diagonal.Variances(
            np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4]))
        firstKey = initial.keys().at(0)
        graph.add(gtsam.PriorFactorPose3(0, gtsam.Pose3(), priorModel))

        # Initializing Pose3 - chordal relaxation"
        initialization = gtsam.InitializePose3.initialize(graph)

        print(initialization)



