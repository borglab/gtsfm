"""Base class for the rotation averaging component of the pipeline.

Authors: Jing Wu, Ayush Baid
"""
import abc
from typing import Dict, List, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Rot3


class RotationAveragingBase(metaclass=abc.ABCMeta):
    """Base class for rotation averaging.

    This class generates global rotation estimates from the pairwise relative
    rotations.
    """
    # ignored-abstractmethod
    @abc.abstractmethod
    def run(self,
            num_poses: int,
            relative_rotations: Dict[Tuple[int, int], Rot3]
            ) -> List[Rot3]:
        """Run the rotation averaging.

        Args:
            num_poses (int): number of poses.
            relative_rotations (Dict[Tuple[int, int], Rot3]): relative
                                                              rotations between
                                                              camera poses.

        Returns:
            List[Rot3]: global rotations for each camera pose.
        """

    def create_computation_graph(
            self,
            num_poses: int,
            relative_rotations: Delayed
    ) -> Delayed:
        """Create the computation graph for performing rotation averaging.

        Args:
            num_poses (int): number of poses.
            relative_rotations (Delayed): the dictionary of relative rotations
                                          wrapped up in Delayed.

        Returns:
            Delayed: global rotations wrapped using dask.delayed.
        """

        return dask.delayed(self.run)(num_poses,
                                      relative_rotations)
