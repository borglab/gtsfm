"""Base class for the rotation averaging component of the GTSFM pipeline.

Authors: Jing Wu, Ayush Baid
"""
import abc
from typing import Dict, List, Optional, Tuple

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
            i1_R_i2_dict: Dict[Tuple[int, int], Optional[Rot3]]
            ) -> List[Optional[Rot3]]:
        """Run the rotation averaging.

        Args:
            num_poses: number of poses.
            i1_R_i2_dict: relative rotations between pairs of camera poses (
                          rotation of i2^th pose in i1^th frame for various
                          pairs of (i1, i2). The pairs serve as keys of the
                          dictionary).

        Returns:
            Global rotations for each camera pose, i.e. w_R_i
        """

    def create_computation_graph(
            self,
            num_poses: int,
            i1_R_i2_dict: Delayed
    ) -> Delayed:
        """Create the computation graph for performing rotation averaging.

        Args:
            num_poses: number of poses.
            i1_R_i2_dict: the dictionary of relative rotations wrapped up in
                          Delayed.

        Returns:
            Delayed: global rotations wrapped using dask.delayed.
        """

        return dask.delayed(self.run)(num_poses, i1_R_i2_dict)
