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
            num_images: int,
            i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]]
            ) -> List[Optional[Rot3]]:
        """Run the rotation averaging.

        Args:
            num_images: number of poses.
            i2Ri1_dict: relative rotations as dictionaries where keys (i2, i1)
                        are pose pairs.

        Returns:
            Global rotations for each camera pose, i.e. w_R_i, as a list. The
                number of entries in the list is `num_images`. The list may
                contain `None` where the global rotation could not be computed
                (either underconstrained system or ill-constrained system).
        """

    def create_computation_graph(
            self,
            num_images: int,
            i2Ri1_graph: Delayed
    ) -> Delayed:
        """Create the computation graph for performing rotation averaging.

        Args:
            num_images: number of poses.
            i2Ri1_graph: dictionary of relative rotations as a delayed task.

        Returns:
            global rotations wrapped using dask.delayed.
        """

        return dask.delayed(self.run)(num_images, i2Ri1_graph)
