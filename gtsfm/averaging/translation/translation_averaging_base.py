"""Base class for the translation averaging component of the GTSFM pipeline.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Optional, Tuple

import dask
from dask.delayed import Delayed
from gtsam import Point3, Rot3, Unit3


class TranslationAveragingBase(metaclass=abc.ABCMeta):
    """Base class for translation averaging.

    This class generates global unit translation estimates from
    pairwise relative unit translation and global rotations.
    """
    # ignored-abstractmethod
    @abc.abstractmethod
    def run(self,
            num_images: int,
            i2Ui1_dict: Dict[Tuple[int, int], Optional[Unit3]],
            wRi_list: List[Optional[Rot3]],
            scale_factor: float = 1.0
            ) -> List[Optional[Point3]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i2Ui1_dict: relative unit-trans as dictionary (i1, i2): i2Ui1.
            wRi_list: global rotations for each camera pose in the world
                      coordinates.
            scale_factor: non-negative global scaling factor.

        Returns:
            Global translation wti for each camera pose. The number of entries
                in the list is `num_images`. The list may contain `None` where
                the global translations could not be computed (either
                underconstrained system or ill-constrained system).
        """

    def create_computation_graph(self,
                                 num_images: int,
                                 i2Ui1_graph: Delayed,
                                 wRi_graph: Delayed,
                                 scale_factor: float = 1.0) -> Delayed:
        """Create the computation graph for performing translation averaging.

        Args:
            num_images: number of camera poses.
            i2Ui1_graph: dictionary of relative unit translations as a delayed
                         task.
            wRi_graph: list of global rotations wrapped up in Delayed.
            scale_factor: non-negative global scaling factor.

        Returns:
            Delayed: global unit translations wrapped using dask.delayed.
        """

        return dask.delayed(self.run)(
            num_images, i2Ui1_graph, wRi_graph, scale_factor)
