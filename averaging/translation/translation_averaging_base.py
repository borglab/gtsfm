"""Base class for the translation averaging component of the pipeline.

Authors: Ayush Baid
"""
import abc
from typing import Dict, List, Tuple, Union

import dask
from dask.delayed import Delayed
from gtsam import Rot3, Unit3


class TranslationAveragingBase(metaclass=abc.ABCMeta):
    """Base class for translation averaging.

    This class generates global unit translation estimates from
    pairwise relative unit translation and global rotations.
    """
    # ignored-abstractmethod
    @abc.abstractmethod
    def run(self,
            num_poses: int,
            i1ti2_dict: Dict[Tuple[int, int], Union[Unit3, None]],
            wRi_list: List[Rot3]
            ) -> List[Unit3]:
        """Run the translation averaging.

        Args:
            num_poses: number of poses.
            i1ti2_dict: relative unit translation between camera poses (from i1 
                        to i2).
            wRi_list: global rotations.
        Returns:
            List[Unit3]: global unit translation for each camera pose.
        """

    def create_computation_graph(
            self,
            num_poses: int,
            i1ti2_graph: Dict[Tuple[int, int], Delayed],
            wRi_graph: Delayed
    ) -> Delayed:
        """Create the computation graph for performing translation averaging.

        Args:
            num_poses: number of poses.
            i1ti2_graph: dictionary of relative unit translations wrapped 
                         up in Delayed.
            wRi_graph: list of global rotations wrapped up in Delayed.

        Returns:
            Delayed: global unit translations wrapped using dask.delayed.
        """

        return dask.delayed(self.run)(num_poses, i1ti2_graph, wRi_graph)
