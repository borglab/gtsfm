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
            i1_t_i2_dict: Dict[Tuple[int, int], Optional[Unit3]],
            w_R_i_list: List[Optional[Rot3]],
            scale_factor: float = 1.0
            ) -> List[Optional[Point3]]:
        """Run the translation averaging.

        Args:
            num_images: number of camera poses.
            i1_t_i2_dict: relative unit translations between pairs of camera
                          poses (direction of translation of i2^th pose in
                          i1^th frame for various pairs of (i1, i2). The pairs
                          serve as keys of the dictionary).
            w_R_i_list: global rotations for each camera pose in the world
                        coordinates.
            scale_factor: non-negative global scaling factor.

        Returns:
            global translation for each camera pose.
        """

    def create_computation_graph(
            self,
            num_images: int,
            i1_t_i2_graph: Dict[Tuple[int, int], Delayed],
            w_R_i_graph: Delayed,
            scale_factor: float = 1.0
    ) -> Delayed:
        """Create the computation graph for performing translation averaging.

        Args:
            num_images: number of camera poses.
            i1_t_i2_graph: graph of relative unit-translations, stored as a    
                           dict.
            w_R_i_graph: list of global rotations wrapped up in Delayed.
            scale_factor: non-negative global scaling factor.

        Returns:
            Delayed: global unit translations wrapped using dask.delayed.
        """

        return dask.delayed(self.run)(
            num_images, i1_t_i2_graph, w_R_i_graph, scale_factor)
