"""
Base class for bundle adjustment module.

Accepts intrinsics, point-correspondence constraints, and initialization
of global rotations and translations, to optimize final camera poses
and point locations.

Authors: John Lambert
"""
import abc
from typing import Dict, List, Tuple

import gtsam


class BundleAdjustmentBase(metaclass=abc.ABCMeta):
    """Base class for all rotation averaging algorithms."""

    @abc.abstractmethod
    def run(
    	self,
    	intrinsics: Dict[int, np.ndarray],
    	global_rotations: List[gtsam.Rot3],
    	global_translations: List[gtsam.Point3],
    	correspondences: Dict[Tuple[int,int],np.ndarray],
    	) -> Tuple[List[gtsam.Pose3], np.ndarray]:
        """
        	Args:
        		intrinsics: 3x3 matrices for each camera ID
        		global_rotations:
        		global_translations:
        		correspondences: dictionary from (i,j) camera
        			pair to Nx4 array, with each matrix row
        			representing x_i,y_i,x_j,y_j

        	Returns:
        		optimized_poses: 
				point_cloud: Numpy array of shape Nx3
        """
        pass