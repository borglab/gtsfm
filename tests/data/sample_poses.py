"""Sample poses for testing the averaging algorithms.

The visualizations of this poses are stored in the folder: tests/data/viz_sample_poses

Authors: Ayush Baid
"""
import copy
from typing import Dict, List, Tuple

import numpy as np
from gtsam import Cal3_S2, Point3, Pose3, Rot3, Unit3
from gtsam.examples import SFMdata

DEFAULT_ROTATION = Rot3.RzRyRx(0, np.deg2rad(10), 0)
DEFAULT_TRANSLATION = np.array([0, 2, 0])


def generate_relative_from_global(
    wTi_list: List[Pose3], pair_indices: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], Pose3]:
    """Generate relative poses from global poses.

    Args:
        wTi_list: global poses.
        pair_indices: pairs (i1, i2) to construct relative poses for.

    Returns:
        Dictionary (i1, i2) -> i2Ti1 for all requested pairs.
    """
    return {(i1, i2): wTi_list[i2].between(wTi_list[i1]) for i1, i2 in pair_indices}


"""4 poses in the circle of radius 5m, all looking at the center of the circle.

For relative poses, each pose has just 2 edges, connecting to the immediate neighbors.
"""
CIRCLE_TWO_EDGES_GLOBAL_POSES = SFMdata.createPoses(Cal3_S2(fx=1, fy=1, s=0, u0=0, v0=0))[::2]

CIRCLE_TWO_EDGES_RELATIVE_POSES = generate_relative_from_global(
    CIRCLE_TWO_EDGES_GLOBAL_POSES, [(0, 1), (1, 2), (2, 3), (0, 3)]
)

"""4 poses in the circle of radius 5m, all looking at the center of the circle.

For relative poses, each pose is connected to every other (3) pose.
"""
CIRCLE_ALL_EDGES_GLOBAL_POSES = copy.copy(CIRCLE_TWO_EDGES_GLOBAL_POSES)

CIRCLE_ALL_EDGES_RELATIVE_POSES = generate_relative_from_global(
    CIRCLE_TWO_EDGES_GLOBAL_POSES, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
)

"""3 poses on a line, simulating forward motion, with large translations and no relative rotation.

For relative poses, we have a fully connected graph.
"""
LINE_LARGE_EDGES_GLOBAL_POSES = [
    Pose3(DEFAULT_ROTATION, DEFAULT_TRANSLATION + np.array([0, 0, 0])),
    Pose3(DEFAULT_ROTATION, DEFAULT_TRANSLATION + np.array([0, 0, 5])),
    Pose3(DEFAULT_ROTATION, DEFAULT_TRANSLATION + np.array([0, 0, 10])),
]

LINE_LARGE_EDGES_RELATIVE_POSES = generate_relative_from_global(LINE_LARGE_EDGES_GLOBAL_POSES, [(0, 1), (0, 2), (1, 2)])

"""3 poses on a line, simulating forward motion, with small translations and no relative rotation.

For relative poses, we have a fully connected graph.
"""
LINE_SMALL_EDGES_GLOBAL_POSES = [
    Pose3(DEFAULT_ROTATION, DEFAULT_TRANSLATION + np.array([0, 0, 0])),
    Pose3(DEFAULT_ROTATION, DEFAULT_TRANSLATION + np.array([0, 0, 1e-3])),
    Pose3(DEFAULT_ROTATION, DEFAULT_TRANSLATION + np.array([0, 0, 5e-3])),
]

LINE_SMALL_EDGES_RELATIVE_POSES = generate_relative_from_global(LINE_SMALL_EDGES_GLOBAL_POSES, [(0, 1), (0, 2), (1, 2)])

"""3 poses in a panorama (i.e. 3 translation values being the same but large relative rotations i.e. pitch varies)

For relative poses, we have a fully connected graph.
"""
PANORAMA_GLOBAL_POSES = [
    Pose3(Rot3.RzRyRx(0, np.deg2rad(-30), 0), DEFAULT_TRANSLATION),
    Pose3(Rot3.RzRyRx(0, 0, 0), DEFAULT_TRANSLATION),
    Pose3(Rot3.RzRyRx(0, np.deg2rad(+30), 0), DEFAULT_TRANSLATION),
]

PANORAMA_RELATIVE_POSES = generate_relative_from_global(PANORAMA_GLOBAL_POSES, [(0, 1), (0, 2), (1, 2)])


def convert_data_for_rotation_averaging(
    wTi_list: List[Pose3], i2Ti1_dict: Dict[Tuple[int, int], Pose3]
) -> Tuple[Dict[Tuple[int, int], Rot3], List[Rot3]]:
    """Converts the poses to inputs and expected outputs for a rotation averaging algorithm.

    Args:
        wTi_list: List of global poses.
        i2Ti1_dict: Dictionary of (i1, i2) -> i2Ti1 relative poses.

    Returns:
        i2Ti1_dict's values mapped to relative rotations i2Ri1.
        wTi_list mapped to global rotations.
    """

    wRi_list = [x.rotation() for x in wTi_list]
    i2Ri1_dict = {k: v.rotation() for k, v in i2Ti1_dict.items()}

    return i2Ri1_dict, wRi_list


def convert_data_for_translation_averaging(
    wTi_list: List[Pose3], i2Ti1_dict: Dict[Tuple[int, int], Pose3]
) -> Tuple[List[Rot3], Dict[Tuple[int, int], Unit3], List[Point3]]:
    """Converts the poses to inputs and expected outputs for a translation averaging algorithm.

    Args:
        wTi_list: List of global poses.
        i2Ti1_dict: Dictionary of (i1, i2) -> i2Ti1 relative poses.

    Returns:
        wTi_list mapped to global rotations.
        i2Ti1_dict's values mapped to relative unit translations i2Ui1.
        wTi_list mapped to global translations.
    """

    wRi_list = [x.rotation() for x in wTi_list]
    wti_list = [x.translation() for x in wTi_list]
    i2Ui1_dict = {k: Unit3(v.translation()) for k, v in i2Ti1_dict.items()}

    return wRi_list, i2Ui1_dict, wti_list
