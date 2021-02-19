"""Sample poses for testing the averaging algorithms.

Authors: Ayush Baid
"""
import copy
from typing import Dict, List, Tuple

import numpy as np
from gtsam import Pose3, Rot3


def generate_relative_from_global(
    wTi_list: List[Pose3], pose_pairs: List[Tuple[int, int]]
) -> Dict[Tuple[int, int], Pose3]:
    """Generate relative poses from global poses.

    Args:
        wTi_list: global poses.
        pose_pairs: pairs (i1, i2) to construct relative poses for.

    Returns:
        Relative pose i2Ti1 for all requested pairs, as a dictionary.
    """
    return {(i1, i2): wTi_list[i2].between(wTi_list[i1]) for i1, i2 in pose_pairs}


"""4 poses in the circle of radius 5m, all looking at the center of the circle.

For relative poses, each pose has just 2 edges, connecting to the immediate neighbors.
"""
CIRCLE_TWO_EDGES_GLOBAL_POSES = [
    Pose3(Rot3.RzRyRx(0, 0, 0), np.array([0, 0, 0])),
    Pose3(Rot3.RzRyRx(np.deg2rad(90), 0, 0), np.array([0, 5, 5])),
    Pose3(Rot3.RzRyRx(np.deg2rad(180), 0, 0), np.array([0, 0, 10])),
    Pose3(Rot3.RzRyRx(np.deg2rad(270), 0, 0), np.array([0, -5, 5])),
]

CIRCLE_TWO_EDGES_RELATIVE_POSES = generate_relative_from_global(
    CIRCLE_TWO_EDGES_GLOBAL_POSES, [(1, 0), (2, 1), (3, 2), (0, 3)]
)

"""4 poses in the circle of radius 5m, all looking at the center of the circle.

For relative poses, each pose is connected to every other (3) pose.
"""
CIRCLE_ALL_EDGES_GLOBAL_POSES = copy.copy(CIRCLE_TWO_EDGES_GLOBAL_POSES)

CIRCLE_ALL_EDGES_RELATIVE_POSES = generate_relative_from_global(
    CIRCLE_TWO_EDGES_GLOBAL_POSES, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
)

"""3 poses simulating a forward motion with large translations, with no relative rotation.

For relative poses, we have a fully connected graph.
"""
FORWARD_MOTION_LARGE_TRANSLATIONS_GLOBAL_POSES = [
    Pose3(Rot3(), np.array([0, 0, 0])),
    Pose3(Rot3(), np.array([0, 0, 5])),
    Pose3(Rot3(), np.array([0, 0, 10])),
]

FORWARD_MOTION_LARGE_TRANSLATIONS_RELATIVE_POSES = generate_relative_from_global(
    FORWARD_MOTION_LARGE_TRANSLATIONS_GLOBAL_POSES, [(0, 1), (0, 2), (1, 2)]
)

"""3 poses simulating a forward motion with small translations, with no relative rotation.

For relative poses, we have a fully connected graph.
"""
FORWARD_MOTION_SMALL_TRANSLATIONS_GLOBAL_POSES = [
    Pose3(Rot3(), np.array([0, 0, 0])),
    Pose3(Rot3(), np.array([0, 0, 1e-3])),
    Pose3(Rot3(), np.array([0, 0, 5e-3])),
]

FORWARD_MOTION_SMALL_TRANSLATIONS_RELATIVE_POSES = generate_relative_from_global(
    FORWARD_MOTION_SMALL_TRANSLATIONS_GLOBAL_POSES, [(0, 1), (0, 2), (1, 2)]
)

"""3 poses in a panorama (i.e. no translations but large relative rotations)

For relative poses, we have a fully connected graph.
"""
PANORAMA_GLOBAL_POSES = [
    Pose3(Rot3.RzRyRx(0, np.deg2rad(-30), 0), np.zeros((3, 1))),
    Pose3(Rot3.RzRyRx(0, 0, 0), np.zeros((3, 1))),
    Pose3(Rot3.RzRyRx(0, np.deg2rad(+30), 0), np.zeros((3, 1))),
]

PANORAMA_RELATIVE_POSES = generate_relative_from_global(PANORAMA_GLOBAL_POSES, [(0, 1), (0, 2), (1, 2)])
