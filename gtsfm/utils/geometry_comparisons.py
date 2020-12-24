"""Utility functions for comparing different types related to geometry.

Authors: Ayush Baid
"""
from typing import List, Optional

import numpy as np
from gtsam import Pose3, Rot3


def compare_rotations(
    wRi_list: List[Optional[Rot3]], wRi_list_: List[Optional[Rot3]]
) -> bool:
    """Helper function to compare two lists of global Rot3, considering the
    origin as ambiguous.

    Notes:
    1. The input lists have the rotations in the same order, and can contain
       None entries.
    2. To resolve global origin ambiguity, we will fix one image index as
       origin in both the inputs and transform both the lists to the new
       origins.

    Args:
        wRi_list: 1st list of rotations.
        wRi_list_: 2nd list of rotations.

    Returns:
        result of the comparison.
    """

    if len(wRi_list) != len(wRi_list_):
        return False

    # check the presense of valid Rot3 objects in the same location
    wRi_valid = [i for (i, wRi) in enumerate(wRi_list) if wRi is not None]
    wRi_valid_ = [i for (i, wRi) in enumerate(wRi_list_) if wRi is not None]
    if wRi_valid != wRi_valid_:
        return False

    if len(wRi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    # fix the origin for both inputs lists
    origin = wRi_list[wRi_valid[0]]
    origin_ = wRi_list_[wRi_valid_[0]]

    # transform all other valid Pose3 entries to the new coordinate frame
    wRi_list = [wRi_list[i].between(origin) for i in wRi_valid[1:]]
    wRi_list_ = [wRi_list_[i].between(origin_) for i in wRi_valid_[1:]]

    return all(
        [wRi.equals(wRi_, 1e-1) for (wRi, wRi_) in zip(wRi_list, wRi_list_)]
    )


def compare_global_poses(
    wTi_list: List[Optional[Pose3]],
    wTi_list_: List[Optional[Pose3]],
    rot_err_thresh: float = 1e-3,
    trans_err_thresh: float = 1e-1,
) -> bool:
    """Helper function to compare two lists of global Pose3, considering the
    origin and scale ambiguous.

    Notes:
    1. The input lists have the poses in the same order, and can contain
       None entries.
    2. To resolve global origin ambiguity, we will fix one image index as
       origin in both the inputs and transform both the lists to the new
       origins.
    3. As there is a scale ambiguity, we use the median scaling factor to
       resolve the ambiguity.

    Args:
        wTi_list: 1st list of poses.
        wTi_list_: 2nd list of poses.
        rot_err_thresh (optional): error threshold for rotations. Defaults to
                                   1e-3.
        trans_err_thresh (optional): relative error threshold for translation.
                                     Defaults to 1e-1.

    Returns:
        results of the comparison.
    """

    # check the length of the input lists
    if len(wTi_list) != len(wTi_list_):
        return False

    # check the presense of valid Pose3 objects in the same location
    wTi_valid = [i for (i, wTi) in enumerate(wTi_list) if wTi is not None]
    wTi_valid_ = [i for (i, wTi) in enumerate(wTi_list_) if wTi is not None]
    if wTi_valid != wTi_valid_:
        return False

    if len(wTi_valid) <= 1:
        # we need >= two entries going forward for meaningful comparisons
        return False

    # fix the origin for both inputs lists
    origin = wTi_list[wTi_valid[0]]
    origin_ = wTi_list_[wTi_valid_[0]]

    # transform all other valid Pose3 entries to the new coordinate frame
    wTi_list = [wTi_list[i].between(origin) for i in wTi_valid[1:]]
    wTi_list_ = [wTi_list_[i].between(origin_) for i in wTi_valid_[1:]]

    # get the scale factor by using the median of scale for each index
    scaling_factors = [
        np.linalg.norm(wTi_list[i].translation())
        / (np.linalg.norm(wTi_list_[i].translation()) + np.finfo(float).eps)
        for i in range(len(wTi_list))
    ]

    # use the median to get the scale factor between two lists
    scale_factor_2to1 = np.median(scaling_factors)

    # scale the pose in the second list
    wTi_list_ = [
        Pose3(x.rotation(), x.translation() * scale_factor_2to1)
        for x in wTi_list_
    ]

    return all(
        [
            (
                wTi.rotation().equals(wTi_.rotation(), rot_err_thresh)
                and np.allclose(
                    wTi.translation(),
                    wTi_.translation(),
                    rtol=trans_err_thresh,
                )
            )
            for (wTi, wTi_) in zip(wTi_list, wTi_list_)
        ]
    )
