"""
Metrics to evaluate sparse reconstruction results
Author: Sushmita Warrier
"""

import numpy as np
from typing import Dict, List, Tuple

import gtsam


def avg_reprojection_error(
    calibration: gtsam.Cal3_S2, 
    pose_estimates : List[List[gtsam.Pose3]], 
    landmark_dict : Dict[gtsam.Point3, List[Tuple[int, gtsam.Point2]]]
    ) -> float:
    """
    Compute average reprojection error across dataset after BA
    Args: 
        calibration: camera calibration gtsam.Cal3_S2
        pose estimates: list of list of gtsam Pose3 i.e. list of camera poses for each landmark
        landmark_dict: dict with 3D pt as key and landmark map as value
            --> landmark_map: [(i,Point2()), (j,Point2())...], where 
            (i,Point2()) = (image idx, imgPt) of all features that are matched for a particular landmark
    Returns: 
        average reprojection error: float
    """
    # TODO: CURRENTLY SAME CALIBRATION ASSUMED FOR ALL. FIX REQD
    initial_estimates = gtsam.Values()
    # Assuming all poses currently given are valid poses - so no cases of failure to triangulate
    landmark_idx = 0
    for landmark_3d_pt, landmark_map in landmark_dict.items():
        initial_estimates.insert(gtsam.symbol('p',landmark_idx), landmark_3d_pt) 
        if len(landmark_map) != len(pose_estimates):
            raise Exception('Number of images and poses must be equal. Number of images was: {} and number of poses was: {}'.format(len(landmark_map), len(pose_estimates)))

        for (pose_idx, _) in landmark_map:
            initial_estimates.insert(gtsam.symbol('x', pose_idx),pose_estimates[pose_idx])
        landmark_idx += 1

    sigma = 1.0
    measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, sigma)
    total_reproj_error = 0
    idx = 0
    for landmark_3d_pt, landmark_map in landmark_dict.items():
        for (pose_idx, keypoint) in landmark_map:  # loop through the observations       
            temp_factor = gtsam.GenericProjectionFactorCal3_S2(
                keypoint, 
                measurement_noise, 
                gtsam.symbol('x', pose_idx), 
                gtsam.symbol('p', idx), 
                calibration
                )
            total_reproj_error += temp_factor.error(initial_estimates) 
        idx += 1  # this is now the nb of landmark pts

    nb_landmark_pts = idx
    mean_computed_error = total_reproj_error/ nb_landmark_pts # total_error/nb_3d_pts
    return mean_computed_error
    

def get_avg_track_length(
    landmark_dict:  Dict[gtsam.Pose3, List[Tuple[int, gtsam.Point2]]]
    ) -> float:
    """
    Get average track length for all features (nb of images across which a feature point is tracked)
    Args: 
        landmark_dict - dict with 3D pt as key and landmark map as value
            --> landmark_map - [(i,Point2()), (j,Point2())...] - (image idx, imgPt) of all features that are matched for a particular landmark
    Returns: 
        average track length
    """
    track_length = [len(v) for k,v in landmark_dict.items()] 
    return sum(track_length)/len(track_length)


def nb_pts():
    """
    Number of feature points that conform to the avg feature track length. (In the order of millions)
    """
    pass

def get_timing_benchmark():
    """
    Time required for reconstruction
    """
    pass

