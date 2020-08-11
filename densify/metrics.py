"""
Metrics to evaluate sparse reconstruction results
Author: Sushmita Warrier
"""

import numpy as np
from typing import Dict, List, Tuple

import gtsam

# from bundle import BA_class (for triangulated pts)

def avg_reprojection_error(
    calibration, 
    pose_estimates : List[List[gtsam.Pose3]], 
    landmark_dict : Dict[gtsam.Pose3, List[Tuple[int, gtsam.Point2]]]
    ) -> float:
    """
    Compute average reprojection error across dataset after BA
    Args: 
        1. calibration - camera calibration gtsam.Cal3_S2
        2. pose estimates - list of list of gtsam Pose3 ie. list of camera poses for each landmark
        3. landmark_dict - dict with 3d pt as key and landmark map as value
            --> landmark_map - [(i,Point2()), (j,Point2())...] - (image idx, imgPt) of all features that are matched for a particular landmark
    Returns: 
        average reprojection error - float
    """
    # NEEDS A FIX: CURRENTLY SAME CALIBRATION ASSUMED FOR ALL. 
    initial_estimates = gtsam.Values()
    # print("pe:", pose_estimates)
    # Assuming all poses currently given are valid poses - so no cases of failure to triangulate
    landmark_idx = 0
    for key, val in landmark_dict.items():
        landmark_3d_pt = key
        landmark_map = val   # [[(i,Point2()), (j,Point2())...]...] -> matched pts
        initial_estimates.insert(gtsam.symbol(ord('p'),landmark_idx), landmark_3d_pt)  #
        assert len(landmark_map) == len(pose_estimates), "Nb of images and nb of poses must be equal"
        for obs in landmark_map:
            pose_idx = obs[0]  #nb of image points
            initial_estimates.insert(gtsam.symbol(ord('x'), pose_idx),pose_estimates[pose_idx])
        landmark_idx += 1

    sigma = 1.0
    measurement_noise = gtsam.noiseModel_Isotropic.Sigma(2, sigma)
    projection_error = np.empty((len(pose_estimates), len(landmark_dict)))  # len(pose_estimates) or pose_estimates[i] for each landmark or something
    total_reproj_error = 0
    factor_map = [[None]*(landmark_idx) for i in range (len(pose_estimates))]
    idx = 0
    for key, val in landmark_dict.items():
        landmark_3d_pt = key
        landmark_map = val 
        for obs in landmark_map:
            keypoint = obs[1]
            #print("keypoint", keypoint)
            pose_idx = obs[0]
            # print("pose_idx", pose_idx, keypoint)
            # ord func in python returns a unicode representation of a string of length 1
            temp_factor = gtsam.GenericProjectionFactorCal3_S2(
                keypoint, 
                measurement_noise, 
                gtsam.symbol(ord('x'), pose_idx), 
                gtsam.symbol(ord('p'), idx), 
                calibration
                )
            total_reproj_error += temp_factor.error(initial_estimates) # unsure if this is correct
            # factor_map[pose_idx][idx] = temp_factor
            factor_map = temp_factor
        idx += 1  # this is now the nb of landmark pts

    for pose_idx in range(projection_error.shape[0]):
        for landmark_idx in range(projection_error.shape[1]):
            projection_error[pose_idx][landmark_idx] = factor_map.error(initial_estimates)
            # not sure how to compute avg error from this- ideas welcome

    nb_landmark_pts = idx
    mean_computed_error = np.sqrt(total_reproj_error/ nb_landmark_pts) # total_error/nb_3d_pts
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

if __name__ == '__main__':
    K = np.array([
        [718.856 ,   0.  ,   607.1928],
        [  0.  ,   718.856 , 185.2157],
        [  0.  ,     0.   ,    1.    ]
        ])
    calibration = gtsam.Cal3_S2(
        fx=K[0, 0],
        fy=K[1, 1],
        s=0,
        u0=K[0, 2],
        v0=K[1, 2],
        )
    
    pose_estimates = []
    R1 = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [0., 0., 1.]
        ])
    R2 = np.array([
        [ 0.99999183 ,-0.00280829 ,-0.00290702],
        [ 0.0028008  , 0.99999276, -0.00257697],
        [ 0.00291424 , 0.00256881 , 0.99999245]
        ])
    t1_gt = gtsam.Point3(0.0,0.0,0.0)
    t2_gt = gtsam.Point3(-0.02182627, 0.00733316, 0.99973488)
    pose_estimates.append(gtsam.Pose3(gtsam.Rot3(R1), t1_gt))
    pose_estimates.append(gtsam.Pose3(gtsam.Rot3(R2), t2_gt))
    # avg_reprojection_error(calibration,pose_estimates)

