"""
Toy case to checking average reprojection error func
Author: Sushmita Warrier
"""
import unittest
import logging
import sys
import numpy as np
import cv2 as cv

import gtsam
from gtsam.utils.test_case import GtsamTestCase

from densify.metrics import avg_reprojection_error

class TestReprojectionError(GtsamTestCase):

    def test_triangulation(self):
        log= logging.getLogger( "Triangulation_test" )
        # we want dummy camera and matched points
        # 2 corresponding image pts, with same camera assumed. 
        # fx = 2 for both cameras
        K = np.array([[ 2. , 0.  , 0. ],
        [ 0.  , 2. , 0. ],
        [ 0.  , 0. , 1. ]])
        # camera poses for both pts -  R2 is rotated by 30deg
        R1 = np.array([[1., 0., 0.],
            [0., 1., 0.],
            [0., 0., 1.]])
        # R2 = np.array([[ 1.0 , 0.0 , 0.0],
        #     [ 0.0  , 1.0, 0.],
        #     [ 0.0 , 0. , 1.]])
        R2 = np.array([[ 1.0 , 0.0 , 0.0],
           [ 0.0  , 0.866, -0.5],
           [ 0.0 , 0.5 , 0.866]])

        # opencv's (which is used as ground truth) method uses array input
        t1_gt = np.array([[0.0], [0.], [0.]])
        t2_gt = np.array([[1.0], [ 0.5], [ 0.]])
        # not sure how to convert array to Point3 directly
        t1 = gtsam.Point3(2.0, 0.0, 0.0)
        t2 = gtsam.Point3(12.0, 0.0, 0.0)

        # Corresponding image points
        imgPt1 = gtsam.Point2(100.0, 120.0)
        imgPt2 = gtsam.Point2(102.0, 103.2)
        imgPt1_gt = np.array([100.0, 120.0])
        imgPt2_gt = np.array([102.0, 103.2])
        calibration = gtsam.Cal3_S2(
        fx=K[0, 0],
        fy=K[1, 1],
        s=0,
        u0=K[0, 2],
        v0=K[1, 2],
        )

        ''' This section computes the triangulated pt and verifies it using opencv's
        method. Can be removed and have directly a triangulated point hardcoded here
        for brevity.
        '''
        obs_list, pose_estimates = [], []
        # set up projection matrices for both cameras
        # R, t are assumed given for both image points
        P1 = np.hstack([R1.T, -R1.T.dot(t1_gt)])
        P2 = np.hstack([R2.T, -R2.T.dot(t2_gt)])
        P1 = K.dot(P1)
        P2 = K.dot(P2)

        pose_estimates.append(gtsam.Pose3(gtsam.Rot3(R1), t1))
        pose_estimates.append(gtsam.Pose3(gtsam.Rot3(R2), t2))
        obs_list.append((0, imgPt1))
        obs_list.append((1, imgPt2))
        poses = gtsam.Pose3Vector()
        keypoints = gtsam.Point2Vector()
        for observation in obs_list:
            key_point = observation[1]   # image point
            pose_idx = observation[0]
            pose = pose_estimates[pose_idx]
            keypoints.push_back(key_point)
            poses.push_back(pose)
        optimize = False
        rank_tol = 1e-9
        computed_Pt3D = gtsam.triangulatePoint3(poses, calibration, keypoints, rank_tol, optimize)
        self.assertIsNotNone(computed_Pt3D, "Triangulated pt missing")

        expected_Pt3D = cv.triangulatePoints(P1, P2, imgPt1_gt, imgPt2_gt).T
        expected_Pt3D = expected_Pt3D[:, :3] / expected_Pt3D[:, 3:4]   # array of array [[ , , ]]
        # convert to gtsam.Point3 format to compare with computed point
        expected_Pt3D_gt = gtsam.Point3(expected_Pt3D[0][0], expected_Pt3D[0][1], expected_Pt3D[0][2])

        self.gtsamAssertEquals(expected_Pt3D_gt,computed_Pt3D)

        # Testing avg_reproj_error func
        landmark_dict = {computed_Pt3D : obs_list}
        mean_computed_error = avg_reprojection_error(calibration, pose_estimates, landmark_dict)

        # Ground truth way to calculate reproj error
        rvec, tvec, imgPt = [], [], []
        rvec1, _ = cv.Rodrigues(R1.T)
        rvec2, _ = cv.Rodrigues(R2.T)
        rvec.append(rvec1)
        rvec.append(rvec2)
        tvec.append(-t1_gt)
        tvec.append(-t2_gt)
        imgPt.append(imgPt1_gt)
        imgPt.append(imgPt2_gt)

        tot_error, total_points = 0, 0
        list_of_3d_pts = [expected_Pt3D]  
        for pt in range(len(list_of_3d_pts)):
            reprojected_pt,_ = cv.projectPoints(list_of_3d_pts[pt], rvec[pt], tvec[pt], K, distCoeffs=None)
            reprojected_pt=reprojected_pt.reshape(-1,2)
            tot_error+=np.sum(np.abs(imgPt[pt]-reprojected_pt)**2)
            total_points+=len(list_of_3d_pts[pt])  # nb of 3d pts

        mean_error_gt = np.sqrt(tot_error/total_points)

        self.assertAlmostEqual(mean_computed_error, mean_error_gt, 3)

if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "Triangulation_test" ).setLevel( logging.DEBUG )
    unittest.main()