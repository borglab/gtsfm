""" 
Test to check average reprojection error function. 
Triangulation example from https://github.com/borglab/gtsam/blob/ca4daa0894abb6e979384302075d5fc3b61119f3/matlab/gtsam_tests/testTriangulation.m

Author: Sushmita Warrier
"""
import logging
import math
import numpy as np
import sys
import unittest

import gtsam
from gtsam.utils.test_case import GtsamTestCase

from gtsfm.densify.metrics import avg_reprojection_error

class TestReprojectionError(GtsamTestCase):

    def test_reprojection_error(self):
        log= logging.getLogger( "Triangulation_test" )
        # we want dummy camera and matched points
        # 2 corresponding image pts, with same camera assumed. 

        # (fx, fy, s, u0, v0)
        sharedCal = gtsam.Cal3_S2(1500, 1200, 0, 640, 480) 
        
        # camera poses for both pts - Looking along X-axis, 1 meter above ground plane (x-y)
        upright = gtsam.Rot3.Ypr(-math.pi / 2, 0., -math.pi / 2)
        pose1 = gtsam.Pose3(upright, gtsam.Point3(0., 0., 1.))
        camera1 = gtsam.PinholeCameraCal3_S2(pose1, sharedCal)
        # create second camera 1 meter to the right of first camera
        pose2 = pose1.compose(gtsam.Pose3(gtsam.Rot3(), gtsam.Point3(1., 0., 0.)))
        camera2 = gtsam.PinholeCameraCal3_S2(pose2, sharedCal)

        # landmark ~5 meters infront of camera
        landmark = gtsam.Point3(5, 0.5, 1.2)

        # Project two landmarks into two cameras and triangulate
        z1 = camera1.project(landmark)
        z2 = camera2.project(landmark)

        poses = gtsam.Pose3Vector()
        measurements = gtsam.Point2Vector()
        poses.append(pose1)
        poses.append(pose2)
        measurements.append(z1)
        measurements.append(z2)

        optimize = False
        rank_tol = 1e-9

        triangulated_landmark = gtsam.triangulatePoint3(poses, sharedCal, measurements, rank_tol, optimize)
        self.gtsamAssertEquals(landmark,triangulated_landmark)

        obs_list = []
        obs_list.append((0, z1))  # z1, z2: image points
        obs_list.append((1, z2))

        # Testing avg_reproj_error func
        landmark_dict = {tuple(triangulated_landmark) : obs_list}
        mean_computed_error = avg_reprojection_error(sharedCal, poses, landmark_dict)

        # Reprojection error ground truth
        errors = camera1.project(landmark) - z1
        # errors returns a 1x2 array
        # converting to pixel error
        pixel_error = np.linalg.norm(errors, ord=None)
        self.assertAlmostEqual(pixel_error, mean_computed_error)

if __name__ == "__main__":
    logging.basicConfig( stream=sys.stderr )
    logging.getLogger( "Triangulation_test" ).setLevel( logging.DEBUG )
    unittest.main()

    # TODO: Integrate Dask functionality?