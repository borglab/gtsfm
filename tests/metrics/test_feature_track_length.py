"""
Toy case for checking avg track length func
Author: Sushmita Warrier
"""
import unittest

import gtsam
from gtsam.utils.test_case import GtsamTestCase

from densify.metrics import get_avg_track_length


class TestReprojectionError(GtsamTestCase):

    def test_track_length(self):
        """ 
        Test average track length computation function using arbitrary values.
        Since only the number of landmarks per point matters, we use arbitrary
        3d points and landmarks to populate this test.
        """
        lndmrk_map1 =  [
            (4, gtsam.Point2(23.0, 4.6)), (5, gtsam.Point2(3.2, 2.8))
            ]
        lndmrk_map2 = [
            (6, gtsam.Point2(9.8, 4.4)), (7, gtsam.Point2(2.3, 8.7)), 
            (8, gtsam.Point2(45.7, 22.1)), (9, gtsam.Point2(2.4,1.9))
            ]
        lndmrk_map3 = [
            (6, gtsam.Point2(9.8, 4.4)), (7, gtsam.Point2(2.3, 8.7)), 
            (8, gtsam.Point2(45.7, 22.1)), (7, gtsam.Point2(2.3, 8.7)), 
            (8, gtsam.Point2(45.7, 22.1)), (9, gtsam.Point2(2.4,1.9))]

        # Dictionary with 3 landmark points, each corresponding to one of the above landmark maps
        landmark_dict = {
            gtsam.Point3(3.2,1.4,5.6) : lndmrk_map1, 
            gtsam.Point3(2.2, 3.3, 4.4): lndmrk_map2, 
            gtsam.Point3(1.2, 2.2, 3.3) : lndmrk_map3 }

        track_length = get_avg_track_length(landmark_dict)

        # Average track length will be:
        # [track_length for 1st landmark + track_length for 2nd landmark \
        # + track_length for 3rd landmark] / # of landmarks
        self.assertEqual(track_length, 4)
    
if __name__ == "__main__":
    unittest.main()

        