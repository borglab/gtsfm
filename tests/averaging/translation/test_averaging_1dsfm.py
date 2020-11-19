"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import unittest

from gtsam import Cal3_S2, Unit3
from gtsam.examples import SFMdata

import tests.averaging.translation.test_translation_averaging_base as \
    test_translation_averaging_base
from averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM


class TestTranslationAveraging1DSFM(
        test_translation_averaging_base.TestTranslationAveragingBase):
    """Test class for 1DSFM rotation averaging.

    All unit test functions defined in TestTranslationAveragingBase are run
    automatically.
    """

    def setUp(self):
        super().setUp()

        self.obj = TranslationAveraging1DSFM()

    def test_simple(self):
        """Test a simple case with 8 camera poses.

        The camera poses are aranged on the circle and point towards the center
        of the circle. The poses of 8 cameras are obtained from SFMdata and the
        unit translations directions between some camera pairs are computed from their global translations.

        This test is copied from GTSAM's TranslationAveragingExample.
        """

        fx, fy, s, u0, v0 = 50.0, 50.0, 0.0, 50.0, 50.0
        global_pose_list = SFMdata.createPoses(Cal3_S2(fx, fy, s, u0, v0))

        expected_w_t_i_list = [x.translation() for x in global_pose_list]

        w_R_i_list = [x.rotation() for x in global_pose_list]

        # create relative translation directions between a pose index and the
        # next two poses
        i1_t_i2_dict = {}
        for i1 in range(len(global_pose_list)-1):
            for i2 in range(i1+1, min(len(global_pose_list), i1+3)):
                # create relative translations using global rotations and
                # translations.
                i1_t_i2_dict[(i1, i2)] = Unit3(
                    w_R_i_list[i1].unrotate(
                        expected_w_t_i_list[i2] - expected_w_t_i_list[i1]))

        computed_wti_list = self.obj.run(
            len(w_R_i_list), i1_t_i2_dict, w_R_i_list)

        # compare the entries
        self.assert_equal_upto_scale(
            expected_w_t_i_list, computed_wti_list)


if __name__ == '__main__':
    unittest.main()
