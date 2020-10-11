"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
from gtsam import Cal3_S2, Unit3
from gtsam.examples import SFMdata

import tests.averaging.translation.test_translation_averaging_base as \
    test_translation_averaging_base
from averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM


class TestTranslationAveraging1DSFM(
        test_translation_averaging_base.TestTranslationAveragingBase):
    """Test class for 1DSFM rotation averaging."""

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

        expected_wti_list = [x.translation() for x in global_pose_list]

        wRi_list = [x.rotation() for x in global_pose_list]

        # create relative translation directions between a pose index and the
        # next two poses
        i1ti2_dict = {}
        for i1 in range(len(global_pose_list)-2):
            for i2 in range(i1+1, i1+3):
                i1ti2_dict[(i1, i2)] = Unit3(
                    wRi_list[i1].unrotate(
                        expected_wti_list[i2] - expected_wti_list[i1]))

        computed_wti_list = self.obj.run(len(wRi_list), i1ti2_dict, wRi_list)

        # compare the entries
        for i in range(1, len(wRi_list)):
            computed_direction = Unit3(computed_wti_list[i].point3() -
                                       computed_wti_list[0].point3())

            expected_direction = Unit3(expected_wti_list[i] -
                                       expected_wti_list[0])

            self.assertTrue(expected_direction.equals(
                computed_direction, 1e-5))
