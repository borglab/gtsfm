"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

from gtsam import Cal3_S2, Pose3, Unit3
from gtsam.examples import SFMdata

import tests.averaging.translation.test_translation_averaging_base as test_translation_averaging_base
from averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from loader.folder_loader import FolderLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / 'data'


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

        The camera poses are arranged on the circle and point towards the center
        of the circle. The poses of 8 cameras are obtained from SFMdata and the
        unit translations directions between some camera pairs are computed from their global translations.

        This test is copied from GTSAM's TranslationAveragingExample.
        """

        fx, fy, s, u0, v0 = 50.0, 50.0, 0.0, 50.0, 50.0
        expected_wTi_list = SFMdata.createPoses(Cal3_S2(fx, fy, s, u0, v0))

        wRi_list = [x.rotation() for x in expected_wTi_list]

        # create relative translation directions between a pose index and the
        # next two poses
        i2Ui1_dict = {}
        for i1 in range(len(expected_wTi_list)-1):
            for i2 in range(i1+1, min(len(expected_wTi_list), i1+3)):
                # create relative translations using global R and T.
                i2Ui1_dict[(i1, i2)] = Unit3(
                    expected_wTi_list[i2].between(
                        expected_wTi_list[i1]
                    ).translation())

        wti_list = self.obj.run(len(wRi_list), i2Ui1_dict, wRi_list)
        wTi_list = [Pose3(wRi, wti)
                    if wti is not None else None
                    for (wRi, wti) in zip(wRi_list, wti_list)]

        # compare the entries
        self.assert_equal_upto_scale(wTi_list, expected_wTi_list)

    def test_lund_door(self):
        loader = FolderLoader(
            str(DATA_ROOT_PATH / 'set1_lund_door'), image_extension='JPG')

        expected_wTi_list = [loader.get_camera_pose(
            x) for x in range(len(loader))]
        wRi_list = [x.rotation() for x in expected_wTi_list]

        i2Ui1_dict = dict()
        for (i1, i2) in loader.get_valid_pairs():
            i2Ti1 = expected_wTi_list[i2].between(expected_wTi_list[i1])

            i2Ui1_dict[(i1, i2)] = Unit3((i2Ti1.translation()))

        wti_list = self.obj.run(len(loader), i2Ui1_dict, wRi_list)

        wTi_list = [Pose3(wRi, wti)
                    if wti is not None else None
                    for (wRi, wti) in zip(wRi_list, wti_list)]

        self.assert_equal_upto_scale(wTi_list, expected_wTi_list)


if __name__ == '__main__':
    unittest.main()
