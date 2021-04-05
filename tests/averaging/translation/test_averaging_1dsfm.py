"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

from gtsam import Pose3, Unit3

import tests.averaging.translation.test_translation_averaging_base as test_translation_averaging_base
import gtsfm.utils.geometry_comparisons as geometry_comparisons
from gtsfm.averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from gtsfm.loader.olsson_loader import OlssonLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"


class TestTranslationAveraging1DSFM(test_translation_averaging_base.TestTranslationAveragingBase):
    """Test class for 1DSFM rotation averaging.

    All unit test functions defined in TestTranslationAveragingBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.obj = TranslationAveraging1DSFM()

    def test_lund_door(self):
        loader = OlssonLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")

        # we will use ground truth poses to generate relative rotations and relative unit translations
        expected_wTi_list = [loader.get_camera_pose(x) for x in range(len(loader))]
        wRi_list = [x.rotation() for x in expected_wTi_list]

        i2Ui1_dict = dict()
        for (i1, i2) in loader.get_valid_pairs():
            i2Ti1 = expected_wTi_list[i2].between(expected_wTi_list[i1])

            i2Ui1_dict[(i1, i2)] = Unit3((i2Ti1.translation()))

        # ensure we can recover absolute translations (up to a scale) from relative unit translations
        wti_list = self.obj.run(len(loader), i2Ui1_dict, wRi_list)

        # combine recovered translations with ground truth absolute rotations
        wTi_list = [Pose3(wRi, wti) if wti is not None else None for (wRi, wti) in zip(wRi_list, wti_list)]
        self.assertTrue(geometry_comparisons.compare_global_poses(wTi_list, expected_wTi_list))


if __name__ == "__main__":
    unittest.main()
