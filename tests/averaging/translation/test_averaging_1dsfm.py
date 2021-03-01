"""Tests for 1DSFM translation averaging.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path

from gtsam import Unit3

import tests.averaging.translation.test_translation_averaging_base as test_translation_averaging_base
import gtsfm.utils.geometry_comparisons as geometry_comparisons
from gtsfm.averaging.translation.averaging_1dsfm import (
    TranslationAveraging1DSFM,
)
from gtsfm.loader.folder_loader import FolderLoader

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "data"


class TestTranslationAveraging1DSFM(test_translation_averaging_base.TestTranslationAveragingBase):
    """Test class for 1DSFM rotation averaging.

    All unit test functions defined in TestTranslationAveragingBase are run automatically.
    """

    def setUp(self):
        super().setUp()

        self.obj = TranslationAveraging1DSFM()

    def test_lund_door(self):
        loader = FolderLoader(str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG")

        expected_wTi_list = [loader.get_camera_pose(x) for x in range(len(loader))]
        wRi_list = [x.rotation() for x in expected_wTi_list]

        i2Ui1_dict = dict()
        for (i1, i2) in loader.get_valid_pairs():
            i2Ti1 = expected_wTi_list[i2].between(expected_wTi_list[i1])

            i2Ui1_dict[(i1, i2)] = Unit3((i2Ti1.translation()))

        wti_computed = self.obj.run(len(loader), i2Ui1_dict, wRi_list)
        wti_expected = [x.translation() for x in expected_wTi_list]

        # TODO: using a v high value for translation relative threshold. Fix it
        self.assertTrue(geometry_comparisons.align_and_compare_translations(wti_computed, wti_expected, 1e-1, 9e-2))


if __name__ == "__main__":
    unittest.main()
