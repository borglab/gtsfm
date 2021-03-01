"""Tests for translation averaging base class.

Authors: Ayush Baid
"""
import pickle
import unittest

import dask
from gtsam import Cal3_S2, Unit3
from gtsam.examples import SFMdata

import gtsfm.utils.geometry_comparisons as geometry_comparisons
from gtsfm.averaging.translation.dummy_translation_averaging import (
    DummyTranslationAveraging,
)


class TestTranslationAveragingBase(unittest.TestCase):
    """Main tests for translation averaging base class.

    This class should be inherited by all unit tests for translation averaging implementations.
    """

    def setUp(self):
        super().setUp()

        self.obj = DummyTranslationAveraging()

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection of relative unit-translations."""

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
        for i1 in range(len(expected_wTi_list) - 1):
            for i2 in range(i1 + 1, min(len(expected_wTi_list), i1 + 3)):
                # create relative translations using global R and T.
                i2Ui1_dict[(i1, i2)] = Unit3(expected_wTi_list[i2].between(expected_wTi_list[i1]).translation())

        # use the `run` API to get expected results
        expected_wti_list = self.obj.run(len(wRi_list), i2Ui1_dict, wRi_list)

        # form computation graph and execute
        i2Ui1_graph = dask.delayed(i2Ui1_dict)
        wRi_graph = dask.delayed(wRi_list)
        computation_graph = self.obj.create_computation_graph(len(wRi_list), i2Ui1_graph, wRi_graph)
        with dask.config.set(scheduler="single-threaded"):
            wti_list = dask.compute(computation_graph)[0]
        # compare the entries
        self.assertTrue(geometry_comparisons.align_and_compare_translations(wti_list, expected_wti_list))

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == "__main__":
    unittest.main()
