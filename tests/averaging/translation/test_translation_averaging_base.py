"""Tests for translation averaging base class.

Authors: Ayush Baid
"""
import pickle
import unittest
from typing import List

import dask
import numpy as np
from gtsam import Point3, Rot3, Unit3

from averaging.translation.dummy_translation_averaging import \
    DummyTranslationAveraging


class TestTranslationAveragingBase(unittest.TestCase):
    """Main tests for translation averaging base class.

    This class should be inherited by all unit tests for translation averaging
    implementations.
    """

    def setUp(self):
        super().setUp()

        self.obj = DummyTranslationAveraging()

    def assert_equal_upto_scale(self,
                                wTi_list1: List[Point3],
                                wTi_list2: List[Point3]):
        """Helper function to assert that two lists of global Point3 are equal
        (upto global scale ambiguity)."""

        self.assertEqual(len(wTi_list1), len(wTi_list2),
                         'two lists to compare have unequal lengths')

        for i in range(1, len(wTi_list1)):
            # accounting for ambiguity in origin of the coordinate system.
            iU0_1 = Unit3(wTi_list1[i] - wTi_list1[0])

            iU0_2 = Unit3(wTi_list2[i] - wTi_list2[0])
            self.assertTrue(iU0_1.equals(iU0_2, 1e-1))

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection
            of relative unit-translations."""

        num_images = 3

        i2Ui1_dict = {
            (1, 0): Unit3(np.array([0, 0.2, 0])),
            (2, 1): Unit3(np.array([0, 0.1, 0.3])),
        }

        wRi_list = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, 30*np.pi/180, 0),
            Rot3.RzRyRx(0, 0, 20*np.pi/180),
        ]

        i2Ui1_graph = dask.delayed(i2Ui1_dict)

        wRi_graph = dask.delayed(wRi_list)

        # use the GTSAM API directly (without dask) for translation averaging
        wTi_expected = self.obj.run(
            num_images, i2Ui1_dict, wRi_list)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_images, i2Ui1_graph, wRi_graph)

        with dask.config.set(scheduler='single-threaded'):
            wTi_computed = dask.compute(computation_graph)[0]

        self.assert_equal_upto_scale(wTi_expected, wTi_computed)

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == '__main__':
    unittest.main()
