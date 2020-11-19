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
                                w_T_i_list_1: List[Point3],
                                w_T_i_list_2: List[Point3]):
        """Helper function to assert that two lists of global Point3 are equal
        (upto global scale ambiguity)."""

        self.assertEqual(len(w_T_i_list_1), len(w_T_i_list_2),
                         'two lists to compare have unequal lengths')

        for idx in range(1, len(w_T_i_list_1)):
            # accounting for ambiguity in origin of the coordinate system.
            idx_t_0_direction1 = Unit3(w_T_i_list_1[idx] - w_T_i_list_1[0])

            idx_t_0_direction2 = Unit3(w_T_i_list_2[idx] - w_T_i_list_2[0])

            self.assertTrue(idx_t_0_direction1.equals(
                idx_t_0_direction2, 1e-2))

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection
            of relative unit-translations."""

        num_images = 3

        i1_t_i2_dict = {
            (0, 1): Unit3(np.array([0, 0.2, 0])),
            (1, 2): Unit3(np.array([0, 0.1, 0.3])),
        }

        w_R_i_list = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, 30*np.pi/180, 0),
            Rot3.RzRyRx(0, 0, 20*np.pi/180),
        ]

        i1_t_i2_graph = {
            (0, 1): dask.delayed(Unit3)(np.array([0, 0.2, 0])),
            (1, 2): dask.delayed(Unit3)(np.array([0, 0.1, 0.3])),
        }

        w_R_i_graph = dask.delayed(w_R_i_list)

        # use the GTSAM API directly (without dask) for translation averaging
        expected_result = self.obj.run(
            num_images, i1_t_i2_dict, w_R_i_list)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_images, i1_t_i2_graph, w_R_i_graph)

        with dask.config.set(scheduler='single-threaded'):
            dask_result = dask.compute(computation_graph)[0]

        self.assert_equal_upto_scale(expected_result, dask_result)

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == '__main__':
    unittest.main()
