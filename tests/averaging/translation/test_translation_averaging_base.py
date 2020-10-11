"""Tests for translation averaging base class.

Authors: Ayush Baid
"""
import pickle
import unittest

import dask
import numpy as np
from gtsam import Rot3, Unit3

from averaging.translation.dummy_translation_averaging import \
    DummyTranslationAveraging


class TestTranslationAveragingBase(unittest.TestCase):
    """Main tests for translation averaging base class."""

    def setUp(self):
        super().setUp()

        self.obj = DummyTranslationAveraging()

    def test_computation_graph(self):
        """Test the dask computation graph execution using a valid collection
            of relative unit-translations."""

        num_poses = 3

        i1ti2_dict_normal = {
            (0, 1): Unit3(np.array([0, 0.2, 0])),
            (1, 2): Unit3(np.array([0, 0.1, 0.3])),
        }

        wRi_list_normal = [
            Rot3.RzRyRx(0, 0, 0),
            Rot3.RzRyRx(0, 30*np.pi/180, 0),
            Rot3.RzRyRx(0, 0, 20*np.pi/180),
        ]

        i1ti2_dict_dask = {
            (0, 1): dask.delayed(Unit3)(np.array([0, 0.2, 0])),
            (1, 2): dask.delayed(Unit3)(np.array([0, 0.1, 0.3])),
        }

        wRi_list_dask = [
            dask.delayed(Rot3.RzRyRx)(0, 0, 0),
            dask.delayed(Rot3.RzRyRx)(0, 30*np.pi/180, 0),
            dask.delayed(Rot3.RzRyRx)(0, 0, 20*np.pi/180),
        ]

        # use the normal API (without dask) for rotation averaging
        normal_result = self.obj.run(
            num_poses, i1ti2_dict_normal, wRi_list_normal)

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_poses,
            i1ti2_dict_dask,
            wRi_list_dask
        )

        with dask.config.set(scheduler='single-threaded'):
            dask_result = dask.compute(computation_graph)[0]

        # compare the two entries
        for idx in range(1, num_poses):
            normal_relative_direction = Unit3(normal_result[idx].point3() -
                                              normal_result[0].point3())

            dask_relative_direction = Unit3(dask_result[idx].point3() -
                                            dask_result[0].point3())

            self.assertTrue(dask_relative_direction.equals(
                normal_relative_direction, 1e-5))

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""
        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")
