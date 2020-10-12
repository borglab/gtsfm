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
    """Main tests for translation averaging base class."""

    def setUp(self):
        super().setUp()

        self.obj = DummyTranslationAveraging()

    def check_equals(self,
                     wTi_list_1: List[Point3],
                     wTi_list_2: List[Point3]):
        """Compares if two lists of global roations are equal (upto global
        scale ambiguity)."""

        self.assertEqual(len(wTi_list_1), len(wTi_list_2))

        for idx in range(1, len(wTi_list_1)):
            # accounting for ambuiguity in origin of the coordinate system.
            direction_w_0tidx_1 = Unit3(wTi_list_1[idx] - wTi_list_1[0])

            direction_w_0tidx_2 = Unit3(wTi_list_2[idx] - wTi_list_2[0])

            self.assertTrue(direction_w_0tidx_1.equals(
                direction_w_0tidx_2, 1e-5))

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

        self.check_equals(normal_result, dask_result)

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""
        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")
