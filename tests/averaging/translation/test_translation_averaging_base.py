"""Tests for translation averaging base class.

Authors: Ayush Baid
"""
import pickle
import unittest

import dask
import numpy as np
from gtsam import Pose3, Rot3, Unit3

import utils.geometry_comparisons as geometry_comparisons
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
        expected_wti_list = self.obj.run(num_images, i2Ui1_dict, wRi_list)
        expected_wTi_list = [Pose3(wRi, wti)
                             if wti is not None else None
                             for (wRi, wti) in zip(wRi_list, expected_wti_list)]

        # use dask's computation graph
        computation_graph = self.obj.create_computation_graph(
            num_images, i2Ui1_graph, wRi_graph)

        with dask.config.set(scheduler='single-threaded'):
            wti_list = dask.compute(computation_graph)[0]

        wTi_list = [Pose3(wRi, wti)
                    if wti is not None else None
                    for (wRi, wti) in zip(wRi_list, wti_list)]

        self.assertTrue(geometry_comparisons.compare_global_poses(
            wTi_list, expected_wTi_list))

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == '__main__':
    unittest.main()
