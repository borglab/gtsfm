"""Tests for translation averaging base class.

Authors: Ayush Baid
"""
import pickle
import unittest
from typing import List

import dask
import numpy as np
from gtsam import Pose3, Rot3, Unit3

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
                                wTi_list1: List[Pose3],
                                wTi_list2: List[Pose3]):
        """Helper function to assert that two lists of global Pose3 are equal,
        upto global pose ambiguity."""

        self.assertEqual(len(wTi_list1), len(wTi_list2),
                         'two lists to compare have unequal lengths')

        # get a referenc pose to resolve ambiguity
        reference_idx = -1
        for i in range(len(wTi_list1)):
            if wTi_list1[i] is not None and wTi_list2[i] is not None:
                reference_idx = i
                break

        if reference_idx == -1:
            # all entries in the list should be none
            for pose in wTi_list1:
                self.assertIsNone(pose)

            for pose in wTi_list2:
                self.assertIsNone(pose)

            return

        scale_factor_2to1 = None

        for i in range(len(wTi_list1)):
            if i == reference_idx:
                continue

            if wTi_list1[i] is None:
                self.assertIsNone(wTi_list2[i])
            else:
                pose_1 = wTi_list1[i].between(wTi_list1[reference_idx])
                pose_2 = wTi_list2[i].between(wTi_list2[reference_idx])

            if scale_factor_2to1 is None:
                # resolve the scale factor by using one measurement
                scale_factor_2to1 = np.linalg.norm(pose_1.translation()) /\
                    (np.linalg.norm(pose_2.translation()) + np.finfo(float).eps)

            # assert equality upto scale
            self.assertTrue(pose_1.rotation().equals(pose_2.rotation(), 1e-3))
            np.testing.assert_allclose(
                pose_1.translation(),
                pose_2.translation()*scale_factor_2to1,
                atol=1e-1,
                rtol=1e-1)

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

        self.assert_equal_upto_scale(wTi_list, expected_wTi_list)

    def test_pickleable(self):
        """Tests that the object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.obj)
        except TypeError:
            self.fail("Cannot dump rotation averaging object using pickle")


if __name__ == '__main__':
    unittest.main()
