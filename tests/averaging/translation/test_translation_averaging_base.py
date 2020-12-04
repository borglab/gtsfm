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
                                wTi_list: List[Pose3],
                                wTi_list_: List[Pose3]):
        """Helper function to assert that two lists of global Pose3 are equal,
        upto global origin and scale ambiguity.

        Notes:
        1. The input lists have the poses in the same order, and can contain
           None entries.
        2. To resolve global origin ambiguity, we will fix one image index as
           origin in both the inputs and transform both the lists to the new
           origins.
        3. As there is a scale ambiguity, we will use one image index to fix
           the scale ambiguity.
        """

        # check the length of the input lists
        self.assertEqual(len(wTi_list), len(wTi_list_),
                         'two lists to compare have unequal lengths')

        # check the presense of valid Pose3 objects in the same location
        wTi_valid = [i for (i, wTi) in enumerate(wTi_list) if wTi is not None]
        wTi_valid_ = [i for (i, wTi) in enumerate(wTi_list_) if wTi is not None]
        self.assertListEqual(wTi_valid, wTi_valid_)

        if len(wTi_valid) <= 1:
            # we need >= two entries going forward for meaningful comparisons
            return

        # fix the origin for both inputs lists
        origin = wTi_list[wTi_valid[0]]
        origin_ = wTi_list_[wTi_valid_[0]]

        # transform all other valid Pose3 entries to the new coordinate frame
        wTi_list = [wTi_list[i].between(origin) for i in wTi_valid[1:]]
        wTi_list_ = [wTi_list_[i].between(origin_) for i in wTi_valid_[1:]]

        # use the first entry to get the scale factor between two lists
        scale_factor_2to1 = np.linalg.norm(wTi_list[0].translation()) / \
            (np.linalg.norm(wTi_list_[0].translation()) + np.finfo(float).eps)

        # map the poses in the 2nd list using the scale factor on translations
        wTi_list_ = [Pose3(x.rotation(), x.translation() * scale_factor_2to1)
                     for x in wTi_list_]

        for (wTi, wTi_) in zip(wTi_list, wTi_list_):
            self.assertTrue(wTi.equals(wTi_, 1e-1))

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
