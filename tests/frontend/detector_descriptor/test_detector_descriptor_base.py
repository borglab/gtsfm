"""
Tests for frontend's base detector-descriptor class.

Authors: Ayush Baid
"""
import unittest

import dask
import numpy as np

import tests.frontend.detector.test_detector_base as test_detector_base
from frontend.descriptor.dummy_descriptor import DummyDescriptor
from frontend.detector.detector_from_joint_detector_descriptor import \
    DetectorFromDetectorDescriptor
from frontend.detector.dummy_detector import DummyDetector
from frontend.detector_descriptor.combination_detector_descriptor import \
    CombinationDetectorDescriptor


class TestDetectorDescriptorBase(test_detector_base.TestDetectorBase):
    """
    Main test class for detector-description combination base class in frontend.

    We re-use detector specific test cases from TestDetectorBase
    """

    def setUp(self):
        """Setup the attributes for the tests."""
        super().setUp()
        self.detector_descriptor = CombinationDetectorDescriptor(
            DummyDetector(),
            DummyDescriptor()
        )

        # explicitly set the detector
        self.detector = DetectorFromDetectorDescriptor(
            self.detector_descriptor)

    def test_detect_and_describe_shape(self):
        """
        Tests that the number of keypoints and descriptors are the same.
        """

        # test on random indexes
        test_indices = [0, 5]
        for idx in test_indices:
            kps, descs = \
                self.detector_descriptor.detect_and_describe(
                    self.loader.get_image(idx))

            if len(kps) == 0:
                # test-case for empty results
                self.assertEqual(0, descs.size)
            else:
                # number of descriptors and features should be equal
                self.assertEqual(len(kps), descs.shape[0])

    def test_computation_graph(self):
        """Test the dask's computation graph formation."""

        image_graph = self.loader.create_computation_graph_for_images()
        detection_graph, description_graph = \
            self.detector_descriptor.create_computation_graph(image_graph)

        with dask.config.set(scheduler='single-threaded'):
            # TODO(ayush): check how many times detection is performed
            detection_results = dask.compute(detection_graph)[0]
            description_results = dask.compute(description_graph)[0]

        # check the number of entries in results
        self.assertEqual(
            len(detection_results), len(self.loader),
            "Dask workflow does not return the same number of results"
        )
        self.assertEqual(
            len(description_results), len(self.loader),
            "Dask workflow does not return the same number of results"
        )

        # compare the result values for a particular image
        idx_under_test = 0

        # check the results via normal workflow and dask workflow for an image
        expected_kps, expected_descs = \
            self.detector_descriptor.detect_and_describe(
                self.loader.get_image(idx_under_test))

        self.assertEqual(detection_results[idx_under_test], expected_kps)
        np.testing.assert_array_equal(
            description_results[idx_under_test], expected_descs)


if __name__ == '__main__':
    unittest.main()
