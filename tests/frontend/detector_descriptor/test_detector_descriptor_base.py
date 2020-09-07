"""
Tests for frontend's base detector-descriptor class.

Authors: Ayush Baid
"""
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
        Tests that the number of features and descriptors are the same.
        """

        # test on random indexes
        test_indices = [0, 5]
        for idx in test_indices:
            features, descriptors = self.detector_descriptor.detect_and_describe(
                self.loader.get_image(idx))

            if features.size == 0:
                # test-case for empty results
                self.assertEqual(0, descriptors.size)
            else:
                # number of descriptors and features should be equal
                self.assertEqual(features.shape[0], descriptors.shape[0])

    def test_computation_graph(self):
        """
        Test the dask's computation graph formation using a single image.
        """

        loader_graph = self.loader.create_computation_graph()
        computation_graph = self.detector_descriptor.create_computation_graph(
            loader_graph)

        results = []
        with dask.config.set(scheduler='single-threaded'):
            results = dask.compute(computation_graph)[0]

        # check the number of results
        self.assertEqual(len(results), len(self.loader),
                         "Dask workflow does not return the same number of results"
                         )

        # check the results via normal workflow and dask workflow for an image
        normal_features, normal_descriptors = self.detector_descriptor.detect_and_describe(
            self.loader.get_image(0))
        dask_features, dask_descriptors = results[0]

        np.testing.assert_allclose(normal_features, dask_features)
        np.testing.assert_allclose(normal_descriptors, dask_descriptors)
