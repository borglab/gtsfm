"""Unit tests for the CppDsfTracksEstimator class."""


from gtsfm.data_association.cpp_dsf_tracks_estimator import CppDsfTracksEstimator
from tests.data_association.test_dsf_tracks_estimator import TestDsfTracksEstimator


class TestCppDsfTracksEstimator(TestDsfTracksEstimator):
    """ """

    def setUp(self):
        self.estimator = CppDsfTracksEstimator()
