"""
"""

from gtsam.utils.test_case import GtsamTestCase

from gtsfm.data_association.cpp_dsf_tracks_estimator import CppDsfTracksEstimator
from tests.data_association.test_dsf_tracks_estimator import TestDsfTracksEstimator


class TestDsfTracksEstimator(TestDsfTracksEstimator):
    """ """

    def setUp(self):
        self.estimator = CppDsfTracksEstimator()
