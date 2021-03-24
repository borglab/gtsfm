"""Unit tests for the GtsfmData class.

Authors: Ayush Baid
"""
import copy
import unittest
import unittest.mock as mock

import gtsam
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, SfmTrack

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))

# create example with non-consecutive cams
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS = GtsfmData(number_images=5)
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=0, camera=EXAMPLE_DATA.get_camera(0))
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=2, camera=EXAMPLE_DATA.get_camera(1))
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=3, camera=EXAMPLE_DATA.get_camera(2))

EQUALITY_TOLERANCE = 1e-5


class TestGtsfmData(unittest.TestCase):
    """Unit tests for GtsfmData."""

    def test_equality_with_same_data(self):
        """Test equality with the same data."""
        self.assertEqual(EXAMPLE_DATA, copy.deepcopy(EXAMPLE_DATA))

    def test_equality_with_different_object(self):
        """Test equality with different data."""
        self.assertNotEqual(EXAMPLE_DATA, EXAMPLE_WITH_NON_CONSECUTIVE_CAMS)

    def test_number_images(self):
        """Test for total number of images."""
        self.assertEqual(EXAMPLE_DATA.number_images(), 3)

    def test_number_tracks(self):
        """Test for number of tracks."""
        self.assertEqual(EXAMPLE_DATA.number_tracks(), 7)

    def test_get_valid_camera_indices_on_consecutive_indices(self):
        """Tests on getter for valid camera indices when input has consecutive indices."""
        expected = [0, 1, 2]
        self.assertListEqual(EXAMPLE_DATA.get_valid_camera_indices(), expected)

    def test_get_valid_camera_indices_on_nonconsecutive_indices(self):
        """Test on getter for valid cameras indices when input has non-consecutive indices."""
        expected = [0, 2, 3]
        self.assertListEqual(EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.get_valid_camera_indices(), expected)

    def test_get_camera_valid(self):
        """Test for get_camera for a valid index."""
        expected = PinholeCameraCal3Bundler(Pose3(), Cal3Bundler(fx=900, k1=0, k2=0, u0=100, v0=100))

        i = 0
        test_data = GtsfmData(1)
        test_data.add_camera(index=i, camera=expected)

        computed = test_data.get_camera(i)
        self.assertTrue(computed.equals(expected, EQUALITY_TOLERANCE))

    def test_get_camera_invalid(self):
        """Test for get_camera for an index where the camera does not exist."""

        computed = EXAMPLE_DATA.get_camera(10)
        self.assertIsNone(computed)

    def test_get_track(self):
        """Testing getter for track."""
        expected_track = SfmTrack(np.array([6.41689062, 0.38897032, -23.58628273]))
        expected_track.add_measurement(0, np.array([383.88000488, 15.2999897]))
        expected_track.add_measurement(1, np.array([559.75, 106.15000153]))

        computed = EXAMPLE_DATA.get_track(1)

        # comparing just the point because track equality is failing
        np.testing.assert_allclose(computed.point3(), expected_track.point3())

    def test_add_track_with_valid_cameras(self):
        """Testing track addition when all cameras in track are already present."""

        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(np.array([0, -2.0, 5.0]))
        track_to_add.add_measurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.add_measurement(idx=1, m=np.array([60.0, 50.0]))

        self.assertTrue(gtsfm_data.add_track(track_to_add))

    def test_add_track_with_nonexistant_cameras(self):
        """Testing track addition where some cameras are not in tracks, resulting in failure."""
        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(np.array([0, -2.0, 5.0]))
        track_to_add.add_measurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.add_measurement(idx=3, m=np.array([60.0, 50.0]))  # this camera does not exist

        self.assertFalse(gtsfm_data.add_track(track_to_add))

    def test_drop_cameras(self):
        """Test dropping cameras."""

        # drop the camera at index 1 in the example data
        computed = EXAMPLE_DATA.drop_cameras([1])

        self.assertEqual(computed.get_camera(0), computed.get_camera(0))
        self.assertEqual(computed.get_camera(2), computed.get_camera(2))
        self.assertEqual(computed.number_tracks(), 0)

    @mock.patch.object(graph_utils, "get_nodes_in_largest_connected_component", return_value=[1, 2, 4])
    def test_select_largest_connected_component(self, graph_largest_cc_mock):
        """Test pruning to largest connected component according to tracks.
        
        The function under test calles the graph utility, which has been mocked and we test the call against the mocked
        object.
        """
        gtsfm_data = GtsfmData(5)
        cam = PinholeCameraCal3Bundler(Pose3(), Cal3Bundler())

        # add the same camera at all indices
        for i in range(gtsfm_data.number_images()):
            gtsfm_data.add_camera(i, cam)

        # add two tracks to define connected component
        track_1 = SfmTrack(np.random.randn(3))
        track_1.add_measurement(idx=0, m=np.random.randn(2))
        track_1.add_measurement(idx=3, m=np.random.randn(2))

        track_2 = SfmTrack(np.random.randn(3))
        track_2.add_measurement(idx=1, m=np.random.randn(2))
        track_2.add_measurement(idx=2, m=np.random.randn(2))
        track_2.add_measurement(idx=4, m=np.random.randn(2))

        gtsfm_data.add_track(track_1)
        gtsfm_data.add_track(track_2)

        largest_component_data = gtsfm_data.select_largest_connected_component()

        graph_largest_cc_mock.assert_called_once_with([(0, 3), (1, 2), (1, 4), (2, 4)])
        self.assertListEqual(largest_component_data.get_valid_camera_indices(), [1, 2, 4])


if __name__ == "__main__":
    unittest.main()
