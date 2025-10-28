"""Unit tests for the GtsfmData class.

Authors: Ayush Baid
"""

import copy
import unittest
import unittest.mock as mock
from pathlib import Path

import gtsam  # type: ignore
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point2, Point3, Pose3, SfmData, SfmTrack

import gtsfm.utils.graph as graph_utils
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"  # Example data with 3 cameras and 7 tracks.
EXAMPLE_DATA = GtsfmData.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))

NULL_DATA = SfmData()

# create example with non-consecutive cams
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS = GtsfmData(number_images=5)
cam0 = EXAMPLE_DATA.get_camera(0)
cam1 = EXAMPLE_DATA.get_camera(1)
cam2 = EXAMPLE_DATA.get_camera(2)
assert cam0 is not None and cam1 is not None and cam2 is not None, "Camera(s) not found in EXAMPLE_DATA"
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=0, camera=cam0)
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=2, camera=cam1)
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=3, camera=cam2)

EQUALITY_TOLERANCE = 1e-5

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


class TestGtsfmData(unittest.TestCase):
    """Unit tests for GtsfmData."""

    def test_equality_with_same_data(self) -> None:
        """Test equality with the same data (same value but not same object)"""
        self.assertEqual(EXAMPLE_DATA, copy.deepcopy(EXAMPLE_DATA))

    def test_equality_with_different_object(self) -> None:
        """Test equality with different data."""
        self.assertNotEqual(EXAMPLE_DATA, EXAMPLE_WITH_NON_CONSECUTIVE_CAMS)

    def testEqualsWithDifferentObject(self) -> None:
        """Test the equality function with different object, expecting false result."""
        other_example_file = "dubrovnik-1-1-pre.txt"
        other_data = GtsfmData.read_bal(gtsam.findExampleDataFile(other_example_file))

        self.assertNotEqual(EXAMPLE_DATA, other_data)

    def testEqualsWithNullObject(self) -> None:
        """Tests equality of null object with itself and other valid object."""
        self.assertEqual(NULL_DATA, NULL_DATA)
        self.assertNotEqual(NULL_DATA, EXAMPLE_DATA)

    def test_number_images(self) -> None:
        """Test for total number of images."""
        self.assertEqual(EXAMPLE_DATA.number_images(), 3)

    def test_number_tracks(self) -> None:
        """Test for number of tracks."""
        self.assertEqual(EXAMPLE_DATA.number_tracks(), 7)

    def test_get_valid_camera_indices_on_consecutive_indices(self) -> None:
        """Tests on getter for valid camera indices when input has consecutive indices."""
        expected = [0, 1, 2]
        self.assertListEqual(EXAMPLE_DATA.get_valid_camera_indices(), expected)

    def test_get_valid_camera_indices_on_nonconsecutive_indices(self) -> None:
        """Test on getter for valid cameras indices when input has non-consecutive indices."""
        expected = [0, 2, 3]
        self.assertListEqual(EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.get_valid_camera_indices(), expected)

    def test_get_camera_valid(self) -> None:
        """Test for get_camera for a valid index."""
        expected = PinholeCameraCal3Bundler(Pose3(), Cal3Bundler(fx=900, k1=0, k2=0, u0=100, v0=100))

        i = 0
        test_data = GtsfmData(1)
        test_data.add_camera(index=i, camera=expected)

        computed = test_data.get_camera(i)
        assert computed is not None
        self.assertTrue(computed.equals(expected, EQUALITY_TOLERANCE))  # type: ignore

    def test_get_camera_invalid(self):
        """Test for get_camera for an index where the camera does not exist."""

        computed = EXAMPLE_DATA.get_camera(10)
        self.assertIsNone(computed)

    def test_get_track(self) -> None:
        """Testing getter for track."""
        expected_track = SfmTrack(Point3(6.41689062, 0.38897032, -23.58628273))
        expected_track.addMeasurement(0, Point2(383.88000488, 15.2999897))
        expected_track.addMeasurement(1, Point2(559.75, 106.15000153))

        computed = EXAMPLE_DATA.get_track(1)

        # comparing just the point because track equality is failing
        np.testing.assert_allclose(computed.point3(), expected_track.point3())

    def test_add_track_with_valid_cameras(self) -> None:
        """Testing track addition when all cameras in track are already present."""

        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(Point3(0, -2.0, 5.0))
        track_to_add.addMeasurement(idx=0, m=Point2(20.0, 5.0))
        track_to_add.addMeasurement(idx=1, m=Point2(60.0, 50.0))

        self.assertTrue(gtsfm_data.add_track(track_to_add))

    def test_add_track_with_nonexistent_cameras(self) -> None:
        """Testing track addition where some cameras are not in tracks, resulting in failure."""
        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(Point3(0, -2.0, 5.0))
        track_to_add.addMeasurement(idx=0, m=Point2(20.0, 5.0))
        track_to_add.addMeasurement(idx=3, m=Point2(60.0, 50.0))  # this camera does not exist

        self.assertFalse(gtsfm_data.add_track(track_to_add))

    def testGetTrackLengthStatistics(self) -> None:
        """Test computation of mean and median track length."""
        expected_mean_length = 2.7142857142857144
        expected_median_length = 3.0

        # 7 tracks have length [3,2,3,3,3,2,3]
        gtsfm_data = GtsfmData.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))
        mean_length, median_length = gtsfm_data.get_track_length_statistics()

        self.assertEqual(mean_length, expected_mean_length)
        self.assertEqual(median_length, expected_median_length)

    def test_pick_cameras(self) -> None:
        """Test picking cameras."""

        obj = copy.deepcopy(EXAMPLE_DATA)
        # add a new track with just camera 0 and 2
        track_to_add = SfmTrack(Point3(0, -2.0, 5.0))
        track_to_add.addMeasurement(idx=0, m=Point2(20.0, 5.0))
        track_to_add.addMeasurement(idx=2, m=Point2(60.0, 50.0))
        obj.add_track(track_to_add)

        # pick the cameras at index 0 and 2, and hence dropping camera at index 1.
        cams_to_pick = [0, 2]
        computed = GtsfmData.from_selected_cameras(obj, cams_to_pick)

        # test the camera has actually been dropped
        self.assertListEqual(computed.get_valid_camera_indices(), cams_to_pick)

        # test the camera objects
        self.assertEqual(computed.get_camera(0), obj.get_camera(0))
        self.assertEqual(computed.get_camera(2), obj.get_camera(2))

        # check the track
        self.assertEqual(computed.number_tracks(), 1)
        self.assertTrue(computed.get_track(0).equals(track_to_add, EQUALITY_TOLERANCE))

    @mock.patch.object(graph_utils, "get_nodes_in_largest_connected_component", return_value=[1, 2, 4])
    def test_select_largest_connected_component(self, graph_largest_cc_mock):
        """Test pruning to largest connected component according to tracks.

        The function under test calls the graph utility,
        which has been mocked and we test the call against the mocked object.
        """
        gtsfm_data = GtsfmData(5)
        cam = PinholeCameraCal3Bundler(Pose3(), Cal3Bundler())

        # add the same camera at all indices
        for i in range(gtsfm_data.number_images()):
            gtsfm_data.add_camera(i, cam)

        # add two tracks to create two connected components
        track_1 = SfmTrack(np.random.randn(3))  # track with 2 cameras, which will be dropped
        track_1.addMeasurement(idx=0, m=np.random.randn(2))
        track_1.addMeasurement(idx=3, m=np.random.randn(2))

        track_2 = SfmTrack(np.random.randn(3))  # track with 3 cameras, which will be retained
        track_2.addMeasurement(idx=1, m=np.random.randn(2))
        track_2.addMeasurement(idx=2, m=np.random.randn(2))
        track_2.addMeasurement(idx=4, m=np.random.randn(2))

        gtsfm_data.add_track(track_1)
        gtsfm_data.add_track(track_2)

        largest_component_data = gtsfm_data.select_largest_connected_component()

        # check the graph util function called with the edges defined by tracks
        graph_largest_cc_mock.assert_called_once_with([(0, 3), (1, 2), (1, 4), (2, 4)])

        # check the expected cameras coming just from track_2
        expected_camera_indices = [1, 2, 4]
        computed_camera_indices = largest_component_data.get_valid_camera_indices()
        self.assertListEqual(computed_camera_indices, expected_camera_indices)

        # check that there is just one track
        expected_num_tracks = 1
        computed_num_tracks = largest_component_data.number_tracks()
        self.assertEqual(computed_num_tracks, expected_num_tracks)

        # check the exact track
        computed_track = largest_component_data.get_track(0)
        self.assertTrue(computed_track.equals(track_2, EQUALITY_TOLERANCE))

    def test_filter_landmarks(self) -> None:
        """Tests filtering of SfmData based on reprojection error."""
        max_reproj_error = 15

        VALID_TRACK_INDICES = [0, 1, 5]

        # construct expected data w/ tracks with reprojection errors below the
        # threshold
        expected_data = GtsfmData(EXAMPLE_DATA.number_images())
        for i in EXAMPLE_DATA.get_valid_camera_indices():
            camera_i = EXAMPLE_DATA.get_camera(i)
            assert camera_i is not None
            expected_data.add_camera(i, camera_i)

        for j in VALID_TRACK_INDICES:
            expected_data.add_track(EXAMPLE_DATA.get_track(j))

        # run the fn under test
        filtered_sfm_data, valid_mask = EXAMPLE_DATA.filter_landmarks(max_reproj_error)
        self.assertEqual(sum(valid_mask), 3)

        # compare the SfmData objects
        self.assertEqual(filtered_sfm_data, expected_data)

    def test_read_bal(self) -> None:
        """Check that read_bal creates correct GtsfmData object."""
        filename: str = gtsam.findExampleDataFile("5pointExample1.txt")
        data: GtsfmData = GtsfmData.read_bal(filename)
        self.assertEqual(data.number_images(), 2)
        self.assertEqual(data.number_tracks(), 5)

    def test_read_bundler(self) -> None:
        """Check that read_bundler creates correct GtsfmData object."""
        filename: str = gtsam.findExampleDataFile("Balbianello.out")
        data: GtsfmData = GtsfmData.read_bundler(filename)
        self.assertEqual(data.number_images(), 5)
        self.assertEqual(data.number_tracks(), 544)

    def test_read_colmap(self) -> None:
        """Test reading a COLMAP scene using GtsfmData.read_colmap."""
        data_dir = TEST_DATA_ROOT / "crane_mast_8imgs_colmap_output"
        gtsfm_data = GtsfmData.read_colmap(str(data_dir))
        # Check number of images and tracks
        self.assertEqual(gtsfm_data.number_images(), 8)
        self.assertEqual(gtsfm_data.number_tracks(), 2122)
        # Check camera types and poses
        for i in range(gtsfm_data.number_images()):
            camera_i = gtsfm_data.get_camera(i)
            assert camera_i is not None
            self.assertIsNotNone(camera_i)
            self.assertIsInstance(camera_i.pose(), Pose3)

        # Check track types
        self.assertTrue(all(isinstance(track, SfmTrack) for track in gtsfm_data.get_tracks()))
        # Check image dimensions for first camera
        cam0 = gtsfm_data.get_camera(0)
        self.assertIsNotNone(cam0)
        assert cam0 is not None
        self.assertEqual(cam0.calibration().px(), 2028)
        self.assertEqual(cam0.calibration().py(), 1520)


if __name__ == "__main__":
    unittest.main()
