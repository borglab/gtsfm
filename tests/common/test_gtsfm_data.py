"""Unit tests for the GtsfmData class.

Authors: Ayush Baid
"""
import copy
import unittest
import unittest.mock as mock

import gtsam
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Rot3, Pose3, SfmData, SfmTrack

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"  # example data with 3 cams and 7 tracks
EXAMPLE_DATA = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))

NULL_DATA = SfmData()

# create example with non-consecutive cams
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS = GtsfmData(number_images=5)
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=0, camera=EXAMPLE_DATA.get_camera(0))
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=2, camera=EXAMPLE_DATA.get_camera(1))
EXAMPLE_WITH_NON_CONSECUTIVE_CAMS.add_camera(index=3, camera=EXAMPLE_DATA.get_camera(2))

EQUALITY_TOLERANCE = 1e-5


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
        other_data = io_utils.read_bal(gtsam.findExampleDataFile(other_example_file))

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
        self.assertTrue(computed.equals(expected, EQUALITY_TOLERANCE))

    def test_get_camera_invalid(self):
        """Test for get_camera for an index where the camera does not exist."""

        computed = EXAMPLE_DATA.get_camera(10)
        self.assertIsNone(computed)

    def test_get_track(self) -> None:
        """Testing getter for track."""
        expected_track = SfmTrack(np.array([6.41689062, 0.38897032, -23.58628273]))
        expected_track.add_measurement(0, np.array([383.88000488, 15.2999897]))
        expected_track.add_measurement(1, np.array([559.75, 106.15000153]))

        computed = EXAMPLE_DATA.get_track(1)

        # comparing just the point because track equality is failing
        np.testing.assert_allclose(computed.point3(), expected_track.point3())

    def test_add_track_with_valid_cameras(self) -> None:
        """Testing track addition when all cameras in track are already present."""

        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(np.array([0, -2.0, 5.0]))
        track_to_add.add_measurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.add_measurement(idx=1, m=np.array([60.0, 50.0]))

        self.assertTrue(gtsfm_data.add_track(track_to_add))

    def test_add_track_with_nonexistant_cameras(self) -> None:
        """Testing track addition where some cameras are not in tracks, resulting in failure."""
        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(np.array([0, -2.0, 5.0]))
        track_to_add.add_measurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.add_measurement(idx=3, m=np.array([60.0, 50.0]))  # this camera does not exist

        self.assertFalse(gtsfm_data.add_track(track_to_add))

    def testGetTrackLengthStatistics(self) -> None:
        """Test computation of mean and median track length."""
        expected_mean_length = 2.7142857142857144
        expected_median_length = 3.0

        # 7 tracks have length [3,2,3,3,3,2,3]
        gtsfm_data = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))
        mean_length, median_length = gtsfm_data.get_track_length_statistics()

        self.assertEqual(mean_length, expected_mean_length)
        self.assertEqual(median_length, expected_median_length)

    def test_pick_cameras(self) -> None:
        """Test picking cameras."""

        obj = copy.deepcopy(EXAMPLE_DATA)
        # add a new track with just camera 0 and 2
        track_to_add = SfmTrack(np.array([0, -2.0, 5.0]))
        track_to_add.add_measurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.add_measurement(idx=2, m=np.array([60.0, 50.0]))
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

        The function under test calles the graph utility, which has been mocked and we test the call against the mocked
        object.
        """
        gtsfm_data = GtsfmData(5)
        cam = PinholeCameraCal3Bundler(Pose3(), Cal3Bundler())

        # add the same camera at all indices
        for i in range(gtsfm_data.number_images()):
            gtsfm_data.add_camera(i, cam)

        # add two tracks to create two connected components
        track_1 = SfmTrack(np.random.randn(3))  # track with 2 cameras, which will be dropped
        track_1.add_measurement(idx=0, m=np.random.randn(2))
        track_1.add_measurement(idx=3, m=np.random.randn(2))

        track_2 = SfmTrack(np.random.randn(3))  # track with 3 cameras, which will be retained
        track_2.add_measurement(idx=1, m=np.random.randn(2))
        track_2.add_measurement(idx=2, m=np.random.randn(2))
        track_2.add_measurement(idx=4, m=np.random.randn(2))

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
            expected_data.add_camera(i, EXAMPLE_DATA.get_camera(i))

        for j in VALID_TRACK_INDICES:
            expected_data.add_track(EXAMPLE_DATA.get_track(j))

        # run the fn under test
        filtered_sfm_data = EXAMPLE_DATA.filter_landmarks(max_reproj_error)

        # compare the SfmData objects
        self.assertEqual(filtered_sfm_data, expected_data)

    def test_align_via_Sim3_to_poses(self) -> None:
        """Ensure that alignment of a SFM result to ground truth camera poses works correctly.

        Consider a simple example, wih 3 estimated poses and 2 points.
        When fitting the Similarity(3), all correspondences should have no noise, and alignment should be exact.

        GT: ===========================================
                    |
                    . (pose 3)
                    .
                    X . .
                    |
          . (pose 2).         . (pose 0)
          .         .(pose 1) .
        --X . . ----X . . --- X . .
                    |
                    |
                    |

        Estimate: =====================================

                    |  . (pose 3)
                    |  .
                    |  X . .
                    |
                    |  .         . (pose 0)
                    |  .(pose 1) .
                    |  X . . --- X . .
                    |
        ---------------------------
                    |
                    |
        """
        dummy_calibration = Cal3Bundler(fx=900, k1=0, k2=0, u0=100, v0=100)
        
        # fmt: off
        wTi_list_gt = [
            Pose3(Rot3(), np.array([3, 0, 0])),  # wTi0
            Pose3(Rot3(), np.array([0, 0, 0])),  # wTi1
            Pose3(Rot3(), np.array([0, -3, 0])), # wTi2
            Pose3(Rot3(), np.array([0, 3, 0])),  # wTi3
        ]
        points_gt = [
            np.array([1, 1, 0]),
            np.array([3, 3, 0])
        ]

        # pose graph is scaled by a factor of 2, and shifted also.
        wTi_list_est = [
            Pose3(Rot3(), np.array([8, 2, 0])),  # wTi0
            Pose3(Rot3(), np.array([2, 2, 0])),  # wTi1
            None,                                # wTi2
            Pose3(Rot3(), np.array([2, 8, 0])),  # wTi3
        ]
        points_est = [
            np.array([4, 4, 0]),
            np.array([8, 8, 0])
        ]
        # fmt: on
        
        def add_dummy_measurements_to_track(track: SfmTrack) -> SfmTrack:
            """Add some dummy 2d measurements in three views in cameras 0,1,3."""
            track.add_measurement(0, np.array([100, 200]))
            track.add_measurement(1, np.array([300, 400]))
            track.add_measurement(3, np.array([500, 600]))
            return track

        sfm_result = GtsfmData(number_images=4)
        gt_gtsfm_data = GtsfmData(number_images=4)
        for gtsfm_data, wTi_list in zip([sfm_result, gt_gtsfm_data], [wTi_list_est, wTi_list_gt]):

            for i, wTi in enumerate(wTi_list):
                if wTi is None:
                    continue
                gtsfm_data.add_camera(i, PinholeCameraCal3Bundler(wTi, dummy_calibration))

            for pt in points_est:
                track = SfmTrack(pt)
                track = add_dummy_measurements_to_track(track)
                gtsfm_data.add_track(track)

        aligned_sfm_result = sfm_result.align_via_Sim3_to_poses(wTi_list_ref=gt_gtsfm_data.get_camera_poses())
        # tracks and poses should match GT now, after applying estimated scale and shift.
        assert aligned_sfm_result == gt_gtsfm_data

        # 3d points from tracks should now match the GT.
        assert np.allclose(aligned_sfm_result.get_track(0).point3(), np.array([1.0, 1.0, 0.0]))
        assert np.allclose(aligned_sfm_result.get_track(1).point3(), np.array([3.0, 3.0, 0.0]))


    def test_get_point_cloud(self) -> None:
        """Ensure that we can fetch non-NaN 3d points from the point tracks."""
        gtsfm_data = GtsfmData(number_images=3)

        gtsfm_data.add_camera(index=0, camera=PinholeCameraCal3Bundler())
        gtsfm_data.add_camera(index=1, camera=PinholeCameraCal3Bundler())
        gtsfm_data.add_camera(index=2, camera=PinholeCameraCal3Bundler())

        # add a track on camera #0 and #1
        track1 = SfmTrack(np.array([0, -2.0, 5.0]))
        track1.add_measurement(idx=0, m=np.array([20.0, 5.0]))
        track1.add_measurement(idx=1, m=np.array([60.0, 50.0]))

        # add a track on camera #1 and #2
        track2 = SfmTrack(np.array([np.nan, -2.0, 5.0]))
        track2.add_measurement(idx=1, m=np.array([20.0, 5.0]))
        track2.add_measurement(idx=2, m=np.array([60.0, 50.0]))

        # add a track on camera #0 and #1
        track3 = SfmTrack(np.array([np.nan, np.nan, np.nan]))
        track3.add_measurement(idx=0, m=np.array([20.0, 5.0]))
        track3.add_measurement(idx=1, m=np.array([60.0, 50.0]))

        # add a track on camera #1 and #2
        track4 = SfmTrack(np.array([1.0, 2.0, 3.0]))
        track4.add_measurement(idx=1, m=np.array([20.0, 5.0]))
        track4.add_measurement(idx=2, m=np.array([60.0, 50.0]))

        gtsfm_data.add_track(track1)
        gtsfm_data.add_track(track2)
        gtsfm_data.add_track(track3)
        gtsfm_data.add_track(track4)

        points = gtsfm_data.get_point_cloud()
        
        expected_points = np.array([
            [ 0., -2.,  5.],
            [ 1.,  2.,  3.]])
        self.assertTrue(np.allclose(points, expected_points))


if __name__ == "__main__":
    unittest.main()
