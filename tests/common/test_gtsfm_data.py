"""Unit tests for the GtsfmData class.

Authors: Ayush Baid
"""
import copy
import unittest
import unittest.mock as mock

import gtsam
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Rot3, Point3, Pose3, SfmData, SfmTrack

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.metrics as metrics_utils
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

    def test_align_via_Sim3_to_poses_skydio32(self) -> None:
        """Real data, from Skydio-32 sequence with the SIFT front-end.
        
        All tracks should have NaN reprojection error.
        """

        fx, k1, k2, px, py = 609.94, -8.82687675e-07, -4.25111629e-06, 507, 380
        calib = Cal3Bundler(fx, k1, k2, px, py)

        wTi2 = Pose3(
            Rot3(
                [
                    [0.95813172, 0.00187844098, -0.286321635],
                    [-0.00477926501, 0.999944088, -0.00943285149],
                    [0.286287907, 0.0104063212, 0.958087127],
                ]
            ),
            Point3(-0.763405555, -0.266165811, -0.118153783),
        )
        wTi3 = Pose3(
            Rot3(
                [
                    [0.999985712, 0.00523414604, -0.00108604007],
                    [-0.00524091331, 0.999966264, -0.00632478431],
                    [0.00105289859, 0.00633038578, 0.999979409],
                ]
            ),
            Point3(-0.76321922 - 0.266172481 - 0.118146617),
        )
        wTi4 = Pose3(
            Rot3(
                [
                    [0.963327067, 0.00797924699, 0.268211287],
                    [-0.0070859796, 0.999965656, -0.00429831651],
                    [-0.268236373, 0.00224014493, 0.963350523],
                ]
            ),
            Point3(-0.76303985 - 0.266175812 - 0.11817041),
        )
        wTi5 = Pose3(
            Rot3(
                [
                    [0.965261642, 0.00828550155, 0.261153812],
                    [-0.00485679326, 0.99989337, -0.0137717378],
                    [-0.261240071, 0.0120249602, 0.965198956],
                ]
            ),
            Point3(-0.762848965 - 0.266179234 - 0.118209764),
        )

        wTi30 = Pose3(
            Rot3(
                [
                    [0.761797609, -0.0189987841, 0.647536446],
                    [0.000647893104, 0.999591701, 0.0285659033],
                    [-0.647814775, -0.0213419024, 0.761498877],
                ]
            ),
            Point3(-0.780003719, -0.266108015, -0.154604541),
        )

        aligned_cameras = {
            2: PinholeCameraCal3Bundler(wTi2, calib),
            3: PinholeCameraCal3Bundler(wTi3, calib),
            4: PinholeCameraCal3Bundler(wTi4, calib),
            5: PinholeCameraCal3Bundler(wTi5, calib),
            30: PinholeCameraCal3Bundler(wTi30, calib),
        }

        """
        , 6: PinholeCameraCal3Bundler(
            Pose3(
                Rot3([
            [0.997285023, 0.00579915148, 0.0734094893],
            [-0.00521261679, 0.999952965, -0.00817897076],
            [-0.0734534676, 0.00777410951, 0.997268345]
        ]
        Point3(   -0.7626597 -0.266172655 -0.118249339)
        calibration
        , 7: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.999955717, 0.00349457331, 0.0087379326],
            [-0.00342733793, 0.999964498, -0.00769782645],
            [-0.00876452301, 0.00766753772, 0.999932194]
        ]
        Point3( -0.762475887  -0.26617569 -0.118282864)
        calibration
        , 8: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.994325635, 0.00242057877, 0.106351643],
            [-0.00195182615, 0.999987919, -0.00451143707],
            [-0.106361279, 0.00427825761, 0.994318347]
        ]),
        Point3( -0.762282917 -0.266174532 -0.118281214)
        calibration
        , 9: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.90271482, -0.0122811439, 0.430064096],
            [0.0143824913, 0.999895229, -0.0016356421],
            [-0.42999895, 0.0076619115, 0.902796875]
        ]),
        Point3(  -0.76211788 -0.266175935  -0.11833603)
        calibration
        


        , 10: PinholeCameraCal3Bundler(
            Pose3(
                Rot3([
            [0.752320757, -0.0393479506, 0.657620877],
            [0.0304635491, 0.999224759, 0.0249369914],
            [-0.658092281, 0.00127284959, 0.752936205]
        ]
        t:  -1.03831226  0.422848075 -0.334846473
        calibration
        , 17: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.96425842, 0.00583707364, 0.264899279],
            [-0.00562481022, 0.999982964, -0.00155985167],
            [-0.264903871, 1.40919399e-05, 0.964274825]
        ]
        t: -0.763034493   -0.2664337 -0.118220314
        calibration
        , 18: PinholeCameraCal3Bundler(
            Pose3(Rot3([
            [0.965579561, 0.0154878197, 0.259646374],
            [-0.0155965006, 0.999877019, -0.00164166671],
            [-0.259639869, -0.00246441501, 0.965702369]
        ])
        t: -0.762870012  -0.26643361 -0.118312972
        )
        calibration
        , 20: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.993825932, 0.0123869074, -0.110256886],
            [-0.0114383725, 0.999891967, 0.00923133309],
            [0.110359323, -0.00791317888, 0.993860253]
        ]),
        t: -0.461535656  0.649651321  0.740870096
        calibration
        , 21: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.999743772, 0.00399265577, -0.0222811461],
            [-0.0037456451, 0.999931191, 0.0111168284],
            [0.0223239986, -0.0110305226, 0.999689935]
        ])
        t: -0.762295127 -0.266429784 -0.118308125
        calibration
        , 22: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.934365821, -0.0117451199, 0.356121558],
            [0.0118014686, 0.999928331, 0.00201445032],
            [-0.356119695, 0.00232052386, 0.934437466]
        ])
        t: -0.762136054 -0.266417059 -0.118342729
        calibration
        , 23: PinholeCameraCal3Bundler(Pose3(Rot3([
            [0.80144847, -0.0291248233, 0.597354246],
            [0.0173399444, 0.999525219, 0.0254688464],
            [-0.597812409, -0.0100538786, 0.80157298]
        ])
        t: -0.780007851 -0.266111924   -0.1546234
        calibration
        , 29: PinholeCameraCal3Bundler(Pose3(Rot3([
            0.842740249, 0.00134121124, 0.538318748],
            -0.0106388274, 0.999843085, 0.014164038],
            -0.538215281, -0.0176636852, 0.842622279]
        ])
        t:  -1.03545775  0.171553331 -0.257474986
        calibration
        , 
        }
        """
        t0 = SfmTrack(pt=[-0.7627727,  -0.26624048, -0.11879795])
        t0.add_measurement(2, [184.08586121, 441.31314087])
        t0.add_measurement(4, [ 18.98637581, 453.21853638])

        t1 = SfmTrack(pt=[-0.76277714, -0.26603358, -0.11884205])
        t1.add_measurement(2, [213.51266479, 288.06637573])
        t1.add_measurement(4, [ 50.23059464, 229.30541992])

        t2 = SfmTrack(pt=[-0.7633115,  -0.2662322,  -0.11826181])
        t2.add_measurement(2, [227.52420044, 695.15087891])
        t2.add_measurement(3, [996.67608643, 705.03125   ])

        t3 = SfmTrack(pt=[-0.76323087, -0.26629859, -0.11836833])
        t3.add_measurement(2, [251.37863159, 702.97064209])
        t3.add_measurement(3, [537.9753418,  732.26025391])

        t4 = SfmTrack(pt=[-0.70450081, -0.28115719, -0.19063382])
        t4.add_measurement(2, [253.17749023, 490.47991943])
        t4.add_measurement(3, [ 13.17782784, 507.57717896])

        t5 = SfmTrack(pt=[-0.52781989, -0.31926005, -0.40763909])
        t5.add_measurement(2, [253.52301025, 478.41384888])
        t5.add_measurement(3, [ 10.92995739, 493.31018066])

        t6 = SfmTrack(pt=[-0.74893948, -0.27132075, -0.1360136 ])
        t6.add_measurement(2, [254.64611816, 533.04730225])
        t6.add_measurement(3, [ 18.78449249, 557.05041504])

        aligned_tracks = [t0, t1, t2, t3, t4, t5, t6]
        aligned_filtered_data = GtsfmData.from_cameras_and_tracks(cameras=aligned_cameras, tracks=aligned_tracks, number_images=32)
        metrics = metrics_utils.get_stats_for_sfmdata(aligned_filtered_data, suffix="_filtered")

        assert metrics[0].name == 'number_cameras'
        assert np.isclose(metrics[0]._data, np.array(5., dtype=np.float32))

        assert metrics[1].name == 'number_tracks_filtered'
        assert np.isclose(metrics[1]._data, np.array(7., dtype=np.float32))

        assert metrics[2].name == '3d_track_lengths_filtered'
        assert metrics[2].summary == {'min': 2, 'max': 2, 'median': 2.0, 'mean': 2.0, 'stddev': 0.0, 'histogram': {'1': 7}}

        assert metrics[3].name == 'reprojection_errors_filtered_px'
        assert metrics[3].summary == {'min': np.nan, 'max': np.nan, 'median': np.nan, 'mean': np.nan, 'stddev': np.nan}


if __name__ == "__main__":
    unittest.main()
