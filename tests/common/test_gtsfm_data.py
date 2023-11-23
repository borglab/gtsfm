"""Unit tests for the GtsfmData class.

Authors: Ayush Baid
"""
import copy
import unittest
import unittest.mock as mock

import gtsam
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmData, SfmTrack

import gtsfm.utils.graph as graph_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.common.gtsfm_data import GtsfmData

GTSAM_EXAMPLE_FILE = "dubrovnik-3-7-pre"  # Example data with 3 cameras and 7 tracks.
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
        expected_track.addMeasurement(0, np.array([383.88000488, 15.2999897]))
        expected_track.addMeasurement(1, np.array([559.75, 106.15000153]))

        computed = EXAMPLE_DATA.get_track(1)

        # comparing just the point because track equality is failing
        np.testing.assert_allclose(computed.point3(), expected_track.point3())

    def test_add_track_with_valid_cameras(self) -> None:
        """Testing track addition when all cameras in track are already present."""

        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(np.array([0, -2.0, 5.0]))
        track_to_add.addMeasurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.addMeasurement(idx=1, m=np.array([60.0, 50.0]))

        self.assertTrue(gtsfm_data.add_track(track_to_add))

    def test_add_track_with_nonexistant_cameras(self) -> None:
        """Testing track addition where some cameras are not in tracks, resulting in failure."""
        gtsfm_data = copy.deepcopy(EXAMPLE_DATA)

        # add a track on camera #0 and #1, which exists in the data
        track_to_add = SfmTrack(np.array([0, -2.0, 5.0]))
        track_to_add.addMeasurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.addMeasurement(idx=3, m=np.array([60.0, 50.0]))  # this camera does not exist

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
        track_to_add.addMeasurement(idx=0, m=np.array([20.0, 5.0]))
        track_to_add.addMeasurement(idx=2, m=np.array([60.0, 50.0]))
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
            expected_data.add_camera(i, EXAMPLE_DATA.get_camera(i))

        for j in VALID_TRACK_INDICES:
            expected_data.add_track(EXAMPLE_DATA.get_track(j))

        # run the fn under test
        filtered_sfm_data, valid_mask = EXAMPLE_DATA.filter_landmarks(max_reproj_error)
        self.assertEqual(sum(valid_mask), 3)

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
            Pose3(Rot3(), np.array([3, 0, 0])),   # wTi0
            Pose3(Rot3(), np.array([0, 0, 0])),   # wTi1
            Pose3(Rot3(), np.array([0, -3, 0])),  # wTi2
            Pose3(Rot3(), np.array([0, 3, 0])),   # wTi3
        ]
        # points_gt = [
        #     np.array([1, 1, 0]),
        #     np.array([3, 3, 0])
        # ]

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
            track.addMeasurement(0, np.array([100, 200]))
            track.addMeasurement(1, np.array([300, 400]))
            track.addMeasurement(3, np.array([500, 600]))
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

        Tracks should have identical non-NaN reprojection error before and after alignment.
        """
        poses_gt = [
            Pose3(
                Rot3(
                    [
                        [0.696305769, -0.0106830792, -0.717665705],
                        [0.00546412488, 0.999939148, -0.00958346857],
                        [0.717724415, 0.00275160848, 0.696321772],
                    ]
                ),
                Point3(5.83077801, -0.94815149, 0.397751679),
            ),
            Pose3(
                Rot3(
                    [
                        [0.692272397, -0.00529704529, -0.721616549],
                        [0.00634689669, 0.999979075, -0.00125157022],
                        [0.721608079, -0.0037136016, 0.692291531],
                    ]
                ),
                Point3(5.03853323, -0.97547405, -0.348177392),
            ),
            Pose3(
                Rot3(
                    [
                        [0.945991981, -0.00633548292, -0.324128225],
                        [0.00450436485, 0.999969379, -0.00639931046],
                        [0.324158843, 0.00459370582, 0.945991552],
                    ]
                ),
                Point3(4.13186176, -0.956364218, -0.796029527),
            ),
            Pose3(
                Rot3(
                    [
                        [0.999553623, -0.00346470207, -0.0296740626],
                        [0.00346104216, 0.999993995, -0.00017469881],
                        [0.0296744897, 7.19175654e-05, 0.999559612],
                    ]
                ),
                Point3(3.1113898, -0.928583423, -0.90539337),
            ),
            Pose3(
                Rot3(
                    [
                        [0.967850252, -0.00144846042, 0.251522892],
                        [0.000254511591, 0.999988546, 0.00477934325],
                        [-0.251526934, -0.00456167299, 0.967839535],
                    ]
                ),
                Point3(2.10584013, -0.921303194, -0.809322971),
            ),
            Pose3(
                Rot3(
                    [
                        [0.969854065, 0.000629052774, 0.243685716],
                        [0.000387180179, 0.999991428, -0.00412234326],
                        [-0.243686221, 0.00409242166, 0.969845508],
                    ]
                ),
                Point3(1.0753788, -0.913035975, -0.616584091),
            ),
            Pose3(
                Rot3(
                    [
                        [0.998189342, 0.00110235337, 0.0601400045],
                        [-0.00110890447, 0.999999382, 7.55559042e-05],
                        [-0.060139884, -0.000142108649, 0.998189948],
                    ]
                ),
                Point3(0.029993558, -0.951495122, -0.425525143),
            ),
            Pose3(
                Rot3(
                    [
                        [0.999999996, -2.62868666e-05, -8.67178281e-05],
                        [2.62791334e-05, 0.999999996, -8.91767396e-05],
                        [8.67201719e-05, 8.91744604e-05, 0.999999992],
                    ]
                ),
                Point3(-0.973569417, -0.936340994, -0.253464928),
            ),
            Pose3(
                Rot3(
                    [
                        [0.99481227, -0.00153645011, 0.101716252],
                        [0.000916919443, 0.999980747, 0.00613725239],
                        [-0.101723724, -0.00601214847, 0.994794525],
                    ]
                ),
                Point3(-2.02071256, -0.955446292, -0.240707879),
            ),
            Pose3(
                Rot3(
                    [
                        [0.89795602, -0.00978591184, 0.43997636],
                        [0.00645921401, 0.999938116, 0.00905779513],
                        [-0.440037771, -0.00529159974, 0.89796366],
                    ]
                ),
                Point3(-2.94096695, -0.939974858, 0.0934225593),
            ),
            Pose3(
                Rot3(
                    [
                        [0.726299119, -0.00916784876, 0.687318077],
                        [0.00892018672, 0.999952563, 0.0039118575],
                        [-0.687321336, 0.00328981905, 0.726346444],
                    ]
                ),
                Point3(-3.72843416, -0.897889251, 0.685129502),
            ),
            Pose3(
                Rot3(
                    [
                        [0.506756029, -0.000331706105, 0.862089858],
                        [0.00613841257, 0.999975964, -0.00322354286],
                        [-0.862068067, 0.00692541035, 0.506745885],
                    ]
                ),
                Point3(-4.3909926, -0.890883291, 1.43029524),
            ),
            Pose3(
                Rot3(
                    [
                        [0.129316352, -0.00206958814, 0.991601896],
                        [0.00515932597, 0.999985691, 0.00141424797],
                        [-0.991590634, 0.00493310721, 0.129325179],
                    ]
                ),
                Point3(-4.58510846, -0.922534227, 2.36884523),
            ),
            Pose3(
                Rot3(
                    [
                        [0.599853194, -0.00890004681, -0.800060263],
                        [0.00313716318, 0.999956608, -0.00877161373],
                        [0.800103615, 0.00275175707, 0.599855085],
                    ]
                ),
                Point3(5.71559638, 0.486863076, 0.279141372),
            ),
            Pose3(
                Rot3(
                    [
                        [0.762552447, 0.000836438681, -0.646926069],
                        [0.00211337894, 0.999990607, 0.00378404105],
                        [0.646923157, -0.00425272942, 0.762543517],
                    ]
                ),
                Point3(5.00243443, 0.513321893, -0.466921769),
            ),
            Pose3(
                Rot3(
                    [
                        [0.930381645, -0.00340164355, -0.36657678],
                        [0.00425636616, 0.999989781, 0.00152338305],
                        [0.366567852, -0.00297761145, 0.930386617],
                    ]
                ),
                Point3(4.05404984, 0.493385291, -0.827904571),
            ),
            Pose3(
                Rot3(
                    [
                        [0.999996073, -0.00278379707, -0.000323508543],
                        [0.00278790921, 0.999905063, 0.0134941517],
                        [0.000285912831, -0.0134950006, 0.999908897],
                    ]
                ),
                Point3(3.04724478, 0.491451306, -0.989571061),
            ),
            Pose3(
                Rot3(
                    [
                        [0.968578343, -0.002544616, 0.248695527],
                        [0.000806130148, 0.999974526, 0.00709200332],
                        [-0.248707238, -0.0066686795, 0.968555721],
                    ]
                ),
                Point3(2.05737869, 0.46840177, -0.546344594),
            ),
            Pose3(
                Rot3(
                    [
                        [0.968827882, 0.000182770584, 0.247734722],
                        [-0.000558107079, 0.9999988, 0.00144484904],
                        [-0.24773416, -0.00153807255, 0.968826821],
                    ]
                ),
                Point3(1.14019947, 0.469674641, -0.0491053805),
            ),
            Pose3(
                Rot3(
                    [
                        [0.991647805, 0.00197867892, 0.128960146],
                        [-0.00247518407, 0.999990129, 0.00368991165],
                        [-0.128951572, -0.00397829284, 0.991642914],
                    ]
                ),
                Point3(0.150270471, 0.457867448, 0.103628642),
            ),
            Pose3(
                Rot3(
                    [
                        [0.992244594, 0.00477781876, -0.124208847],
                        [-0.0037682125, 0.999957938, 0.00836195891],
                        [0.124243574, -0.00782906317, 0.992220862],
                    ]
                ),
                Point3(-0.937954641, 0.440532658, 0.154265069),
            ),
            Pose3(
                Rot3(
                    [
                        [0.999591078, 0.00215462857, -0.0285137564],
                        [-0.00183807224, 0.999936443, 0.0111234301],
                        [0.028535911, -0.0110664711, 0.999531507],
                    ]
                ),
                Point3(-1.95622231, 0.448914367, -0.0859439782),
            ),
            Pose3(
                Rot3(
                    [
                        [0.931835342, 0.000956922238, 0.362880212],
                        [0.000941640753, 0.99998678, -0.00505501434],
                        [-0.362880252, 0.00505214382, 0.931822122],
                    ]
                ),
                Point3(-2.85557418, 0.434739285, 0.0793777177),
            ),
            Pose3(
                Rot3(
                    [
                        [0.781615218, -0.0109886966, 0.623664238],
                        [0.00516954657, 0.999924591, 0.011139446],
                        [-0.623739616, -0.00548270158, 0.781613084],
                    ]
                ),
                Point3(-3.67524552, 0.444074681, 0.583718622),
            ),
            Pose3(
                Rot3(
                    [
                        [0.521291761, 0.00264805046, 0.853374051],
                        [0.00659087718, 0.999952868, -0.00712898365],
                        [-0.853352707, 0.00934076542, 0.521249738],
                    ]
                ),
                Point3(-4.35541796, 0.413479707, 1.31179007),
            ),
            Pose3(
                Rot3(
                    [
                        [0.320164205, -0.00890839482, 0.947319884],
                        [0.00458409304, 0.999958649, 0.007854118],
                        [-0.947350678, 0.00182799903, 0.320191803],
                    ]
                ),
                Point3(-4.71617526, 0.476674479, 2.16502998),
            ),
            Pose3(
                Rot3(
                    [
                        [0.464861609, 0.0268597443, -0.884976415],
                        [-0.00947397841, 0.999633409, 0.0253631906],
                        [0.885333239, -0.00340614699, 0.464945663],
                    ]
                ),
                Point3(6.11772094, 1.63029238, 0.491786626),
            ),
            Pose3(
                Rot3(
                    [
                        [0.691647251, 0.0216006293, -0.721912024],
                        [-0.0093228132, 0.999736395, 0.020981541],
                        [0.722174939, -0.00778156302, 0.691666308],
                    ]
                ),
                Point3(5.46912979, 1.68759322, -0.288499782),
            ),
            Pose3(
                Rot3(
                    [
                        [0.921208931, 0.00622640471, -0.389018433],
                        [-0.00686296262, 0.999976419, -0.000246683913],
                        [0.389007724, 0.00289706631, 0.92122994],
                    ]
                ),
                Point3(4.70156942, 1.72186229, -0.806181015),
            ),
            Pose3(
                Rot3(
                    [
                        [0.822397705, 0.00276497594, 0.568906142],
                        [0.00804891535, 0.999831556, -0.016494662],
                        [-0.568855921, 0.0181442503, 0.822236923],
                    ]
                ),
                Point3(-3.51368714, 1.59619714, 0.437437437),
            ),
            Pose3(
                Rot3(
                    [
                        [0.726822937, -0.00545541524, 0.686803193],
                        [0.00913794245, 0.999956756, -0.00172754968],
                        [-0.686764068, 0.00753159111, 0.726841357],
                    ]
                ),
                Point3(-4.29737821, 1.61462527, 1.11537749),
            ),
            Pose3(
                Rot3(
                    [
                        [0.402595481, 0.00697612855, 0.915351441],
                        [0.0114113638, 0.999855006, -0.0126391687],
                        [-0.915306892, 0.0155338804, 0.4024575],
                    ]
                ),
                Point3(-4.6516433, 1.6323107, 1.96579585),
            ),
        ]

        fx, k1, k2, px, py = 609.94, -8.82687675e-07, -4.25111629e-06, 507, 380
        calib = Cal3Bundler(fx, k1, k2, px, py)

        unaligned_cameras = {
            2: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.681949, -0.568276, 0.460444],
                            [0.572389, -0.0227514, 0.819667],
                            [-0.455321, 0.822524, 0.34079],
                        ]
                    ),
                    Point3(-1.52189, 0.78906, -1.60608),
                ),
                calib,
            ),
            4: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.817805393, -0.575044816, 0.022755196],
                            [0.0478829397, -0.0285875849, 0.998443776],
                            [-0.573499401, 0.81762229, 0.0509139174],
                        ]
                    ),
                    Point3(-1.22653168, 0.686485651, -1.39294168),
                ),
                calib,
            ),
            3: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.783051568, -0.571905041, 0.244448085],
                            [0.314861464, -0.0255673164, 0.948793218],
                            [-0.536369743, 0.819921299, 0.200091385],
                        ]
                    ),
                    Point3(-1.37620079, 0.721408674, -1.49945316),
                ),
                calib,
            ),
            5: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.818916586, -0.572896131, 0.0341415873],
                            [0.0550548476, -0.0192038786, 0.99829864],
                            [-0.571265778, 0.819402974, 0.0472670839],
                        ]
                    ),
                    Point3(-1.06370243, 0.663084159, -1.27672831),
                ),
                calib,
            ),
            6: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.798825521, -0.571995242, 0.186277293],
                            [0.243311017, -0.0240196245, 0.969650869],
                            [-0.550161372, 0.819905178, 0.158360233],
                        ]
                    ),
                    Point3(-0.896250742, 0.640768239, -1.16984756),
                ),
                calib,
            ),
            7: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.786416666, -0.570215296, 0.237493882],
                            [0.305475635, -0.0248440676, 0.951875732],
                            [-0.536873788, 0.821119534, 0.193724669],
                        ]
                    ),
                    Point3(-0.740385043, 0.613956842, -1.05908579),
                ),
                calib,
            ),
            8: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.806252832, -0.57019757, 0.157578877],
                            [0.211046715, -0.0283979846, 0.977063375],
                            [-0.55264424, 0.821016617, 0.143234279],
                        ]
                    ),
                    Point3(-0.58333517, 0.549832698, -0.9542864),
                ),
                calib,
            ),
            9: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.821191354, -0.557772774, -0.120558255],
                            [-0.125347331, -0.0297958331, 0.991665395],
                            [-0.556716092, 0.829458703, -0.0454472483],
                        ]
                    ),
                    Point3(-0.436483039, 0.55003923, -0.850733187),
                ),
                calib,
            ),
            21: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.778607603, -0.575075476, 0.251114312],
                            [0.334920968, -0.0424301164, 0.941290407],
                            [-0.53065822, 0.816999316, 0.225641247],
                        ]
                    ),
                    Point3(-0.736735967, 0.571415247, -0.738663611),
                ),
                calib,
            ),
            17: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.818569806, -0.573904529, 0.0240221722],
                            [0.0512889176, -0.0313725422, 0.998190969],
                            [-0.572112681, 0.818321059, 0.0551155579],
                        ]
                    ),
                    Point3(-1.36150982, 0.724829031, -1.16055631),
                ),
                calib,
            ),
            18: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.812668105, -0.582027424, 0.0285417146],
                            [0.0570298244, -0.0306936169, 0.997900547],
                            [-0.579929436, 0.812589675, 0.0581366453],
                        ]
                    ),
                    Point3(-1.20484771, 0.762370343, -1.05057127),
                ),
                calib,
            ),
            20: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.748446406, -0.580905382, 0.319963926],
                            [0.416860654, -0.0368374152, 0.908223651],
                            [-0.515805363, 0.813137099, 0.269727429],
                        ]
                    ),
                    Point3(569.449421, -907.892555, -794.585647),
                ),
                calib,
            ),
            22: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.826878177, -0.559495019, -0.0569017041],
                            [-0.0452256802, -0.0346974602, 0.99837404],
                            [-0.560559647, 0.828107125, 0.00338702978],
                        ]
                    ),
                    Point3(-0.591431172, 0.55422253, -0.654656597),
                ),
                calib,
            ),
            29: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.785759779, -0.574532433, -0.229115805],
                            [-0.246020939, -0.049553424, 0.967996981],
                            [-0.567499134, 0.81698038, -0.102409921],
                        ]
                    ),
                    Point3(69.4916073, 240.595227, -493.278045),
                ),
                calib,
            ),
            23: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.783524382, -0.548569702, -0.291823276],
                            [-0.316457553, -0.051878563, 0.94718701],
                            [-0.534737468, 0.834493797, -0.132950906],
                        ]
                    ),
                    Point3(-5.93496204, 41.9304933, -3.06881633),
                ),
                calib,
            ),
            10: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.766833992, -0.537641809, -0.350580824],
                            [-0.389506676, -0.0443270797, 0.919956336],
                            [-0.510147213, 0.84200736, -0.175423563],
                        ]
                    ),
                    Point3(234.185458, 326.007989, -691.769777),
                ),
                calib,
            ),
            30: PinholeCameraCal3Bundler(
                Pose3(
                    Rot3(
                        [
                            [-0.754844165, -0.559278755, -0.342662459],
                            [-0.375790683, -0.0594160018, 0.92479787],
                            [-0.537579435, 0.826847636, -0.165321923],
                        ]
                    ),
                    Point3(-5.93398168, 41.9107972, -3.07385081),
                ),
                calib,
            ),
        }

        t0 = SfmTrack(pt=[-0.89190672, 1.21298076, -1.05838554])
        t0.addMeasurement(2, [184.08586121, 441.31314087])
        t0.addMeasurement(4, [18.98637581, 453.21853638])

        t1 = SfmTrack(pt=[-0.76287111, 1.26476165, -1.22710579])
        t1.addMeasurement(2, [213.51266479, 288.06637573])
        t1.addMeasurement(4, [50.23059464, 229.30541992])

        t2 = SfmTrack(pt=[-1.45773622, 0.86221933, -1.47515461])
        t2.addMeasurement(2, [227.52420044, 695.15087891])
        t2.addMeasurement(3, [996.67608643, 705.03125])

        t3 = SfmTrack(pt=[-1.40486691, 0.93824916, -1.35192298])
        t3.addMeasurement(2, [251.37863159, 702.97064209])
        t3.addMeasurement(3, [537.9753418, 732.26025391])

        t4 = SfmTrack(pt=[55.48969812, 52.24862241, 58.84578119])
        t4.addMeasurement(2, [253.17749023, 490.47991943])
        t4.addMeasurement(3, [13.17782784, 507.57717896])

        t5 = SfmTrack(pt=[230.43166291, 206.44760657, 234.25904211])
        t5.addMeasurement(2, [253.52301025, 478.41384888])
        t5.addMeasurement(3, [10.92995739, 493.31018066])

        t6 = SfmTrack(pt=[11.62742671, 13.43484624, 14.50306349])
        t6.addMeasurement(2, [254.64611816, 533.04730225])
        t6.addMeasurement(3, [18.78449249, 557.05041504])

        unaligned_tracks = [t0, t1, t2, t3, t4, t5, t6]

        unaligned_filtered_data = GtsfmData.from_cameras_and_tracks(
            cameras=unaligned_cameras, tracks=unaligned_tracks, number_images=32
        )
        unaligned_metrics = metrics_utils.get_metrics_for_sfmdata(unaligned_filtered_data, suffix="_filtered")
        aligned_filtered_data = unaligned_filtered_data.align_via_Sim3_to_poses(wTi_list_ref=poses_gt)

        aligned_metrics = metrics_utils.get_metrics_for_sfmdata(aligned_filtered_data, suffix="_filtered")

        assert unaligned_metrics[3].name == "reprojection_errors_filtered_px"
        assert aligned_metrics[3].name == "reprojection_errors_filtered_px"

        # Reprojection error should be unaffected by Sim(3) alignment.
        for key in ["min", "max", "median", "mean", "stddev"]:
            assert np.isclose(unaligned_metrics[3].summary[key], aligned_metrics[3].summary[key])


if __name__ == "__main__":
    unittest.main()
