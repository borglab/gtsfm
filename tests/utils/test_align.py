"""Unit tests for comparison functions for geometry types.

Authors: Ayush Baid
"""

import copy
import unittest
from typing import Sequence

import numpy as np
from gtsam import PinholeCameraCal3Bundler  # type: ignore
from gtsam import Cal3Bundler, Point2, Point3, Pose3, Rot3, SfmTrack, Similarity3
from gtsam.examples import SFMdata  # type: ignore

import tests.data.sample_poses as sample_poses
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import align, transform

POSE_LIST = SFMdata.posesOnCircle(R=40)

ROT3_EULER_ANGLE_ERROR_THRESHOLD = 1e-2
POINT3_RELATIVE_ERROR_THRESH = 1e-1
POINT3_ABS_ERROR_THRESH = 1e-2


def rot3(matrix: np.ndarray | list) -> Rot3:
    """Helper to create Rot3 from a numpy array or list, avoiding type-check errors."""
    R: np.ndarray = np.array(matrix)
    return Rot3(R)


def rot3_compare(R: Rot3, R_: Rot3, msg=None) -> None:
    if not np.allclose(R.xyz(), R_.xyz(), atol=1e-2):
        standardMsg = f"{R} != {R_}"
        raise AssertionError(msg or standardMsg)


def point3_compare(t: np.ndarray, t_: np.ndarray, msg=None) -> None:
    if not np.allclose(t, t_, rtol=POINT3_RELATIVE_ERROR_THRESH, atol=POINT3_ABS_ERROR_THRESH):
        standardMsg = f"{t} != {t_}"
        raise AssertionError(msg or standardMsg)


class TestAlignmentUtils(unittest.TestCase):
    """Unit tests for comparison functions for geometry types."""

    def __assert_equality_on_rot3s(self, computed: Sequence[Rot3 | None], expected: Sequence[Rot3 | None]) -> None:
        self.assertEqual(len(computed), len(expected))

        for i, (R, R_) in enumerate(zip(computed, expected)):
            try:
                self.assertEqual(R, R_)
            except AssertionError as e:
                raise AssertionError(f"Rot3 mismatch at index {i}: {e}") from e

    def __assert_equality_on_point3s(
        self, computed: Sequence[np.ndarray | None], expected: Sequence[np.ndarray | None]
    ) -> None:
        self.assertEqual(len(computed), len(expected))

        for t, t_ in zip(computed, expected):
            if t is not None and t_ is not None:
                np.testing.assert_allclose(t, t_, rtol=POINT3_RELATIVE_ERROR_THRESH, atol=POINT3_ABS_ERROR_THRESH)
            else:
                assert t is None and t_ is None

    def __assert_equality_on_pose3s(self, computed: Sequence[Pose3 | None], expected: Sequence[Pose3 | None]) -> None:
        self.assertEqual(len(computed), len(expected))

        computed_rot3s = [x.rotation() if x is not None else None for x in computed]
        computed_point3s = [x.translation() if x is not None else None for x in computed]
        expected_rot3s = [x.rotation() if x is not None else None for x in expected]
        expected_point3s = [x.translation() if x is not None else None for x in expected]

        self.__assert_equality_on_rot3s(computed_rot3s, expected_rot3s)
        self.__assert_equality_on_point3s(computed_point3s, expected_point3s)

    def setUp(self):
        super().setUp()

        self.addTypeEqualityFunc(Rot3, rot3_compare)
        self.addTypeEqualityFunc(np.ndarray, point3_compare)

    def test_align_rotations(self):
        """Tests the alignment of rotations."""

        # using rotation along just the Y-axis so that angles can be linearly added.
        aRi_list = [
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(-10), 0),
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(30), 0),
        ]
        bRi_list = [
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(-10 + 90), 0),
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(30 + 90), 0),
        ]

        aRb = align.so3_from_optional_Rot3s(aRi_list, bRi_list)
        computed = transform.Rot3s_with_so3(bRi_list, aRb)
        expected = [
            Rot3.RzRyRx(0, np.deg2rad(-10), 0),
            Rot3.RzRyRx(0, np.deg2rad(30), 0),
        ]

        self.__assert_equality_on_rot3s(computed, expected)

    def test_align_rotations_with_no_relative_rotations(self):
        """Tests the alignment of rotations."""

        # using rotation along just the Y-axis so that angles can be linearly added.
        aRi_list = [
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(-10), 0),
            Rot3.RzRyRx(np.deg2rad(0), np.deg2rad(30), 0),
        ]
        bRi_list = copy.deepcopy(aRi_list)

        aRb = align.so3_from_optional_Rot3s(aRi_list, bRi_list)
        computed = transform.Rot3s_with_so3(bRi_list, aRb)
        expected = copy.deepcopy(aRi_list)
        self.__assert_equality_on_rot3s(computed, expected)

    def test_align_poses_after_sim3_transform(self) -> None:
        """Test for alignment of poses after applying a SIM3 transformation."""

        translation_shift: np.ndarray = Point3(5, 10, -5)
        rotation_shift = Rot3.RzRyRx(0, 0, np.deg2rad(30))
        scaling_factor = 0.7

        sim3 = Similarity3(rotation_shift, translation_shift, scaling_factor)
        ref_list = [sim3.transformFrom(x) for x in sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES]

        aSb = align.sim3_from_Pose3s(sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES, ref_list)
        assert isinstance(aSb, Similarity3)
        computed_poses = transform.Pose3s_with_sim3(ref_list, aSb)
        self.__assert_equality_on_pose3s(computed_poses, sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES)

    def test_align_poses_with_outlier(self) -> None:
        """Test for alignment of poses after applying a SIM3 transform and adding an outlier."""
        translation_shift: np.ndarray = Point3(5, 10, -5)
        rotation_shift = Rot3.RzRyRx(0, 0, np.deg2rad(30))
        scaling_factor = 0.7

        sim3 = Similarity3(rotation_shift, translation_shift, scaling_factor)
        ref_list = copy.deepcopy(sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES)
        input_list = [sim3.transformFrom(x) for x in sample_poses.CIRCLE_TWO_EDGES_GLOBAL_POSES]
        random_t: np.ndarray = np.random.rand(3)
        input_list[1] = Pose3(Rot3.RzRyRx(np.deg2rad(60), -np.deg2rad(-30), np.deg2rad(np.deg2rad(-20))), random_t)

        # Note: this test requires exhaustive alignment and will fail with regular alignment.
        aSb = align.sim3_from_Pose3s_robust(ref_list, input_list)
        assert isinstance(aSb, Similarity3)
        computed_poses = transform.Pose3s_with_sim3(input_list, aSb)
        computed_poses_with_outlier_removed = [computed_poses[0]] + computed_poses[2:]
        ref_list_with_outlier_removed = [ref_list[0]] + ref_list[2:]
        self.__assert_equality_on_pose3s(computed_poses_with_outlier_removed, ref_list_with_outlier_removed)

    def test_align_poses_on_panorama_after_sim3_transform(self) -> None:
        """Test for alignment of poses after applying a forward motion transformation."""

        translation_shift: np.ndarray = Point3(0, 5, 0)
        rotation_shift = Rot3.RzRyRx(0, 0, np.deg2rad(30))
        scaling_factor = 1.0

        aTi_list = sample_poses.PANORAMA_GLOBAL_POSES
        bSa = Similarity3(rotation_shift, translation_shift, scaling_factor)
        bTi_list = [bSa.transformFrom(x) for x in aTi_list]

        aSb = align.sim3_from_Pose3s(aTi_list, bTi_list)
        assert isinstance(aSb, Similarity3)
        aTi_list_ = transform.Pose3s_with_sim3(bTi_list, aSb)
        self.__assert_equality_on_pose3s(aTi_list_, aTi_list)

    def test_align_poses_sim3_ignore_missing(self) -> None:
        """Consider a simple cases with 4 poses in a line. Suppose SfM only recovers 2 of the 4 poses."""
        Z_3x1: np.ndarray = np.zeros((3,))
        ones: np.ndarray = np.ones((3,))
        wT0 = Pose3(Rot3(), Z_3x1)
        wT1 = Pose3(Rot3(), ones)
        wT2 = Pose3(Rot3(), ones * 2)
        wT3 = Pose3(Rot3(), ones * 3)

        # `a` frame is the target/reference frame
        aTi_list = [wT0, wT1, wT2, wT3]
        # `b` frame contains the estimates
        bTi_list = [None, wT1, None, wT3]
        aSb = align.sim3_from_optional_Pose3s(aTi_list, bTi_list)
        aTi_list_ = transform.optional_Pose3s_with_sim3(bTi_list, aSb)

        # indices 0 and 2 should still have no estimated pose, even after alignment
        assert aTi_list_[0] is None
        assert aTi_list_[2] is None

        # identity alignment should preserve poses, should still match GT/targets at indices 1 and 3
        self.__assert_equality_on_pose3s(computed=[aTi_list_[1], aTi_list_[3]], expected=[aTi_list[1], aTi_list[3]])

    def test_gtsfm_data_align_via_Sim3_to_poses(self) -> None:
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
            Pose3(Rot3(), Point3(3, 0, 0)),   # wTi0
            Pose3(Rot3(), Point3(0, 0, 0)),   # wTi1
            Pose3(Rot3(), Point3(0, -3, 0)),  # wTi2
            Pose3(Rot3(), Point3(0, 3, 0)),   # wTi3
        ]
        # points_gt = [
        #     Point3(1, 1, 0),
        #     Point3(3, 3, 0)
        # ]

        # pose graph is scaled by a factor of 2, and shifted also.
        wTi_list_est = [
            Pose3(Rot3(), Point3(8, 2, 0)),  # wTi0
            Pose3(Rot3(), Point3(2, 2, 0)),  # wTi1
            None,                                # wTi2
            Pose3(Rot3(), Point3(2, 8, 0)),  # wTi3
        ]
        points_est = [
            Point3(4, 4, 0),
            Point3(8, 8, 0)
        ]
        # fmt: on

        def add_dummy_measurements_to_track(track: SfmTrack) -> SfmTrack:
            """Add some dummy 2d measurements in three views in cameras 0,1,3."""
            track.addMeasurement(0, Point2(100, 200))
            track.addMeasurement(1, Point2(300, 400))
            track.addMeasurement(3, Point2(500, 600))
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
        aligned_sfm_result = sfm_result.align_via_sim3_and_transform(gt_gtsfm_data.get_camera_poses())
        # tracks and poses should match GT now, after applying estimated scale and shift.
        assert aligned_sfm_result == gt_gtsfm_data

        # 3d points from tracks should now match the GT.
        assert np.allclose(aligned_sfm_result.get_track(0).point3(), Point3(1.0, 1.0, 0))
        assert np.allclose(aligned_sfm_result.get_track(1).point3(), Point3(3.0, 3.0, 0))

    def test_ransac_align_poses_sim3_ignore_missing_pure_identity(self) -> None:
        """Ensure that for identity poses, and thus identity Similarity(3), we get back exactly what we started with."""

        aTi_list = [
            Pose3(rot3([[0.771176, -0.36622, 0], [0.636622, 0.771176, 0], [0, 0, 1]]), Point3(6.94918, 2.4749, 0)),
            Pose3(
                rot3([[0.124104, -0.92269, 0], [0.992269, 0.124104, 0], [0, 0, 1]]),
                Point3(6.06848, 4.57841, 0),
            ),
            Pose3(
                rot3([[0.914145, 0.05387, 0], [-0.405387, 0.914145, 0], [0, 0, 1]]),
                Point3(6.47869, 5.29594, 0),
            ),
            Pose3(
                rot3([[0.105365, -0.94434, 0], [0.994434, 0.105365, 0], [0, 0, 1]]),
                Point3(5.59441, 5.22469, 0),
            ),
            Pose3(
                rot3([[-0.991652, -0.2894, 0], [0.12894, -0.991652, 0], [0, 0, 1]]),
                Point3(7.21399, 5.41445, 0),
            ),
        ]
        # Make twice as long.
        aTi_list = aTi_list + aTi_list

        bTi_list = copy.deepcopy(aTi_list)

        aSb = align.sim3_from_optional_Pose3s(aTi_list, bTi_list)
        aligned_bTi_list_est = transform.optional_Pose3s_with_sim3(bTi_list, aSb)

        self.__assert_equality_on_pose3s(aTi_list, aligned_bTi_list_est)

    def test_ransac_align_poses_sim3_ignore_two_missing_estimated_poses(self) -> None:
        """Unit test for simple case of 3 poses (one is an outlier with massive translation error.)"""

        aTi_list = [
            None,
            Pose3(Rot3(), Point3(50, 0, 0)),
            Pose3(Rot3(), Point3(0, 10, 0)),
            Pose3(Rot3(), Point3(0, 0, 20)),
            None,
        ]

        # Below was previously in b's frame. Has a bit of noise compared to pose graph above.
        bTi_list = [
            None,
            Pose3(Rot3(), Point3(50.1, 0, 0)),
            Pose3(Rot3(), Point3(0, 9.9, 0)),
            Pose3(Rot3(), Point3(0, 0, 2000)),
            None,
        ]

        aSb = align.sim3_from_optional_Pose3s(aTi_list, bTi_list)
        aligned_bTi_list_est = transform.optional_Pose3s_with_sim3(bTi_list, aSb)
        assert np.isclose(aSb.scale(), 1.0, atol=1e-2)
        assert aligned_bTi_list_est[1] is not None
        assert aligned_bTi_list_est[2] is not None
        assert np.allclose(aligned_bTi_list_est[1].translation(), [Point3(50.0114, 0.0576299, 0)], atol=1e-3)
        assert np.allclose(aligned_bTi_list_est[2].translation(), [Point3(-0.0113879, 9.94237, 0)], atol=1e-3)

    def test_ransac_align_poses_sim3_if_no_ground_truth_provided(self) -> None:
        aTi_list = [
            None,
            None,
            None,
        ]

        # Below was previously in b's frame. Has a bit of noise compared to pose graph above.
        bTi_list = [
            Pose3(Rot3(), Point3(50.1, 0, 0)),
            Pose3(Rot3(), Point3(0, 9.9, 0)),
            Pose3(Rot3(), Point3(0, 0, 2000)),
        ]

        aSb = align.sim3_from_optional_Pose3s(aTi_list, bTi_list)
        assert isinstance(aSb, Similarity3)

    def test_align_gtsfm_data_via_Sim3_to_poses_skydio32(self) -> None:
        """Real data, from Skydio-32 sequence with the SIFT front-end.

        Tracks should have identical non-NaN reprojection error before and after alignment.
        """
        poses_gt = [
            Pose3(
                rot3(
                    [
                        [0.696305769, -0.0106830792, -0.717665705],
                        [0.00546412488, 0.999939148, -0.00958346857],
                        [0.717724415, 0.00275160848, 0.696321772],
                    ]
                ),
                Point3(5.83077801, -0.94815149, 0.397751679),
            ),
            Pose3(
                rot3(
                    [
                        [0.692272397, -0.00529704529, -0.721616549],
                        [0.00634689669, 0.999979075, -0.00125157022],
                        [0.721608079, -0.0037136016, 0.692291531],
                    ]
                ),
                Point3(5.03853323, -0.97547405, -0.348177392),
            ),
            Pose3(
                rot3(
                    [
                        [0.945991981, -0.00633548292, -0.324128225],
                        [0.00450436485, 0.999969379, -0.00639931046],
                        [0.324158843, 0.00459370582, 0.945991552],
                    ]
                ),
                Point3(4.13186176, -0.956364218, -0.796029527),
            ),
            Pose3(
                rot3(
                    [
                        [0.999553623, -0.00346470207, -0.0296740626],
                        [0.00346104216, 0.999993995, -0.00017469881],
                        [0.0296744897, 7.19175654e-05, 0.999559612],
                    ]
                ),
                Point3(3.1113898, -0.928583423, -0.90539337),
            ),
            Pose3(
                rot3(
                    [
                        [0.967850252, -0.00144846042, 0.251522892],
                        [0.000254511591, 0.999988546, 0.00477934325],
                        [-0.251526934, -0.00456167299, 0.967839535],
                    ]
                ),
                Point3(2.10584013, -0.921303194, -0.809322971),
            ),
            Pose3(
                rot3(
                    [
                        [0.969854065, 0.000629052774, 0.243685716],
                        [0.000387180179, 0.999991428, -0.00412234326],
                        [-0.243686221, 0.00409242166, 0.969845508],
                    ]
                ),
                Point3(1.0753788, -0.913035975, -0.616584091),
            ),
            Pose3(
                rot3(
                    [
                        [0.998189342, 0.00110235337, 0.0601400045],
                        [-0.00110890447, 0.999999382, 7.55559042e-05],
                        [-0.060139884, -0.000142108649, 0.998189948],
                    ]
                ),
                Point3(0.029993558, -0.951495122, -0.425525143),
            ),
            Pose3(
                rot3(
                    [
                        [0.999999996, -2.62868666e-05, -8.67178281e-05],
                        [2.62791334e-05, 0.999999996, -8.91767396e-05],
                        [8.67201719e-05, 8.91744604e-05, 0.999999992],
                    ]
                ),
                Point3(-0.973569417, -0.936340994, -0.253464928),
            ),
            Pose3(
                rot3(
                    [
                        [0.99481227, -0.00153645011, 0.101716252],
                        [0.000916919443, 0.999980747, 0.00613725239],
                        [-0.101723724, -0.00601214847, 0.994794525],
                    ]
                ),
                Point3(-2.02071256, -0.955446292, -0.240707879),
            ),
            Pose3(
                rot3(
                    [
                        [0.89795602, -0.00978591184, 0.43997636],
                        [0.00645921401, 0.999938116, 0.00905779513],
                        [-0.440037771, -0.00529159974, 0.89796366],
                    ]
                ),
                Point3(-2.94096695, -0.939974858, 0.0934225593),
            ),
            Pose3(
                rot3(
                    [
                        [0.726299119, -0.00916784876, 0.687318077],
                        [0.00892018672, 0.999952563, 0.0039118575],
                        [-0.687321336, 0.00328981905, 0.726346444],
                    ]
                ),
                Point3(-3.72843416, -0.897889251, 0.685129502),
            ),
            Pose3(
                rot3(
                    [
                        [0.506756029, -0.000331706105, 0.862089858],
                        [0.00613841257, 0.999975964, -0.00322354286],
                        [-0.862068067, 0.00692541035, 0.506745885],
                    ]
                ),
                Point3(-4.3909926, -0.890883291, 1.43029524),
            ),
            Pose3(
                rot3(
                    [
                        [0.129316352, -0.00206958814, 0.991601896],
                        [0.00515932597, 0.999985691, 0.00141424797],
                        [-0.991590634, 0.00493310721, 0.129325179],
                    ]
                ),
                Point3(-4.58510846, -0.922534227, 2.36884523),
            ),
            Pose3(
                rot3(
                    [
                        [0.599853194, -0.00890004681, -0.800060263],
                        [0.00313716318, 0.999956608, -0.00877161373],
                        [0.800103615, 0.00275175707, 0.599855085],
                    ]
                ),
                Point3(5.71559638, 0.486863076, 0.279141372),
            ),
            Pose3(
                rot3(
                    [
                        [0.762552447, 0.000836438681, -0.646926069],
                        [0.00211337894, 0.999990607, 0.00378404105],
                        [0.646923157, -0.00425272942, 0.762543517],
                    ]
                ),
                Point3(5.00243443, 0.513321893, -0.466921769),
            ),
            Pose3(
                rot3(
                    [
                        [0.930381645, -0.00340164355, -0.36657678],
                        [0.00425636616, 0.999989781, 0.00152338305],
                        [0.366567852, -0.00297761145, 0.930386617],
                    ]
                ),
                Point3(4.05404984, 0.493385291, -0.827904571),
            ),
            Pose3(
                rot3(
                    [
                        [0.999996073, -0.00278379707, -0.000323508543],
                        [0.00278790921, 0.999905063, 0.0134941517],
                        [0.000285912831, -0.0134950006, 0.999908897],
                    ]
                ),
                Point3(3.04724478, 0.491451306, -0.989571061),
            ),
            Pose3(
                rot3(
                    [
                        [0.968578343, -0.002544616, 0.248695527],
                        [0.000806130148, 0.999974526, 0.00709200332],
                        [-0.248707238, -0.0066686795, 0.968555721],
                    ]
                ),
                Point3(2.05737869, 0.46840177, -0.546344594),
            ),
            Pose3(
                rot3(
                    [
                        [0.968827882, 0.000182770584, 0.247734722],
                        [-0.000558107079, 0.9999988, 0.00144484904],
                        [-0.24773416, -0.00153807255, 0.968826821],
                    ]
                ),
                Point3(1.14019947, 0.469674641, -0.0491053805),
            ),
            Pose3(
                rot3(
                    [
                        [0.991647805, 0.00197867892, 0.128960146],
                        [-0.00247518407, 0.999990129, 0.00368991165],
                        [-0.128951572, -0.00397829284, 0.991642914],
                    ]
                ),
                Point3(0.150270471, 0.457867448, 0.103628642),
            ),
            Pose3(
                rot3(
                    [
                        [0.992244594, 0.00477781876, -0.124208847],
                        [-0.0037682125, 0.999957938, 0.00836195891],
                        [0.124243574, -0.00782906317, 0.992220862],
                    ]
                ),
                Point3(-0.937954641, 0.440532658, 0.154265069),
            ),
            Pose3(
                rot3(
                    [
                        [0.999591078, 0.00215462857, -0.0285137564],
                        [-0.00183807224, 0.999936443, 0.0111234301],
                        [0.028535911, -0.0110664711, 0.999531507],
                    ]
                ),
                Point3(-1.95622231, 0.448914367, -0.0859439782),
            ),
            Pose3(
                rot3(
                    [
                        [0.931835342, 0.000956922238, 0.362880212],
                        [0.000941640753, 0.99998678, -0.00505501434],
                        [-0.362880252, 0.00505214382, 0.931822122],
                    ]
                ),
                Point3(-2.85557418, 0.434739285, 0.0793777177),
            ),
            Pose3(
                rot3(
                    [
                        [0.781615218, -0.0109886966, 0.623664238],
                        [0.00516954657, 0.999924591, 0.011139446],
                        [-0.623739616, -0.00548270158, 0.781613084],
                    ]
                ),
                Point3(-3.67524552, 0.444074681, 0.583718622),
            ),
            Pose3(
                rot3(
                    [
                        [0.521291761, 0.00264805046, 0.853374051],
                        [0.00659087718, 0.999952868, -0.00712898365],
                        [-0.853352707, 0.00934076542, 0.521249738],
                    ]
                ),
                Point3(-4.35541796, 0.413479707, 1.31179007),
            ),
            Pose3(
                rot3(
                    [
                        [0.320164205, -0.00890839482, 0.947319884],
                        [0.00458409304, 0.999958649, 0.007854118],
                        [-0.947350678, 0.00182799903, 0.320191803],
                    ]
                ),
                Point3(-4.71617526, 0.476674479, 2.16502998),
            ),
            Pose3(
                rot3(
                    [
                        [0.464861609, 0.0268597443, -0.884976415],
                        [-0.00947397841, 0.999633409, 0.0253631906],
                        [0.885333239, -0.00340614699, 0.464945663],
                    ]
                ),
                Point3(6.11772094, 1.63029238, 0.491786626),
            ),
            Pose3(
                rot3(
                    [
                        [0.691647251, 0.0216006293, -0.721912024],
                        [-0.0093228132, 0.999736395, 0.020981541],
                        [0.722174939, -0.00778156302, 0.691666308],
                    ]
                ),
                Point3(5.46912979, 1.68759322, -0.288499782),
            ),
            Pose3(
                rot3(
                    [
                        [0.921208931, 0.00622640471, -0.389018433],
                        [-0.00686296262, 0.999976419, -0.000246683913],
                        [0.389007724, 0.00289706631, 0.92122994],
                    ]
                ),
                Point3(4.70156942, 1.72186229, -0.806181015),
            ),
            Pose3(
                rot3(
                    [
                        [0.822397705, 0.00276497594, 0.568906142],
                        [0.00804891535, 0.999831556, -0.016494662],
                        [-0.568855921, 0.0181442503, 0.822236923],
                    ]
                ),
                Point3(-3.51368714, 1.59619714, 0.437437437),
            ),
            Pose3(
                rot3(
                    [
                        [0.726822937, -0.00545541524, 0.686803193],
                        [0.00913794245, 0.999956756, -0.00172754968],
                        [-0.686764068, 0.00753159111, 0.726841357],
                    ]
                ),
                Point3(-4.29737821, 1.61462527, 1.11537749),
            ),
            Pose3(
                rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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
                    rot3(
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

        t0 = SfmTrack(pt=Point3(-0.89190672, 1.21298076, -1.05838554))
        t0.addMeasurement(2, Point2(184.08586121, 441.31314087))
        t0.addMeasurement(4, Point2(18.98637581, 453.21853638))

        t1 = SfmTrack(pt=Point3(-0.76287111, 1.26476165, -1.22710579))
        t1.addMeasurement(2, Point2(213.51266479, 288.06637573))
        t1.addMeasurement(4, Point2(50.23059464, 229.30541992))

        t2 = SfmTrack(pt=Point3(-1.45773622, 0.86221933, -1.47515461))
        t2.addMeasurement(2, Point2(227.52420044, 695.15087891))
        t2.addMeasurement(3, Point2(996.67608643, 705.03125))

        t3 = SfmTrack(pt=Point3(-1.40486691, 0.93824916, -1.35192298))
        t3.addMeasurement(2, Point2(251.37863159, 702.97064209))
        t3.addMeasurement(3, Point2(537.9753418, 732.26025391))

        t4 = SfmTrack(pt=Point3(55.48969812, 52.24862241, 58.84578119))
        t4.addMeasurement(2, Point2(253.17749023, 490.47991943))
        t4.addMeasurement(3, Point2(13.17782784, 507.57717896))

        t5 = SfmTrack(pt=Point3(230.43166291, 206.44760657, 234.25904211))
        t5.addMeasurement(2, Point2(253.52301025, 478.41384888))
        t5.addMeasurement(3, Point2(10.92995739, 493.31018066))

        t6 = SfmTrack(pt=Point3(11.62742671, 13.43484624, 14.50306349))
        t6.addMeasurement(2, Point2(254.64611816, 533.04730225))
        t6.addMeasurement(3, Point2(18.78449249, 557.05041504))

        unaligned_tracks = [t0, t1, t2, t3, t4, t5, t6]

        unaligned_filtered_data = GtsfmData.from_cameras_and_tracks(
            cameras=unaligned_cameras, tracks=unaligned_tracks, number_images=32
        )
        unaligned_metrics = unaligned_filtered_data.get_metrics(suffix="_filtered")
        aligned_filtered_data = unaligned_filtered_data.align_via_sim3_and_transform(poses_gt)

        aligned_metrics = aligned_filtered_data.get_metrics(suffix="_filtered")

        assert unaligned_metrics[3].name == "reprojection_errors_filtered_px"
        assert aligned_metrics[3].name == "reprojection_errors_filtered_px"

        # Reprojection error should be unaffected by Sim(3) alignment.
        for key in ["min", "max", "median", "mean", "stddev"]:
            assert np.isclose(unaligned_metrics[3].summary[key], aligned_metrics[3].summary[key])


if __name__ == "__main__":
    unittest.main()
