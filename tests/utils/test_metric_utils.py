"""Unit tests for metrics utilities.

Authors: Travis Driver, John Lambert
"""

import unittest

import numpy as np
import trimesh
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmTrack

import gtsfm.utils.metrics as metric_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints


class TestMetricUtils(unittest.TestCase):
    """Class containing all unit tests for metric utils."""

    def test_mesh_inlier_correspondences(self) -> None:
        """Tests `compute_keypoint_intersections()` function.

        We arrange four cameras in the x-z plane around a cube centered at the origin with side length 1. These cameras
        are placed at (2, 0, 0), (-2, 0, 0), (0, 0, 2) and (0, 0, -2). We project a single 3d point located at the
        origin into each camera. Since the cube has unit length on each dimension, we expect a keypoint located at the
        center of each image to be found at the boundary of the cube -- 0.5 meters from the origin for each side on the
        z-x plane.
        """
        # Create cube mesh with side length one centered at origin.
        box = trimesh.primitives.Box()

        # Create arrangement of two cameras pointing at the center of one of the cube's faces.
        fx, k1, k2, u0, v0 = 10, 0, 0, 1, 1
        calibration = Cal3Bundler(fx, k1, k2, u0, v0)
        cam_pos = [[2, 1, 0], [2, -1, 0]]
        target_pos = [0.5, 0, 0]
        up_vector = [0, -1, 0]
        cam_i1 = PinholeCameraCal3Bundler().Lookat(cam_pos[0], target_pos, up_vector, calibration)
        cam_i2 = PinholeCameraCal3Bundler().Lookat(cam_pos[1], target_pos, up_vector, calibration)
        keypoints_i1 = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))
        keypoints_i2 = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))

        # Project keypoint at center of each simulated image and record intersection.
        is_inlier, reproj_err = metric_utils.mesh_inlier_correspondences(
            keypoints_i1, keypoints_i2, cam_i1, cam_i2, box, dist_threshold=0.1
        )
        assert np.count_nonzero(is_inlier) == 1
        assert reproj_err[0] < 1e-4

    def test_compute_keypoint_intersections(self) -> None:
        """Tests `compute_keypoint_intersections()` function."""
        # Create cube mesh with side length one centered at origin.
        box = trimesh.primitives.Box()

        # Create arrangement of 4 cameras in x-z plane pointing at the cube.
        fx, k1, k2, u0, v0 = 10, 0, 0, 1, 1
        calibration = Cal3Bundler(fx, k1, k2, u0, v0)
        cam_pos = [[2, 0, 0], [-2, 0, 0], [0, 0, 2], [0, 0, -2]]
        target_pos = [0, 0, 0]
        up_vector = [0, -1, 0]
        cams = [PinholeCameraCal3Bundler().Lookat(c, target_pos, up_vector, calibration) for c in cam_pos]

        # Project keypoint at center of each simulated image and record intersection.
        kpt = Keypoints(coordinates=np.array([[1, 1]]).astype(np.float32))
        expected_intersections = [[0.5, 0, 0], [-0.5, 0, 0], [0, 0, 0.5], [0, 0, -0.5]]
        estimated_intersections = []
        for cam in cams:
            _, intersection = metric_utils.compute_keypoint_intersections(kpt, cam, box, verbose=True)
            estimated_intersections.append(intersection.flatten().tolist())
        np.testing.assert_allclose(expected_intersections, estimated_intersections)

    def test_get_stats_for_sfmdata_skydio32(self) -> None:
        """Verifies that track reprojection errors are returned as NaN if given degenerate input.

        The data used below corresponds to camera poses aligned to GT from Skydio-32 sequence with the SIFT front-end,
        if the Sim(3) object used for alignment is corrupted with negative scale.
        All tracks should have NaN reprojection error, from cheirality errors from the negative scale.
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

        t0 = SfmTrack(pt=[-0.7627727, -0.26624048, -0.11879795])
        t0.addMeasurement(2, [184.08586121, 441.31314087])
        t0.addMeasurement(4, [18.98637581, 453.21853638])

        t1 = SfmTrack(pt=[-0.76277714, -0.26603358, -0.11884205])
        t1.addMeasurement(2, [213.51266479, 288.06637573])
        t1.addMeasurement(4, [50.23059464, 229.30541992])

        t2 = SfmTrack(pt=[-0.7633115, -0.2662322, -0.11826181])
        t2.addMeasurement(2, [227.52420044, 695.15087891])
        t2.addMeasurement(3, [996.67608643, 705.03125])

        t3 = SfmTrack(pt=[-0.76323087, -0.26629859, -0.11836833])
        t3.addMeasurement(2, [251.37863159, 702.97064209])
        t3.addMeasurement(3, [537.9753418, 732.26025391])

        t4 = SfmTrack(pt=[-0.70450081, -0.28115719, -0.19063382])
        t4.addMeasurement(2, [253.17749023, 490.47991943])
        t4.addMeasurement(3, [13.17782784, 507.57717896])

        t5 = SfmTrack(pt=[-0.52781989, -0.31926005, -0.40763909])
        t5.addMeasurement(2, [253.52301025, 478.41384888])
        t5.addMeasurement(3, [10.92995739, 493.31018066])

        t6 = SfmTrack(pt=[-0.74893948, -0.27132075, -0.1360136])
        t6.addMeasurement(2, [254.64611816, 533.04730225])
        t6.addMeasurement(3, [18.78449249, 557.05041504])

        aligned_tracks = [t0, t1, t2, t3, t4, t5, t6]
        aligned_filtered_data = GtsfmData.from_cameras_and_tracks(
            cameras=aligned_cameras, tracks=aligned_tracks, number_images=32
        )
        metrics = metric_utils.get_stats_for_sfmdata(aligned_filtered_data, suffix="_filtered")

        assert metrics[0].name == "number_cameras"
        assert np.isclose(metrics[0]._data, np.array(5.0, dtype=np.float32))

        assert metrics[1].name == "number_tracks_filtered"
        assert np.isclose(metrics[1]._data, np.array(7.0, dtype=np.float32))

        assert metrics[2].name == "3d_track_lengths_filtered"
        assert metrics[2].summary == {
            "min": 2,
            "max": 2,
            "median": 2.0,
            "mean": 2.0,
            "stddev": 0.0,
            "histogram": {"1": 7},
            "len": 7,
            "invalid": 0,
        }

        assert metrics[3].name == "reprojection_errors_filtered_px"
        assert metrics[3].summary == {"min": np.nan, "max": np.nan, "median": np.nan, "mean": np.nan, "stddev": np.nan}


def test_compute_percentage_change_improve() -> None:
    """Ensure that percentage change is computed correctly for a 50% improvement over 100."""
    x = 100
    y = 150
    change_percent = metric_utils.compute_percentage_change(x, y)
    assert np.isclose(change_percent, 50)


def test_compute_percentage_change_static() -> None:
    """Ensure that percentage change is computed correctly for no change in a value."""
    x = 100
    y = 100
    change_percent = metric_utils.compute_percentage_change(x, y)
    assert np.isclose(change_percent, 0)


def test_compute_percentage_change_regression() -> None:
    """Ensure that percentage change is computed correctly for a 99% regression against 100."""
    x = 100
    y = 1
    change_percent = metric_utils.compute_percentage_change(x, y)
    assert np.isclose(change_percent, -99)


def test_pose_auc1() -> None:
    """Area under curve resembles one triangle on the left, and then a rectangle to its right."""
    errors = np.ones(5) * 5.0
    thresholds = [5, 10, 20]

    aucs = metric_utils.pose_auc(errors, thresholds)

    # Sum triangles and rectangles.
    # AUC @ 5 deg thresh: 0. (no cameras under this threshold).
    # AUC @ 10 deg thresh: (5 * 0.2 * (1/2) + 1 * 5.0) / 10
    # AUC @ 20 deg thresh: (5 * 0.2 * (1/2) + 1 * 15.0) / 20
    expected_aucs = [0.0, 0.55, 0.775]

    assert np.allclose(aucs, expected_aucs)


def test_pose_auc_all_zero_errors_perfect_auc() -> None:
    errors = np.zeros(5)

    thresholds = [5, 10, 20]
    aucs = metric_utils.pose_auc(errors, thresholds)
    expected_aucs = [1.0, 1.0, 1.0]
    assert np.allclose(aucs, expected_aucs)


def test_pose_auc_all_errors_exceed_threshold_zero_auc() -> None:
    errors = np.ones(5) * 25.0
    thresholds = [5, 10, 20]
    aucs = metric_utils.pose_auc(errors, thresholds)
    expected_aucs = [0.0, 0.0, 0.0]
    assert np.allclose(aucs, expected_aucs)


def test_pose_auc_works_for_nan_error() -> None:
    thresholds = [1, 2.5, 5, 10, 20]

    pose_errors = np.array(
        [
            0.19827642,
            0.24861091,
            0.3637053,
            0.46981758,
            0.4993183,
            0.9036343,
            1.6756403,
            0.83505183,
            0.28977838,
            0.42773795,
            0.48842856,
            6.654587,
            np.nan,
            0.2563246,
            0.28247333,
            0.32443768,
            0.13612723,
            0.3027622,
            0.2385458,
            2.9347303,
            1.421731,
            0.3097524,
            0.57351094,
            0.5353638,
            0.5003734,
            0.4324144,
            0.22548386,
            0.06921609,
            0.25950527,
            0.9956337,
            0.76206225,
            0.5812757,
        ]
    )

    aucs = metric_utils.pose_auc(pose_errors, thresholds)

    # Note recall is roughly (27 / 32) since exclude 5 errors above 1 deg -> (1.422, 1.676, 2.935, 6.655,   nan)
    # If we drew triangle up to recall point, we would get 0.84 * 0.5 -> 0.42, but more
    # mass lies above diagonal, so get to roughly 0.5 AUC.
    expected_auc_at_1_deg = 0.4996
    assert np.isclose(aucs[0], expected_auc_at_1_deg, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
