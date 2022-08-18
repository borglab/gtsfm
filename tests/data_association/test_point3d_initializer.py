"""Unit tests for initialization of 3D landmark from tracks of 2D measurements across cameras. We use example SFM data
from gtsam (found at gtsam/python/gtsam/examples/SFMdata.py) which creates 8 cameras uniformly spaced around a circle
with radius 40m.

Authors: Ayush Baid
"""
import copy
import pickle
import unittest
from pathlib import Path
from typing import List

import numpy as np
from gtsam import Cal3_S2, Cal3Bundler, PinholeCameraCal3Bundler, Point2, Point3, Pose3, Rot3
from gtsam.examples import SFMdata

from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (Point3dInitializer, TriangulationOptions,
                                                        TriangulationSamplingMode)
from gtsfm.loader.olsson_loader import OlssonLoader

# path for data used in this test
DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DOOR_TRACKS_PATH = DATA_ROOT_PATH / "tracks2d_door.pickle"
DOOR_DATASET_PATH = DATA_ROOT_PATH / "set1_lund_door"

# focal length set to 50 px, with `px`, `py` set to zero
CALIBRATION = Cal3Bundler(50, 0, 0, 0, 0)
# Generate 8 camera poses arranged in a circle of radius 40 m
CAMERAS = {
    i: PinholeCameraCal3Bundler(pose, CALIBRATION)
    for i, pose in enumerate(
        SFMdata.createPoses(Cal3_S2(CALIBRATION.fx(), CALIBRATION.fx(), 0, CALIBRATION.px(), CALIBRATION.py()))
    )
}
LANDMARK_POINT = Point3(0.0, 0.0, 0.0)
MEASUREMENTS = [SfmMeasurement(i, cam.project(LANDMARK_POINT)) for i, cam in CAMERAS.items()]


def get_track_with_one_outlier() -> List[SfmMeasurement]:
    """Generates a track with outlier measurement."""
    # perturb one measurement
    idx_to_perturb = 5

    perturbed_measurements = copy.deepcopy(MEASUREMENTS)

    original_measurement = perturbed_measurements[idx_to_perturb]
    perturbed_measurements[idx_to_perturb] = SfmMeasurement(
        original_measurement.i, perturbed_measurements[idx_to_perturb].uv + Point2(20.0, -10.0)
    )

    return perturbed_measurements


def get_track_with_duplicate_measurements() -> List[SfmMeasurement]:
    """Generates a track with 2 measurements in an image."""

    new_measurements = copy.deepcopy(MEASUREMENTS)

    new_measurements.append(SfmMeasurement(new_measurements[0].i, new_measurements[0].uv + Point2(2.0, -3.0)))

    return new_measurements


class TestTriangulationOptions(unittest.TestCase):
    """Unit tests for TriangulationOptions"""

    def test_options_ransac(self) -> None:
        """Asserts values of default RANSAC options."""
        triangulation_options = TriangulationOptions(
            reproj_error_threshold=5, mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM
        )
        assert triangulation_options.num_ransac_hypotheses() == 2749

    def test_options_ransac_min_hypotheses(self) -> None:
        """Assert that number of hypotheses is overwritten if less than minimum."""
        triangulation_options = TriangulationOptions(
            reproj_error_threshold=5, mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM, min_num_hypotheses=10000
        )
        assert triangulation_options.num_ransac_hypotheses() == 10000

    def test_options_ransac_max_hypotheses(self) -> None:
        """Assert that number of hypotheses is overwritten if greater than maximum."""
        triangulation_options = TriangulationOptions(
            reproj_error_threshold=5,
            mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM,
            min_inlier_ratio=1e-4,
            max_num_hypotheses=1000,
        )
        assert triangulation_options.num_ransac_hypotheses() == 1000


class TestPoint3dInitializer(unittest.TestCase):
    """Unit tests for Point3dInitializer."""

    def setUp(self):
        super().setUp()

        self.simple_triangulation_initializer = Point3dInitializer(
            CAMERAS, TriangulationOptions(reproj_error_threshold=5, mode=TriangulationSamplingMode.NO_RANSAC)
        )

        self.ransac_uniform_sampling_initializer = Point3dInitializer(
            CAMERAS,
            TriangulationOptions(
                reproj_error_threshold=5, mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM, min_num_hypotheses=100
            ),
        )

    def __runWithCorrectMeasurements(self, obj: Point3dInitializer) -> bool:
        """Run the initialization with a track with all correct measurements, and checks for correctness of the
        recovered 3D point."""

        sfm_track, _, _ = obj.triangulate(SfmTrack2d(MEASUREMENTS))
        point3d = sfm_track.point3()

        return np.allclose(point3d, LANDMARK_POINT)

    def __runWithTwoMeasurements(self, obj: Point3dInitializer) -> bool:
        """Run the initialization with a track with all correct measurements, and checks for correctness of the
        recovered 3D point."""

        sfm_track, _, _ = obj.triangulate(SfmTrack2d(MEASUREMENTS[:2]))
        point3d = sfm_track.point3()

        return np.allclose(point3d, LANDMARK_POINT)

    def __runWithOneMeasurement(self, obj: Point3dInitializer) -> bool:
        """Run initialization with a track with all correct measurements, and checks for a None track as a result."""
        sfm_track, _, _ = obj.triangulate(SfmTrack2d(MEASUREMENTS[:1]))

        return sfm_track is None

    def __runWithSingleOutlier(self, obj: Point3dInitializer) -> bool:
        """Run the initialization for a track with all inlier measurements except one, and checks for correctness of
        the estimated point."""

        sfm_track, _, _ = obj.triangulate(SfmTrack2d(get_track_with_one_outlier()))
        point3d = sfm_track.point3()

        return np.array_equal(point3d, LANDMARK_POINT)

    def __runWithCheiralityException(self, obj: Point3dInitializer) -> bool:
        """Run the initialization in a a-cheiral setup, and check that the result is a None track."""

        cameras = obj.track_camera_dict

        # flip the cameras first
        yaw = np.pi
        camera_flip_pose = Pose3(Rot3.RzRyRx(yaw, 0, 0), np.zeros((3, 1)))
        flipped_cameras = {
            i: PinholeCameraCal3Bundler(cam.pose().compose(camera_flip_pose), cam.calibration())
            for i, cam in cameras.items()
        }

        obj_with_flipped_cameras = Point3dInitializer(flipped_cameras, obj.options)

        sfm_track, _, _ = obj_with_flipped_cameras.triangulate(SfmTrack2d(MEASUREMENTS))

        return sfm_track is None

    def __runWithDuplicateMeasurements(self, obj: Point3dInitializer) -> bool:
        """Run the initialization for a track with all inlier measurements except one, and checks for correctness of
        the estimated point."""

        sfm_track, _, _ = obj.triangulate(SfmTrack2d(get_track_with_duplicate_measurements()))
        point3d = sfm_track.point3()

        return np.allclose(point3d, LANDMARK_POINT, atol=1, rtol=1e-1)

    def testSimpleTriangulationWithCorrectMeasurements(self):
        self.assertTrue(self.__runWithCorrectMeasurements(self.simple_triangulation_initializer))

    def testSimpleTriangulationWithTwoMeasurements(self):
        self.assertTrue(self.__runWithTwoMeasurements(self.simple_triangulation_initializer))

    def testSimpleTriangulationWithOneMeasurement(self):
        self.assertTrue(self.__runWithOneMeasurement(self.simple_triangulation_initializer))

    def testSimpleTriangulationWithOutlierMeasurements(self):
        sfm_track, _, _ = self.simple_triangulation_initializer.triangulate(SfmTrack2d(get_track_with_one_outlier()))
        self.assertIsNone(sfm_track)

    def testSimpleTriangulationWithCheiralityException(self):
        self.assertTrue(self.__runWithCheiralityException(self.simple_triangulation_initializer))

    def testSimpleTriangulationWithDuplicateMeaseurements(self):
        self.assertTrue(self.__runWithDuplicateMeasurements(self.simple_triangulation_initializer))

    def testRansacUniformSamplingWithCorrectMeasurements(self):
        self.assertTrue(self.__runWithCorrectMeasurements(self.ransac_uniform_sampling_initializer))

    def testRansacUniformSamplingWithTwoMeasurements(self):
        self.assertTrue(self.__runWithTwoMeasurements(self.simple_triangulation_initializer))

    def testRansacUniformSamplingWithOneMeasurement(self):
        self.assertTrue(self.__runWithOneMeasurement(self.simple_triangulation_initializer))

    def testRansacUniformSamplingWithOutlierMeasurements(self):
        self.assertTrue(self.__runWithSingleOutlier(self.ransac_uniform_sampling_initializer))

    def testRansacUniformSamplingWithCheiralityException(self):
        self.assertTrue(self.__runWithCheiralityException(self.ransac_uniform_sampling_initializer))

    def testRansacUniformSamplingWithDuplicateMeaseurements(self):
        self.assertTrue(self.__runWithDuplicateMeasurements(self.ransac_uniform_sampling_initializer))

    def testSimpleTriangulationOnDoorDataset(self):
        """Test the tracks of the door dataset using simple triangulation initialization. Using computed tracks with
        ground truth camera params.

        Expecting failures on 2 tracks which have incorrect matches.
        """
        with open(DOOR_TRACKS_PATH, "rb") as handle:
            tracks = pickle.load(handle)

        loader = OlssonLoader(DOOR_DATASET_PATH, image_extension="JPG", max_resolution=1296)

        camera_dict = {
            i: PinholeCameraCal3Bundler(loader.get_camera_pose(i), loader.get_camera_intrinsics(i))
            for i in range(len(loader))
        }

        initializer = Point3dInitializer(
            camera_dict, TriangulationOptions(mode=TriangulationSamplingMode.NO_RANSAC, reproj_error_threshold=1e5)
        )

        # tracks which have expected failures
        # (both tracks have incorrect measurements)
        expected_failures = [
            SfmTrack2d(
                measurements=[
                    SfmMeasurement(i=1, uv=np.array([1252.22729492, 1487.29431152])),
                    SfmMeasurement(i=2, uv=np.array([1170.96679688, 1407.35876465])),
                    SfmMeasurement(i=4, uv=np.array([263.32104492, 1489.76965332])),
                ]
            ),
            SfmTrack2d(
                measurements=[
                    SfmMeasurement(i=6, uv=np.array([1142.34545898, 735.92169189])),
                    SfmMeasurement(i=7, uv=np.array([1179.84155273, 763.04095459])),
                    SfmMeasurement(i=9, uv=np.array([216.54107666, 774.74017334])),
                ]
            ),
        ]

        for track_2d in tracks:
            triangulated_track, _, _ = initializer.triangulate(track_2d)

            if triangulated_track is None:
                # assert we have failures which are already expected
                self.assertIn(track_2d, expected_failures)


class TestPoint3dInitializerUnestimatedCameras(unittest.TestCase):
    """Unit tests for Point3dInitializer when a camera pose could not be estimated."""

    def setUp(self):
        """Data taken from Skydio-CraneMast-8, with images resized to 760 px resolution."""
        super().setUp()

        fx, k1, k2, u0, v0 = 583.1175, 0, 0, 507, 380
        calibration = Cal3Bundler(fx, k1, k2, u0, v0)

        wRi1 = np.array(
            [
                [-0.736357028, -0.589757459, 0.331608908],
                [0.565525823, -0.805544778, -0.176856308],
                [0.371428151, 0.0573040157, 0.926691631],
            ]
        )
        wti1 = np.array([-0.649658109, 0.656963354, -0.382548681])
        wTi1 = Pose3(Rot3(wRi1), wti1)

        wRi2 = np.array(
            [
                [-0.802732442, -0.594660358, 0.0447178336],
                [0.59626145, -0.80157983, 0.0440688035],
                [0.009638943, 0.0620389785, 0.998027182],
            ]
        )
        wti2 = np.array([-1.95517763, 1.44100216, -0.442089696])
        wTi2 = Pose3(Rot3(wRi2), wti2)

        # Should be 3 cameras, but one is unestimated. Since camera 0 is unestimated, triangulation
        # cannot succeed later if only 2 views are provided and one of them is from camera 0.
        cameras = {1: PinholeCameraCal3Bundler(wTi1, calibration), 2: PinholeCameraCal3Bundler(wTi2, calibration)}

        self.triangulator = Point3dInitializer(
            cameras, TriangulationOptions(mode=TriangulationSamplingMode.NO_RANSAC, reproj_error_threshold=5)
        )

    def test_extract_measurements_unestimated_camera(self) -> None:
        """Ensure triangulation args are None for length-2 tracks where one or more measurements come from
        unestimated cameras.

        In the corresponding camera data, we have only 1 valid view within the track.
        The function `extract_measurements()` should return None for the GTSAM primitives it generates.
        """
        inlier_track = SfmTrack2d(
            measurements=[
                SfmMeasurement(i=0, uv=np.array([229.0, 500.0], dtype=np.float32)),
                SfmMeasurement(i=1, uv=np.array([69.0, 532.0], dtype=np.float32)),
            ]
        )
        track_cameras, track_measurements = self.triangulator.extract_measurements(inlier_track)
        assert track_cameras is None
        assert track_measurements is None
