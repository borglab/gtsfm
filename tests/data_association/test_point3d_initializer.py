"""Unit tests for initialization of 3D landmark from tracks of 2D measurements
across cameras. We use example SFM data from gtsam (found at
gtsam/python/gtsam/examples/SFMdata.py) which creates 8 cameras uniformly
spaced around a circle with radius 40m.

Authors: Ayush Baid
"""
import copy
import unittest
from typing import List

import numpy as np
from gtsam import (
    Cal3_S2,
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point2,
    Point3,
    Pose3,
    Rot3,
)
from gtsam.examples import SFMdata

from gtsfm.data_association.feature_tracks import SfmMeasurement, SfmTrack
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationParam,
)

# focal length set to 50 px, with `px`, `py` set to zero
CALIBRATION = Cal3Bundler(50, 0, 0, 0, 0)
# Generate 8 camera poses arranged in a circle of radius 40 m
CAMERAS = {
    i: PinholeCameraCal3Bundler(pose, CALIBRATION)
    for i, pose in enumerate(
        SFMdata.createPoses(
            Cal3_S2(
                CALIBRATION.fx(),
                CALIBRATION.fx(),
                0,
                CALIBRATION.px(),
                CALIBRATION.py(),
            )
        )
    )
}
LANDMARK_POINT = Point3(0.0, 0.0, 0.0)
MEASUREMENTS = [
    SfmMeasurement(i, cam.project(LANDMARK_POINT)) for i, cam in CAMERAS.items()
]


def get_track_with_one_outlier() -> List[SfmMeasurement]:
    """Generates a track with outlier measurement."""
    # perturb one measurement
    idx_to_perturb = 5

    perturbed_measurements = copy.deepcopy(MEASUREMENTS)

    original_measurement = perturbed_measurements[idx_to_perturb]
    perturbed_measurements[idx_to_perturb] = SfmMeasurement(
        original_measurement.i,
        perturbed_measurements[idx_to_perturb].uv + Point2(20.0, -10.0),
    )

    return perturbed_measurements


def get_track_with_duplicate_measurements() -> List[SfmMeasurement]:
    """Generates a track with 2 measurements in an image."""

    new_measurements = copy.deepcopy(MEASUREMENTS)

    new_measurements.append(
        SfmMeasurement(
            new_measurements[0].i,
            new_measurements[0].uv + Point2(2.0, -3.0),
        )
    )

    return new_measurements


class TestPoint3dInitializer(unittest.TestCase):
    """Unit tests for Point3dInitializer."""

    def setUp(self):
        super().setUp()

        self.simple_triangulation_initializer = Point3dInitializer(
            CAMERAS, TriangulationParam.NO_RANSAC
        )

        self.ransac_uniform_sampling_initializer = Point3dInitializer(
            CAMERAS,
            TriangulationParam.RANSAC_SAMPLE_UNIFORM,
            num_ransac_hypotheses=100,
            reproj_error_thresh=5,
        )

    def __runWithCorrectMeasurements(self, obj: Point3dInitializer) -> bool:
        """Run the initialization with a track with all correct measurements,
        and checks for correctness of the recovered 3D point."""

        sfm_track = obj.triangulate(SfmTrack(MEASUREMENTS))
        point3d = sfm_track.landmark

        return np.allclose(point3d, LANDMARK_POINT)

    def __runWithTwoMeasurements(self, obj: Point3dInitializer) -> bool:
        """Run the initialization with a track with all correct measurements,
        and checks for correctness of the recovered 3D point."""

        sfm_track = obj.triangulate(SfmTrack(MEASUREMENTS[:2]))
        point3d = sfm_track.landmark

        return np.allclose(point3d, LANDMARK_POINT)

    def __runWithOneMeasurement(self, obj: Point3dInitializer) -> bool:
        """Run the initialization with a track with all correct measurements,
        and checks for a None track as a result."""
        sfm_track = obj.triangulate(SfmTrack(MEASUREMENTS[:1]))

        return sfm_track is None

    def __runWithSingleOutlier(self, obj: Point3dInitializer) -> bool:
        """Run the initialization for a track with all inlier measurements
        except one, and checks for correctness of the estimated point."""

        sfm_track = obj.triangulate(SfmTrack(get_track_with_one_outlier()))
        point3d = sfm_track.landmark

        return np.array_equal(point3d, LANDMARK_POINT)

    def __runWithCheiralityException(self, obj: Point3dInitializer) -> bool:
        """Run the initialization in a a-cheiral setup, and check that the
        result is a None track."""

        cameras = obj.track_camera_dict

        # flip the cameras first
        yaw = np.pi
        camera_flip_pose = Pose3(Rot3.RzRyRx(yaw, 0, 0), np.zeros((3, 1)))
        flipped_cameras = {
            i: PinholeCameraCal3Bundler(
                cam.pose().compose(camera_flip_pose), cam.calibration()
            )
            for i, cam in cameras.items()
        }

        obj_with_flipped_cameras = Point3dInitializer(
            flipped_cameras,
            obj.mode,
            obj.num_ransac_hypotheses,
            obj.reproj_error_thresh,
        )

        sfm_track = obj_with_flipped_cameras.triangulate(SfmTrack(MEASUREMENTS))

        return sfm_track is None

    def __runWithDuplicateMeasurements(self, obj: Point3dInitializer) -> bool:
        """Run the initialization for a track with all inlier measurements
        except one, and checks for correctness of the estimated point."""

        sfm_track = obj.triangulate(
            SfmTrack(get_track_with_duplicate_measurements())
        )
        point3d = sfm_track.landmark

        return np.allclose(point3d, LANDMARK_POINT, atol=1, rtol=1e-1)

    def testSimpleTriangulationWithCorrectMeasurements(self):
        self.assertTrue(
            self.__runWithCorrectMeasurements(
                self.simple_triangulation_initializer
            )
        )

    def testSimpleTriangulationWithTwoMeasurements(self):
        self.assertTrue(
            self.__runWithTwoMeasurements(self.simple_triangulation_initializer)
        )

    def testSimpleTriangulationWithOneMeasurement(self):
        self.assertTrue(
            self.__runWithOneMeasurement(self.simple_triangulation_initializer)
        )

    def testSimpleTriangulationWithOutlierMeasurements(self):
        self.assertFalse(
            self.__runWithSingleOutlier(self.simple_triangulation_initializer)
        )

    def testSimpleTriangulationWithCheiralityException(self):
        self.assertTrue(
            self.__runWithCheiralityException(
                self.simple_triangulation_initializer
            )
        )

    def testSimpleTriangulationWithDuplicateMeaseurements(self):
        self.assertTrue(
            self.__runWithDuplicateMeasurements(
                self.simple_triangulation_initializer
            )
        )

    def testRansacUniformSamplingWithCorrectMeasurements(self):
        self.assertTrue(
            self.__runWithCorrectMeasurements(
                self.ransac_uniform_sampling_initializer
            )
        )

    def testRansacUniformSamplingWithTwoMeasurements(self):
        self.assertTrue(
            self.__runWithTwoMeasurements(self.simple_triangulation_initializer)
        )

    def testRansacUniformSamplingWithOneMeasurement(self):
        self.assertTrue(
            self.__runWithOneMeasurement(self.simple_triangulation_initializer)
        )

    def testRansacUniformSamplingWithOutlierMeasurements(self):
        self.assertTrue(
            self.__runWithSingleOutlier(
                self.ransac_uniform_sampling_initializer
            )
        )

    def testRansacUniformSamplingWithCheiralityException(self):
        self.assertTrue(
            self.__runWithCheiralityException(
                self.ransac_uniform_sampling_initializer
            )
        )

    def testRansacUniformSamplingWithDuplicateMeaseurements(self):
        self.assertTrue(
            self.__runWithDuplicateMeasurements(
                self.ransac_uniform_sampling_initializer
            )
        )
