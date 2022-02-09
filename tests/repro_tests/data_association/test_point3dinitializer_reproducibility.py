"""Reproducibility tests for Point3D initializer, in different configurations.

Authors: Ayush Baid
"""
import unittest
from typing import Dict, Optional, Tuple

import gtsam
import numpy as np
from gtsam import PinholeCameraCal3Bundler, SfmTrack

import gtsfm.utils.io as io_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationOptions,
    TriangulationSamplingMode,
    TriangulationExitCode,
)
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

# Load ground truth data with 3 images and 7 tracks.
GTSAM_EXAMPLE_FILE: str = "dubrovnik-3-7-pre"
GROUND_TRUTH_DATA: GtsfmData = io_utils.read_bal(gtsam.findExampleDataFile(GTSAM_EXAMPLE_FILE))

MEASUREMENTS_NOISE_STANDARD_DEVIATION = 1  # in pixels

TRIANGULATION_RESULT_TYPE = Tuple[Optional[SfmTrack], Optional[float], TriangulationExitCode]


class TestPoint3dInitializerNoRansac(ReproducibilityTestBase, unittest.TestCase):
    """Reproducibility tests for point3d initializer with no RANSAC."""

    def setUp(
        self,
        triangulation_options: TriangulationOptions = TriangulationOptions(
            reproj_error_threshold=3, mode=TriangulationSamplingMode.NO_RANSAC
        ),
    ) -> None:
        camera_dict: Dict[int, PinholeCameraCal3Bundler] = {
            i: GROUND_TRUTH_DATA.get_camera(i) for i in range(GROUND_TRUTH_DATA.number_images())
        }
        self._point3d_initializer: Point3dInitializer = Point3dInitializer(camera_dict, triangulation_options)
        track_2d_noisefree: SfmTrack2d = track_utils.cast_3dtrack_to_2dtrack(GROUND_TRUTH_DATA.get_track(0))

        # add noise to all the points
        self._track_2d: SfmTrack2d = SfmTrack2d(
            measurements=[
                SfmMeasurement(
                    measurement.i, measurement.uv + MEASUREMENTS_NOISE_STANDARD_DEVIATION * np.random.randn(2)
                )
                for measurement in track_2d_noisefree.measurements
            ]
        )

    def run_once(self) -> TRIANGULATION_RESULT_TYPE:
        return self._point3d_initializer.triangulate(self._track_2d)

    def assert_results(self, results_a: TRIANGULATION_RESULT_TYPE, results_b: TRIANGULATION_RESULT_TYPE) -> None:
        self.__assert_track3d(results_a[0], results_b[0])
        self.assertAlmostEqual(results_a[1], results_b[1])
        self.assertEqual(results_a[2], results_b[2])

    def __assert_track3d(self, track3d_a: Optional[SfmTrack], track3d_b: Optional[SfmTrack]) -> None:
        # all tracks will be non-None for the data in this test.
        self.assertIsNotNone(track3d_a)
        self.assertIsNotNone(track3d_b)
        np.testing.assert_allclose(track3d_a.point3(), track3d_b.point3())
        self.assertEqual(track3d_a.number_measurements(), track3d_b.number_measurements())
        for measurement_idx in range(track3d_a.number_measurements()):
            measurement_a: Tuple[int, np.ndarray] = track3d_a.measurement(measurement_idx)
            measurement_b: Tuple[int, np.ndarray] = track3d_b.measurement(measurement_idx)
            self.assertEqual(measurement_a[0], measurement_b[0])
            np.testing.assert_allclose(measurement_a[1], measurement_b[1])


class TestPoint3dInitializerRansacSampleUniform(TestPoint3dInitializerNoRansac):
    """Reproducibility tests for point3d initializer in RANSAC with uniform sampling mode."""

    def setUp(
        self,
        triangulation_options: TriangulationOptions = TriangulationOptions(
            reproj_error_threshold=3, mode=TriangulationSamplingMode.RANSAC_SAMPLE_UNIFORM
        ),
    ) -> None:
        super().setUp(triangulation_options)

        # replace one measurement with an outlier
        measurements_with_one_outlier = self._track_2d.measurements
        measurements_with_one_outlier[2] = SfmMeasurement(measurements_with_one_outlier[2].i, np.array([50, 30]))
        self._track_2d = SfmTrack2d(measurements=measurements_with_one_outlier)


class TestPoint3dInitializerRansacTopKBaselines(TestPoint3dInitializerRansacSampleUniform):
    """Reproducibility tests for point3d initializer in RANSAC with top-k baselines mode."""

    def setUp(self) -> None:
        super().setUp(
            triangulation_options=TriangulationOptions(
                reproj_error_threshold=3, mode=TriangulationSamplingMode.RANSAC_TOPK_BASELINES
            )
        )


class TestPoint3dInitializerRansacSampleBiasedBaseline(TestPoint3dInitializerRansacSampleUniform):
    """Reproducibility tests for point3d initializer in RANSAC with biased baseline sampling."""

    def setUp(self) -> None:
        super().setUp(
            triangulation_options=TriangulationOptions(
                reproj_error_threshold=3, mode=TriangulationSamplingMode.RANSAC_SAMPLE_BIASED_BASELINE
            )
        )
