""""Reproducibility tests for verifiers.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path
from typing import Tuple

from gtsam import Rot3, Unit3

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import gtsfm.utils.io as io_utils
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from tests.repro_tests.test_repro_base import ReproducibilityTestBase

# defining the path for test data
TEST_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data"
IMG1_PATH = TEST_DATA_ROOT_PATH / "crane_mast_8imgs_colmap_output" / "images" / "crane_mast_1.jpg"
IMG2_PATH = TEST_DATA_ROOT_PATH / "crane_mast_8imgs_colmap_output" / "images" / "crane_mast_2.jpg"

VERIFIER_RESULT_TYPE = Tuple[Rot3, Unit3]

ROT3_DIFF_ANGLE_THRESHOLD_DEG = 0.001
UNIT3_DIFF_ANGLE_THRESHOLD_DEG = 0.001


class VerifierReproducibilityBase(ReproducibilityTestBase, unittest.TestCase):
    def setUp(
        self, verifier: VerifierBase = Ransac(use_intrinsics_in_verification=False, estimation_threshold_px=3)
    ) -> None:
        super().setUp()
        self._verifier: VerifierBase = verifier
        detector_descriptor = SIFTDetectorDescriptor()
        matcher = TwoWayMatcher()
        self._image_i1 = io_utils.load_image(str(IMG1_PATH))
        self._image_i2 = io_utils.load_image(str(IMG2_PATH))
        self._keypoints_i1, descriptors_i1 = detector_descriptor.apply(self._image_i1)
        self._keypoints_i2, descriptors_i2 = detector_descriptor.apply(self._image_i2)
        self._match_indices = matcher.apply(
            self._keypoints_i1,
            self._keypoints_i2,
            descriptors_i1,
            descriptors_i2,
            (self._image_i1.height, self._image_i1.width),
            (self._image_i2.height, self._image_i2.width),
        )

    def run_once(self) -> VERIFIER_RESULT_TYPE:
        return self._verifier.verify(
            keypoints_i1=self._keypoints_i1,
            keypoints_i2=self._keypoints_i2,
            match_indices=self._match_indices,
            camera_intrinsics_i1=self._image_i1.get_intrinsics_from_exif(),
            camera_intrinsics_i2=self._image_i2.get_intrinsics_from_exif(),
        )[:2]

    def assert_results(self, results_a: VERIFIER_RESULT_TYPE, results_b: VERIFIER_RESULT_TYPE) -> None:
        print(results_a[0])
        print(results_b[0])
        self.assertLessEqual(
            geometry_comparisons.compute_relative_rotation_angle(results_a[0], results_b[0]),
            ROT3_DIFF_ANGLE_THRESHOLD_DEG,
        )
        self.assertLessEqual(
            geometry_comparisons.compute_relative_unit_translation_angle(results_a[1], results_b[1]),
            UNIT3_DIFF_ANGLE_THRESHOLD_DEG,
        )
