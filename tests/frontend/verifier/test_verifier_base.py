"""Tests for frontend's base verifier class.

Authors: Ayush Baid
"""

import pickle
import random
import unittest
from typing import Optional, Tuple, Union

import dask
import numpy as np
from gtsam import (
    Cal3Bundler,
    EssentialMatrix,
    PinholeCameraCal3Bundler,
    Pose3,
    Rot3,
    Unit3,
)

import gtsfm.utils.features as feature_utils
import gtsfm.utils.geometry_comparisons as geom_comp_utils
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.dummy_verifier import DummyVerifier

RANDOM_SEED = 15

ROTATION_ANGULAR_ERROR_DEG_THRESHOLD = 2
DIRECTION_ANGULAR_ERROR_DEG_THRESHOLD = 2


class TestVerifierBase(unittest.TestCase):
    """Unit tests for the Base Verifier class.

    Should be inherited by all verifier unit tests.
    """

    def setUp(self):
        super().setUp()

        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        self.verifier = DummyVerifier()

    def __execute_test(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        match_indices: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
        i2Ri1_expected: Rot3,
        i2Ui1_expected: Unit3,
        verified_indices_expected: np.ndarray,
        use_intrinsics_in_verification_vals: Union[Tuple[bool], Tuple[bool, bool]] = (True, False),
    ) -> None:
        """Execute the test both w/ and w/o intrinsics in verification."""

        for use_intrinsics_in_verification in use_intrinsics_in_verification_vals:
            i2Ri1_computed, i2Ui1_computed, verified_indices_computed = self.verifier.verify(
                keypoints_i1,
                keypoints_i2,
                match_indices,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
                use_intrinsics_in_verification,
            )

            if i2Ri1_expected is None:
                self.assertIsNone(i2Ri1_computed)
            else:
                self.assertLess(
                    geom_comp_utils.compute_relative_rotation_angle(i2Ri1_expected, i2Ri1_computed),
                    ROTATION_ANGULAR_ERROR_DEG_THRESHOLD,
                )
            if i2Ui1_expected is None:
                self.assertIsNone(i2Ui1_computed)
            else:
                self.assertLess(
                    geom_comp_utils.compute_relative_unit_translation_angle(i2Ui1_expected, i2Ui1_computed),
                    DIRECTION_ANGULAR_ERROR_DEG_THRESHOLD,
                )
            np.testing.assert_array_equal(verified_indices_computed, verified_indices_expected)

    def test_simple_scene(self):
        """Test a simple scene with 10 points, 5 each on 2 planes, so that RANSAC family of methods do not
        get trapped into a degenerate sample."""
        if isinstance(self.verifier, DummyVerifier):
            self.skipTest("Cannot check correctness for dummy verifier")

        # obtain the keypoints and the ground truth essential matrix.
        keypoints_i1, keypoints_i2, i2Ei1_expected = simulate_two_planes_scene(5, 5)

        # match keypoints row by row
        match_indices = np.vstack((np.arange(len(keypoints_i1)), np.arange(len(keypoints_i1)))).T

        # run the test w/ and w/o using intrinsics in verification
        self.__execute_test(
            keypoints_i1,
            keypoints_i2,
            match_indices,
            Cal3Bundler(),
            Cal3Bundler(),
            i2Ei1_expected.rotation(),
            i2Ei1_expected.direction(),
            match_indices,
            use_intrinsics_in_verification_vals=(False,),
        )

    def test_valid_verified_indices(self):
        """Test if valid indices in output."""

        # Repeat the experiment 10 times as we might not have successful
        # verification every time.

        for _ in range(10):
            (
                _,
                _,
                verified_indices,
                keypoints_i1,
                keypoints_i2,
            ) = self.__verify_random_inputs_using_intrinsics_in_verification()

            if verified_indices.size > 0:
                # check that the indices are not out of bounds
                self.assertTrue(np.all(verified_indices >= 0))
                self.assertTrue(np.all(verified_indices[:, 0] < len(keypoints_i1)))
                self.assertTrue(np.all(verified_indices[:, 1] < len(keypoints_i2)))
            else:
                # we have a meaningless test
                self.skipTest("No valid results found")

    def test_verify_empty_matches(self):
        """Tests the output when there are no match indices."""

        keypoints_i1 = feature_utils.generate_random_keypoints(10, [250, 300])
        keypoints_i2 = feature_utils.generate_random_keypoints(12, [400, 300])
        match_indices = np.array([], dtype=np.int32)
        intrinsics_i1 = Cal3Bundler()
        intrinsics_i2 = Cal3Bundler()

        self.__execute_test(
            keypoints_i1,
            keypoints_i2,
            match_indices,
            intrinsics_i1,
            intrinsics_i2,
            i2Ri1_expected=None,
            i2Ui1_expected=None,
            verified_indices_expected=np.array([], dtype=np.uint32),
        )

    def test_create_computation_graph(self):
        """Checks that the dask computation graph produces the same results as direct APIs."""

        # creating inputs for verification
        (keypoints_i1, keypoints_i2, matches_i1i2, intrinsics_i1, intrinsics_i2,) = generate_random_input_for_verifier()

        # and use GTSFM's direct API to get expected results

        for use_intrinsics_in_verification in (True, False):

            expected_results = self.verifier.verify(
                keypoints_i1, keypoints_i2, matches_i1i2, intrinsics_i1, intrinsics_i2, use_intrinsics_in_verification
            )

            expected_i2Ri1 = expected_results[0]
            expected_i2Ui1 = expected_results[1]
            expected_v_corr_idxs = expected_results[2]

            # generate the computation graph for the verifier
            (delayed_i2Ri1, delayed_i2Ui1, delayed_v_corr_idxs,) = self.verifier.create_computation_graph(
                dask.delayed(keypoints_i1),
                dask.delayed(keypoints_i2),
                dask.delayed(matches_i1i2),
                dask.delayed(intrinsics_i1),
                dask.delayed(intrinsics_i2),
                use_intrinsics_in_verification,
            )

            with dask.config.set(scheduler="single-threaded"):
                i2Ri1, i2Ui1, v_corr_idxs = dask.compute(delayed_i2Ri1, delayed_i2Ui1, delayed_v_corr_idxs)

            if expected_i2Ri1 is None:
                self.assertIsNone(i2Ri1)
            else:
                self.assertTrue(expected_i2Ri1.equals(i2Ri1, 1e-2))
            if expected_i2Ui1 is None:
                self.assertIsNone(i2Ui1)
            else:
                self.assertTrue(expected_i2Ui1.equals(i2Ui1, 1e-2))
            np.testing.assert_array_equal(v_corr_idxs, expected_v_corr_idxs)

    def test_pickleable(self):
        """Tests that the verifier object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.verifier)
        except TypeError:
            self.fail("Cannot dump verifier using pickle")

    def __verify_random_inputs_using_intrinsics_in_verification(
        self,
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray, Keypoints, Keypoints]:
        """Generates random inputs for pair (#i1, #i2) and perform verification using the intrinsics.

        Returns:
            Computed relative rotation i2Ri1.
            Computed translation direction i2Ui1.
            Indices of keypoints which are verified correspondences.
            Keypoints from #i1 which were input to the verifier.
            Keypoints from #i2 which were input to the verifier.
        """

        (
            keypoints_i1,
            keypoints_i2,
            match_indices,
            intrinsics_i1,
            intrinsics_i2,
        ) = generate_random_input_for_verifier()

        (i2Ri1, i2Ui1, verified_indices,) = self.verifier.verify(
            keypoints_i1, keypoints_i2, match_indices, intrinsics_i1, intrinsics_i2, use_intrinsics_in_verification=True
        )

        return i2Ri1, i2Ui1, verified_indices, keypoints_i1, keypoints_i2


def generate_random_input_for_verifier() -> Tuple[Keypoints, Keypoints, np.ndarray, Cal3Bundler, Cal3Bundler]:
    """Generates random inputs for verification.

    Returns:
        Keypoints for image #i1.
        Keypoints for image #i2.
        Indices of match between image pair (#i1, #i2).
        Intrinsics for image #i1.
        Intrinsics for image #i2.
    """

    # Randomly generate number of keypoints
    num_keypoints_i1 = random.randint(0, 100)
    num_keypoints_i2 = random.randint(0, 100)

    # randomly generate image shapes
    image_shape_i1 = (random.randint(100, 400), random.randint(100, 400))
    image_shape_i2 = (random.randint(100, 400), random.randint(100, 400))

    # generate intrinsics from image shapes
    intrinsics_i1 = Cal3Bundler(
        fx=min(image_shape_i1[0], image_shape_i1[1]), k1=0, k2=0, u0=image_shape_i1[0] / 2, v0=image_shape_i1[1] / 2,
    )

    intrinsics_i2 = Cal3Bundler(
        fx=min(image_shape_i2[0], image_shape_i2[1]), k1=0, k2=0, u0=image_shape_i2[0] / 2, v0=image_shape_i2[1] / 2,
    )

    # randomly generate the keypoints
    keypoints_i1 = feature_utils.generate_random_keypoints(num_keypoints_i1, image_shape_i1)
    keypoints_i2 = feature_utils.generate_random_keypoints(num_keypoints_i2, image_shape_i2)

    # randomly generate matches
    num_matches = random.randint(1, min(num_keypoints_i1, num_keypoints_i2))
    if num_matches == 0:
        matching_indices_i1i2 = np.array([], dtype=np.uint32)
    else:
        matching_indices_i1i2 = np.empty((num_matches, 2), dtype=np.uint32)
        matching_indices_i1i2[:, 0] = np.random.choice(num_keypoints_i1, size=(num_matches,), replace=False)
        matching_indices_i1i2[:, 1] = np.random.choice(num_keypoints_i2, size=(num_matches,), replace=False)

    return (
        keypoints_i1,
        keypoints_i2,
        matching_indices_i1i2,
        intrinsics_i1,
        intrinsics_i2,
    )


def sample_points_on_plane(
    plane_coefficients: Tuple[float, float, float, float],
    range_x_coordinate: Tuple[float, float],
    range_y_coordinate: Tuple[float, float],
    num_points: int,
) -> np.ndarray:
    """Sample random points on a 3D plane ax + by + cz + d = 0.

    Args:
        plane_coefficients: coefficients (a,b,c,d) of the plane equation.
        range_x_coordinate: desired range of the x coordinates of samples.
        range_y_coordinate: desired range of the y coordinates of samples.
        num_points: number of points to sample.

    Returns:
        3d points on the plane, of shape (num_points, 3).
    """

    a, b, c, d = plane_coefficients

    if c == 0:
        raise ValueError("z-coefficient for the plane should not be zero")

    # sample x and y coordinates randomly
    x = np.random.uniform(low=range_x_coordinate[0], high=range_x_coordinate[1], size=(num_points, 1),)
    y = np.random.uniform(low=range_y_coordinate[0], high=range_y_coordinate[1], size=(num_points, 1),)

    # calculate z coordinates using equation of the plane
    z = -(a * x + b * y + d) / c

    pts = np.hstack([x, y, z])

    # assert points are on the plane
    pts_residuals = pts @ np.array(plane_coefficients[:3]).reshape(3, 1) + plane_coefficients[3]
    np.testing.assert_almost_equal(pts_residuals, np.zeros((num_points, 1)))

    return pts


def simulate_two_planes_scene(M: int, N: int) -> Tuple[Keypoints, Keypoints, EssentialMatrix]:
    """Generate a scene where 3D points are on two planes, and projects the points to the 2 cameras. There are M points
    on plane 1, and N points on plane 2.

    The two planes in this test are:
    1. -10x -y -20z +150 = 0
    2. 15x -2y -35z +200 = 0

    Args:
        M: number of points on 1st plane.
        N: number of points on 2nd plane.

    Returns:
        keypoints for image i1, of length (M+N).
        keypoints for image i2, of length (M+N).
        Essential matrix i2Ei1.
    """
    # range of 3D points
    range_x_coordinate = (-5, 7)
    range_y_coordinate = (-10, 10)

    # define the plane equation
    plane1_coeffs = (-10, -1, -20, 150)
    plane2_coeffs = (15, -2, -35, 200)

    # sample the points from planes
    plane1_points = sample_points_on_plane(plane1_coeffs, range_x_coordinate, range_y_coordinate, M)
    plane2_points = sample_points_on_plane(plane2_coeffs, range_x_coordinate, range_y_coordinate, N)

    points_3d = np.vstack((plane1_points, plane2_points))

    # define the camera poses and compute the essential matrix
    wti1 = np.array([0.1, 0, -20])
    wti2 = np.array([1, -2, -20.4])

    wRi1 = Rot3.RzRyRx(np.pi / 20, 0, 0.0)
    wRi2 = Rot3.RzRyRx(0.0, np.pi / 6, 0.0)

    wTi1 = Pose3(wRi1, wti1)
    wTi2 = Pose3(wRi2, wti2)
    i2Ti1 = wTi2.between(wTi1)

    i2Ei1 = EssentialMatrix(i2Ti1.rotation(), Unit3(i2Ti1.translation()))

    # project 3D points to 2D image measurements
    intrinsics = Cal3Bundler()
    camera_i1 = PinholeCameraCal3Bundler(wTi1, intrinsics)
    camera_i2 = PinholeCameraCal3Bundler(wTi2, intrinsics)

    uv_im1 = []
    uv_im2 = []
    for point in points_3d:
        uv_im1.append(camera_i1.project(point))
        uv_im2.append(camera_i2.project(point))

    uv_im1 = np.vstack(uv_im1)
    uv_im2 = np.vstack(uv_im2)

    # return the points as keypoints and the essential matrix
    return Keypoints(coordinates=uv_im1), Keypoints(coordinates=uv_im2), i2Ei1


if __name__ == "__main__":
    unittest.main()
