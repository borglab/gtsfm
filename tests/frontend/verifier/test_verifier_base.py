"""Tests for frontend's base verifier class.

Authors: Ayush Baid
"""

import pickle
import random
import unittest
from typing import Tuple

import dask
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Pose3, Rot3, Unit3

from common.keypoints import Keypoints
from frontend.verifier.dummy_verifier import DummyVerifier

RANDOM_SEED = 15


class TestVerifierBase(unittest.TestCase):
    """Unit tests for the Base Verifier class.

    Should be inherited by all verifier unit tests.
    """

    def setUp(self):
        super().setUp()

        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        self.verifier = DummyVerifier()

    def test_simple_scene(self):
        """Test a simple scene with 8 points, 4 on each plane, so that
        RANSAC family of methods do not get trapped into a degenerate sample.
        """
        if isinstance(self.verifier, DummyVerifier):
            self.skipTest('Cannot check correctness for dummy verifier')
        keypoints_i1, keypoints_i2, expected_i2Ei1 = \
            simulate_two_planes_scene(4, 4)

        # match keypoints row by row
        match_indices = np.vstack((
            np.arange(len(keypoints_i1)),
            np.arange(len(keypoints_i1)))).T


<< << << < HEAD
        i2Ri1, i2Ui1, verified_indices = \
            self.verifier.verify_with_approximate_intrinsics(
                keypoints_i1,
                keypoints_i2,
                match_indices,
                Cal3Bundler(),
                Cal3Bundler()
            )

        self.assertTrue(i2Ri1.equals(expected_i2Ei1.rotation(), 1e-2))
        self.assertTrue(i2Ui1.equals(expected_i2Ei1.direction(), 1e-2))
== == == =
        computed_i2Ri1, computed_i2Ui1, verified_indices = self.verifier.verify_with_approximate_intrinsics(
            keypoints_i1,
            keypoints_i2,
            match_indices,
            Cal3Bundler(),
            Cal3Bundler()
        )

        self.assertTrue(computed_i2Ri1.equals(
            expected_i2Ei1.rotation(), 1e-2))
        self.assertTrue(computed_i2Ui1.equals(
            expected_i2Ei1.direction(), 1e-2))
>>>>>> > returning rot3 and unit3 instead of essential matrix
        np.testing.assert_array_equal(verified_indices, match_indices)

    def test_valid_verified_indices(self):
        """Test if valid indices in output."""

        # Repeat the experiment 10 times as we might not have successful
        # verification every time.

        for _ in range(10):
            _, _, verified_indices, keypoints_i1, keypoints_i2 = \
                self.__verify_random_inputs_with_exact_intrinsics()

            if verified_indices.size > 0:
                # check that the indices are not out of bounds
                self.assertTrue(np.all(verified_indices >= 0))
                self.assertTrue(
                    np.all(verified_indices[:, 0] < len(keypoints_i1)))
                self.assertTrue(
                    np.all(verified_indices[:, 1] < len(keypoints_i2)))
            else:
                # we have a meaningless test
                self.skipTest('No valid results found')

    def test_verify_empty_matches(self):
        """Tests the output when there are no match indices."""

        keypoints_i1 = generate_random_keypoints(10, [250, 300])
        keypoints_i2 = generate_random_keypoints(12, [400, 300])
        match_indices = np.array([], dtype=np.int32)
        intrinsics_i1 = Cal3Bundler()
        intrinsics_i2 = Cal3Bundler()

        i2Ri1, i2Ui1, verified_indices = \
            self.verifier.verify_with_exact_intrinsics(
                keypoints_i1, keypoints_i2, match_indices, intrinsics_i1, intrinsics_i2
            )

        self.assertIsNone(i2Ri1)
        self.assertIsNone(i2Ui1)
        self.assertEqual(0, verified_indices.size)

    def test_create_computation_graph(self):
        """Checks that the dask computation graph produces the same results as
        direct APIs."""

        # Set up 3 pairs of inputs to the verifier
        num_images = 6
        image_indices = [(0, 1), (4, 3), (2, 5)]

        # creating inputs for verification and use GTSFM's direct API to get
        # expected results
        keypoints_list = [None]*num_images
        matches_dict = dict()
        intrinsics_list = [None]*num_images

        expected_relative_rotations = dict()
        expected_relative_unit_translations = dict()
        expected_verified_correspondences_indices = dict()
        for (i1, i2) in image_indices:
            keypoints_i1, keypoints_i2, matches_i1i2, \
                intrinsics_i1, intrinsics_i2 = \
                generate_random_input_for_verifier()

            keypoints_list[i1] = keypoints_i1
            keypoints_list[i2] = keypoints_i2

            matches_dict[(i1, i2)] = matches_i1i2

            intrinsics_list[i1] = intrinsics_i1
            intrinsics_list[i2] = intrinsics_i2

            verification_result_i1i2 = \
                self.verifier.verify_with_exact_intrinsics(
                    keypoints_i1,
                    keypoints_i2,
                    matches_i1i2,
                    intrinsics_i1,
                    intrinsics_i2
                )

            expected_relative_rotations[(i1, i2)] = verification_result_i1i2[0]
            expected_relative_unit_translations[(i1, i2)] = \
                verification_result_i1i2[1]
            expected_verified_correspondences_indices[(i1, i2)] = \
                verification_result_i1i2[2]

        # Convert the inputs to computation graphs
        detection_graph = [dask.delayed(x) for x in keypoints_list]
        matcher_graph = {image_indices: dask.delayed(match) for
                         (image_indices, match) in matches_dict.items()}
        intrinsics_graph = [dask.delayed(x) for x in intrinsics_list]

        # generate the computation graph for the verifier
        rotations_graph, unit_translations_graph, \
            verified_correspondence_indices_graph = \
            self.verifier.create_computation_graph(
                detection_graph,
                matcher_graph,
                intrinsics_graph,
                exact_intrinsics_flag=True
            )

        with dask.config.set(scheduler='single-threaded'):
            computed_relative_rotations = dask.compute(rotations_graph)[0]
            computed_relative_unit_translations = dask.compute(
                unit_translations_graph)[0]
            computed_verified_correspondences_indices = \
                dask.compute(verified_correspondence_indices_graph)[0]

        # compare the length of results
        self.assertEqual(len(computed_relative_rotations),
                         len(computed_relative_rotations))
        self.assertEqual(len(computed_relative_unit_translations),
                         len(expected_relative_unit_translations))
        self.assertEqual(len(computed_verified_correspondences_indices),
                         len(expected_verified_correspondences_indices))

        # compare the values
        for indices_i1i2 in computed_relative_rotations.keys():
            computed_i2Ri1 = computed_relative_rotations[indices_i1i2]
            computed_i2Ui1 = computed_relative_unit_translations[indices_i1i2]
            computed_verified_indices_i1i2 = \
                computed_verified_correspondences_indices[indices_i1i2]

            expected_i2Ri1 = expected_relative_rotations[indices_i1i2]
            expected_i2Ui1 = expected_relative_unit_translations[indices_i1i2]
            expected_verified_indices_i1i2 = \
                expected_verified_correspondences_indices[indices_i1i2]

            if expected_i2Ri1 is None:
                self.assertIsNone(computed_i2Ri1)
            else:
                self.assertTrue(expected_i2Ri1.equals(computed_i2Ri1, 1e-2))

            if expected_i2Ui1 is None:
                self.assertIsNone(computed_i2Ui1)
            else:
                self.assertTrue(expected_i2Ui1.equals(computed_i2Ui1, 1e-2))

            np.testing.assert_array_equal(
                computed_verified_indices_i1i2, expected_verified_indices_i1i2)

    def test_pickleable(self):
        """Tests that the verifier object is pickleable (required for dask)."""

        try:
            pickle.dumps(self.verifier)
        except TypeError:
            self.fail("Cannot dump verifier using pickle")

    def __verify_random_inputs_with_exact_intrinsics(self) -> \
            Tuple[Rot3, Unit3, np.ndarray, np.ndarray, np.ndarray]:
        """Generates random inputs for pair (#i1, #i2) and perform verification
        by treating intrinsics as exact.

        Returns:
            Computed relative rotation i2Ri1.
            Computed translation direction i2Ui1.
            Indices of keypoints which are verified correspondences.
            Keypoints from #i1 which were input to the verifier.
            Keypoints from #i2 which were input to the verifier.
        """

        keypoints_i1, keypoints_i2, match_indices, \
            intrinsics_i1, intrinsics_i2 = \
            generate_random_input_for_verifier()

        i2Ri1, i2Ui1, verified_indices = \
            self.verifier.verify_with_exact_intrinsics(
                keypoints_i1,
                keypoints_i2,
                match_indices,
                intrinsics_i1,
                intrinsics_i2
            )

        return i2Ri1, i2Ui1, verified_indices, keypoints_i1, keypoints_i2


def generate_random_keypoints(num_keypoints: int,
                              image_shape: Tuple[int, int]) -> Keypoints:
    """Generates random features within the image bounds.

    Args:
        num_keypoints: number of features to generate.
        image_shape: size of the image.

    Returns:
        generated features.
    """

    if num_keypoints == 0:
        return np.array([])

    return Keypoints(
        coordinates=np.random.randint(
            [0, 0], high=image_shape, size=(num_keypoints, 2)
        ).astype(np.float32))


def generate_random_input_for_verifier() -> \
        Tuple[Keypoints, Keypoints, np.ndarray, Cal3Bundler, Cal3Bundler]:
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
    image_shape_i1 = [random.randint(100, 400), random.randint(100, 400)]
    image_shape_i2 = [random.randint(100, 400), random.randint(100, 400)]

    # generate intrinsics from image shapes
    intrinsics_i1 = Cal3Bundler(
        fx=min(image_shape_i1[0], image_shape_i1[1]),
        k1=0,
        k2=0,
        u0=image_shape_i1[0]/2,
        v0=image_shape_i1[1]/2,
    )

    intrinsics_i2 = Cal3Bundler(
        fx=min(image_shape_i2[0], image_shape_i2[1]),
        k1=0,
        k2=0,
        u0=image_shape_i2[0]/2,
        v0=image_shape_i2[1]/2,
    )

    # randomly generate the keypoints
    keypoints_i1 = generate_random_keypoints(
        num_keypoints_i1, image_shape_i1)
    keypoints_i2 = generate_random_keypoints(
        num_keypoints_i2, image_shape_i2)

    # randomly generate matches
    num_matches = random.randint(0, min(num_keypoints_i1, num_keypoints_i2))
    if num_matches == 0:
        matching_indices_i1i2 = np.array([], dtype=np.int32)
    else:
        matching_indices_i1i2 = np.empty((num_matches, 2), dtype=np.int32)
        matching_indices_i1i2[:, 0] = np.random.choice(
            num_keypoints_i1, size=(num_matches,), replace=False)
        matching_indices_i1i2[:, 1] = np.random.choice(
            num_keypoints_i2, size=(num_matches,), replace=False)

    return keypoints_i1, keypoints_i2, matching_indices_i1i2, \
        intrinsics_i1, intrinsics_i2


def sample_points_on_plane(
        plane_coefficients: Tuple[float, float, float, float],
        range_x_coordinate: Tuple[float, float],
        range_y_coordinate: Tuple[float, float],
        num_points: int) -> np.ndarray:
    """Sample random points on a 3D plane ax + by + cz + d = 0.

    Args:
        plane_coefficients: coefficients (a,b,c,d) of the plane equation.
        range_x_coordinate: desired range of the x coordinates of samples.
        range_y_coordinate: desired range of the x coordinates of samples.
        num_points: number of points to sample

    Returns:
        3d points, of shape (num_points, 3).
    """

    if plane_coefficients[3] == 0:
        raise ValueError('z-coefficient for the plane should not be zero')

    pts = np.empty((num_points, 3))

    # sample x coordinates randomly
    pts[:, 0] = np.random.rand(num_points) * \
        (range_x_coordinate[1] - range_x_coordinate[0]) + \
        range_x_coordinate[0]

    # sample y coordinates randomly
    pts[:, 1] = np.random.rand(num_points) * \
        (range_y_coordinate[1] - range_y_coordinate[0]) + \
        range_y_coordinate[0]

    # calculate z coordinates using equation of the plane
    pts[:, 2] = (plane_coefficients[0] * pts[:, 0] +
                 plane_coefficients[1] * pts[:, 1] +
                 plane_coefficients[3]) / plane_coefficients[2]

    return pts


def simulate_two_planes_scene(num_points_plane1: int,
                              num_points_plane2: int
                              ) -> Tuple[Keypoints, Keypoints, EssentialMatrix]:
    """The world coordinate system is the same as coordinate system of the
    first camera.

    The two planes in this test are:
    1. -10x -y -20z +150 = 0
    2. 15x -2y -35z +200 = 0

    Args:
        num_points_plane1: number of points on 1st plane.
        num_points_plane2: number of points on 2nd plane.

    Returns:
        keypoints for image i1, of length num_points_plane1+num_points_plane2.
        keypoints for image i2, of length num_points_plane1+num_points_plane2.
        Essential matrix i2Ei1.
    """
    # range of 3D points
    range_x_coordinate = (-5, 7)
    range_y_coordinate = (-10, 10)

    # define the plane equation
    plane1_coeffs = (-10, -1, -20, 150)
    plane2_coeffs = (15, -2, -35, 200)

    # sample the points from planes
    plane1_points = sample_points_on_plane(
        plane1_coeffs,
        range_x_coordinate,
        range_y_coordinate,
        num_points_plane1)
    plane2_points = sample_points_on_plane(
        plane2_coeffs,
        range_x_coordinate,
        range_y_coordinate,
        num_points_plane2)

    points_3d = np.vstack((plane1_points, plane2_points))

    # convert to homogenous coordinates
    points_3d = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # project the 3D points to both the cameras
    wTi1 = np.array([0.1, 0, 0])
    wTi2 = np.array([1, -2, -0.4])

    wRi1 = Rot3.RzRyRx(np.pi/20, 0, 0.0)
    wRi2 = Rot3.RzRyRx(0.0, np.pi/6, 0.0)

    wPi1 = Pose3(wRi1, wTi1)
    wPi2 = Pose3(wRi2, wTi2)
    i2Pi1 = wPi2.between(wPi1)

    i2Ei1 = EssentialMatrix(i2Pi1.rotation(), Unit3(i2Pi1.translation()))

    intrinsics = Cal3Bundler()

    extrinsics_i1 = wRi1.inverse().matrix() @ np.concatenate(
        (np.eye(3), -wTi1.reshape(-1, 1)),
        axis=1
    )

    extrinsics_i2 = wRi2.inverse().matrix() @ np.concatenate(
        (np.eye(3), -wTi2.reshape(-1, 1)),
        axis=1
    )

    features_im1 = (intrinsics.K() @ extrinsics_i1 @ points_3d.T).T
    features_im2 = (intrinsics.K() @ extrinsics_i2 @ points_3d.T).T

    features_im1[:, :2] = features_im1[:, :2]/features_im1[:, 2:3]
    features_im2[:, :2] = features_im2[:, :2]/features_im2[:, 2:3]

    return Keypoints(coordinates=features_im1[:, :2]), \
        Keypoints(coordinates=features_im2[:, :2]), \
        i2Ei1


if __name__ == "__main__":
    unittest.main()
