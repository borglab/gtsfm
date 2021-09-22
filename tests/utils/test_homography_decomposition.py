"""

Test cases come from COLMAP https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc
which in turn take values from OpenCV.

Author: John Lambert (Python)
"""

import numpy as np

import gtsfm.utils.homography_decomposition as homography_utils


def test_decompose_homography_matrix() -> None:
    """Ensure the 4 possible rotation, translation, normal vector options can be computed from H.

    See https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc#L44
    # As noted by COLMAP: test case values are obtained from OpenCV.
    """
    H = np.array(
        [
            [2.649157564634028, 4.583875997496426, 70.694447785121326],
            [-1.072756858861583, 3.533262150437228, 1513.656999614321649],
            [0.001303887589576, 0.003042206876298, 1],
        ]
    )
    H *= 3

    # fmt: off
    K = np.array(
        [
            [640, 0, 320],
            [0, 640, 240],
            [0, 0, 1]
        ]
    )
    # fmt: on

    R_cmbs, t_cmbs, n_cmbs = homography_utils.decompose_homography_matrix(H, K1=K, K2=K)

    assert len(R_cmbs) == 4
    assert len(t_cmbs) == 4
    assert len(n_cmbs) == 4

    R_ref = np.array(
        [
            [0.43307983549125, 0.545749113549648, -0.717356090899523],
            [-0.85630229674426, 0.497582023798831, -0.138414255706431],
            [0.281404038139784, 0.67421809131173, 0.682818960388909],
        ]
    )
    t_ref = np.array([1.826751712278038, 1.264718492450820, 0.195080809998819])
    n_ref = np.array([-0.244875830334816, -0.480857890778889, -0.841909446789566])

    ref_solution_exists = False

    for i in range(4):
        if np.allclose(R_cmbs[i], R_ref) and np.allclose(t_cmbs[i], t_ref) and np.allclose(n_cmbs[i], n_ref):
            ref_solution_exists = True

    assert ref_solution_exists


# def test_pose_from_homography_matrix() -> None:
#     """

#     See: https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc#L120
#     """
#     K1 = np.eye(3)
#     K2 = np.eye(3)
#     R_ref = np.eye(3)
#     t_ref = np.array([1, 0, 0])
#     n_ref = np.array([-1, 0, 0])
#     d_ref = 1
#     H = homography_utils.homography_matrix_from_pose(K1, K2, R_ref, t_ref, n_ref, d_ref)

#     # fmt: off
#     points1 = np.array(
#     [
#     [0.1, 0.4],
#     [0.2, 0.3],
#     [0.3, 0.2],
#     [0.4, 0.1]
#     ]
#     )
#     # fmt: on

#     points2 = np.zeros((0, 2))
#     for point1 in points1:
#         # affine to homogeneous
#         point2 = H @ np.array([point1[0], point1[1], 1.0])
#         # convert homogenous to affine
#         point2 /= point2[2]
#         points2.append(point2)

#     R, t, n, points3D = homography_utils.pose_from_homography_matrix(
#         H,
#         K1,
#         K2,
#         points1,
#         points2,
#     )

#     np.testing.assert_almost_equal(R, R_ref)
#     np.testing.assert_almost_equal(t, t_ref)
#     np.testing.assert_almost_equal(n, n_ref)
#     assert points3D.shape == points1.shape


# def test_check_cheirality() -> None:
#     """ """
#     R = ""
#     t = ""
#     points1 = ""
#     points2 = ""
#     points3D = check_cheirality(R, t, points1, points2)

#     assert False


def test_compute_opposite_of_minor_M00() -> None:
    """Ensure negative of lower-right 2x2 determinant is returned (M00)."""
    matrix = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    )
    neg_minor = homography_utils.compute_opposite_of_minor(matrix, row=0, col=0)

    # det is -3 = 45 - 48 = 5*9 - 6 * 8
    ref_neg_minor = 3
    assert neg_minor == ref_neg_minor

def test_compute_opposite_of_minor_M22() -> None:
    """Ensure negative of upper-left 2x2 determinant is returned (M22)."""
    matrix = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    )
    neg_minor = homography_utils.compute_opposite_of_minor(matrix, row=2, col=2)

    # det is -3 = 5 - 8 = 1*5 - 2*4
    ref_neg_minor = 3
    assert neg_minor == ref_neg_minor

def test_compute_opposite_of_minor_M11() -> None:
    """Ensure negative of 2x2 determinant is returned (M11)."""
    matrix = np.array(
        [
            [1,2,3],
            [4,5,6],
            [7,8,9]
        ]
    )
    neg_minor = homography_utils.compute_opposite_of_minor(matrix, row=1, col=1)

    # det is -12 = 9 - 21 = 1*9 - 3*7 
    ref_neg_minor = 12
    assert neg_minor == ref_neg_minor


