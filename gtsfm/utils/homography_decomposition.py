"""
Decompose an homography matrix into the possible rotations, translations,
and plane normal vectors.

Based off of homography decomposition implementation in COLMAP:
    https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix.cc
    https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix.h
See how it is used in COLMAP:
https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L198

COLMAP and OpenCV's implementationsare based off of:
Ezio Malis, Manuel Vargas, and others. Deeper understanding of the homography decomposition for vision-based control. 2007.
https://hal.inria.fr/inria-00174036/PDF/RR-6303.pdf

OpenCV does not support the case of intrinsics from two separate cameras, however.

Authors: John Lambert (Python), from original C++
"""

from typing import List, Tuple

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, Unit3

from gtsfm.common.sfm_track import SfmTrack2d, SfmMeasurement
from gtsfm.data_association.point3d_initializer import Point3dInitializer, TriangulationParam, TriangulationExitCode

"""
PoseFromHomographyMatrix(
        H, camera1.CalibrationMatrix(), camera2.CalibrationMatrix(),
        inlier_points1_normalized, inlier_points2_normalized, &R, &tvec, &n,
        &points3D);

  if (points3D.empty()) {
    tri_angle = 0;
  } else {
    tri_angle = Median(CalculateTriangulationAngles(
        Eigen::Vector3d::Zero(), -R.transpose() * tvec, points3D));
  }

  if (config == PLANAR_OR_PANORAMIC) {
    if (tvec.norm() == 0) {
      config = PANORAMIC;
      tri_angle = 0;
    } else {
      config = PLANAR;
    }
"""


def pose_from_homography_matrix(
    H: np.ndarray,
    camera_intrinsics_i1: Cal3Bundler,
    camera_intrinsics_i2: Cal3Bundler,
    points1: np.ndarray,
    points2: np.ndarray,
) -> Tuple[Rot3, Unit3, np.ndarray, np.ndarray]:
    """Recover the most probable pose from the given homography matrix.

    Args:
        H: array of shape (3,3)
        camera_intrinsics_i1: camera 1's intrinsics
        camera_intrinsics_i2: camera 2's intrinsics
        points1: array of shape (N,2)
        points2: array of shape (N,2)

    Returns:
        i2Ri1: relative rotation matrix.
        i2Ui1: translation direction.
        n: array of shape (3,) representing plane normal vector.
        points3D: array of shape (N,3) representing triangulated 3d points.
    """
    if points1.shape != points2.shape:
        raise RuntimeError("Coordinates of 2d correspondences must have the same shape.")

    K1 = camera_intrinsics_i1.K()
    K2 = camera_intrinsics_i2.K()
    i2Ri1_cmbs, i2ti1_cmbs, n_cmbs = decompose_homography_matrix(H, K1, K2)

    points3D = []
    for i in range(len(i2Ri1_cmbs)):
        points3D_cmb = check_cheirality(
            i2Ri1_cmbs[i], i2ti1_cmbs[i], camera_intrinsics_i1, camera_intrinsics_i2, points1, points2
        )
        print(f"hypothesis {i}  -> {len(points3D_cmb)}")
        print(i2ti1_cmbs[i])
        if len(points3D_cmb) >= len(points3D):
            i2Ri1 = i2Ri1_cmbs[i]
            i2ti1 = i2ti1_cmbs[i]
            n = n_cmbs[i]
            points3D = points3D_cmb

    points3D = np.array(points3D)
    print("Translation direction: ", i2ti1)
    print(f"Triangulated {points3D.shape} points from {points1.shape} correspondences.")

    # Note: we cannot blindly cast the translation direction to Unit3, as zero translation is valid
    # from a rotation, and should be noted in 1dsfm.
    return i2Ri1, Unit3(i2ti1), n, points3D


def check_cheirality(
    i2Ri1: Rot3,
    i2ti1: np.ndarray,
    camera_intrinsics_i1: Cal3Bundler,
    camera_intrinsics_i2: Cal3Bundler,
    points1: np.ndarray,
    points2: np.ndarray,
) -> np.ndarray:
    """
    Args:
        R: array of shape (3,3)
        t: array of shape (3,)
        points1:
        points2:

    Returns:
        points3D: array of shape (N,3)
    """
    if points1.shape != points2.shape:
        raise RuntimeError("Coordinates of 2d correspondences must have the same shape.")

    
    i2Ti1 = Pose3(i2Ri1, i2ti1)
    camera_dict = {
        1: PinholeCameraCal3Bundler(i2Ti1, camera_intrinsics_i1),
        2: PinholeCameraCal3Bundler(Pose3(), camera_intrinsics_i2),
    }

    # set the camera 2 pose as the global coordinate system
    # COLMAP states that wTc for camera 2 is equal to (R,t) from homography.
    # given wTi1 and wTi2, then if
    # i2Ti1 = wTi2.inverse() * wTi1
    # thus, wTi2 is identity, and wTi1 = i2Ti1
    triangulator = Point3dInitializer(
        track_camera_dict=camera_dict, mode=TriangulationParam.NO_RANSAC, reproj_error_thresh=float("inf")
    )

    points3d = []
    # try triangulating each point
    for point1, point2 in zip(points1, points2):

        track_2d = SfmTrack2d(measurements=[SfmMeasurement(i=1, uv=point1), SfmMeasurement(i=2, uv=point2)])
        track_3d, _, exit_code = triangulator.triangulate(track_2d)
        if exit_code == TriangulationExitCode.CHEIRALITY_FAILURE:
            continue
        points3d.append(track_3d.point3())

    return points3d


def decompose_homography_matrix(
    H: np.ndarray, K1: np.ndarray, K2: np.ndarray
) -> Tuple[List[Rot3], List[Unit3], List[Unit3]]:
    """Decompose an homography matrix into the possible rotations, translations, and plane normal vectors.

    Based off of the OpenCV and COLMAP implementations:
    `HomographyDecompInria::findRmatFrom_tstar_n()` at
    https://github.com/opencv/opencv/blob/master/modules/calib3d/src/homography_decomp.cpp#L325

    Args:
        H: array of shape (3,3)
        K1: array of shape (3,3) representing camera 1's intrinsics
        K2: array of shape (3,3) representing camera 2's intrinsics

    Returns:
        R_cmbs: list representing combinations of possible R matrices of shape (3,3). If H
           corresponds to a pure rotation, then this list will only have 1 entry. Otherwise,
           a list of 4 possible rotations is returned.
        t_cmbs: list representing combinations of possible t directions of shape (3,).
           Note: `t` is not necessarily a unit vector, as it can have zero norm (pure rotation).
        n_cmbs: list representing combinations of possible plane normals vectors of shape (3,).
    """
    # Remove calibration from homography.
    H_normalized = np.linalg.inv(K2) @ H @ K1

    # Remove scale from normalized homography.
    _, S, _ = np.linalg.svd(H_normalized)

    # Singular values are always sorted in decreasing order (same as in Eigen)
    # Use the median of the singular values to compute \gammma
    H_normalized /= S[1]

    # Ensure that we always return rotations, and never reflections.
    #
    # It's enough to take det(H_normalized) > 0.
    #
    # To see this:
    # - In the paper: R := H_normalized * (Id + x y^t)^{-1} (page 32).
    # - Can check that this implies that R is orthogonal: RR^t = Id.
    # - To return a rotation, we also need det(R) > 0.
    # - By Sylvester's idenitity: det(Id + x y^t) = (1 + x^t y), which
    #   is positive by choice of x and y (page 24).
    # - So det(R) and det(H_normalized) have the same sign.
    if np.linalg.det(H_normalized) < 0:
        H_normalized *= -1

    # See Section 4.1 (Page 14)
    S = H_normalized.T @ H_normalized - np.eye(3)

    # Check if H is rotation matrix.
    kMinInfinityNorm = 1e-3
    # matrix infinity norm is max(sum(abs(x), axis=1)) in numpy
    # and we want the vector infinity norm max(abs(x)), so flatten.
    if np.linalg.norm(S.flatten(), ord=np.inf) < kMinInfinityNorm:
        R_cmbs = [ Rot3(H_normalized) ]
        t_cmbs = [np.zeros(3)]
        n_cmbs = [np.zeros(3)]
        return R_cmbs, t_cmbs, n_cmbs

    M00 = compute_opposite_of_minor(S, row=0, col=0)
    M11 = compute_opposite_of_minor(S, row=1, col=1)
    M22 = compute_opposite_of_minor(S, row=2, col=2)

    rtM00 = np.sqrt(M00)
    rtM11 = np.sqrt(M11)
    rtM22 = np.sqrt(M22)

    M01 = compute_opposite_of_minor(S, row=0, col=1)
    M12 = compute_opposite_of_minor(S, row=1, col=2)
    M02 = compute_opposite_of_minor(S, row=0, col=2)

    e12 = np.sign(M12)
    e02 = np.sign(M02)
    e01 = np.sign(M01)

    nS00 = np.absolute(S[0, 0])
    nS11 = np.absolute(S[1, 1])
    nS22 = np.absolute(S[2, 2])

    nS = np.array([nS00, nS11, nS22])
    # use the alternative among the three given corresponding to the s_ii with largest absolute value
    # (the most well conditioned option)
    idx = np.argmax(nS)

    # See equations 11,12,13.
    # fmt: off
    if idx == 0:
        np1 = np.array(
            [
                S[0, 0],
                S[0, 1] + rtM22,
                S[0, 2] + e12 * rtM11
            ]
        )
        np2 = np.array(
            [
                S[0, 0],
                S[0, 1] - rtM22,
                S[0, 2] - e12 * rtM11
            ]
        )
    elif idx == 1:
        np1 = np.array(
            [
                S[0, 1] + rtM22,
                S[1, 1],
                S[1, 2] - e02 * rtM00
            ]
        )
        np2 = np.array(
            [
                S[0, 1] - rtM22,
                S[1, 1],
                S[1, 2] + e02 * rtM00
            ]
        )
    elif idx == 2:
        np1 = np.array(
            [
                S[0, 2] + e01 * rtM11,
                S[1, 2] + rtM00,
                S[2, 2]
            ]
        )
        np2 = np.array(
            [
                S[0, 2] - e01 * rtM11,
                S[1, 2] - rtM00,
                S[2, 2]
            ]
        )
    # fmt: on

    traceS = np.trace(S)
    v = 2.0 * np.sqrt(1.0 + traceS - M00 - M11 - M22)

    ESii = np.sign(S[idx, idx])

    r_2 = 2 + traceS + v  # this is \rho^2 + trace(S) + \nu
    nt_2 = 2 + traceS - v

    r = np.sqrt(r_2)
    n_t = np.sqrt(nt_2)

    # normalize
    n1 = np1 / np.linalg.norm(np1)
    n2 = np2 / np.linalg.norm(np2)

    half_nt = 0.5 * n_t
    esii_t_r = ESii * r

    # Equations 16,17
    t1_star = half_nt * (esii_t_r * n2 - n_t * n1)
    t2_star = half_nt * (esii_t_r * n1 - n_t * n2)

    R1 = compute_homography_rotation(H_normalized, t1_star, n1, v)
    # See Equation 20
    t1 = R1 @ t1_star

    R2 = compute_homography_rotation(H_normalized, t2_star, n2, v)
    # See Equation 21
    t2 = R2 @ t2_star

    # combinations differ from OpenCV's implementations (using COLMAP's)
    R_cmbs = [ Rot3(R1), Rot3(R1), Rot3(R2), Rot3(R2)]
    t_cmbs = [t1, -t1, t2, -t2]
    n_cmbs = [-n1, n1, -n2, n2]
    return R_cmbs, t_cmbs, n_cmbs


def compute_opposite_of_minor(matrix: np.ndarray, row: int, col: int) -> float:
    """Compute the opposite of a 3x3 matrix's 2x2 minor (multiply determinant by -1).

    If A is a square matrix, then the minor of the entry in the i'th row and j'th column
    (also called the (i, j) minor, or a first minor[1]) is the determinant of the submatrix
    formed by deleting the i'th row and j'th column.

    Reference: https://en.wikipedia.org/wiki/Minor_(linear_algebra)

    Implemented as OpenCV:
    https://github.com/opencv/opencv/blob/master/modules/calib3d/src/homography_decomp.cpp#L299
    and COLMAP do so.

    Args:
        matrix: array of shape (3,3)
        row: row index.
        col: column index.

    Returns:
        float representing opposite of matrix minor.
    """
    col1 = 1 if col == 0 else 0
    col2 = 1 if col == 2 else 2
    row1 = 1 if row == 0 else 0
    row2 = 1 if row == 2 else 2
    return matrix[row1, col2] * matrix[row2, col1] - matrix[row1, col1] * matrix[row2, col2]


def compute_homography_rotation(H_normalized: np.ndarray, tstar: np.ndarray, n: np.ndarray, v: float) -> np.ndarray:
    """Returns 3x3 matrix

    See Equation 99 on Page 32 of https://hal.inria.fr/inria-00174036/PDF/RR-6303.pdf
    Reference: See OpenCV's `HomographyDecompInria::findRmatFrom_tstar_n()`
    https://github.com/opencv/opencv/blob/master/modules/calib3d/src/homography_decomp.cpp#L310

    Args:
        H_normalized: array of shape (3,3)
        tstar: array of shape (3,)
        n: array of shape (3,) representing normal vector.
        v:

    Returns:
        array of shape (3,3) representing rotation matrix
    """
    I = np.eye(3)
    tstar = tstar.reshape(3, 1)
    n = n.reshape(3, 1)

    # fmt: off
    R = H_normalized @ (I - (2.0/v) * tstar @ n.T) # noqa
    # fmt: on
    return R


def homography_matrix_from_pose(
    camera_intrinsics_i1: Cal3Bundler,
    camera_intrinsics_i2: Cal3Bundler,
    R: Rot3,
    t: np.ndarray,
    n: np.ndarray,
    d: float,
) -> np.ndarray:
    """Compute a homography matrix from a known relative pose.

    Args:
        camera_intrinsics_i1: camera 1's intrinsics.
        camera_intrinsics_i2: camera 2's intrinsics.
        R: 3x3 rotation matrix
        t: array of shape (3,) representing translation vector.
        n: array of shape (3,) representing normal vector.
        d: Orthogonal distance from plane.

    Returns:
        H: array of shape (3,3) representing homography matrix.
    """
    if d <= 0:
        raise RuntimeError("Orthogonal distance from plane `d` must be positive.")

    K1 = camera_intrinsics_i1.K()
    K2 = camera_intrinsics_i2.K()

    # normalize
    n /= np.linalg.norm(n)

    t = t.reshape(3, 1)
    n = n.reshape(3, 1)

    return K2 @ (R - t @ n.T / d) @ np.linalg.inv(K1)
