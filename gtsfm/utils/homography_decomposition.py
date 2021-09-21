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

from typing import Tuple

from gtsam import Rot3, Unit3

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
    K1: np.ndarray,
    K2: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    n: np.ndarray,
    points3D: np.ndarray,
) -> Tuple[Rot3, Unit3, np.ndarray, np.ndarray]:
    """Recover the most probable pose from the given homography matrix.

    Args:
        H: array of shape (3,3)
        K1: array of shape (3,3) representing camera 1's intrinsics
        K2: array of shape (3,3) representing camera 2's intrinsics
        points1: array of shape (N,2)
        points2: array of shape (N,2)

    Returns:
        R: relative rotation matrix.
        t: translation direction.
        n: array of shape (3,) representing plane normal vector.
        points3D: array of shape (N,3) representing triangulated 3d points.
    """
    if points1.shape != points2.shape:
        raise RuntimeError("Coordinates of 2d correspondences must have the same shape.")

    R_cmbs, t_cmbs, n_cmbs = decompose_homography_matrix(H, K1, K2)

    points3d = np.zeros((0, 3))
    for i in range(len(R_cmbs)):
        points3D_cmb = check_cheirality(R_cmbs[i], t_cmbs[i], points1, points2)
        if len(points3D_cmb) >= len(points3D):
            R = R_cmbs[i]
            t = t_cmbs[i]
            n = n_cmbs[i]
            points3D = points3D_cmb

    return R, t, n, points3D


def check_cheirality(R: Rot3, t: np.ndarray, points1, points2) -> np.ndarray:
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

    # const Eigen::Matrix3x4d proj_matrix1 = Eigen::Matrix3x4d::Identity();
    # const Eigen::Matrix3x4d proj_matrix2 = ComposeProjectionMatrix(R, t);
    # const double kMinDepth = std::numeric_limits<double>::epsilon();
    # const double max_depth = 1000.0f * (R.transpose() * t).norm();
    # points3D->clear();
    # for (size_t i = 0; i < points1.size(); ++i) {
    #   const Eigen::Vector3d point3D =
    #       TriangulatePoint(proj_matrix1, proj_matrix2, points1[i], points2[i]);
    #   const double depth1 = CalculateDepth(proj_matrix1, point3D);
    #   if (depth1 > kMinDepth && depth1 < max_depth) {
    #     const double depth2 = CalculateDepth(proj_matrix2, point3D);
    #     if (depth2 > kMinDepth && depth2 < max_depth) {
    #       points3D->push_back(point3D);
    #     }
    #   }
    # }
    # return !points3D->empty();


def decompose_homography_matrix(
    H: np.ndarray, K1: np.ndarray, K2: np.ndarray
) -> Tuple[List[Rot3], List[Unit3], List[Unit3]]:
    """Decompose an homography matrix into the possible rotations, translations, and plane normal vectors.

    Args:
        H: array of shape (3,3)
        K1: array of shape (3,3) representing camera 1's intrinsics
        K2: array of shape (3,3) representing camera 2's intrinsics

    Returns:
        R_cmbs: list representing combinations of possible R matrices of shape (3,3).
        t_cmbs: list representing combinations of possible t directions of shape (3,).
        n_cmbs: list representing combinations of possible plane normals vectors of shape (3,).
    """
    # Remove calibration from homography.
    H_normalized = np.linalg.inv(K2) @ H @ K1

    # Remove scale from normalized homography.
    _, S, _ = np.linalg.svd(H_normalized)

    # Singular values are always sorted in decreasing order (same as in Eigen)
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

    S = H_normalized.T * H_normalized - np.eye(3)

    # Check if H is rotation matrix.
    kMinInfinityNorm = 1e-3
    if np.linalg.norm(S, ord=inf) < kMinInfinityNorm:
        R_cmbs = [H_normalized]
        t_cmbs = [np.zeros(3)]
        n_cmbs = [np.zeros(3)]
        return R_cmbs, t_cmbs, n_cmbs

    M00 = ComputeOppositeOfMinor(S, 0, 0)
    M11 = ComputeOppositeOfMinor(S, 1, 1)
    M22 = ComputeOppositeOfMinor(S, 2, 2)

    rtM00 = np.sqrt(M00)
    rtM11 = np.sqrt(M11)
    rtM22 = np.sqrt(M22)

    M01 = ComputeOppositeOfMinor(S, 0, 1)
    M12 = ComputeOppositeOfMinor(S, 1, 2)
    M02 = ComputeOppositeOfMinor(S, 0, 2)

    e12 = SignOfNumber(M12)
    e02 = SignOfNumber(M02)
    e01 = SignOfNumber(M01)

    nS00 = np.absolute(S[0, 0])
    nS11 = np.absolute(S[1, 1])
    nS22 = np.absolute(S[2, 2])

    nS = np.array([nS00, nS11, nS22])
    # count the number of elements between the first element in the array, and the largest element.
    idx = np.argmax(nS)

    np1 = np.zeros(3)
    np2 = np.zeros(3)
    if idx == 0:
        np1[0] = S[0, 0]
        np2[0] = S[0, 0]
        np1[1] = S[0, 1] + rtM22
        np2[1] = S[0, 1] - rtM22
        np1[2] = S[0, 2] + e12 * rtM11
        np2[2] = S[0, 2] - e12 * rtM11
    elif idx == 1:
        np1[0] = S[0, 1] + rtM22
        np2[0] = S[0, 1] - rtM22
        np1[1] = S[1, 1]
        np2[1] = S[1, 1]
        np1[2] = S[1, 2] - e02 * rtM00
        np2[2] = S[1, 2] + e02 * rtM00
    elif idx == 2:
        np1[0] = S[0, 2] + e01 * rtM11
        np2[0] = S[0, 2] - e01 * rtM11
        np1[1] = S[1, 2] + rtM00
        np2[1] = S[1, 2] - rtM00
        np1[2] = S[2, 2]
        np2[2] = S[2, 2]

    traceS = np.trace(S)
    v = 2.0 * np.sqrt(1.0 + traceS - M00 - M11 - M22)

    ESii = SignOfNumber(S[idx, idx])
    r_2 = 2 + traceS + v
    nt_2 = 2 + traceS - v

    r = np.sqrt(r_2)
    n_t = np.sqrt(nt_2)

    # normalize
    n1 = np1 / np.linalg.norm(np1)
    n2 = np2 / np.linalg.norm(np2)

    half_nt = 0.5 * n_t
    esii_t_r = ESii * r

    t1_star = half_nt * (esii_t_r * n2 - n_t * n1)
    t2_star = half_nt * (esii_t_r * n1 - n_t * n2)

    R1 = compute_homography_rotation(H_normalized, t1_star, n1, v)
    t1 = R1 * t1_star

    R2 = compute_homography_rotation(H_normalized, t2_star, n2, v)
    t2 = R2 * t2_star

    R_cmbs = [R1, R1, R2, R2]
    t_cmbs = [t1, -t1, t2, -t2]
    n_cmbs = [-n1, n1, -n2, n2]
    return R_cmbs, t_cmbs, n_cmbs


def ComputeOppositeOfMinor(matrix: np.ndarray, row: int, col: int) -> float:
    """
    Args:
        matrix: array of shape
        row: row index.
        col: column index.

    Returns:
        float representing ...
    """
    col1 = 1 if col == 0 else 0
    col2 = 1 if col == 2 else 2
    row1 = 1 if row == 0 else 0
    row2 = 1 if row == 2 else 2
    return matrix[row1, col2] * matrix[row2, col1] - matrix[row1, col1] * matrix[row2, col2]


def compute_homography_rotation(H_normalized: np.ndarray, tstar: np.ndarray, n: np.ndarray, v: float) -> np.ndarray:
    """Returns 3x3 matrix

    See Equation 99 on Page 32 of https://hal.inria.fr/inria-00174036/PDF/RR-6303.pdf

    Args:
        H_normalized: array of shape (3,3)
        tstar: array of shape (3,)
        n: array of shape (3,)
        v:

    Returns:
        array of shape (3,3) representing rotation matrix
    """
    I = np.eye(3)
    return H_normalized @ (I - (2.0 / v) @ tstar @ n.T)
