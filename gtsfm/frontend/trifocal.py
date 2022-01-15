"""
Trifocal tensor estimation based on Julia17psivt

Reference:
https://github.com/LauraFJulia/TFT_vs_Fund/blob/master/TFT_methods/linearTFT.m

Author: John Lambert
"""

from typing import Tuple

import numpy as np


def create_trifocal_data_matrix(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Populate data matrix representing homogeneous system.

    Args:
        p1: (N,2) array of keypoints in view 1
        p2: (N,2) array of keypoints in view 2
        p3: (N,2) array of keypoints in view 3

    Returns:
        A: (4*N, 27) array representing data matrix
    """
    N = p1.shape[0]
    A = np.zeros((4 * N, 27))  # matrix of the linear system on the parameters of the TFT

    for i in range(N):
        x1, y1 = p1[i, :2]
        x2, y2 = p2[i, :2]
        x3, y3 = p3[i, :2]

        # fmt: off
        # A has 27 columns for Ax=b
        A[4 * i , :] = np.array([
            x1, 0, -x1 * x2, 0, 0, 0, -x1 * x3, 0, x1 * x2 * x3,
            y1, 0, -x2 * y1, 0, 0, 0, -x3 * y1, 0, x2 * x3 * y1, 1,
            0, -x2, 0, 0, 0, -x3, 0, x2 * x3,
        ])
        A[4 * i + 1, :] = np.array([
            0, x1, -x1 * y2, 0, 0, 0, 0, -x1 * x3, x1 * x3 * y2,
            0, y1, -y1 * y2, 0, 0, 0, 0, -x3 * y1, x3 * y1 * y2,
            0, 1, -y2, 0, 0, 0, 0, -x3, x3 * y2,
        ])
        A[4 * i + 2, :] = np.array([
            0, 0, 0, x1, 0, -x1 * x2, -x1 * y3, 0, x1 * x2 * y3,
            0, 0, 0, y1, 0, -x2 * y1, -y1 * y3, 0, x2 * y1 * y3,
            0, 0, 0, 1, 0, -x2, -y3, 0, x2 * y3,
        ])
        A[4 * i + 3, :] = np.array([
            0, 0, 0, 0, x1, -x1 * y2, 0, -x1 * y3, x1 * y2 * y3,
            0, 0, 0, 0, y1, -y1 * y2, 0, -y1 * y3, y1 * y2 * y3,
            0, 0, 0, 0, 1, -y2, 0, -y3, y2 * y3,
        ])
        # fmt: on

    return A


def linearTFT(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Linear estimation of the TFT for normalized image coordinates (whitened and ...)

    Estimation of the trifocal tensor from triplets of corresponding image
    points by algebraic minimization of the linear contraints given by the
    incidence relationships. The three projection matrices are also computed
    using the TFT.

    See page 4 of Amnon Shashua:
    http://robotics.stanford.edu/~shashua/alg-pami-final.pdf

    Equation 2 of:
    https://hal-enpc.archives-ouvertes.fr/hal-01700686/document

    Args:
        p1, p2, p3: 3 vectors (N,3) or (N,2) containing image points for images 1, 2, and 3
            respectively in homogeneous or cartesian coordinates.

    Returns:
        T: 3x3x3 array containing the trifocal tensor associated to
                    the triplets of corresponding points
        P1, P2, P3: three estimated projection matrices 3x4
    """
    if p1.shape[1] == 3:
        # normalize via broadcasting
        p1 = p1[:, :2] / p1[:, 2].reshape(-1, 1)
        p2 = p2[:, :2] / p2[:, 2].reshape(-1, 1)
        p3 = p3[:, :2] / p3[:, 2].reshape(-1, 1)

    A = create_trifocal_data_matrix(p1, p2, p3)

    _, _, Vt = np.linalg.svd(A)

    t = Vt[-1, :]

    # equivalent to column-major reshape of t in MATLAB
    T = np.reshape(a=t, newshape=(3, 3, 3), order="F")

    # compute valid tensor
    # epipoles
    _, _, Vt = np.linalg.svd(T[:, :, 0])
    v1 = Vt[-1, :]
    _, _, Vt = np.linalg.svd(T[:, :, 1])
    v2 = Vt[-1, :]
    _, _, Vt = np.linalg.svd(T[:, :, 2])
    v3 = Vt[-1, :]
    _, _, Vt = np.linalg.svd(np.stack([v1, v2, v3], axis=1).T)
    epi31 = Vt[-1, :].reshape(3, 1)

    _, _, Vt = np.linalg.svd(T[:, :, 0].T)
    v1 = Vt[-1, :]
    _, _, Vt = np.linalg.svd(T[:, :, 1].T)
    v2 = Vt[-1, :]
    _, _, Vt = np.linalg.svd(T[:, :, 2].T)
    v3 = Vt[-1, :]
    _, _, Vt = np.linalg.svd(np.stack([v1, v2, v3], axis=1).T)
    epi21 = Vt[-1, :].reshape(3, 1)

    # using matrices' parameters
    E = np.hstack([np.kron(np.eye(3), np.kron(epi31, np.eye(3))), -np.kron(np.eye(9), epi21)])

    U, S, Vt = np.linalg.svd(E)
    V = Vt.T
    rank_E = np.linalg.matrix_rank(E)
    Up = U[:, :rank_E]
    Vp = V[:, :rank_E]
    Sp = np.diag(S)[:rank_E, :rank_E]

    _, _, Vt = np.linalg.svd(A @ Up)
    tp = Vt[-1, :]
    t = Up @ tp
    a = Vp @ np.linalg.inv(Sp) @ tp

    P1 = np.eye(3, 4)
    P2 = np.hstack([np.reshape(a[:9], newshape=(3, 3), order="F"), epi21])
    P3 = np.hstack([np.reshape(a[9:], newshape=(3, 3), order="F"), epi31])
    T = np.reshape(a=t, newshape=(3, 3, 3), order="F")

    return T, P1, P2, P3


def test_linearTFT() -> None:
    """ """
    p1 = np.array(
        [
            [56.0, 499.0],
            [507.0, 503.0],
            [284.0, 685.0],
            [415.0, 503.0],
            [542.0, 1071.0],
            [237.0, 484.0],
            [227.0, 472.0],
            [139.0, 312.0],
            [438.0, 322.0],
            [553.0, 859.0],
            [279.0, 556.0],
            [647.0, 518.0],
            [628.0, 627.0],
            [389.0, 814.0],
            [297.0, 487.0],
            [476.0, 87.0],
            [267.0, 206.0],
            [653.0, 981.0],
            [382.0, 147.0],
            [182.0, 142.0],
        ]
    )
    p2 = np.array(
        [
            [22.0, 463.0],
            [467.0, 504.0],
            [229.0, 683.0],
            [358.0, 499.0],
            [620.0, 1080.0],
            [191.0, 465.0],
            [180.0, 450.0],
            [119.0, 265.0],
            [422.0, 316.0],
            [527.0, 867.0],
            [224.0, 545.0],
            [688.0, 533.0],
            [658.0, 639.0],
            [313.0, 822.0],
            [246.0, 473.0],
            [511.0, 85.0],
            [275.0, 170.0],
            [701.0, 985.0],
            [407.0, 129.0],
            [174.0, 85.0],
        ]
    )
    p3 = np.array(
        [
            [35.0, 453.0],
            [417.0, 526.0],
            [197.0, 702.0],
            [308.0, 514.0],
            [666.0, 1092.0],
            [172.0, 467.0],
            [160.0, 451.0],
            [139.0, 246.0],
            [403.0, 335.0],
            [492.0, 891.0],
            [190.0, 554.0],
            [683.0, 566.0],
            [650.0, 667.0],
            [253.0, 848.0],
            [216.0, 479.0],
            [522.0, 118.0],
            [299.0, 169.0],
            [707.0, 996.0],
            [426.0, 145.0],
            [196.0, 64.0],
        ]
    )

    T, P1, P2, P3 = linearTFT(p1, p2, p3)

    expected_T = np.zeros((3, 3, 3))
    # fmt: off
    expected_T[:, :, 0] = np.array(
        [
            [0.0042, -0.0002, 0.0000],
            [0.0012, 0.0001, 0.0000],
            [0.0000, 0.0000, 0.0000]
        ]
    )
    expected_T[:, :, 1] = np.array(
        [
            [-0.0000, -0.0042, 0.0000],
            [0.0080, 0.0009, 0.0000],
            [-0.0000, -0.0000, 0.0000]
        ]
    )
    expected_T[:, :, 2] = np.array(
        [
            [-0.1939, 0.6066, -0.0035],
            [-0.7678, -0.0692, -0.0005],
            [0.0072, 0.0012, 0.0000],
        ]
    )
    # fmt: on
    assert np.allclose(T, -expected_T, atol=1e-3)


def crossM(v: np.ndarray) -> np.ndarray:
    """Produces the 3x3 matrix corresponding to the cross product of vector v."""
    # fmt: off
    M = np.array(
        [
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ]
    )
    # fmt: on
    return M


def convert_to_homogenous_coordinates(coords: np.ndarray) -> np.ndarray:
    """Convert coordinates to homogenous system (by appending a column of ones)."""
    N = coords.shape[0]
    return np.hstack((coords, np.ones((N, 1))))


def compute_trifocal_errors(T: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> np.ndarray:
    """Measure deviations from trifocal tensor contraint for each 3-view correspondence.

    Args:
        T: trifocal tensor for unnormalized image coordinates:
        p1: (N,2) points in view 1.
        p2: (N,2) points in view 2.
        p3: (N,2) points in view 3.

    Returns:
        errors:
    """
    p1_h = convert_to_homogenous_coordinates(p1)
    p2_h = convert_to_homogenous_coordinates(p2)
    p3_h = convert_to_homogenous_coordinates(p3)

    # TODO: find the bug below!
    errors = []
    for x1, x2, x3 in zip(p1_h, p2_h, p3_h):
        sum_T = x1[0] * T[:, :, 0] + x1[1] * T[:, :, 1] + x1[2] * T[:, :, 2]
        # this constraint should be equal to zero, under ideal conditions.
        error = crossM(x2) @ (sum_T) @ crossM(x3)
        # get 3x3 matrix of errors.
        errors.append(error.mean())

    T_flattened = np.reshape(a=T, newshape=(27, 1), order="F")

    A = create_trifocal_data_matrix(p1, p2, p3)
    # import pdb; pdb.set_trace()
    errors_ = A @ T_flattened
    errors_ = errors_.reshape(-1, 4).mean(axis=1)
    return errors_


def transform_TFT(T_old: np.ndarray, M1: np.ndarray, M2: np.ndarray, M3: np.ndarray, inverse: bool = False):
    """Convert a Trifocal Tensor that was computed in normalized image coordinates, to
    instead operate on standard image coordiantes.

    (an algebraic transformation has been applied to the image points.

    if inverse==0 :
      from a TFT T_old associated to P1_old, P2_old, P3_old, find the new TFT
      T_new associated to P1_new=M1*P1_old, P2_new=M2*P2_old, P3_new=M3*P3_old.
    if inverse==1 :
      from a TFT T_old associated to P1_old, P2_old, P3 _old, find the new TFT
      T_new associated to M1*P1_new=P1_old, M2*P2_new=P2_old, M3*P3_new=P3_old.

    Args:
        T_old: trifocal tensor that operates on normalized points.
        M1: normalization matrix for points in view 1.
        M2: normalization matrix for points in view 2.
        M3: normalization matrix for points in view 3.

    Returns:
        T_new: trifocal tensor that operates on unnormalized points.
    """
    if not inverse:
        M1i = np.linalg.inv(M1)
        T_new = np.zeros((3, 3, 3))
        T_new[:, :, 0] = (
            M2 @ (M1i[0, 0] * T_old[:, :, 0] + M1i[1, 0] * T_old[:, :, 1] + M1i[2, 0] * T_old[:, :, 2]) @ M3.T
        )
        T_new[:, :, 1] = (
            M2 @ (M1i[0, 1] * T_old[:, :, 0] + M1i[1, 1] * T_old[:, :, 1] + M1i[2, 1] * T_old[:, :, 2]) @ M3.T
        )
        T_new[:, :, 2] = (
            M2 @ (M1i[0, 2] * T_old[:, :, 0] + M1i[1, 2] * T_old[:, :, 1] + M1i[2, 2] * T_old[:, :, 2]) @ M3.T
        )

    else:
        M2i = np.linalg.inv(M2)
        M3i = np.linalg.inv(M3)
        T_new = np.zeros((3, 3, 3))
        T_new[:, :, 0] = (
            M2i @ (M1[0, 0] * T_old[:, :, 0] + M1[1, 0] * T_old[:, :, 1] + M1[2, 0] * T_old[:, :, 2]) @ M3i.T
        )
        T_new[:, :, 1] = (
            M2i @ (M1[0, 1] * T_old[:, :, 0] + M1[1, 1] * T_old[:, :, 1] + M1[2, 1] * T_old[:, :, 2]) @ M3i.T
        )
        T_new[:, :, 2] = (
            M2i @ (M1[0, 2] * T_old[:, :, 0] + M1[1, 2] * T_old[:, :, 1] + M1[2, 2] * T_old[:, :, 2]) @ M3i.T
        )

    T_new = T_new / np.linalg.norm(T_new)
    return T_new


def Normalize2Ddata(points) -> Tuple[np.ndarray, np.ndarray]:
    """Isometric Normalization of 2D points.

    Given a set of points in R^2, outputs a normalization matrix that, applied
    to the points (in homogeneous coordinates), transforms them into having
    mean (0,0) and mean distance to the center equal to sqrt(2).

    Note: a more standard formulation is:
    T = np.asarray([[1.0 / std[0], 0, -mean[0] / std[0]], [0, 1.0 / std[1], -mean[1] / std[1]], [0, 0, 1]])

    Args:
        points: (n,2)-vector of n points of dimension 2

    Returns:
        N: isometric normalization 3x3-matrix
        new_points: (n,3)-vector of the n normalized points of dimension 2 in homogeneous coordinates.
    """
    n = points.shape[0]

    mean = np.mean(points, axis=0)
    norm0 = np.linalg.norm(points - mean.reshape(1, 2), axis=1).mean()
    # norm0 = mean(sqrt(sum((points-repmat(points0,1,n)).^2,1)))
    N_matrix = np.diag(np.array([np.sqrt(2) / norm0, np.sqrt(2) / norm0, 1]))
    N_matrix[:2, 2] = -np.sqrt(2) * mean / norm0

    new_points = N_matrix[:2, :] @ np.vstack([points.T, np.ones((1, n))])

    return new_points.T, N_matrix


def test_normalize2Ddata() -> None:
    """ """
    p1 = np.array(
        [
            [56.0, 499.0],
            [507.0, 503.0],
            [284.0, 685.0],
            [415.0, 503.0],
            [542.0, 1071.0],
            [237.0, 484.0],
            [227.0, 472.0],
            [139.0, 312.0],
            [438.0, 322.0],
            [553.0, 859.0],
            [279.0, 556.0],
            [647.0, 518.0],
            [628.0, 627.0],
            [389.0, 814.0],
            [297.0, 487.0],
            [476.0, 87.0],
            [267.0, 206.0],
            [653.0, 981.0],
            [382.0, 147.0],
            [182.0, 142.0],
        ]
    )
    x1, Normal1 = Normalize2Ddata(p1)
    expected_x1 = np.array(
        [
            [-1.6355, -0.0745],
            [0.6418, -0.0543],
            [-0.4842, 0.8647],
            [0.1772, -0.0543],
            [0.8185, 2.8138],
            [-0.7216, -0.1502],
            [-0.7721, -0.2108],
            [-1.2164, -1.0187],
            [0.2934, -0.9682],
            [0.8741, 1.7433],
            [-0.5095, 0.2133],
            [1.3487, 0.0215],
            [1.2528, 0.5718],
            [0.0459, 1.5161],
            [-0.4186, -0.1351],
            [0.4853, -2.1548],
            [-0.5701, -1.5540],
            [1.3790, 2.3593],
            [0.0106, -1.8519],
            [-0.9993, -1.8771],
        ]
    )
    assert np.allclose(expected_x1, x1, atol=1e-3)


def compute_trifocal_tensor_inliers(Corresp: np.ndarray) -> None:
    """
    Args:
        Corresp: (N,6) array

    Returns:
        T: trifocal tensor
        errors:
    """
    # TODO: add RANSAC loop

    # Normalization of the data
    x1, Normal1 = Normalize2Ddata(Corresp[:, :2])
    x2, Normal2 = Normalize2Ddata(Corresp[:, 2:4])
    x3, Normal3 = Normalize2Ddata(Corresp[:, 4:])

    # Model to estimate T: linear equations
    T, P1, P2, P3 = linearTFT(x1, x2, x3)

    # tensor denormalization
    T = transform_TFT(T, Normal1, Normal2, Normal3, inverse=True)

    errors = compute_trifocal_errors(T, p1=Corresp[:, :2], p2=Corresp[:, 2:4], p3=Corresp[:, 4:])

    return T, errors


def test_compute_trifocal_tensor_inliers() -> None:
    """ """

    p1 = np.array(
        [
            [56.0, 499.0],
            [507.0, 503.0],
            [284.0, 685.0],
            [415.0, 503.0],
            [542.0, 1071.0],
            [237.0, 484.0],
            [227.0, 472.0],
            [139.0, 312.0],
            [438.0, 322.0],
            [553.0, 859.0],
            [279.0, 556.0],
            [647.0, 518.0],
            [628.0, 627.0],
            [389.0, 814.0],
            [297.0, 487.0],
            [476.0, 87.0],
            [267.0, 206.0],
            [653.0, 981.0],
            [382.0, 147.0],
            [182.0, 142.0],
        ]
    )
    p2 = np.array(
        [
            [22.0, 463.0],
            [467.0, 504.0],
            [229.0, 683.0],
            [358.0, 499.0],
            [620.0, 1080.0],
            [191.0, 465.0],
            [180.0, 450.0],
            [119.0, 265.0],
            [422.0, 316.0],
            [527.0, 867.0],
            [224.0, 545.0],
            [688.0, 533.0],
            [658.0, 639.0],
            [313.0, 822.0],
            [246.0, 473.0],
            [511.0, 85.0],
            [275.0, 170.0],
            [701.0, 985.0],
            [407.0, 129.0],
            [174.0, 85.0],
        ]
    )
    p3 = np.array(
        [
            [35.0, 453.0],
            [417.0, 526.0],
            [197.0, 702.0],
            [308.0, 514.0],
            [666.0, 1092.0],
            [172.0, 467.0],
            [160.0, 451.0],
            [139.0, 246.0],
            [403.0, 335.0],
            [492.0, 891.0],
            [190.0, 554.0],
            [683.0, 566.0],
            [650.0, 667.0],
            [253.0, 848.0],
            [216.0, 479.0],
            [522.0, 118.0],
            [299.0, 169.0],
            [707.0, 996.0],
            [426.0, 145.0],
            [196.0, 64.0],
        ]
    )

    correspondences = np.hstack([p1, p2, p3])
    errors = compute_trifocal_tensor_inliers(correspondences)


def test_transform_TFT() -> None:
    """ """
    # fmt: off
    T_normalized = np.zeros((3,3,3))
    T_normalized[:,:,0] = np.array(
        [
            [-0.295298011846379,   0.015710965461611,   0.004538735841168],
            [-0.035135213948350,  -0.000712219021949,  -0.001080621572178],
            [-0.016909725638198,   0.000380094632833,  -0.000066272289526]
        ]
    )

    T_normalized[:,:,1] = np.array(
        [
            [ 0.003205333125675,   0.318966211040013,  -0.001825598175232],
            [-0.579679842644145,  -0.017749684151412,  -0.028359399895284],
            [ 0.000779391316483,   0.012807267650718,  -0.000041344963317]
        ]
    )

    T_normalized[:,:,2] = np.array(
        [
            [ 0.003138155201618,   0.015192926068021,   0.331687588008719],
            [-0.013778244434547,  -0.000172174484972,   0.010423071661180],
            [-0.600250244153915,  -0.028825495597683,  -0.016018330467374]
        ]
    )
    Normal1 = np.array(
        [
            [0.005049434794292,                   0,  -1.918280278351590],
            [                0,   0.005049434794292,  -2.594147125567595],
            [                0,                   0,   1.000000000000000]
        ])
    Normal2 = np.array(
        [
            [0.004656043231928,                   0,  -1.706905448824881],
            [                0,   0.004656043231928,  -2.341524141336696],
            [                0,                   0,   1.000000000000000]
        ])
    Normal3 = np.array(
        [
            [0.004545232186843,                   0,  -1.620602536218864],
            [                0,   0.004545232186843,  -2.336931128865317],
            [                0,                   0,   1.000000000000000]
        ]
    )
    expected_T_unnormalized = np.zeros((3,3,3))
    expected_T_unnormalized[:,:,0] = np.array(
        [

           [-0.0040,  0.0003,  0.0000],
           [-0.0010, -0.0000, -0.0000],
           [-0.0000,  0.0000, -0.0000]
        ]
    )
    expected_T_unnormalized[:,:,1] = np.array(
        [
            [  0.0000,  0.0043, -0.0000 ],
            [ -0.0079, -0.0007, -0.0000 ],
            [  0.0000,  0.0000, -0.0000 ]
        ]
    )
    expected_T_unnormalized[:,:,2] = np.array(
        [
            [  0.1911, -0.6195,  0.0034 ],
            [  0.7603,  0.0377,  0.0006 ],
            [ -0.0070, -0.0012, -0.0000 ]
        ]
    )
    # fmt: on
    T_unnormalized = transform_TFT(T_old=T_normalized, M1=Normal1, M2=Normal2, M3=Normal3, inverse=True)
    assert np.allclose(T_unnormalized, expected_T_unnormalized, atol=1e-3)


if __name__ == "__main__":
    test_linearTFT()
    test_normalize2Ddata()
    test_transform_TFT()
    test_compute_trifocal_tensor_inliers()
