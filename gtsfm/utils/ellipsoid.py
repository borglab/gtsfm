"""Algorithms to fit Ellisoid geometry to 3D points and extract the rotation matrix necessary to align point cloud with
x, y, and z axes. Used in React Three Fiber Visualization Tool.

Authors: Adi Singh
"""

import numpy as np
from numpy.linalg import eig, inv
from scipy.spatial import ConvexHull, convex_hull_plot_2d


def center_point_cloud(pointsList: list) -> np.ndarray:
    """Centers a point cloud with respect using mean values of x, y, and z.

    Args:
        pointCloud: list(length N) of lists(length 3) (representing the point cloud)

    Returns:
        Centered point cloud, of shape Nx3

    Raises:
        TypeError: if points list is not 3 dimensional
    """
    if len(pointsList[0]) != 3:
        raise TypeError("Points list should be 3D")

    points = np.array([np.array(p) for p in pointsList])

    means = np.mean(points, axis=0)
    points_centered = points - means
    return points_centered


def compute_convex_hull(points_np: np.ndarray) -> np.ndarray:
    """Returns a set of vertices which represent the 3D Convex Hull (outer perimeter) of our point cloud.

    Args:
        points_np: point cloud of shape N x 3

    Returns:
        a reduced collection of points representing the convex hull, of shape M x 3 (M <= N)

    Raises:
        TypeError: if points_np is not 3 dimensional
    """
    if points_np.shape[1] != 3:
        raise TypeError("Point Cloud should be 3D")

    hull = ConvexHull(points_np)
    hullSet = set()
    for simplex in hull.simplices:
        hullSet.add(tuple(points_np[simplex[0]]))
        hullSet.add(tuple(points_np[simplex[1]]))

    hull_set = np.array([np.array(p) for p in hullSet])
    return hull_set


def fit_ls_ellipsoid(hull_set: np.ndarray) -> np.ndarray:
    """Fits an ellipsoid to a given set (x,y,z) 3D points using a least squares approximation approach. 
    Uses the formula for a general quadric surface: 
    Ax^2 + By^2 + Cz^2 +  Dxy +  Exz +  Fyz +  Gx +  Hy +  Iz + J = 0

    See https://www.physics.smu.edu/~scalise/SMUpreprints/SMU-HEP-10-14.pdf for more details on the 
    mathematical approach to ellipsoidal fitting.

    Args:
        hull_set: the convex hull of the original point cloud, of shape N x 3

    Returns:
        Array, of length 9, for parameters A, B, ..., J

    Raises:
        TypeError: if hull_set is not 3 dimensional
    """
    if hull_set.shape[1] != 3:
        raise TypeError("Point Cloud should be 3D")

    x = hull_set[:, 0].reshape(hull_set.shape[0], 1)
    y = hull_set[:, 1].reshape(hull_set.shape[0], 1)
    z = hull_set[:, 2].reshape(hull_set.shape[0], 1)

    A = np.hstack((x * x, y * y, z * z, 2 * x * y, 2 * x * z, 2 * y * z, 2 * x, 2 * y, 2 * z))
    b = np.ones_like(x)

    # solve normal equations to find the 9 parameters
    ATA = A.T @ A
    ATA_inv = np.linalg.inv(ATA)
    params = ATA_inv @ (AT @ b)
    params = np.append(params, -1)

    return params


def extract_params_from_poly(params: np.ndarray) -> np.ndarray:
    """Extracts the center, axes, and rotation matrix for an ellipsoid given the quadric surface polynomial equation.

    Args:
        params: parameters (A, B, ..., J) for the polynomial equation

    Returns:
        Center, Axes Lengths, and Rotation Matrix for the Ellipsoid.

    Raises:
        TypeError: if params is not of the length 10
    """
    if params.shape[0] != 10:
        raise TypeError("Params should of length 10")

    M = np.array(
        [
            [params[0], params[3] / 2.0, params[4] / 2.0, params[6] / 2.0],
            [params[3] / 2.0, params[1], params[5] / 2.0, params[7] / 2.0],
            [params[4] / 2.0, params[5] / 2.0, params[2], params[8] / 2.0],
            [params[6] / 2.0, params[7] / 2.0, params[8] / 2.0, params[9]],
        ]
    )

    # find center of ellipsoid
    A3 = M[0:3, 0:3]
    A3_inv = inv(A3)
    ofs = params[6:9] / 2.0
    center = -(A3_inv @ ofs)

    # center ellipsoid at origin
    Tofs = np.eye(4)
    Tofs[3, 0:3] = center
    R = Tofs @ (M @ Tofs.T)

    # eigen decomp
    R3 = R[0:3, 0:3]
    s1 = -R[3, 3]
    R3S = R3 / s1
    (el, ec) = eig(R3S)

    recip = 1.0 / np.abs(el)
    axes = np.sqrt(recip)
    inve = inv(ec)

    return (center, axes, inve)


def apply_ellipsoid_rotation(rot: np.ndarray, centered_pc: np.ndarray):
    """Applies a rotation on the centered point cloud.

    Args:
        rot: rotation matrix, shape 3x3
        centered_pc: centered point cloud, shape Nx3

    Returns:
        a modified point cloud entirely in list form

    Raises:
        TypeError: if rot matrix isn't 3x3 or centered_pc is not of dimension 3
    """
    if rot.shape[0] != 3 or rot.shape[1] != 3:
        raise TypeError("Rotation Matrix should be of shape 3x3")
    if centered_pc.shape[1] != 3:
        raise TypeError("Point Cloud shoud be 3 dimensional")

    rotated_points = rot @ centered_pc.T
    rotated_points_list = (rotated_points.T).tolist()
    return rotated_points_list