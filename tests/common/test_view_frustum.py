import matplotlib.pyplot as plt
import numpy as np
from gtsam import Point3, Pose3, Rot3
from matplotlib.axes._axes import Axes  # noqa: F401
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.spatial.transform import Rotation

import gtsfm.utils.viz as viz_utils
from gtsfm.common.view_frustum import ViewFrustum, compute_pixel_ray_directions_vectorized


def test_compute_pixel_ray_directions_vectorized():
    """ """
    uv = np.array([[0, 0], [1, 1], [3, 3]])
    img_h = 4
    img_w = 4
    fx = 5
    ray_dirs = compute_pixel_ray_directions_vectorized(uv, fx, img_w, img_h)

    expected_ray_dirs = np.array([[-2.0, -2.0, 5.0], [-1.0, -1.0, 5.0], [1.0, 1.0, 5.0]])

    expected_ray_dirs[0] /= np.linalg.norm(expected_ray_dirs[0])
    expected_ray_dirs[1] /= np.linalg.norm(expected_ray_dirs[1])
    expected_ray_dirs[2] /= np.linalg.norm(expected_ray_dirs[2])

    assert np.allclose(ray_dirs, expected_ray_dirs, atol=1e-4)


def test_get_mesh_edges_camframe():
    """Verify we can plot the 8 edges of the frustum"""
    fx = 1392.1069298937407
    img_w = 1920
    img_h = 1200
    frustum_obj = ViewFrustum(fx, img_w, img_h)
    edges_camfr = frustum_obj.get_mesh_edges_camframe()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for edge_camfr in edges_camfr:

        # start and end vertices
        vs = edge_camfr[0]
        ve = edge_camfr[1]
        ax.plot3D([vs[0], ve[0]], [vs[1], ve[1]], [vs[2], ve[2]], c="b")

    viz_utils.set_axes_equal(ax)
    # uncomment line below to see viz
    # plt.show()
    plt.close("all")


def test_get_mesh_edges_worldframe():
    """
    Calibration from Argoverse:
        train1/273c1883-673a-36bf-b124-88311b1a80be
    """
    fx = 1392.1069298937407
    img_w = 1920
    img_h = 1200
    frustum_obj = ViewFrustum(fx, img_w, img_h)

    # quaternion in (qw, qx, qy, qz) order
    qw, qx, qy, qz = [
        0.06401399257908719,
        -0.06266155729362148,
        -0.7078861012523953,
        0.7006232979606847,
    ]
    quat_xyzw = [qx, qy, qz, qw]
    wtc = np.array([1.294530313917792, -0.28519924870913804, 1.3701008006525792])

    wRc = Rotation.from_quat(quat_xyzw).as_matrix()

    # actually egovehicle_SE3_camera
    wTc = Pose3(Rot3(wRc), Point3(wtc))
    edges_worldfr = frustum_obj.get_mesh_edges_worldframe(wTc)

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    for edge_worldfr in edges_worldfr:

        # start and end vertices
        vs = edge_worldfr[0]
        ve = edge_worldfr[1]
        ax.plot3D([vs[0], ve[0]], [vs[1], ve[1]], [vs[2], ve[2]], c="b")

    viz_utils.set_axes_equal(ax)
    # uncomment line below to see viz
    # plt.show()
    plt.close("all")


if __name__ == "__main__":
    # test_get_mesh_edges_camframe()
    test_get_mesh_edges_worldframe()
