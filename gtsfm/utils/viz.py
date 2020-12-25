"""Functions to visualize outputs at different stages of GTSFM.

Authors: Ayush Baid
"""
from typing import List, Optional, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Pose3, SfmData
from matplotlib.axes._axes import Axes

import gtsfm.utils.images as image_utils
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)


def set_axes_equal(ax: Axes):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Ref: https://github.com/borglab/gtsam/blob/develop/python/gtsam/utils/plot.py#L13

    Args:
        ax: axis for the plot.
    """
    # get the min and max value for each of (x, y, z) axes as 3x2 matrix.
    # This gives us the bounds of the minimum volume cuboid encapsulating all
    # data.
    limits = np.array(
        [
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d(),
        ]
    )

    # find the centroid of the cuboid
    centroid = np.mean(limits, axis=1)

    # pick the largest edge length for this cuboid
    largest_edge_length = np.max(np.abs(limits[:, 1] - limits[:, 0]))

    # set new limits to draw a cube using the largest edge length
    radius = 0.5 * largest_edge_length
    ax.set_xlim3d([centroid[0] - radius, centroid[0] + radius])
    ax.set_ylim3d([centroid[1] - radius, centroid[1] + radius])
    ax.set_zlim3d([centroid[2] - radius, centroid[2] + radius])


def draw_circle_cv2(
    image: Image,
    x: int,
    y: int,
    color: Tuple[int, int, int],
    circle_size: int = 10,
) -> Image:
    """Draw a solid circle on the image.

    Args:
        image: image to draw the circle on.
        x: x coordinate of the center of the circle.
        y: y coordinate of the center of the circle.
        color: RGB color of the circle.

    Returns:
        Image: image with the circle drawn on it.
    """
    return Image(
        cv.circle(
            image.value_array,
            center=(x, y),
            radius=circle_size,
            color=color,
            thickness=-1,  # solid circle
        )
    )


def draw_line_cv2(
    image: Image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    line_color: Tuple[int, int, int],
    line_thickness: int = 10,
) -> Image:
    """Draw a line on the image from coordinates (x1, y1) to (x2, y2).

    Args:
        image: image to draw the line on.
        x1: x coordinate of start of the line.
        y1: y coordinate of start of the line.
        x2: x coordinate of end of the line.
        y2: y coordinate of end of the line.
        line_color: color of the line.
        line_thickness (optional): line thickness. Defaults to 5.

    Returns:
        Image: image with the line drawn on it.
    """
    return Image(
        cv.line(
            image.value_array,
            (x1, y1),
            (x2, y2),
            line_color,
            line_thickness,
            cv.LINE_AA,
        )
    )


def plot_twoview_correspondences(
    image_i1: Image,
    image_i2: Image,
    kps_i1: Keypoints,
    kps_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    dot_color: Tuple[int, int, int] = (0, 0, 0),
    max_corrs: Optional[int] = 50,
) -> Image:
    """Plot correspondences between two images.

    Args:
        image_i1: first image.
        image_i2: second image.
        kps_i1: keypoints for image_i1.
        kps_i2: keypoints for image_i2.
        corr_idxs_i1i2: indices of correspondences between i1 and i2.
        inlier_mask (optional): inlier mask for correspondences as boolean
                                array. Defaults to None.
        dot_color (optional): color for keypoints. Defaults to (0, 0, 0).
        max_corrs (optional): max number of correspondences to plot. Defaults
                              to 50.

    Returns:
        image visualizing correspondences between two images.
    """
    image_i1, image_i2, scale_i1, scale_i2 = image_utils.match_image_widths(
        image_i1, image_i2
    )

    result = image_utils.vstack_images(image_i1, image_i2)

    if max_corrs is not None:
        # subsample matches
        corr_idxs_i1i2 = corr_idxs_i1i2[
            np.random.choice(corr_idxs_i1i2.shape[0], max_corrs)
        ]

    for corr_idx in range(corr_idxs_i1i2.shape[0]):
        # mark the points in both images as circles, and draw connecting line
        idx_i1, idx_i2 = corr_idxs_i1i2[corr_idx]

        x_i1 = (kps_i1.coordinates[idx_i1, 0] * scale_i1[0]).astype(np.int32)
        y_i1 = (kps_i1.coordinates[idx_i1, 1] * scale_i1[1]).astype(np.int32)
        x_i2 = (kps_i2.coordinates[idx_i2, 0] * scale_i2[0]).astype(np.int32)
        y_i2 = (kps_i2.coordinates[idx_i2, 1] * scale_i2[1]).astype(
            np.int32
        ) + image_i1.height

        result = draw_circle_cv2(result, x_i1, y_i1, dot_color)
        result = draw_circle_cv2(result, x_i2, y_i2, dot_color)

        # drawing correspondences with optional inlier mask
        if inlier_mask is None:
            line_color = tuple([ int(c) for c in np.random.randint(0,255+1,3)])
        elif inlier_mask[corr_idx]:
            line_color = COLOR_GREEN
        else:
            line_color = COLOR_RED

        result = draw_line_cv2(result, x_i1, y_i1, x_i2, y_i2, line_color)

    return result


def plot_sfm_data_3d(sfm_data: SfmData, ax: Axes) -> None:
    """Plot the camera poses and landmarks in 3D matplotlib plot.

    Args:
        sfm_data: SfmData object with camera and tracks.
        ax: axis to plot on.
    """
    # extract camera poses
    camera_poses = []
    for i in range(sfm_data.number_cameras()):
        camera_poses.append(sfm_data.camera(i).pose())

    plot_poses_3d(camera_poses, ax)

    # plot 3D points
    for j in range(sfm_data.number_tracks()):
        landmark = sfm_data.track(j).point3()

        ax.plot(landmark[0], landmark[1], landmark[2], "g.", markersize=1)


def plot_poses_3d(
    wTi_list: List[Pose3], ax: Axes, center_marker_color: str = "k"
) -> None:
    """Plot poses in 3D as dots for centers and lines denoting the orthonormal
    coordinate system for each camera.

    Color convention: R -> x axis, G -> y axis, B -> z axis.

    Args:
        wTi_list: list of poses to plot.
        ax: axis to plot on.
        center_marker_color (optional): color for camera center marker.
                                        Defaults to "k".
    """
    spec = "{}.".format(center_marker_color)

    for wTi in wTi_list:
        x, y, z = wTi.translation().squeeze()

        ax.plot(x, y, z, spec, markersize=10)

        R = wTi.rotation().matrix()

        # getting the direction of the coordinate system (x, y, z axes)
        default_axis_length = 0.5
        v1 = R[:, 0] * default_axis_length
        v2 = R[:, 1] * default_axis_length
        v3 = R[:, 2] * default_axis_length

        ax.plot3D([x, x + v1[0]], [y, y + v1[1]], [z, z + v1[2]], c="r")
        ax.plot3D([x, x + v2[0]], [y, y + v2[1]], [z, z + v2[2]], c="g")
        ax.plot3D([x, x + v3[0]], [y, y + v3[1]], [z, z + v3[2]], c="b")


def plot_and_compare_poses_3d(
    wTi_list: List[Pose3], wTi_list_: List[Pose3]
) -> None:
    """Plots two sets poses in 3D with different markers to compare.

    The markers are colored black (k) and cyan (c) for the two lists.

    Args:
        wTi_list: first set of poses.
        wTi_list_: second set of poses.
    """
    fig = plt.figure()
    ax = fig.gca(projection="3d")

    plot_poses_3d(wTi_list, ax, center_marker_color="k")
    plot_poses_3d(wTi_list_, ax, center_marker_color="c")
    set_axes_equal(ax)

    plt.show()
