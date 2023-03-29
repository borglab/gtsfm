"""Functions to visualize outputs at different stages of GTSFM.

Authors: Ayush Baid
"""
import os
from typing import List, Optional, Tuple

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Pose3
from matplotlib.axes._axes import Axes

import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.images as image_utils
import gtsfm.utils.io as io_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints

COLOR_RED = (255, 0, 0)
COLOR_GREEN = (0, 255, 0)


def set_axes_equal(ax: Axes):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres, cubes as cubes, etc..  This is one
    possible solution to Matplotlib's ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Ref: https://github.com/borglab/gtsam/blob/develop/python/gtsam/utils/plot.py#L13

    Args:
        ax: axis for the plot.
    """
    # get the min and max value for each of (x, y, z) axes as 3x2 matrix.
    # This gives us the bounds of the minimum volume cuboid encapsulating all
    # data.
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])

    # find the centroid of the cuboid
    centroid = np.mean(limits, axis=1)

    # pick the largest edge length for this cuboid
    largest_edge_length = np.max(np.abs(limits[:, 1] - limits[:, 0]))

    # set new limits to draw a cube using the largest edge length
    radius = 0.5 * largest_edge_length
    ax.set_xlim3d([centroid[0] - radius, centroid[0] + radius])
    ax.set_ylim3d([centroid[1] - radius, centroid[1] + radius])
    ax.set_zlim3d([centroid[2] - radius, centroid[2] + radius])


def draw_circle_cv2(image: Image, x: int, y: int, color: Tuple[int, int, int], circle_size: int = 10) -> Image:
    """Draw a solid circle on the image.

    Args:
        image: image to draw the circle on.
        x: x coordinate of the center of the circle.
        y: y coordinate of the center of the circle.
        color: RGB color of the circle.
        circle_size (optional): the size of the circle (in pixels). Defaults to 10.

    Returns:
        Image: image with the circle drawn on it.
    """
    return Image(
        cv.circle(image.value_array, center=(x, y), radius=circle_size, color=color, thickness=-1)  # solid circle
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
        line_thickness (optional): line thickness. Defaults to 10.

    Returns:
        Image: image with the line drawn on it.
    """
    return Image(cv.line(image.value_array, (x1, y1), (x2, y2), line_color, line_thickness, cv.LINE_AA))


def plot_twoview_correspondences(
    image_i1: Image,
    image_i2: Image,
    kps_i1: Keypoints,
    kps_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    inlier_mask: Optional[np.ndarray] = None,
    dot_color: Optional[Tuple[int, int, int]] = None,
    max_corrs: Optional[int] = 50,
) -> Image:
    """Plot correspondences between two images as lines between two circles.

    Args:
        image_i1: first image.
        image_i2: second image.
        kps_i1: keypoints for image_i1.
        kps_i2: keypoints for image_i2.
        corr_idxs_i1i2: indices of correspondences between i1 and i2.
        inlier_mask (optional): inlier mask for correspondences as boolean array. Defaults to None.
        dot_color (optional): color for keypoints. Defaults to (0, 0, 0).
        max_corrs (optional): max number of correspondences to plot. Defaults to 50.

    Returns:
        image visualizing correspondences between two images.
    """
    image_i1, image_i2, scale_i1, scale_i2 = image_utils.match_image_widths(image_i1, image_i2)

    result = image_utils.vstack_image_pair(image_i1, image_i2)

    if max_corrs is not None and corr_idxs_i1i2.shape[0] > max_corrs:
        # subsample matches
        corr_idxs_i1i2 = corr_idxs_i1i2[np.random.choice(corr_idxs_i1i2.shape[0], max_corrs)]

    for corr_idx in range(corr_idxs_i1i2.shape[0]):
        # mark the points in both images as circles, and draw connecting line
        idx_i1, idx_i2 = corr_idxs_i1i2[corr_idx]

        x_i1 = (kps_i1.coordinates[idx_i1, 0] * scale_i1[0]).astype(np.int32)
        y_i1 = (kps_i1.coordinates[idx_i1, 1] * scale_i1[1]).astype(np.int32)
        x_i2 = (kps_i2.coordinates[idx_i2, 0] * scale_i2[0]).astype(np.int32)
        y_i2 = (kps_i2.coordinates[idx_i2, 1] * scale_i2[1]).astype(np.int32) + image_i1.height

        # drawing correspondences with optional inlier mask
        if inlier_mask is None:
            line_color = tuple([int(c) for c in np.random.randint(0, 255 + 1, 3)])
        elif inlier_mask[corr_idx]:
            line_color = COLOR_GREEN
        else:
            line_color = COLOR_RED

        result = draw_line_cv2(result, x_i1, y_i1, x_i2, y_i2, line_color, line_thickness=2)

        if dot_color is None:
            dot_color = line_color
        result = draw_circle_cv2(result, x_i1, y_i1, dot_color, circle_size=2)
        result = draw_circle_cv2(result, x_i2, y_i2, dot_color, circle_size=2)

    return result


def plot_sfm_data_3d(sfm_data: GtsfmData, ax: Axes, max_plot_radius: float = 50) -> None:
    """Plot the camera poses and landmarks in 3D matplotlib plot.

    Args:
        sfm_data: SfmData object with camera and tracks.
        ax: axis to plot on.
        max_plot_radius: maximum distance threshold away from any camera for which a point
            will be plotted
    """
    camera_poses = [sfm_data.get_camera(i).pose() for i in sfm_data.get_valid_camera_indices()]
    plot_poses_3d(camera_poses, ax)

    num_tracks = sfm_data.number_tracks()
    # Restrict 3d points to some radius of camera poses
    points_3d = np.array([list(sfm_data.get_track(j).point3()) for j in range(num_tracks)])

    nearby_points_3d = comp_utils.get_points_within_radius_of_cameras(camera_poses, points_3d, max_plot_radius)

    # plot 3D points
    for landmark in nearby_points_3d:
        ax.plot(landmark[0], landmark[1], landmark[2], "g.", markersize=1)


def plot_poses_3d(
    wTi_list: List[Optional[Pose3]], ax: Axes, center_marker_color: str = "k", label_name: Optional[str] = None
) -> None:
    """Plot poses in 3D as dots for centers and lines denoting the orthonormal
    coordinate system for each camera.

    Color convention: R -> x axis, G -> y axis, B -> z axis.

    Args:
        wTi_list: list of poses to plot.
        ax: axis to plot on.
        center_marker_color (optional): color for camera center marker. Defaults to "k".
        name:
    """
    spec = "{}.".format(center_marker_color)

    is_label_added = False
    for wTi in wTi_list:
        if wTi is None:
            continue

        if is_label_added:
            # for the rest of iterations, set label to None (otherwise would be duplicated in legend)
            label_name = None

        x, y, z = wTi.translation().squeeze()
        ax.plot(x, y, z, spec, markersize=10, label=label_name)
        is_label_added = True

        R = wTi.rotation().matrix()

        # getting the direction of the coordinate system (x, y, z axes)
        default_axis_length = 0.5
        v1 = R[:, 0] * default_axis_length
        v2 = R[:, 1] * default_axis_length
        v3 = R[:, 2] * default_axis_length

        ax.plot3D([x, x + v1[0]], [y, y + v1[1]], [z, z + v1[2]], c="r")
        ax.plot3D([x, x + v2[0]], [y, y + v2[1]], [z, z + v2[2]], c="g")
        ax.plot3D([x, x + v3[0]], [y, y + v3[1]], [z, z + v3[2]], c="b")


def plot_and_compare_poses_3d(wTi_list: List[Pose3], wTi_list_: List[Pose3]) -> None:
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


def save_twoview_correspondences_viz(
    image_i1: Image,
    image_i2: Image,
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    inlier_mask: Optional[np.ndarray],
    file_path: str,
) -> None:
    """Visualize correspondences between pairs of images.

    Args:
        image_i1: image #i1.
        image_i2: image #i2.
        keypoints_i1: detected Keypoints for image #i1.
        keypoints_i2: detected Keypoints for image #i2.
        corr_idxs_i1i2: correspondence indices.
        two_view_report: front-end metrics and inlier/outlier info for image pair.
        file_path: file path to save the visualization.
    """
    plot_img = plot_twoview_correspondences(
        image_i1, image_i2, keypoints_i1, keypoints_i2, corr_idxs_i1i2, inlier_mask=inlier_mask
    )

    io_utils.save_image(plot_img, file_path)


def save_sfm_data_viz(sfm_data: GtsfmData, folder_name: str) -> None:
    """Visualize the camera poses and 3d points in SfmData.

    Args:
        sfm_data: data to visualize.
        folder_name: folder to save the visualization at.
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    plot_sfm_data_3d(sfm_data, ax)
    set_axes_equal(ax)

    # save the 3D plot in the original view
    fig.savefig(os.path.join(folder_name, "3d.png"))

    # save the BEV representation
    default_camera_elevation = 100  # in metres above ground
    ax.view_init(azim=0, elev=default_camera_elevation)
    fig.savefig(os.path.join(folder_name, "bev.png"))

    plt.close(fig)


def save_camera_poses_viz(
    pre_ba_sfm_data: GtsfmData, post_ba_sfm_data: GtsfmData, gt_pose_graph: List[Optional[Pose3]], folder_name: str
) -> None:
    """Visualize the camera pose and save to disk.

    Args:
        pre_ba_sfm_data: data input to bundle adjustment.
        post_ba_sfm_data: output of bundle adjustment.
        gt_pose_graph: ground truth poses.
        folder_name: folder to save the visualization at.
    """
    # extract camera poses
    pre_ba_poses = []
    for i in pre_ba_sfm_data.get_valid_camera_indices():
        pre_ba_poses.append(pre_ba_sfm_data.get_camera(i).pose())

    post_ba_poses = []
    for i in post_ba_sfm_data.get_valid_camera_indices():
        post_ba_poses.append(post_ba_sfm_data.get_camera(i).pose())

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    plot_poses_3d(gt_pose_graph, ax, center_marker_color="m", label_name="GT")
    plot_poses_3d(pre_ba_poses, ax, center_marker_color="c", label_name="Pre-BA")
    plot_poses_3d(post_ba_poses, ax, center_marker_color="k", label_name="Post-BA")

    ax.legend(loc="upper left")
    set_axes_equal(ax)

    # save the 3D plot in the original view
    fig.savefig(os.path.join(folder_name, "poses_3d.png"))

    # save the BEV representation
    default_camera_elevation = 100  # in metres above ground
    ax.view_init(azim=0, elev=default_camera_elevation)
    fig.savefig(os.path.join(folder_name, "poses_bev.png"))

    plt.close(fig)
