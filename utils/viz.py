"""Functions to visualize.

Authors: Ayush Baid
"""
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from gtsam import Pose3, SfmData
from matplotlib.axes._axes import Axes
import cv2 as cv
from numpy.ma.core import dot

from common.image import Image
from common.keypoints import Keypoints


def vstack_images(
    image_i1_array: np.ndarray, image_i2_array: np.ndarray
) -> np.ndarray:
    """
    Stacks 2 images top-bottom
    :param imgA:
    :param imgB:
    :return:

    NOTE: copied from Frank's assignment
    """
    new_height = image_i1_array.shape[0] + image_i2_array.shape[0]
    new_width = max(image_i1_array.shape[1], image_i2_array.shape[1])

    new_image = np.ones(
        (new_height, new_width, 3),
        dtype=image_i1_array.dtype,
    )

    if np.issubdtype(new_image.dtype, np.integer):
        new_image *= 255

    new_image[
        : image_i1_array.shape[0], : image_i1_array.shape[1], :
    ] = image_i1_array
    new_image[
        image_i1_array.shape[0] :, : image_i2_array.shape[1], :
    ] = image_i2_array

    return new_image


def plot_twoview_correspondences(
    image_i1: Image,
    image_i2: Image,
    keypoints_i1: Keypoints,
    keypoints_i2: Keypoints,
    corr_idxs_i1i2: np.ndarray,
    match_width: bool = True,
    inlier_mask: Optional[np.ndarray] = None,
    dot_color: Tuple[int, int, int] = (0, 0, 0),
    max_keypoints: Optional[int] = 50,
) -> Image:
    """

    NOTE: copied from Frank's assignment and modified
    """
    scale_i1 = 1.0
    scale_i2 = 1.0
    image_i1_array = image_i1.value_array
    image_i2_array = image_i2.value_array
    if match_width:
        max_width = int(max(image_i1.width, image_i2.width))
        if image_i1.width != max_width:
            scale_i1 = float(max_width) / image_i1.width
            image_i1_array = cv.resize(
                image_i1.value_array,
                (max_width, int(image_i1.height * scale_i1)),
                interpolation=cv.INTER_CUBIC,
            )
        elif image_i2.width != max_width:
            scale_i2 = float(max_width) / image_i2.width

            image_i2_array = cv.resize(
                image_i2.value_array,
                (max_width, int(image_i1.height * scale_i2)),
                interpolation=cv.INTER_CUBIC,
            )

    stacked_image_array = vstack_images(image_i1_array, image_i2_array)

    shift_y = image_i1.height

    if max_keypoints is not None:
        corr_idxs_i1i2 = corr_idxs_i1i2[
            np.random.choice(corr_idxs_i1i2.shape[0], max_keypoints)
        ]

    for corr_idx in range(corr_idxs_i1i2.shape[0]):
        # mark the points in both images
        idx_i1, idx_i2 = corr_idxs_i1i2[corr_idx]

        x_i1, y_i1 = (keypoints_i1.coordinates[idx_i1] * scale_i1).astype(
            np.int32
        )
        x_i2, y_i2 = (keypoints_i2.coordinates[idx_i2] * scale_i2).astype(
            np.int32
        )
        y_i2 += shift_y

        stacked_image_array = cv.circle(
            stacked_image_array,
            (x_i1, y_i1),
            10,
            dot_color,
            -1,
        )
        stacked_image_array = cv.circle(
            stacked_image_array,
            (x_i2, y_i2),
            10,
            dot_color,
            -1,
        )

        # drawing correspondences with optional inlier mask
        if inlier_mask is None:
            line_color = (0, 0, 255)
        elif inlier_mask[corr_idx]:
            line_color = (0, 255, 0)
        else:
            line_color = (255, 0, 0)

        stacked_image_array = cv.line(
            stacked_image_array,
            (x_i1, y_i1),
            (x_i2, y_i2),
            line_color,
            5,
            cv.LINE_AA,
        )

    # scale the image to high-res
    stacked_image_array = cv.resize(
        stacked_image_array,
        (stacked_image_array.shape[1] * 2, stacked_image_array.shape[0] * 2),
        interpolation=cv.INTER_CUBIC,
    )

    return Image(stacked_image_array)


def plot_sfm_data(sfm_data: SfmData, ax: Axes) -> None:

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
    wTi_list: List[Pose3], ax: Axes, camera_marker_color="k"
) -> None:
    for wTi in wTi_list:
        camera_center = wTi.translation().squeeze()

        ax.scatter(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            c=camera_marker_color,
            marker="x",
            s=20,
            depthshade=0,
        )

        R = wTi.rotation().matrix()

        v1 = R[:, 0] * 1
        v2 = R[:, 1] * 1
        v3 = R[:, 2] * 1

        cc0, cc1, cc2 = camera_center

        ax.plot3D(
            [cc0, cc0 + v1[0]], [cc1, cc1 + v1[1]], [cc2, cc2 + v1[2]], c="r"
        )
        ax.plot3D(
            [cc0, cc0 + v2[0]], [cc1, cc1 + v2[1]], [cc2, cc2 + v2[2]], c="g"
        )
        ax.plot3D(
            [cc0, cc0 + v3[0]], [cc1, cc1 + v3[1]], [cc2, cc2 + v3[2]], c="b"
        )


def plot_to_compare_poses_3d(
    wTi_list: List[Pose3], wTi_list_: List[Pose3]
) -> None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for wTi in wTi_list:
        camera_center = wTi.translation().squeeze()

        ax.scatter(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            c="k",
            marker="x",
            s=20,
            depthshade=0,
        )

        R = wTi.rotation().matrix()

        v1 = R[:, 0] * 1
        v2 = R[:, 1] * 1
        v3 = R[:, 2] * 1

        cc0, cc1, cc2 = camera_center

        ax.plot3D(
            [cc0, cc0 + v1[0]], [cc1, cc1 + v1[1]], [cc2, cc2 + v1[2]], c="r"
        )
        ax.plot3D(
            [cc0, cc0 + v2[0]], [cc1, cc1 + v2[1]], [cc2, cc2 + v2[2]], c="g"
        )
        ax.plot3D(
            [cc0, cc0 + v3[0]], [cc1, cc1 + v3[1]], [cc2, cc2 + v3[2]], c="b"
        )

    for wTi in wTi_list_:
        camera_center = wTi.translation().squeeze()

        ax.scatter(
            camera_center[0],
            camera_center[1],
            camera_center[2],
            c="c",
            marker="x",
            s=20,
            depthshade=0,
        )

        R = wTi.rotation().matrix()

        v1 = R[:, 0] * 1
        v2 = R[:, 1] * 1
        v3 = R[:, 2] * 1

        cc0, cc1, cc2 = camera_center

        ax.plot3D(
            [cc0, cc0 + v1[0]], [cc1, cc1 + v1[1]], [cc2, cc2 + v1[2]], c="r"
        )
        ax.plot3D(
            [cc0, cc0 + v2[0]], [cc1, cc1 + v2[1]], [cc2, cc2 + v2[2]], c="g"
        )
        ax.plot3D(
            [cc0, cc0 + v3[0]], [cc1, cc1 + v3[1]], [cc2, cc2 + v3[2]], c="b"
        )

    plt.show()
