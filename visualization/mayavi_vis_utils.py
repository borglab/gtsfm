"""
Utilities for rendering camera frustums and 3d point clouds using Mayavi mlab.

Author: John Lambert
"""

import argparse
from typing import List

import mayavi
import numpy as np
from colour import Color
from gtsam import Cal3Bundler, Pose3
from mayavi import mlab


from gtsfm.common.view_frustum import ViewFrustum


def draw_point_cloud_mayavi(
    args: argparse.Namespace, fig: mayavi.core.scene.Scene, point_cloud: np.ndarray, rgb: np.ndarray
) -> None:
    """Render a point cloud as a collection of spheres, using Mayavi.

    Args:
        args: rendering options.
        fig: Mayavi figure object.
        point_cloud: array of shape (N,3) representing 3d points.
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255]
    """
    n = point_cloud.shape[0]
    x, y, z = point_cloud.T
    alpha = np.ones((n, 1)).astype(np.uint8) * 255  # no transparency
    rgba = np.hstack([rgb, alpha]).astype(np.uint8)

    pts = mlab.pipeline.scalar_scatter(x, y, z)  # plot the points
    pts.add_attribute(rgba, "colors")  # assign the colors to each point
    pts.data.point_data.set_active_scalars("colors")
    g = mlab.pipeline.glyph(pts)
    g.glyph.glyph.scale_factor = args.sphere_radius  # set scaling for all the points
    g.glyph.scale_mode = "data_scaling_off"  # make all the points same size


def draw_cameras_mayavi(
    zcwTw: Pose3, fig: mayavi.core.scene.Scene, calibrations: List[Cal3Bundler], wTi_list: List[Pose3]
) -> None:
    """Render camera frustums as collections of line segments, using Mayavi mlab.

    Args:
        zcwTw: transforms world points to a new world frame where the point cloud is zero-centered
        fig: Mayavi mlab figure object.
        calibrations: calibration object for each camera
        wTi_list: list of camera poses for each image
    """
    colormap = np.array(
        [[color_obj.rgb] for color_obj in Color("red").range_to(Color("green"), len(wTi_list))]
    ).squeeze()

    for i, (K, wTi) in enumerate(zip(calibrations, wTi_list)):
        wTi = zcwTw.compose(wTi)

        color = tuple(colormap[i].tolist())

        K = K.K()
        fx = K[0, 0]

        # Use 2*principal point as proxy measure for image height and width
        # TODO (in future PR): use the real image height and width
        px = K[0, 2]
        py = K[1, 2]

        img_w = px * 2
        img_h = py * 2
        frustum_obj = ViewFrustum(fx, img_w, img_h)

        edges_worldfr = frustum_obj.get_mesh_edges_worldframe(wTi)
        for edge_worldfr in edges_worldfr:

            # start and end vertices
            vs = edge_worldfr[0]
            ve = edge_worldfr[1]

            # TODO: consider adding line_width
            mlab.plot3d(  # type: ignore
                [vs[0], ve[0]],
                [vs[1], ve[1]],
                [vs[2], ve[2]],
                color=color,
                tube_radius=None,
                figure=fig,
            )


def draw_scene_mayavi(
    args: argparse.Namespace,
    point_cloud: np.ndarray,
    rgb: np.ndarray,
    calibrations: List[Cal3Bundler],
    wTi_list: List[Pose3],
    zcwTw: Pose3,
) -> None:
    """Render camera frustums and a 3d point cloud against a white background, using Mayavi.

    Args:
        args: rendering options.
        point_cloud
        rgb: uint8 array of shape (N,3) representing colors in RGB order, in the range [0,255].
        calibrations: calibration object for each camera
        wTi_list: list of camera poses for each image
        zcwTw: transforms world points to a new world frame where the point cloud is zero-centered
    """
    bgcolor = (1, 1, 1)
    fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))  # type: ignore
    draw_cameras_mayavi(zcwTw, fig, calibrations, wTi_list)
    draw_point_cloud_mayavi(args, fig, point_cloud, rgb)
    mlab.show()
