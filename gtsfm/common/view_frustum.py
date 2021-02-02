
from typing import Tuple

import numpy as np
from gtsam import Point3, Pose3

FRUSTUM_RAY_LEN = 0.3  # meters, arbitrary


class ViewFrustum:
    """ Generates edges of a 5-face mesh for drawing pinhole camera in 3d"""

    def __init__(self, fx: float, img_w: int, img_h: int) -> None:
        """
        Args:
        fx: focal length in x-direction, assuming square pixels (fx == fy)
        img_w: image width (in pixels)
        img_h: image height (in pixels)
        """
        self.fx = fx
        self.img_w = img_w
        self.img_h = img_h

    def get_frustum_vertices(self) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """Obtain 3d positions of all 5 frustum vertices

          (x,y,z)               (x,y,z)                (x,y,z)              (x,y,z)
              \\=================//                      \\                   //
               \\               //                        \\ 1-------------2 //
        (-w/2,-h/2,fx)       (w/2,-h/2,fx)                 \\| IMAGE PLANE |//
                 1-------------2                             |             |/
                 |\\         //| IMAGE PLANE  (-w/2, h/2,fx) 3-------------4 (w/2, h/2,fx)
                 | \\       // | IMAGE PLANE                  \\         //
                 3--\\-----//--4                               \\       //
                     \\   //                                    \\     //
                      \\ //                                      \\   //
                        O PINHOLE                                 \\ //
                                                                    O PINHOLE
        """
        uv = np.array(
            [
                [self.img_w // 2, self.img_h // 2],  # optical center
                [0, 0],  # top-left
                [self.img_w - 1, 0],  # top-right
                [0, self.img_h - 1],  # bottom-left
                [self.img_w - 1, self.img_h - 1],  # bottom-right
            ]
        )

        ray_dirs = compute_pixel_ray_directions_vectorized(
            uv, self.fx, self.img_w, self.img_h
        )
        v0 = ray_dirs[0] * 0
        v1 = ray_dirs[1] * FRUSTUM_RAY_LEN
        v2 = ray_dirs[2] * FRUSTUM_RAY_LEN
        v3 = ray_dirs[3] * FRUSTUM_RAY_LEN
        v4 = ray_dirs[4] * FRUSTUM_RAY_LEN

        return v0, v1, v2, v3, v4

    def get_mesh_edges_camframe(self) -> np.ndarray:
        """Return 8 edges defining the frustum mesh, in camera coordinate frame.

           Edge ordering below:

        0
            .-------.
           3|\\1   /|4
            ._\\ 2/_.
            \\ \\/ /
             5\\../6

           Returns:
               edges_camfr: array of shape (8,2,2) representing 8 polylines in camera frame
        """
        v0, v1, v2, v3, v4 = self.get_frustum_vertices()

        # top-left to top-right (e0)
        e0 = np.array([v1, v2])

        # optical center to top-left (e1)
        e1 = np.array([v0, v1])

        # optical center to top-right (e2)
        e2 = np.array([v0, v2])

        # bottom-left to top-left (e3)
        e3 = np.array([v3, v1])

        # bottom-right to top-right (e4)
        e4 = np.array([v4, v2])

        # optical center to bottom-left (e5)
        e5 = np.array([v0, v3])

        # optical center to bottom-right (e6)
        e6 = np.array([v0, v4])

        # bottom-left to bottom-right (e7)
        e7 = np.array([v3, v4])

        edges_camfr = np.stack([e0, e1, e2, e3, e4, e5, e6, e7])
        return edges_camfr

    def get_mesh_edges_worldframe(self, wTc: Pose3) -> np.ndarray:
        """Return 8 edges defining the frustum mesh, in world/global frame.

        Returns:
            edges_camfr: array of shape (8,2,2) representing 8 polylines in world frame
        """
        edges_camfr = self.get_mesh_edges_camframe()
        edges_worldfr = []

        for edge_camfr in edges_camfr:

            v_start = edge_camfr[0]
            v_end = edge_camfr[1]

            edge_worldfr = [wTc.transformFrom(Point3(v_start)), wTc.transformFrom(Point3(v_end))]
            edges_worldfr += [edge_worldfr]

        return edges_worldfr


def compute_pixel_ray_directions_vectorized(
    uv: np.ndarray, fx: float, img_w: int, img_h: int
) -> np.ndarray:
    """Given (u,v) coordinates and intrinsics, generate pixels rays in cam. coord frame
    Assume +z points out of the camera, +y is downwards, and +x is across the imager.

    Args:
        uv: array of shape (N,2) with (u,v) coordinates
        fx: focal length in x-direction, assuming square pixels (fx == fy)
        img_w: image width (in pixels)
        img_h: image height (in pixels)

    Returns:
        ray_dirs: Array of shape (N,3) with ray directions in camera frame
    """
    assert uv.shape[1] == 2
    assert uv.ndim == 2

    # assuming principal point at center of images now
    px = img_w / 2
    py = img_h / 2

    u = uv[:, 0]
    v = uv[:, 1]
    num_rays = uv.shape[0]

    # offset from center
    x_center_offs = u - px
    y_center_offs = v - py

    x_center_offs = x_center_offs.reshape(num_rays, 1)
    y_center_offs = y_center_offs.reshape(num_rays, 1)
    fx_tiled = np.tile(fx, (num_rays)).reshape(num_rays, 1)

    ray_dirs = np.hstack([x_center_offs, y_center_offs, fx_tiled])

    # elementwise multiplication of scalars requires last dim to match
    ray_dirs = (ray_dirs.T / np.linalg.norm(ray_dirs, axis=1)).T
    assert ray_dirs.shape[1] == 3
    return ray_dirs

