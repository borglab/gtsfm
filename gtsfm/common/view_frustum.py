from typing import List, Tuple

import numpy as np
from gtsam import Point3, Pose3

DEFAULT_FRUSTUM_RAY_LEN = 0.3  # meters, arbitrary


class ViewFrustum:
    """Generates edges of a 5-face mesh for drawing pinhole camera in 3d"""

    def __init__(
        self,
        fx: float,
        img_w: int,
        img_h: int,
        frustum_ray_len: float = DEFAULT_FRUSTUM_RAY_LEN,
    ) -> None:
        """
        Args:
            fx: focal length in x-direction, assuming square pixels (fx == fy)
            img_w: image width (in pixels)
            img_h: image height (in pixels)
            frustum_ray_len: extent to which extend frustum rays away from optical center
                (increase length for large-scale scenes to make frustums visible)
        """
        self.fx_ = fx
        self.img_w_ = img_w
        self.img_h_ = img_h
        self.frustum_ray_len_ = frustum_ray_len

    def get_frustum_vertices_camfr(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Obtain 3d positions of all 5 frustum vertices in the camera frame

          (x,y,z)               (x,y,z)                (x,y,z)              (x,y,z)
              \\=================//                      \\                   //
               \\               //                        \\ 1-------------2 //
        (-w/2,-h/2,fx)       (w/2,-h/2,fx)                 \\| IMAGE PLANE |//
                 1-------------2                             |             |/
                 |\\         //| IMAGE PLANE  (-w/2, h/2,fx) 4-------------3 (w/2, h/2,fx)
                 | \\       // | IMAGE PLANE                  \\         //
                 4--\\-----//--3                               \\       //
                     \\   //                                    \\     //
                      \\ //                                      \\   //
                        O PINHOLE                                 \\ //
                                                                    O PINHOLE
        """
        uv = np.array(
            [
                [self.img_w_ // 2, self.img_h_ // 2],  # optical center
                [0, 0],  # top-left
                [self.img_w_ - 1, 0],  # top-right
                [self.img_w_ - 1, self.img_h_ - 1],  # bottom-right
                [0, self.img_h_ - 1],  # bottom-left
            ]
        )

        ray_dirs = compute_pixel_ray_directions_vectorized(uv, self.fx_, self.img_w_, self.img_h_)
        v0 = ray_dirs[0] * 0
        v1 = ray_dirs[1] * self.frustum_ray_len_
        v2 = ray_dirs[2] * self.frustum_ray_len_
        v3 = ray_dirs[3] * self.frustum_ray_len_
        v4 = ray_dirs[4] * self.frustum_ray_len_

        return v0, v1, v2, v3, v4

    def get_mesh_edges_from_verts(self, verts: List[np.ndarray]) -> np.ndarray:
        """Given 5 vertices, with v0 being optical center, return edges defining the frustum mesh.

           Connectivity and edge ordering is defined below:
                4
            .-------.
           7|\\0   /|5
            ._\\ 1/_.
            \\ \\/ /
             3\\../2
        Args:
            verts: List of length 5, each element is an array of shape (3,) representing
                mesh vertices either in the camera frame or in the world frame

        Returns:
            edges: array of shape (8,2,2) representing 8 polylines in same frame as vertices
        """
        v0, v1, v2, v3, v4 = verts

        edges = []
        # connect optical center (OC i.e. v0) to 4 points that lie along 4 rays into frustum
        # OC to top-left, OC to top-right, OC to bottom-right, OC to bottom-left
        for v in [v1, v2, v3, v4]:
            edges += [np.array([v0, v])]

        edges += [np.array([v1, v2])]  # top-left to top-right
        edges += [np.array([v2, v3])]  # top-right to bottom-right
        edges += [np.array([v3, v4])]  # bottom-right to bottom-left
        edges += [np.array([v4, v1])]  # bottom-left to top-left

        edges = np.stack(edges)
        return edges

    def get_mesh_edges_camframe(self) -> np.ndarray:
        """Return 8 edges defining the frustum mesh, in the camera coordinate frame.

        Returns:
            edges_camfr: array of shape (8,2,2) representing 8 polylines in camera frame
        """
        v0, v1, v2, v3, v4 = self.get_frustum_vertices_camfr()
        edges_camfr = self.get_mesh_edges_from_verts([v0, v1, v2, v3, v4])
        return edges_camfr

    def get_mesh_edges_worldframe(self, wTc: Pose3) -> np.ndarray:
        """Return 8 edges defining the frustum mesh, in the world/global frame.

        Args:
            wTc: camera pose in world frame

        Returns:
            edges_worldfr: array of shape (8,2,2) representing 8 polylines in world frame
        """
        v0, v1, v2, v3, v4 = self.get_frustum_vertices_camfr()
        verts_worldfr = [wTc.transformFrom(Point3(vc)) for vc in [v0, v1, v2, v3, v4]]
        edges_worldfr = self.get_mesh_edges_from_verts(verts_worldfr)
        return edges_worldfr


def compute_pixel_ray_directions_vectorized(uv: np.ndarray, fx: float, img_w: int, img_h: int) -> np.ndarray:
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
    assert uv.ndim == 2
    assert uv.shape[1] == 2

    # assuming principal point at center of images now
    px = img_w / 2
    py = img_h / 2

    num_rays = uv.shape[0]
    # broadcast (1,2) across (N,2) uv array
    center_offs = uv - np.array([px, py]).reshape(1, -1)
    ray_dirs = np.zeros((num_rays, 3))
    ray_dirs[:, :2] = center_offs
    ray_dirs[:, 2] = fx

    ray_dirs = ray_dirs / np.linalg.norm(ray_dirs, axis=1, keepdims=True)

    assert ray_dirs.shape[1] == 3
    return ray_dirs
