"""
Functions for rendering an interpolated path between the training poses using the gaussian splats after training.

Authors: Harneet Singh Khanuja
"""

from typing import Dict

import cv2
import numpy as np
import scipy
import torch

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.splat.gaussian_splatting import GaussianSplatting
from gtsfm.splat.gs_data import GaussianSplattingData
from gtsfm.utils import logger as logger_utils

logger = logger_utils.get_logger()


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def viewmatrix(lookdir: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct lookat view matrix.

    Args:
        lookdir: the direction the camera is looking (forward vector).
        up: the up vector for the camera.
        position: the position of the camera in world coordinates.

    Returns:
        The lookat view matrix
    """
    vec2 = normalize(lookdir)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


# See https://github.com/nerfstudio-project/gsplat/blob/main/examples/datasets/traj.py
# Helper function for generate_interpolated_path function
def poses_to_points(poses, dist):
    """Converts from pose matrices to (position, lookat, up) format.

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        dist: Scalar distance to place lookat and up points from the camera origin

    Returns:
        Array of points with [position, lookat, up] for each pose.
    """
    pos = poses[:, :3, -1]
    lookat = poses[:, :3, -1] - dist * poses[:, :3, 2]
    up = poses[:, :3, -1] + dist * poses[:, :3, 1]
    return np.stack([pos, lookat, up], 1)


# Helper function for generate_interpolated_path function
def points_to_poses(points):
    """Converts from (position, lookat, up) format to pose matrices.

    Args:
        points: array of points with [position, lookat, up] for each pose.
    Returns:
        Array of new camera poses.
    """
    return np.array([viewmatrix(p - l, u - p, p) for p, l, u in points])


# Helper function for generate_interpolated_path function
def b_spline_interpolate(points, n, k, s):
    """Runs multidimensional B-spline interpolation on the input points.

    Args:
        points: array of points with [position, lookat, up] for each pose.
        n: n_interp * (n - 1) total poses.
        k: polynomial degree of B-spline.
        s: parameter for spline smoothing, 0 forces exact interpolation.

    Returns:
        Interpolated points along the spline in (position, lookat, up) format.
    """
    sh = points.shape
    pts = np.reshape(points, (sh[0], -1))
    k = min(k, sh[0] - 1)
    tck, _ = scipy.interpolate.splprep(pts.T, k=k, s=s)
    u = np.linspace(0, 1, n, endpoint=False)
    new_points = np.array(scipy.interpolate.splev(u, tck))
    new_points = np.reshape(new_points.T, (n, sh[1], sh[2]))
    return new_points


# See https://github.com/nerfstudio-project/gsplat/blob/main/examples/datasets/traj.py
def generate_interpolated_path(
    poses: np.ndarray,
    n_interp: int,
    spline_degree: int = 5,
    smoothness: float = 0.03,
    rot_weight: float = 0.1,
):
    """Creates a smooth spline path between input keyframe camera poses.

    Spline is calculated with poses in format (position, lookat-point, up-point).

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        n_interp: returned path will have n_interp * (n - 1) total poses.
        spline_degree: polynomial degree of B-spline.
        smoothness: parameter for spline smoothing, 0 forces exact interpolation.
        rot_weight: relative weighting of rotation/translation in spline solve.

    Returns:
        Array of new camera poses with shape (n_interp * (n - 1), 3, 4).
    """
    points = poses_to_points(poses, dist=rot_weight)
    new_points = b_spline_interpolate(points, n_interp * (points.shape[0] - 1), k=spline_degree, s=smoothness)
    return points_to_poses(new_points)


@torch.no_grad()
def generate_interpolated_video(
    images_graph: Dict[int, Image], sfm_result_graph: GtsfmData, cfg_result_graph, splats_graph, video_fpath
):
    """
    Renders a video with interpolated poses from the training poses
    Args:
            images_graph: computation graph for images.
            sfm_result_graph: computation graph for SFM output
            cfg_result_graph: computation graph for the training Config parameters
            splats_graph: computation graph for the gaussian splats
            video_fpath: location where the video will be saved
    """
    cfg = cfg_result_graph
    gs = GaussianSplatting(cfg)
    gs.training = False
    splats = splats_graph
    num_frames = cfg.num_frames
    fps = cfg.fps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_dataset = GaussianSplattingData(images_graph, sfm_result_graph)
    camtoworlds_all = generate_interpolated_path(full_dataset._camtoworlds, num_frames)

    camtoworlds_all = np.concatenate(
        [
            camtoworlds_all,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0),
        ],
        axis=1,
    )
    camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
    K = torch.from_numpy(full_dataset._intrinsics[0]).float().to(device)
    height, width = full_dataset.actual_img_dims[0]
    frame_size = (width * 2, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_fpath, fourcc, fps, frame_size)
    for i in range(len(camtoworlds_all)):
        camtoworlds = camtoworlds_all[i : i + 1]
        Ks = K[None]

        renders, _, _ = gs._rasterize_splats(
            splats=splats,
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            render_mode="RGB+ED",
        )  # [1, H, W, 4]
        colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
        depths = renders[..., 3:4]  # [1, H, W, 1]
        depths = (depths - depths.min()) / (depths.max() - depths.min())
        canvas_list = [colors, depths.repeat(1, 1, 1, 3)]

        # write images
        canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().detach().numpy()
        canvas = (canvas * 255).astype(np.uint8)
        canvas_bgr = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        writer.write(canvas_bgr)
    writer.release()
    logger.info(f"Interpolated video saved to {video_fpath}")
