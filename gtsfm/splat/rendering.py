"""
Functions for rendering an interpolated path between the training poses using the gaussian splats after training.

Authors: Harneet Singh Khanuja
"""

from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np
import scipy  # type: ignore
import torch
from gsplat import export_splats

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.splat.gaussian_splatting import GaussianSplatting
from gtsfm.splat.gs_data import GaussianSplattingData
from gtsfm.utils import logger as logger_utils

logger = logger_utils.get_logger()


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalization helper function."""
    return x / np.linalg.norm(x)


def view_matrix(viewing_direction: np.ndarray, up: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Construct look_at view matrix.

    Args:
        viewing_direction: the direction the camera is looking (forward vector).
        up: the up vector for the camera.
        position: the position of the camera in world coordinates.

    Returns:
        The look_at view matrix
    """
    vec2 = normalize(viewing_direction)
    vec0 = normalize(np.cross(up, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, position], axis=1)
    return m


# See https://github.com/nerfstudio-project/gsplat/blob/main/examples/datasets/traj.py
# Helper function for generate_interpolated_path function
def poses_to_points(poses, dist):
    """Converts from pose matrices to (position, look_at, up) format.

    Args:
        poses: (n, 3, 4) array of input pose keyframes.
        dist: Scalar distance to place look_at and up points from the camera origin

    Returns:
        Array of points with [position, look_at, up] for each pose.
    """
    pos = poses[:, :3, -1]
    look_at = poses[:, :3, -1] - dist * poses[:, :3, 2]
    up = poses[:, :3, -1] + dist * poses[:, :3, 1]
    return np.stack([pos, look_at, up], 1)


# Helper function for generate_interpolated_path function
def points_to_poses(points):
    """Converts from (position, look_at, up) format to pose matrices.

    Args:
        points: array of points with [position, look_at, up] for each pose.
    Returns:
        Array of new camera poses.
    """
    return np.array([view_matrix(p - l, u - p, p) for p, l, u in points])


# Helper function for generate_interpolated_path function
def b_spline_interpolate(points, n, k, s):
    """Runs multidimensional B-spline interpolation on the input points.

    Args:
        points: array of points with [position, look_at, up] for each pose.
        n: n_interp * (n - 1) total poses.
        k: polynomial degree of B-spline.
        s: parameter for spline smoothing, 0 forces exact interpolation.

    Returns:
        Interpolated points along the spline in (position, look_at, up) format.
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

    Spline is calculated with poses in format (position, look_at-point, up-point).

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
    images_by_index: Mapping[int, Image],
    gtsfm_data: GtsfmData,
    cfg: Any,
    splats: dict,
    video_fpath: str,
):
    """
    Renders a video with interpolated poses from the training poses
    Args:
            images_by_index: computation graph for images keyed by index.
            gtsfm_data: computation graph for SfM output
            cfg: computation graph for the training Config parameters
            splats: computation graph for the gaussian splats
            video_fpath: location where the video will be saved
    """
    gs = GaussianSplatting(cfg, training=False)
    num_frames = cfg.num_frames
    fps = cfg.fps
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splats = {key: v.to(device) for key, v in splats.items()}
    full_dataset = GaussianSplattingData(images_by_index, gtsfm_data)
    wTi_np = generate_interpolated_path(full_dataset.wTi_tensor, num_frames, spline_degree=2)

    wTi_np = np.concatenate(
        [
            wTi_np,
            np.repeat(np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(wTi_np), axis=0),
        ],
        axis=1,
    )
    wTi_tensor = torch.from_numpy(wTi_np).float().to(device)
    K = torch.from_numpy(full_dataset.intrinsics[0]).float().to(device)
    height, width = full_dataset.actual_img_dims[0]
    frame_size = (width * 2, height)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(video_fpath, fourcc, fps, frame_size)
    for i in range(len(wTi_tensor)):
        wTc_tensor = wTi_tensor[i : i + 1]  # noqa: E203
        Ks = K[None]

        renders, _, _ = gs.rasterize_splats(
            splats=splats,
            wTi_tensor=wTc_tensor,
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
    logger.info("Interpolated video saved to %s", video_fpath)


# See https://github.com/nerfstudio-project/gsplat/blob/main/gsplat/exporter.py
def save_splats(save_path: str | Path, splats):
    """
    Export a Gaussian Splats model to bytes.

    Args:
        save_path: Output folder path.
        splats: 3D Gaussian splats defining the scene
    """
    opacities = splats["opacities"].squeeze()

    export_splats(
        means=splats["means"],
        scales=splats["scales"],
        quats=splats["quats"],
        opacities=opacities,
        sh0=splats["sh0"],
        shN=splats["shN"],
        format="ply",
        save_to=f"{save_path}/gaussian_splats.ply",
    )

    logger.info("Successfully saved Gaussian splats .ply file to %s/gaussian_splats.ply", save_path)
