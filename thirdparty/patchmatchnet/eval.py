"""PatchmatchNet evaluation methods (truncated)
    reference: https://github.com/FangjinhuaWang/PatchmatchNet

"""
from typing import Tuple

import cv2
import numpy as np


def reproject_with_depth(
    depth_ref: np.ndarray,
    intrinsics_ref: np.ndarray,
    extrinsics_ref: np.ndarray,
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    extrinsics_src: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Project the reference points to the source view, then project back to calculate the reprojection error

    Args:
        depth_ref (np.ndarray): depths of points in the reference view, (H, W)
        intrinsics_ref (np.ndarray): camera intrinsic of the reference view, (3, 3)
        extrinsics_ref (np.ndarray): camera extrinsic of the reference view, (4, 4)
        depth_src (np.ndarray): depths of points in the source view, (H, W)
        intrinsics_src (np.ndarray): camera intrinsic of the source view, (3, 3)
        extrinsics_src (np.ndarray): camera extrinsic of the source view, (4, 4)

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            depth_reprojected: reprojected depths of points in the reference view, (H, W)
            x_reprojected: reprojected x coordinates of points in the reference view, (H, W)
            y_reprojected: reprojected y coordinates of points in the reference view, (H, W)
            x_src: x coordinates of points in the source view, (H, W)
            y_src: y coordinates of points in the source view, (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    # step1. project reference pixels to the source view
    # reference view x, y
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])
    # reference 3D space
    xyz_ref = np.matmul(
        np.linalg.inv(intrinsics_ref), np.vstack((x_ref, y_ref, np.ones_like(x_ref))) * depth_ref.reshape([-1])
    )
    # source 3D space
    xyz_src = np.matmul(
        np.matmul(extrinsics_src, np.linalg.inv(extrinsics_ref)), np.vstack((xyz_ref, np.ones_like(x_ref)))
    )[:3]
    # source view x, y
    K_xyz_src = np.matmul(intrinsics_src, xyz_src)
    xy_src = K_xyz_src[:2] / K_xyz_src[2:3]

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[0].reshape([height, width]).astype(np.float32)
    y_src = xy_src[1].reshape([height, width]).astype(np.float32)
    sampled_depth_src = cv2.remap(depth_src, x_src, y_src, interpolation=cv2.INTER_LINEAR)

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = np.matmul(
        np.linalg.inv(intrinsics_src), np.vstack((xy_src, np.ones_like(x_ref))) * sampled_depth_src.reshape([-1])
    )
    # reference 3D space
    xyz_reprojected = np.matmul(
        np.matmul(extrinsics_ref, np.linalg.inv(extrinsics_src)), np.vstack((xyz_src, np.ones_like(x_ref)))
    )[:3]
    # source view x, y, depth
    depth_reprojected = xyz_reprojected[2].reshape([height, width]).astype(np.float32)
    K_xyz_reprojected = np.matmul(intrinsics_ref, xyz_reprojected)
    xy_reprojected = K_xyz_reprojected[:2] / K_xyz_reprojected[2:3]
    x_reprojected = xy_reprojected[0].reshape([height, width]).astype(np.float32)
    y_reprojected = xy_reprojected[1].reshape([height, width]).astype(np.float32)

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src


def check_geometric_consistency(
    depth_ref: np.ndarray,
    intrinsics_ref: np.ndarray,
    extrinsics_ref: np.ndarray,
    depth_src: np.ndarray,
    intrinsics_src: np.ndarray,
    extrinsics_src: np.ndarray,
    geo_pixel_thres: float,
    geo_depth_thres: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Check geometric consistency and return valid points

    Args:
        depth_ref (np.ndarray): depths of points in the reference view, (H, W)
        intrinsics_ref (np.ndarray): camera intrinsic of the reference view, (3, 3)
        extrinsics_ref (np.ndarray): camera extrinsic of the reference view, (4, 4)
        depth_src (np.ndarray): depths of points in the source view, (H, W)
        intrinsics_src (np.ndarray): camera intrinsic of the source view, (3, 3)
        extrinsics_src (np.ndarray): camera extrinsic of the source view, (4, 4)
        geo_pixel_thres (float): geometric pixel threshold
        geo_depth_thres (float): geometric depth threshold

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            mask: mask for points with geometric consistency, (H, W)
            depth_reprojected: reprojected depths of points in the reference view, (H, W)
            x2d_src: x coordinates of points in the source view, (H, W)
            y2d_src: y coordinates of points in the source view, (H, W)
    """
    width, height = depth_ref.shape[1], depth_ref.shape[0]
    x_ref, y_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth(
        depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src, extrinsics_src
    )

    # check |p_reproj-p_1| < 1
    dist = np.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)

    # check |d_reproj-d_1| / d_1 < 0.01
    # depth_ref = np.squeeze(depth_ref, 2)
    depth_diff = np.abs(depth_reprojected - depth_ref)
    relative_depth_diff = depth_diff / depth_ref

    mask = np.logical_and(dist < geo_pixel_thres, relative_depth_diff < geo_depth_thres)
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src
