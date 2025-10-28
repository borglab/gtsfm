"""Helpers to integrate the VGGT submodule with GTSFM.

This module centralizes everything needed to import Facebook's VGGT code that
ships as a Git submodule in ``thirdparty/vggt``.  It exposes small utilities to
resolve the checkpoint path, pick sane default devices/dtypes, and lazily load
the model so that other parts of GTSFM do not have to duplicate this boilerplate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Union
import copy

import torch
import torch.nn.functional as F

import numpy as np
import pycolmap
import trimesh
from gtsfm.utils import logger as logger_utils

PathLike = Union[str, Path]

_LOGGER = logger_utils.get_logger()

REPO_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
VGGT_SUBMODULE_PATH = THIRDPARTY_ROOT / "vggt"
LIGHTGLUE_SUBMODULE_PATH = THIRDPARTY_ROOT / "LightGlue"
DEFAULT_WEIGHTS_PATH = VGGT_SUBMODULE_PATH / "weights" / "model.pt"


def _ensure_submodule_on_path(path: Path, name: str) -> None:
    """Add a vendored thirdparty module to ``sys.path`` if needed."""
    if not path.exists():
        raise ImportError(
            f"Required submodule '{name}' not found at {path}. "
            "Did you run 'git submodule update --init --recursive'?"
        )

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_submodule_on_path(VGGT_SUBMODULE_PATH, "vggt")
_ensure_submodule_on_path(LIGHTGLUE_SUBMODULE_PATH, "LightGlue")

try:
    from vggt.dependency.projection import project_3D_points_np  # type: ignore
    from vggt.dependency.track_predict import predict_tracks  # type: ignore
    from vggt.models.vggt import VGGT  # type: ignore
    from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues  # type: ignore
    from vggt.utils.load_fn import load_and_preprocess_images_square  # type: ignore
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'vggt' Python package could not be imported even after adding the submodule to sys.path."
    ) from exc


def resolve_vggt_weights_path(checkpoint_path: PathLike | None = None) -> Path:
    """Return a concrete path to the VGGT checkpoint, validating that it exists."""
    path = Path(checkpoint_path) if checkpoint_path is not None else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"VGGT checkpoint not found at {path}. " "Please run 'bash download_model_weights.sh' from the repo root."
        )
    return path


def default_vggt_device(device: torch.device | str | None = None) -> torch.device:
    """Pick a reasonable default device for VGGT inference."""
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def default_vggt_dtype(device: torch.device | None = None) -> torch.dtype:
    """Select a dtype that matches the provided device (VGGT prefers fp16/bfloat16 on Ada+ GPUs)."""
    if device is None:
        device = default_vggt_device()

    if device.type == "cuda" and torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability(device)
        return torch.bfloat16 if major >= 8 else torch.float16
    return torch.float32


def load_vggt_model(
    checkpoint_path: PathLike | None = None,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    strict: bool = True,
) -> VGGT:
    """Instantiate VGGT, load weights, move it to the requested device/dtype, and return it."""
    resolved_device = default_vggt_device(device)
    resolved_dtype = dtype
    if resolved_dtype is None and resolved_device.type == "cuda":
        resolved_dtype = default_vggt_dtype(resolved_device)

    weights_path = resolve_vggt_weights_path(checkpoint_path)
    _LOGGER.info("⏳ Loading VGGT checkpoint from %s", weights_path)

    model = VGGT()
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=strict)
    model.eval()
    model = model.to(resolved_device)
    
    # if resolved_dtype is not None:
    #     # dtype casting is only attempted when explicitly requested or inferred for CUDA devices.
    #     model = model.to(dtype=resolved_dtype)

    return model

def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)


    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    model.to("cpu")
    torch.cuda.empty_cache()

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf

def _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        fidx: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intri: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intri = np.array(
            [intrinsics[fidx][0, 0], intrinsics[fidx][1, 1], intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[fidx][0, 0] + intrinsics[fidx][1, 1]) / 2
        pycolmap_intri = np.array([focal, intrinsics[fidx][0, 2], intrinsics[fidx][1, 2], extra_params[fidx][0]])
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intri

def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
    image_id_list=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks
    # image_id_list: global image id list if any

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    if masks.sum(1).min() < min_inlier_per_frame:
        print(f"Not enough inliers per frame, skip BA.")
        return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    for vidx in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)

    num_points3D = len(valid_idx)
    camera = None
    # frame idx
    for fidx in range(N):

        image_idx = image_id_list[fidx]

        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=image_idx
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=image_idx, name=f"image_{image_idx}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(image_idx, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            # image.registered = True
        except:
            print(f"frame {image_idx} is out of BA")
            # image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction, valid_mask

def batch_np_matrix_to_pycolmap_wo_track(
    points3d,
    points_xyf,
    points_rgb,
    extrinsics,
    intrinsics,
    image_size,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    image_id_list=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """
    # points3d: Px3
    # points_xyf: Px3, with x, y coordinates and frame indices
    # points_rgb: Px3, rgb colors
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks
    # image_id_list: global image id list if any

    N = len(extrinsics)
    P = len(points3d)

    assert image_id_list is not None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for vidx in range(P):
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), points_rgb[vidx])

    camera = None
    # frame idx
    for fidx in range(N):

        image_idx = image_id_list[fidx]

        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=image_idx
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=image_idx, name=f"image_{image_idx}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        points_belong_to_fidx = points_xyf[:, 2].astype(np.int32) == fidx
        points_belong_to_fidx = np.nonzero(points_belong_to_fidx)[0]

        for point3D_batch_idx in points_belong_to_fidx:
            point3D_id = point3D_batch_idx + 1
            point2D_xyf = points_xyf[point3D_batch_idx]
            point2D_xy = point2D_xyf[:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            # add element
            track = reconstruction.points3D[point3D_id].track
            track.add_element(image_idx, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            # image.registered = True
        except:
            print(f"frame {image_idx} does not have any points")
            # image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction

def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths,
    original_coords,
    img_size,
    shift_point2d_to_original_res=False,
    shared_camera=False,
    image_id_list=None,
):
    rescale_camera = True

    assert image_id_list is not None

    # for py_image_id in reconstruction.images:
    for local_id in range(len(image_id_list)):
        py_image_id = image_id_list[local_id]
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        py_image = reconstruction.images[py_image_id]
        py_camera = reconstruction.cameras[py_image.camera_id]
        # py_image.name = image_paths[local_id]

        if rescale_camera:
            # Rescale the camera parameters
            pred_params = copy.deepcopy(py_camera.params)

            real_image_size = original_coords[local_id, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp  # center of the image

            py_camera.params = pred_params
            py_camera.width = real_image_size[0]
            py_camera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            # Also shift the point2D to original resolution
            top_left = original_coords[local_id, :2]

            for point2D in py_image.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            # If shared_camera, all images share the same camera
            # no need to rescale any more
            rescale_camera = False

    print("reconstruction.images: ", reconstruction.images)

    return reconstruction

# __all__ = [
#     "VGGT_SUBMODULE_PATH",
#     "DEFAULT_WEIGHTS_PATH",
#     "resolve_vggt_weights_path",
#     "default_vggt_device",
#     "default_vggt_dtype",
#     "load_vggt_model",
#     "batch_np_matrix_to_pycolmap",
#     "predict_tracks",
#     "unproject_depth_map_to_point_map",
#     "create_pixel_coordinate_grid",
#     "randomly_limit_trues",
#     "load_and_preprocess_images_square",
#     "pose_encoding_to_extri_intri",
#     "project_3D_points_np",
#     "run_VGGT",
#     "batch_np_matrix_to_pycolmap_wo_track"
# ]
