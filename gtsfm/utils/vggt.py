"""Helpers to integrate the VGGT submodule with GTSFM.

This module centralizes everything needed to import Facebook's VGGT code that
ships as a Git submodule in ``thirdparty/vggt``.  It exposes small utilities to
resolve the checkpoint path, pick sane default devices/dtypes, and lazily load
the model so that other parts of GTSFM do not have to duplicate this boilerplate.
"""

from __future__ import annotations

import copy
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence, Union

import numpy as np
import pycolmap
import torch
import torch.nn.functional as F
from torch.amp import autocast as amp_autocast  # type: ignore

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
    from vggt.models.vggt import VGGT  # type: ignore
    from vggt.utils.geometry import unproject_depth_map_to_point_map  # type: ignore
    from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues  # type: ignore
    from vggt.utils.load_fn import load_and_preprocess_images_square  # type: ignore
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri  # type: ignore
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "The 'vggt' Python package could not be imported even after adding the submodule to sys.path."
    ) from exc


if TYPE_CHECKING:  # pragma: no cover - import guard for type checkers
    from vggt.dependency.track_predict import predict_tracks as _predict_tracks_type  # noqa: F401


_predict_tracks_impl: Callable[..., Any] | None = None


def _import_predict_tracks() -> Callable[..., Any]:
    """Import the tracker lazily so optional dependencies do not block other utilities."""
    global _predict_tracks_impl

    if _predict_tracks_impl is None:
        try:
            from vggt.dependency.track_predict import predict_tracks as _predict_tracks  # type: ignore
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError(
                "predict_tracks requires the optional LightGlue dependency. "
                "Ensure the LightGlue submodule (including ALIKED) is installed."
            ) from exc

        _predict_tracks_impl = _predict_tracks

    return _predict_tracks_impl


def predict_tracks(*args: Any, **kwargs: Any) -> Any:
    """Proxy for VGGSfM track prediction that loads on demand."""
    return _import_predict_tracks()(*args, **kwargs)


@dataclass(frozen=True)
class VGGTReconstructionConfig:
    """Configuration for the high-level VGGT reconstruction pipeline."""

    use_ba: bool = False
    vggt_fixed_resolution: int = 518
    img_load_resolution: int = 1024
    max_query_pts: int = 1000
    query_frame_num: int = 4
    fine_tracking: bool = True
    vis_thresh: float = 0.2
    max_reproj_error: float = 8.0
    confidence_threshold: float = 5.0
    max_points_for_colmap: int = 100000
    camera_type_ba: str = "SIMPLE_PINHOLE"
    camera_type_feedforward: str = "PINHOLE"
    shared_camera: bool = False
    use_colmap_ba: bool = False
    keypoint_extractor: str = "aliked+sp"


@dataclass
class VGGTReconstructionResult:
    """Outputs from the VGGT reconstruction helper."""

    reconstruction: pycolmap.Reconstruction
    reconstruction_resolution: int
    extrinsic: np.ndarray
    intrinsic: np.ndarray
    depth_map: np.ndarray
    depth_confidence: np.ndarray
    points_3d: np.ndarray
    points_rgb: np.ndarray | None
    points_xyf: np.ndarray | None
    used_ba: bool
    valid_track_mask: np.ndarray | None
    pred_tracks: np.ndarray | None = None
    pred_visibility: np.ndarray | None = None
    pred_confidence: np.ndarray | None = None
    fallback_reason: str | None = None


@dataclass(frozen=True)
class ImageSize:
    """Explicit representation of (height, width)."""

    height: int
    width: int

    def as_numpy(self) -> np.ndarray:
        return np.array([self.height, self.width], dtype=np.int32)


def _to_numpy_confidence(depth_conf: np.ndarray) -> np.ndarray:
    """Ensure the depth confidence map has shape (S, H, W)."""
    if depth_conf.ndim == 4 and depth_conf.shape[-1] == 1:
        return np.squeeze(depth_conf, axis=-1)
    return depth_conf


def run_vggt_reconstruction(
    image_batch: torch.Tensor,
    *,
    image_indices: Sequence[int],
    image_names: Sequence[str] | None = None,
    original_coords: torch.Tensor,
    config: VGGTReconstructionConfig | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
    model: VGGT | None = None,
    weights_path: PathLike | None = None,
) -> VGGTReconstructionResult:
    """High-level helper that runs VGGT and converts its outputs to a COLMAP reconstruction.

    Args:
        image_batch: Tensor of shape (N, 3, H, W) with pixel intensities in [0, 1] and channel order [C,H,W].
        image_indices: Identifiers corresponding to each image in the batch.
        image_names: Optional list of names to assign to the COLMAP images.
        original_coords: Tensor storing the crop/pad metadata returned by the loader.
        config: Optional configuration for the reconstruction pipeline.
        device: Device for inference. Defaults to :func:`default_vggt_device`.
        dtype: Optional dtype. Defaults to :func:`default_vggt_dtype` when running on CUDA.
        model: Optional pre-loaded VGGT model. If provided it will be used directly.
        weights_path: Optional override for the VGGT checkpoint path when ``model`` is ``None``.

    Returns:
        VGGTReconstructionResult containing the reconstructed scene information.
    """
    cfg = config or VGGTReconstructionConfig()

    if len(image_indices) != image_batch.shape[0]:
        raise ValueError("image_indices must have the same length as the batch dimension.")

    resolved_device = default_vggt_device(device)
    resolved_dtype = dtype
    if resolved_dtype is None:
        resolved_dtype = default_vggt_dtype(resolved_device) if resolved_device.type == "cuda" else torch.float32

    if model is None:
        model = load_vggt_model(weights_path, device=resolved_device)
    else:
        model = model.to(resolved_device)

    image_batch = image_batch.to(resolved_device).contiguous()
    if image_batch.ndim != 4 or image_batch.shape[1] != 3:
        raise ValueError("image_batch must have shape (N, 3, H, W); received %s" % (image_batch.shape,))
    height, width = (int(image_batch.shape[-2]), int(image_batch.shape[-1]))
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(
        model, image_batch, resolved_dtype, cfg.vggt_fixed_resolution
    )
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    depth_conf_map = _to_numpy_confidence(depth_conf)

    used_ba = cfg.use_ba
    fallback_reason: str | None = None
    valid_track_mask: np.ndarray | None = None
    pred_tracks: np.ndarray | None = None
    pred_visibility: np.ndarray | None = None
    pred_confidence: np.ndarray | None = None
    points_rgb: np.ndarray | None = None
    points_xyf: np.ndarray | None = None

    reconstruction: pycolmap.Reconstruction | None = None
    reconstruction_resolution = cfg.vggt_fixed_resolution

    if used_ba:
        scale = cfg.img_load_resolution / cfg.vggt_fixed_resolution
        try:
            autocast_ctx = (
                torch.autocast(device_type=resolved_device.type, dtype=resolved_dtype)
                if resolved_device.type == "cuda"
                else nullcontext()
            )
            with autocast_ctx:
                pred_tracks, pred_visibility, pred_confidence, points_3d, points_rgb = predict_tracks(
                    image_batch,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=cfg.max_query_pts,
                    query_frame_num=cfg.query_frame_num,
                    keypoint_extractor=cfg.keypoint_extractor,
                    fine_tracking=cfg.fine_tracking,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            intrinsic[:, :2, :] *= scale
            track_mask = pred_visibility > cfg.vis_thresh
            reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
                points_3d,
                extrinsic,
                intrinsic,
                pred_tracks,
                ImageSize(height=height, width=width),
                masks=track_mask,
                max_reproj_error=cfg.max_reproj_error,
                shared_camera=cfg.shared_camera,
                camera_type=cfg.camera_type_ba,
                points_rgb=points_rgb,
                image_id_list=list(image_indices),
            )

            if reconstruction is None:
                raise ValueError("No reconstruction can be built with BA.")

            if cfg.use_colmap_ba:
                ba_options = pycolmap.BundleAdjustmentOptions()
                pycolmap.bundle_adjustment(reconstruction, ba_options)

            reconstruction_resolution = cfg.img_load_resolution
        except ImportError as exc:
            fallback_reason = f"predict_tracks unavailable: {exc}"
            _LOGGER.warning("%s Falling back to feedforward reconstruction without BA.", fallback_reason)
            reconstruction = None
            used_ba = False

    if not used_ba:
        resized = F.interpolate(
            image_batch,
            size=(cfg.vggt_fixed_resolution, cfg.vggt_fixed_resolution),
            mode="bilinear",
            align_corners=False,
        )
        points_rgb = (resized.detach().cpu().numpy() * 255).astype(np.uint8).transpose(0, 2, 3, 1)
        num_frames, h, w, _ = points_3d.shape
        points_xyf = create_pixel_coordinate_grid(num_frames, h, w)

        conf_mask = depth_conf_map >= cfg.confidence_threshold
        conf_mask = randomly_limit_trues(conf_mask, cfg.max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            ImageSize(height=cfg.vggt_fixed_resolution, width=cfg.vggt_fixed_resolution),
            shared_camera=cfg.shared_camera,
            camera_type=cfg.camera_type_feedforward,
            image_id_list=list(image_indices),
        )
        reconstruction_resolution = cfg.vggt_fixed_resolution

    if reconstruction is None:
        raise ValueError("VGGT reconstruction failed to produce a valid COLMAP reconstruction.")

    if isinstance(original_coords, torch.Tensor):
        original_coords_np = original_coords.detach().cpu().numpy()
    else:
        original_coords_np = np.asarray(original_coords)

    rename_source = list(image_names) if image_names is not None else None

    rename_colmap_recons_and_rescale_camera(
        reconstruction,
        rename_source,
        original_coords_np,
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=cfg.shared_camera,
        image_id_list=list(image_indices),
    )

    return VGGTReconstructionResult(
        reconstruction=reconstruction,
        reconstruction_resolution=reconstruction_resolution,
        extrinsic=extrinsic,
        intrinsic=intrinsic,
        depth_map=depth_map,
        depth_confidence=depth_conf_map,
        points_3d=points_3d,
        points_rgb=points_rgb,
        points_xyf=points_xyf,
        used_ba=used_ba,
        valid_track_mask=valid_track_mask,
        pred_tracks=pred_tracks,
        pred_visibility=pred_visibility,
        pred_confidence=pred_confidence,
        fallback_reason=fallback_reason,
    )


def resolve_vggt_weights_path(weights_path: PathLike | None = None) -> Path:
    """Return a concrete path to the VGGT checkpoint, validating that it exists."""
    path = Path(weights_path) if weights_path is not None else DEFAULT_WEIGHTS_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"VGGT checkpoint not found at {path}. Please run 'scripts/download_model_weights.sh' from the repo root."
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
    weights_path: PathLike | None = None,
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

    weights_path = resolve_vggt_weights_path(weights_path)
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
        if images.device.type == "cuda":
            autocast_ctx = amp_autocast("cuda", dtype=dtype) if dtype is not None else amp_autocast("cuda")
        else:
            autocast_ctx = nullcontext()
        with autocast_ctx:  # type: ignore[arg-type]
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def _build_pycolmap_intrinsics(frame_index, intrinsics, camera_type, extra_params=None):
    """
    Helper function to get camera parameters based on camera type.

    Args:
        frame_index: Frame index
        intrinsics: Camera intrinsic parameters
        camera_type: Type of camera model
        extra_params: Additional parameters for certain camera types

    Returns:
        pycolmap_intrinsics: NumPy array of camera parameters
    """
    if camera_type == "PINHOLE":
        pycolmap_intrinsics = np.array(
            [
                intrinsics[frame_index][0, 0],
                intrinsics[frame_index][1, 1],
                intrinsics[frame_index][0, 2],
                intrinsics[frame_index][1, 2],
            ]
        )
    elif camera_type == "SIMPLE_PINHOLE":
        focal = (intrinsics[frame_index][0, 0] + intrinsics[frame_index][1, 1]) / 2
        pycolmap_intrinsics = np.array([focal, intrinsics[frame_index][0, 2], intrinsics[frame_index][1, 2]])
    elif camera_type == "SIMPLE_RADIAL":
        raise NotImplementedError("SIMPLE_RADIAL is not supported yet")
        focal = (intrinsics[frame_index][0, 0] + intrinsics[frame_index][1, 1]) / 2
        pycolmap_intrinsics = np.array(
            [focal, intrinsics[frame_index][0, 2], intrinsics[frame_index][1, 2], extra_params[frame_index][0]]
        )
    else:
        raise ValueError(f"Camera type {camera_type} is not supported yet")

    return pycolmap_intrinsics


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size: ImageSize,
    image_id_list,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    Convert batched NumPy arrays (with track information) to a PyCOLMAP reconstruction.

    Args mirror the VGGSfM helper but enforce an explicit `ImageSize` instead of a raw tuple.
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: ImageSize, assume all the frames have been padded to the same size
    # image_id_list: global image id list if any

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    height, width = int(image_size.height), int(image_size.width)

    reproj_mask = None
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
        _LOGGER.warning("Not enough inliers per frame, skip BA.")
        return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()  # type: ignore

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    for i in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[i] if points_rgb is not None else np.zeros(3)
        reconstruction.add_point3D(points3d[i], pycolmap.Track(), rgb)  # type: ignore

    num_points3D = len(valid_idx)

    camera = None
    # frame idx
    for frame_index in range(N):

        image_idx = image_id_list[frame_index]

        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intrinsics = _build_pycolmap_intrinsics(frame_index, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(  # type: ignore
                model=camera_type,
                width=width,
                height=height,
                params=pycolmap_intrinsics,
                camera_id=image_idx,
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(  # type: ignore
            pycolmap.Rotation3d(extrinsics[frame_index][:3, :3]), extrinsics[frame_index][:3, 3]  # type: ignore
        )  # Rot and Trans

        image = pycolmap.Image(  # type: ignore
            id=image_idx, name=f"image_{image_idx}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[frame_index][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[frame_index][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))  # type: ignore

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(image_idx, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)  # type: ignore
            # image.registered = True
        except Exception as e:
            _LOGGER.warning("Frame %s is out of BA: %s", image_idx, e)
            # image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction, valid_mask


def batch_np_matrix_to_pycolmap_wo_track(
    points3d,  # Px3
    points_xyf,  # Px3, with x, y coordinates and frame indices
    points_rgb,  # Px3, rgb colors
    extrinsics,  # Nx3x4
    intrinsics,  # Nx3x3
    image_size: ImageSize,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    image_id_list=None,  # global image id list if any
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP
    where N is the number of frames and P is the number of tracks.

    Different from batch_np_matrix_to_pycolmap, this function does not use tracks.

    Assumes all the frames have been padded to the same size.
    It saves points3d to colmap reconstruction format only to serve as init for Gaussians or other nvs methods.

    Do NOT use this for BA.
    """

    N = len(extrinsics)
    P = len(points3d)

    assert image_id_list is not None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    for i in range(P):
        reconstruction.add_point3D(points3d[i], pycolmap.Track(), points_rgb[i])

    camera = None
    for frame_index in range(N):
        image_idx = image_id_list[frame_index]

        # set camera
        if camera is None or (not shared_camera):
            pycolmap_intrinsics = _build_pycolmap_intrinsics(frame_index, intrinsics, camera_type)

            camera = pycolmap.Camera(
                model=camera_type,
                width=image_size.width,
                height=image_size.height,
                params=pycolmap_intrinsics,
                camera_id=image_idx,
            )
            reconstruction.add_camera(camera)

        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[frame_index][:3, :3]), extrinsics[frame_index][:3, 3]
        )
        image = pycolmap.Image(
            id=image_idx, name=f"image_{image_idx}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []
        point2D_idx = 0

        points_belong_to_frame = points_xyf[:, 2].astype(np.int32) == frame_index
        points_belong_to_frame = np.nonzero(points_belong_to_frame)[0]

        for point3D_batch_idx in points_belong_to_frame:
            point3D_id = point3D_batch_idx + 1
            point2D_xy = points_xyf[point3D_batch_idx][:2]
            points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

            track = reconstruction.points3D[point3D_id].track
            track.add_element(image_idx, point2D_idx)
            point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)  # type: ignore
        except Exception as e:
            _LOGGER.warning("Frame %s does not have any points: %s", image_idx, e)

        reconstruction.add_image(image)

    return reconstruction


def rename_colmap_recons_and_rescale_camera(
    reconstruction,
    image_paths=None,
    original_coords=None,
    img_size=None,
    shift_point2d_to_original_res=False,
    shared_camera=False,
    image_id_list=None,
):
    if original_coords is None or img_size is None or image_id_list is None:
        raise ValueError("original_coords, img_size, and image_id_list must be provided.")

    rescale_camera = True

    # for py_image_id in reconstruction.images:
    for local_id in range(len(image_id_list)):
        py_image_id = image_id_list[local_id]
        # Reshaped the padded&resized image to the original size
        # Rename the images to the original names
        py_image = reconstruction.images[py_image_id]
        py_camera = reconstruction.cameras[py_image.camera_id]
        if image_paths is not None:
            py_image.name = str(image_paths[local_id])

        resize_ratio = 1.0
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

    _LOGGER.debug("Reconstruction images: %s", reconstruction.images)

    return reconstruction


__all__ = [
    "VGGT_SUBMODULE_PATH",
    "DEFAULT_WEIGHTS_PATH",
    "ImageSize",
    "VGGTReconstructionConfig",
    "VGGTReconstructionResult",
    "resolve_vggt_weights_path",
    "default_vggt_device",
    "default_vggt_dtype",
    "load_vggt_model",
    "run_vggt_reconstruction",
    "batch_np_matrix_to_pycolmap",
    "predict_tracks",
    "unproject_depth_map_to_point_map",
    "create_pixel_coordinate_grid",
    "randomly_limit_trues",
    "load_and_preprocess_images_square",
    "pose_encoding_to_extri_intri",
    "project_3D_points_np",
    "run_VGGT",
    "batch_np_matrix_to_pycolmap_wo_track",
]
