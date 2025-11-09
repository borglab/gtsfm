"""Helpers to integrate the AnySplat submodule with GTSFM."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Protocol

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch

import gtsfm.utils.torch as torch_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils import logger as logger_utils
from gtsfm.utils.splat import GaussiansProtocol

_SH0_NORMALIZATION_FACTOR = 0.28209479177387814


class DecoderSplattingCUDAProtocol(Protocol):
    """Type protocol for AnySplat DecoderSplattingCUDA to satisfy mypy."""

    def cpu(self) -> DecoderSplattingCUDAProtocol:
        """Move the decoder to CPU."""
        ...

    def to(self, device: torch.device | str) -> DecoderSplattingCUDAProtocol:
        """Move the decoder to the specified device."""
        ...


logger = logger_utils.get_logger()


REPO_ROOT = Path(__file__).resolve().parents[2]
THIRDPARTY_ROOT = REPO_ROOT / "thirdparty"
VGGT_SUBMODULE_PATH = THIRDPARTY_ROOT / "vggt"
ANYSPLAT_SUBMODULE_PATH = THIRDPARTY_ROOT / "AnySplat"
LIGHTGLUE_SUBMODULE_PATH = THIRDPARTY_ROOT / "LightGlue"


def _ensure_submodule_on_path(path: Path, name: str) -> None:
    """Add a vendored thirdparty module to ``sys.path`` if needed."""
    if not path.exists():
        raise ImportError(
            f"Required submodule '{name}' not found at {path}. " "Run 'git submodule update --init --recursive'?"
        )

    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


_ensure_submodule_on_path(VGGT_SUBMODULE_PATH, "vggt")
_ensure_submodule_on_path(ANYSPLAT_SUBMODULE_PATH, "anysplat")
_ensure_submodule_on_path(LIGHTGLUE_SUBMODULE_PATH, "LightGlue")

_SAVE_INTERPOLATED_VIDEO: Callable[..., Any] | None = None
_EXPORT_PLY: Callable[..., Any] | None = None
AnySplat: type[Any] | None = None  # type: ignore[assignment]
DecoderSplattingCUDA: type[DecoderSplattingCUDAProtocol] | Any = Any  # type: ignore[assignment]
Gaussians: type[GaussiansProtocol] | Any = Any  # type: ignore[assignment]
_IMPORT_ERROR: Exception | None = None
batchify_unproject_depth_map_to_point_map: Callable[..., torch.Tensor] | None = None

try:  # pragma: no cover - exercised by integration tests, hard to simulate in unit tests.
    from src.misc.image_io import save_interpolated_video as _save_interpolated_video_impl  # type: ignore
    from src.model.decoder.decoder_splatting_cuda import (
        DecoderSplattingCUDA as _DecoderSplattingCUDAImpl,
    )  # type: ignore
    from src.model.encoder.vggt.utils.geometry import (
        batchify_unproject_depth_map_to_point_map as _batchify_unproject_impl,
    )  # type: ignore
    from src.model.model.anysplat import AnySplat as _AnySplatImpl  # type: ignore
    from src.model.ply_export import export_ply as _export_ply_impl  # type: ignore
    from src.model.types import Gaussians as _GaussiansImpl  # type: ignore
except (ModuleNotFoundError, OSError) as exc:  # pragma: no cover - import guard
    _IMPORT_ERROR = exc
else:
    _SAVE_INTERPOLATED_VIDEO = _save_interpolated_video_impl
    _EXPORT_PLY = _export_ply_impl
    AnySplat = _AnySplatImpl  # type: ignore[assignment]
    DecoderSplattingCUDA = _DecoderSplattingCUDAImpl  # type: ignore[assignment]
    Gaussians = _GaussiansImpl  # type: ignore[assignment]
    batchify_unproject_depth_map_to_point_map = _batchify_unproject_impl


def import_predict_tracks():
    """Return the vendored ``predict_tracks`` helper from the VGGT submodule.
    The tracker lives in ``thirdparty/vggt``. We keep this import behind a small helper so that runtime
    errors surface with a clear explanation when the submodule is missing.
    """

    try:
        from vggt.dependency.track_predict import predict_tracks as _predict_tracks  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only when the submodule is absent
        raise ImportError(
            "Could not import VGGT tracker utilities. Ensure the 'vggt' submodule is checked out by "
            "running `git submodule update --init --recursive`."
        ) from exc
    return _predict_tracks


def _require_thirdparty() -> None:
    """Ensure the AnySplat thirdparty dependencies imported successfully."""

    if _IMPORT_ERROR is not None:
        raise ImportError(
            "The 'anysplat' Python package could not be imported even after adding the submodule to sys.path."
        ) from _IMPORT_ERROR


def save_interpolated_video(*args: Any, **kwargs: Any) -> Any:
    """Proxy to AnySplat's video export helper, ensuring dependencies are present."""

    _require_thirdparty()
    assert _SAVE_INTERPOLATED_VIDEO is not None
    return _SAVE_INTERPOLATED_VIDEO(*args, **kwargs)


def export_ply(*args: Any, **kwargs: Any) -> Any:
    """Proxy to AnySplat's PLY export helper, ensuring dependencies are present."""

    _require_thirdparty()
    assert _EXPORT_PLY is not None
    return _EXPORT_PLY(*args, **kwargs)


@dataclass
class AnySplatReconstructionResult:
    """Outputs from the Anysplat generate splats function."""

    gtsfm_data: GtsfmData
    splats: GaussiansProtocol
    pred_context_pose: dict[str, torch.Tensor]
    height: int
    width: int
    decoder: DecoderSplattingCUDAProtocol


def load_model(
    *,
    device: torch.device | str | None = None,
    checkpoint_path: str | Path | None = None,
) -> Any:
    """Load AnySplat weights optionally moving the model to the requested device."""

    _require_thirdparty()
    assert AnySplat is not None

    target = str(checkpoint_path) if checkpoint_path is not None else "lhjiang/anysplat"

    model = AnySplat.from_pretrained(target)
    model.eval()

    if device is not None:
        resolved_device = torch.device(device)
        model = model.to(resolved_device)

    logger.info("✅ VGGT model weights loaded successfully.")
    return model


def add_tracks_with_gaussian_mean(splats, max_gaussians, gtsfm_data):

    logger.info("Adding Gaussian means to GtsfmData as 3D tracks.")
    splats_means = splats.means[0].cpu().numpy()  # type: ignore
    dc_color = splats.harmonics[..., 0][0]  # type: ignore

    if max_gaussians is not None and splats.opacities.shape[1] > max_gaussians:
        logger.info(f"Filtering Gaussians to {max_gaussians}")

        op = splats.opacities[0]
        K = min(max_gaussians, op.numel())

        topk = torch.topk(op, k=K, largest=True, sorted=True)
        idx = topk.indices.to(splats.means.device).long()

        splats_means = splats.means[:, idx, :][0].cpu().numpy()

        dc_color = splats.harmonics[:, idx, :, :][..., 0][0]

    colors_tensor = (dc_color * _SH0_NORMALIZATION_FACTOR + 0.5).clamp(0.0, 1.0)
    colors_np = (colors_tensor * 255).cpu().numpy()

    if splats_means.size > 0:
        for j, xyz in enumerate(splats_means):
            color = colors_np[j]
            track = torch_utils.colored_track_from_point(xyz, color)
            gtsfm_data.add_track(track)

    logger.info(f"Added {len(splats_means)} tracks from Gaussian means.")

    return gtsfm_data


def visualize_tracks(processed_images, tracks, vis_scores, out_dir, vis_thresh=0.5, max_tracks=80):
    """
    Save per-frame overlays

    Args:
        processed_images: torch.Tensor of shape (1, V, 3, 448, 448) in [-1, 1].
        tracks: np.ndarray of shape (V, N, 2) with pixel coordinates (0..447).
        vis_scores: np.ndarray of shape (V, N) with visibility probabilities.
        out_dir: output directory.
        vis_thresh: minimum visibility score to plot a track sample.
    """
    assert processed_images.ndim == 5 and processed_images.shape[0] == 1, "Unexpected image stack shape."
    processed_images = processed_images.squeeze(0).cpu()  # shape (V, 3, 448, 448)

    V, _, H, W = processed_images.shape
    assert tracks.shape[:1] == (V,), "tracks must have shape (V, N, 2)."

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert images back to [0,1] float for plotting
    image_stack = (processed_images.permute(0, 2, 3, 1).numpy() * 0.5 + 0.5).clip(0.0, 1.0)

    # Choose a subset of tracks to visualize (consistent across frames).
    # We pick the ones with the highest total visibility.
    visibility_scores = vis_scores.sum(axis=0)
    tracked_indices = np.argsort(visibility_scores)[-max_tracks:]

    cmaps = [mpl.cm.get_cmap(n, 20) for n in ("tab20", "tab20b", "tab20c")]
    palette = [mpl.colors.to_hex(c(i)) for c in cmaps for i in range(c.N)]
    colors = [palette[int(idx) % len(palette)] for idx in tracked_indices]

    for frame_idx in range(V):
        img = image_stack[frame_idx]
        uv = tracks[frame_idx, tracked_indices]
        vis = vis_scores[frame_idx, tracked_indices]

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img)
        for track_idx, (uv_track, vis_track, color) in enumerate(zip(uv, vis, colors)):
            ax.scatter(
                uv_track[0],
                uv_track[1],
                s=12,
                c=color,
                edgecolors="black",
                linewidths=0.4,
                alpha=0.9,
                label=f"Track {tracked_indices[track_idx]}",
            )

        ax.set_title(f"Frame {frame_idx}")
        ax.axis("off")

        frame_path = out_dir / f"frame_{frame_idx:03d}.png"
        fig.savefig(frame_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
