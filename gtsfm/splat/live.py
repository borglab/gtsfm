"""Low-overhead file bridge from Dask Gaussian workers to the browser workspace."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Mapping

import torch

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    finally:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass


def publish_training_update(
    splats: Mapping[str, torch.Tensor],
    step: int,
    max_steps: int,
    loss: float,
    *,
    force_preview: bool = False,
) -> None:
    """Publish progress and, periodically, an atomic PLY snapshot.

    The bridge is disabled unless the workspace sets ``GTSFM_LIVE_DIR``. Export
    errors are deliberately non-fatal so observability can never stop training.
    """

    live_dir_value = os.environ.get("GTSFM_LIVE_DIR")
    if not live_dir_value:
        return
    live_dir = Path(live_dir_value)
    completed = min(step + 1, max_steps)
    interval = max(1, int(os.environ.get("GTSFM_LIVE_PREVIEW_INTERVAL", "250")))
    max_preview_splats = max(1, int(os.environ.get("GTSFM_LIVE_PREVIEW_MAX_SPLATS", "50000")))
    should_preview = force_preview or completed == max_steps or completed == 1 or completed % interval == 0
    preview_path = live_dir / "live_splats.ply"
    preview_written = preview_path.exists()

    if should_preview:
        try:
            from gsplat import export_splats

            live_dir.mkdir(parents=True, exist_ok=True)
            temp_path = live_dir / "live_splats.tmp.ply"
            splat_count = int(splats["means"].shape[0])
            if splat_count > max_preview_splats:
                # A full training snapshot can grow beyond 200 MB, which is too
                # slow to commit, transfer, parse, and replace interactively.
                # Keep a deterministic, scene-wide sample for the live viewer;
                # the final export remains complete and downloadable.
                indices = torch.linspace(
                    0,
                    splat_count - 1,
                    steps=max_preview_splats,
                    device=splats["means"].device,
                ).long()
                preview_splats = {name: value.index_select(0, indices) for name, value in splats.items()}
            else:
                preview_splats = splats
            export_splats(
                means=preview_splats["means"].detach().cpu(),
                scales=preview_splats["scales"].detach().cpu(),
                quats=preview_splats["quats"].detach().cpu(),
                opacities=preview_splats["opacities"].detach().cpu().squeeze(),
                sh0=preview_splats["sh0"].detach().cpu(),
                shN=preview_splats["shN"].detach().cpu(),
                format="ply",
                save_to=str(temp_path),
            )
            os.replace(temp_path, preview_path)
            preview_written = True
            logger.info(
                "Published live Gaussian preview at step %d (%d of %d splats)",
                completed,
                min(splat_count, max_preview_splats),
                splat_count,
            )
        except Exception as exc:  # pragma: no cover - depends on CUDA/gsplat runtime
            logger.warning("Unable to export live Gaussian preview: %s", exc)

    _atomic_json(
        live_dir / "status.json",
        {
            "stage": "gaussian_splatting",
            "step": completed,
            "max_steps": max_steps,
            "progress": completed / max_steps if max_steps else 1.0,
            "loss": loss,
            "splat_count": int(splats["means"].shape[0]),
            "preview_splat_count": min(int(splats["means"].shape[0]), max_preview_splats),
            "preview_written": preview_written,
            "updated_at": time.time(),
        },
    )
