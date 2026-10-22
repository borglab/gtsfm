"""Modal deployment definition for the GTSFM GPU workspace.

This file is evaluated by ``modal deploy`` on the user's machine. It packages
the current checkout and exposes the same FastAPI workspace used locally.

Modal also imports this module again inside the container to hydrate the
function. Checkout/token checks and local ``add_local_dir`` mounts are only
valid at deploy time, so they must not run in that remote import path.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


def _in_modal_container() -> bool:
    """True when Modal is importing this file to start a remote function."""

    return os.environ.get("MODAL_IS_REMOTE") == "1" or not modal.is_local()


def _is_public_workspace_read(request: object) -> bool:
    """Allow the browser viewer to load catalogs and reconstruction files."""

    method = str(getattr(request, "method", "")).upper()
    path = str(getattr(getattr(request, "url", None), "path", "") or "")
    if method not in {"GET", "HEAD"}:
        return False
    if path.startswith("/data/"):
        return True
    return path in {
        "/api/configuration",
        "/api/hardware",
        "/api/samples",
        "/api/scenes",
        "/api/setup",
    }


def _build_workspace_image() -> modal.Image:
    """Package the local checkout into the CUDA workspace image."""

    source_root = Path(os.environ.get("GTSFM_SOURCE_ROOT", Path.cwd())).expanduser().resolve()
    gpu_cpu = float(os.environ.get("GTSFM_MODAL_CPU", "8"))
    gpu_memory_mb = int(os.environ.get("GTSFM_MODAL_MEMORY_MB", "65536"))
    remote_api_key = os.environ.get("GTSFM_REMOTE_API_KEY", "")
    if not (source_root / "pyproject.toml").is_file():
        raise RuntimeError("GTSFM_SOURCE_ROOT must point to a GTSFM source checkout")
    if not remote_api_key:
        raise RuntimeError("GTSFM_REMOTE_API_KEY is required for a protected remote workspace")
    if not 1 <= gpu_cpu <= 32:
        raise RuntimeError("GTSFM_MODAL_CPU must be between 1 and 32")
    if not 4096 <= gpu_memory_mb <= 524288:
        raise RuntimeError("GTSFM_MODAL_MEMORY_MB must be between 4096 and 524288")

    common_excludes = [
        "**/.git/**",
        "**/__pycache__/**",
        "**/*.pyc",
        "**/.DS_Store",
        "**/node_modules/**",
        "**/*.ipynb",
    ]
    # Keep gtsfm/assets (sensor database). Skip bulky third-party demo/asset trees.
    thirdparty_excludes = [
        *common_excludes,
        "**/assets/**",
        "**/demo/**",
        "**/demos/**",
        "**/examples/**",
    ]
    runtime_env = {
        # Modal's Python layer can leave CC/CXX pointing at clang even though the
        # The source-build image uses an explicit compiler selection for native dependencies.
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
        "HF_HOME": "/workspace/cache/huggingface",
        "TORCH_HOME": "/workspace/cache/torch",
        "TORCH_EXTENSIONS_DIR": "/workspace/cache/torch-extensions",
        "XDG_CACHE_HOME": "/workspace/cache",
        "PYTHONPATH": "/root",
    }
    image = (
        modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .apt_install(
            "build-essential",
            "git",
            "graphviz",
            "libegl1",
            "libgl1",
            "libglib2.0-0",
            "libgomp1",
            "libx11-6",
            "ninja-build",
        )
        .env(runtime_env)
        .uv_sync(str(source_root), groups=[], frozen=True, extra_options="--no-default-groups")
    )
    return (
        image.add_local_dir(source_root / "gtsfm", "/root/gtsfm", ignore=common_excludes)
        .add_local_dir(source_root / "visualization", "/root/visualization", ignore=common_excludes)
        .add_local_dir(source_root / "thirdparty", "/root/thirdparty", ignore=thirdparty_excludes)
    )


GPU = os.environ.get("GTSFM_MODAL_GPU", "L40S")
MODAL_CPU = float(os.environ.get("GTSFM_MODAL_CPU", "8"))
MODAL_MEMORY_MB = int(os.environ.get("GTSFM_MODAL_MEMORY_MB", "65536"))
REMOTE_API_KEY = os.environ.get("GTSFM_REMOTE_API_KEY", "")
# The baked image is used at deploy time. Container hydration still evaluates
# this module, but the function already has its image; a stub is enough there.
image = (
    modal.Image.debian_slim(python_version="3.12")
    if _in_modal_container()
    else _build_workspace_image()
)

app = modal.App("gtsfm-studio")
workspace = modal.Volume.from_name("gtsfm-studio-data", create_if_missing=True)
workspace_secret = modal.Secret.from_dict({"GTSFM_API_KEY": REMOTE_API_KEY})


@app.function(
    image=image,
    gpu=GPU,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY_MB,
    timeout=24 * 60 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    scaledown_window=10 * 60,
    volumes={"/workspace": workspace},
    secrets=[workspace_secret],
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="gtsfm-workspace")
def gtsfm_workspace():
    """Serve the protected GTSFM API and viewer on a Modal GPU container."""

    import hmac
    from fastapi import Request
    from fastapi.responses import JSONResponse

    from visualization.app import create_app

    web_app = create_app("/workspace/results")

    @web_app.middleware("http")
    async def protect_internet_workspace(request: Request, call_next):
        expected = os.environ["GTSFM_API_KEY"]
        provided = request.headers.get("Authorization", "")
        if _is_public_workspace_read(request) or hmac.compare_digest(provided, f"Bearer {expected}"):
            return await call_next(request)
        if request.url.path.startswith(("/api/", "/data/")):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    return web_app
