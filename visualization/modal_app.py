"""Modal deployment definition for the GTSFM GPU workspace.

This file is evaluated by ``modal deploy`` on the user's machine. It packages
the current checkout and exposes the same FastAPI workspace used locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal


SOURCE_ROOT = Path(os.environ.get("GTSFM_SOURCE_ROOT", Path.cwd())).expanduser().resolve()
GPU = os.environ.get("GTSFM_MODAL_GPU", "L40S")
MODAL_CPU = float(os.environ.get("GTSFM_MODAL_CPU", "8"))
MODAL_MEMORY_MB = int(os.environ.get("GTSFM_MODAL_MEMORY_MB", "65536"))
RUNTIME_IMAGE = os.environ.get("GTSFM_MODAL_RUNTIME_IMAGE", "").strip()
REMOTE_API_KEY = os.environ.get("GTSFM_REMOTE_API_KEY", "")
if not (SOURCE_ROOT / "pyproject.toml").is_file():
    raise RuntimeError("GTSFM_SOURCE_ROOT must point to a GTSFM source checkout")
if not REMOTE_API_KEY:
    raise RuntimeError("GTSFM_REMOTE_API_KEY is required for a protected remote workspace")
if not 1 <= MODAL_CPU <= 32:
    raise RuntimeError("GTSFM_MODAL_CPU must be between 1 and 32")
if not 4096 <= MODAL_MEMORY_MB <= 524288:
    raise RuntimeError("GTSFM_MODAL_MEMORY_MB must be between 4096 and 524288")

EXCLUDES = [
    "**/.git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
    "**/node_modules/**",
    "**/assets/**",
    "**/demo/**",
    "**/demos/**",
    "**/examples/**",
    "**/*.ipynb",
]

RUNTIME_ENV = {
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

if RUNTIME_IMAGE:
    image = modal.Image.from_registry(RUNTIME_IMAGE).entrypoint([]).env(RUNTIME_ENV)
else:
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
        .env(RUNTIME_ENV)
        .uv_sync(str(SOURCE_ROOT), groups=[], frozen=True, extra_options="--no-default-groups")
    )

image = (
    image.add_local_dir(SOURCE_ROOT / "gtsfm", "/root/gtsfm", ignore=EXCLUDES)
    .add_local_dir(SOURCE_ROOT / "visualization", "/root/visualization", ignore=EXCLUDES)
    .add_local_dir(SOURCE_ROOT / "thirdparty", "/root/thirdparty", ignore=EXCLUDES)
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
        if request.url.path.startswith(("/api/", "/data/")) and not hmac.compare_digest(provided, f"Bearer {expected}"):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    return web_app
