"""Modal deployment definition for the GTSFM GPU workspace.

This file is evaluated by ``modal deploy`` on the user's machine. It packages
the current checkout and exposes the same FastAPI workspace used locally.
"""

from __future__ import annotations

import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Mapping

import modal

from visualization.runtime import JobManager, ManagedJob, _utc_now

IS_MODAL_RUNTIME = os.environ.get("MODAL_IS_REMOTE") == "1"
# GTSFM_SOURCE_ROOT is a path on the user's machine and is valid only while
# `modal deploy` packages the checkout. Modal imports this module again inside
# the container, where the packaged source lives under /root instead.
SOURCE_ROOT = (
    Path("/root") if IS_MODAL_RUNTIME else Path(os.environ.get("GTSFM_SOURCE_ROOT", Path.cwd())).expanduser().resolve()
)
GPU = os.environ.get("GTSFM_MODAL_GPU", "L40S")
MODAL_CPU = float(os.environ.get("GTSFM_MODAL_CPU", "8"))
MODAL_MEMORY_MB = int(os.environ.get("GTSFM_MODAL_MEMORY_MB", "65536"))
RUNTIME_IMAGE = os.environ.get("GTSFM_MODAL_RUNTIME_IMAGE", "").strip()
# One derived bearer key protects the workspace API. The local deploy process
# uses it to create a Modal Secret, which injects the same GTSFM_API_KEY into
# the remote container. Modal account credentials are never mounted here.
WORKSPACE_API_KEY = os.environ.get("GTSFM_API_KEY", "")
# Do not expose this path through image-build environment variables. Modal
# extends registry images before mounting Volumes, and tools such as uv/pip may
# eagerly create cache directories from those variables. A non-empty mount
# target makes the container fail before the application can start.
WORKSPACE_MOUNT = "/mnt/gtsfm-studio"
if not IS_MODAL_RUNTIME and not (SOURCE_ROOT / "pyproject.toml").is_file():
    raise RuntimeError("GTSFM_SOURCE_ROOT must point to a GTSFM source checkout")
if not IS_MODAL_RUNTIME and not WORKSPACE_API_KEY:
    raise RuntimeError("GTSFM_API_KEY is required for a protected remote workspace")
if not 1 <= MODAL_CPU <= 32:
    raise RuntimeError("GTSFM_MODAL_CPU must be between 1 and 32")
if not 4096 <= MODAL_MEMORY_MB <= 524288:
    raise RuntimeError("GTSFM_MODAL_MEMORY_MB must be between 4096 and 524288")

SOURCE_EXCLUDES = [
    "**/.git/**",
    "**/__pycache__/**",
    "**/*.pyc",
    "**/.DS_Store",
    "**/node_modules/**",
    "**/demo/**",
    "**/demos/**",
    "**/examples/**",
    "**/*.ipynb",
]
THIRDPARTY_EXCLUDES = [*SOURCE_EXCLUDES, "**/assets/**"]

RUNTIME_ENV = {
    # Modal's Python layer can leave CC/CXX pointing at clang even though the
    # The source-build image uses an explicit compiler selection for native dependencies.
    "CC": "/usr/bin/gcc",
    "CXX": "/usr/bin/g++",
    "PYTHONPATH": "/root",
    "GTSFM_MODAL_GPU": GPU,
    "GTSFM_MODAL_CPU": str(MODAL_CPU),
    "GTSFM_MODAL_MEMORY_MB": str(MODAL_MEMORY_MB),
}
if RUNTIME_IMAGE:
    # Preserve the image choice when Modal imports this definition again in
    # the remote container. Otherwise it would incorrectly enter the slower
    # source-build definition because local deployment variables are absent.
    RUNTIME_ENV["GTSFM_MODAL_RUNTIME_IMAGE"] = RUNTIME_IMAGE

if RUNTIME_IMAGE:
    gpu_image = (
        modal.Image.from_registry(
            RUNTIME_IMAGE,
            setup_dockerfile_commands=[
                # uv-created virtual environments intentionally omit pip, but
                # Modal's legacy registry-image builder requires ``python -m
                # pip`` while installing its runtime dependencies.
                "RUN uv pip install --python /opt/gtsfm-venv/bin/python pip",
            ],
        )
        .entrypoint([])
        .env(RUNTIME_ENV)
        .workdir("/root")
    )
else:
    gpu_image = (
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

gpu_image = (
    gpu_image.add_local_dir(SOURCE_ROOT / "gtsfm", "/root/gtsfm", ignore=SOURCE_EXCLUDES)
    .add_local_dir(SOURCE_ROOT / "visualization", "/root/visualization", ignore=SOURCE_EXCLUDES)
    .add_local_dir(SOURCE_ROOT / "thirdparty", "/root/thirdparty", ignore=THIRDPARTY_EXCLUDES)
)

# Browser traffic, health checks, uploads, status polling, and file downloads
# do not require CUDA. Keep them on a small CPU image so they cannot wake or
# hold an expensive GPU worker. Only ``gtsfm_gpu_job`` below uses gpu_image.
control_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "certifi",
        "fastapi>=0.116,<1.0",
        "pillow>=9.0.0",
        "python-multipart>=0.0.20,<1.0",
        "pyyaml",
        "uvicorn[standard]>=0.35,<1.0",
    )
    .env(RUNTIME_ENV)
    .add_local_dir(SOURCE_ROOT / "gtsfm", "/root/gtsfm", ignore=SOURCE_EXCLUDES)
    .add_local_dir(SOURCE_ROOT / "visualization", "/root/visualization", ignore=SOURCE_EXCLUDES)
)

app = modal.App("gtsfm-studio")
workspace = modal.Volume.from_name("gtsfm-studio-data", create_if_missing=True)
job_state = modal.Dict.from_name("gtsfm-studio-jobs", create_if_missing=True)
# Modal imports this definition both locally and inside its containers. Object
# dependencies must have the same shape in both environments, even though the
# deployed Secret's value is injected only when the function runs.
workspace_secret = modal.Secret.from_dict({"GTSFM_API_KEY": WORKSPACE_API_KEY})


@app.function(
    image=gpu_image,
    gpu=GPU,
    cpu=MODAL_CPU,
    memory=MODAL_MEMORY_MB,
    timeout=24 * 60 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    # Prefer cost savings over a warm GPU. Modal's minimum is two seconds, so
    # the accelerator is released almost immediately after a job completes.
    scaledown_window=2,
    # The target is intentionally absent from the image. Modal requires Volume
    # mount paths to be empty before the container starts.
    volumes={WORKSPACE_MOUNT: workspace},
)
def gtsfm_gpu_job(job_id: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Run exactly one reconstruction on GPU and publish lightweight state."""

    os.environ.update(
        {
            "HF_HOME": f"{WORKSPACE_MOUNT}/cache/huggingface",
            "TORCH_HOME": f"{WORKSPACE_MOUNT}/cache/torch",
            "TORCH_EXTENSIONS_DIR": f"{WORKSPACE_MOUNT}/cache/torch-extensions",
            "XDG_CACHE_HOME": f"{WORKSPACE_MOUNT}/cache",
        }
    )
    manager = JobManager(Path(WORKSPACE_MOUNT) / "results")
    gpu_spec = dict(spec)
    gpu_spec["execution_target"] = "local"
    gpu_spec["hardware"] = "cuda:0"

    def publish(payload: Mapping[str, Any]) -> None:
        state = dict(payload)
        try:
            previous = job_state.get(job_id)
        except KeyError:
            previous = None
        if isinstance(previous, dict) and previous.get("_modal_call_id"):
            state["_modal_call_id"] = previous["_modal_call_id"]
        preview = Path(str(state.get("live_root", ""))) / "live_splats.ply"
        if preview.is_file():
            state["_modal_preview_version"] = preview.stat().st_mtime_ns
        job_state[job_id] = state

    try:
        manager.start(gpu_spec, job_id=job_id)
        last_committed_preview_version = 0
        while True:
            job = manager.get(job_id)
            if job is None:
                raise RuntimeError("GPU worker lost the active job")
            payload = job.public()
            preview_path = Path(job.live_root) / "live_splats.ply"
            preview_version = preview_path.stat().st_mtime_ns if preview_path.is_file() else 0
            if preview_version > last_committed_preview_version:
                # Make the bytes visible to the CPU control service before its
                # job-state notification advertises this preview version.
                workspace.commit()
                last_committed_preview_version = preview_version
            publish(payload)
            if job.status not in {"queued", "running"}:
                workspace.commit()
                publish(job.public())
                return job.public()
            time.sleep(1)
    except Exception as exc:
        failed = {
            "id": job_id,
            "name": str(spec.get("name") or "gtsfm-run"),
            "status": "failed",
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
            "output_root": str(Path(WORKSPACE_MOUNT) / "results" / "runs" / job_id),
            "live_root": str(Path(WORKSPACE_MOUNT) / "results" / ".gtsfm" / "live" / job_id),
            "spec": dict(spec),
            "command": ["gtsfm", "runner"],
            "return_code": None,
            "error": str(exc),
            "log_tail": [str(exc)],
            "pid": None,
            "remote": None,
            "has_live_preview": False,
            "has_final_splat": False,
        }
        publish(failed)
        raise


class ModalGpuJobManager(JobManager):
    """CPU-side job controller that dispatches only pipelines to a GPU Function."""

    def __init__(self, results_root: Path) -> None:
        super().__init__(results_root)
        self._calls: dict[str, str] = {}
        self._preview_versions: dict[str, int] = {}
        self._terminal_reloads: set[str] = set()
        self._restore_jobs()

    def _restore_jobs(self) -> None:
        try:
            job_ids = job_state.get("__job_ids__") or []
        except KeyError:
            job_ids = []
        for job_id in job_ids[-100:]:
            try:
                payload = job_state.get(job_id)
            except KeyError:
                continue
            if isinstance(payload, dict):
                self._restore_payload(payload)

    def _restore_payload(self, payload: Mapping[str, Any]) -> ManagedJob:
        job_id = str(payload["id"])
        job = self._jobs.get(job_id)
        if job is None:
            job = ManagedJob(
                id=job_id,
                name=str(payload.get("name") or "gtsfm-run"),
                status=str(payload.get("status") or "queued"),
                created_at=str(payload.get("created_at") or _utc_now()),
                updated_at=str(payload.get("updated_at") or _utc_now()),
                output_root=str(payload.get("output_root") or self.results_root / "runs" / job_id),
                live_root=str(payload.get("live_root") or self.runtime_root / "live" / job_id),
                spec=dict(payload.get("spec") or {}),
                command=list(payload.get("command") or ["modal", "gtsfm_gpu_job", job_id]),
            )
            self._jobs[job_id] = job
        job.status = str(payload.get("status") or job.status)
        job.updated_at = str(payload.get("updated_at") or job.updated_at)
        job.output_root = str(payload.get("output_root") or job.output_root)
        job.live_root = str(payload.get("live_root") or job.live_root)
        job.return_code = payload.get("return_code")
        job.error = str(payload["error"]) if payload.get("error") else None
        job.log_tail = [str(line) for line in payload.get("log_tail") or []][-250:]
        call_id = str(payload.get("_modal_call_id") or "")
        if call_id:
            self._calls[job_id] = call_id
        return job

    def _sync(self, job: ManagedJob) -> ManagedJob:
        try:
            payload = job_state.get(job.id)
        except KeyError:
            return job
        if not isinstance(payload, dict):
            return job
        with self._lock:
            job = self._restore_payload(payload)
        call_id = self._calls.get(job.id)
        if job.status in {"queued", "running"} and call_id:
            try:
                completed = modal.FunctionCall.from_id(call_id).get(timeout=0)
            except TimeoutError:
                pass
            except Exception as exc:
                with self._lock:
                    job.status = "failed"
                    job.error = f"Modal GPU worker failed: {exc}"
                    job.updated_at = _utc_now()
                    job.log_tail.append(job.error)
                    del job.log_tail[:-250]
                failed_state = job.public()
                failed_state["_modal_call_id"] = call_id
                job_state[job.id] = failed_state
            else:
                if isinstance(completed, dict):
                    with self._lock:
                        job = self._restore_payload(completed)
        preview_version = int(payload.get("_modal_preview_version") or 0)
        should_reload = preview_version > self._preview_versions.get(job.id, 0)
        if job.status not in {"queued", "running"} and job.id not in self._terminal_reloads:
            should_reload = True
            self._terminal_reloads.add(job.id)
        if should_reload:
            try:
                workspace.reload()
                self._preview_versions[job.id] = preview_version
            except Exception as exc:
                with self._lock:
                    job.log_tail.append(f"Result sync is retrying: {exc}")
                    del job.log_tail[:-250]
        return job

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            jobs = list(self._jobs.values())
        for job in jobs:
            self._sync(job)
        return super().list()

    def get(self, job_id: str) -> ManagedJob | None:
        job = super().get(job_id)
        return self._sync(job) if job is not None else None

    def start(self, spec: Mapping[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
        if str(spec.get("execution_target") or "local") != "local":
            raise ValueError("The Modal control service only accepts local worker specifications")
        with self._lock:
            known_jobs = list(self._jobs.values())
        active_jobs = [job for job in known_jobs if self._sync(job).status in {"queued", "running"}]
        if active_jobs:
            raise ValueError("A Modal reconstruction is already running. Stop it or wait for it to finish.")
        job_id = job_id or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
            raise ValueError("Job ID contains unsupported characters")
        name = re.sub(r"[^A-Za-z0-9._-]+", "-", str(spec.get("name") or "gtsfm-run")).strip("-.") or "gtsfm-run"
        public_spec = dict(spec)
        for secret_field in (
            "api_key",
            "modal_api_key",
            "modal_token_id",
            "modal_token_secret",
            "modal_token_command",
            "ssh_private_key",
        ):
            public_spec.pop(secret_field, None)
        if public_spec.get("splat_implementation") == "anysplat":
            public_spec["config_name"] = "anysplat"
        now = _utc_now()
        job = ManagedJob(
            id=job_id,
            name=name,
            status="queued",
            created_at=now,
            updated_at=now,
            output_root=str(self.results_root / "runs" / f"{name}-{job_id}"),
            live_root=str(self.runtime_root / "live" / job_id),
            spec=public_spec,
            command=["modal", "gtsfm_gpu_job", job_id],
        )
        with self._lock:
            self._jobs[job_id] = job
        workspace.commit()
        initial = job.public()
        job_state[job_id] = initial
        try:
            call = gtsfm_gpu_job.spawn(job_id, dict(spec))
        except Exception as exc:
            with self._lock:
                job.status = "failed"
                job.error = f"Unable to start Modal GPU worker: {exc}"
                job.updated_at = _utc_now()
            job_state[job_id] = job.public()
            return job.public()
        self._calls[job_id] = call.object_id
        state = job.public()
        state["_modal_call_id"] = call.object_id
        job_state[job_id] = state
        try:
            known_ids = list(job_state.get("__job_ids__") or [])
        except KeyError:
            known_ids = []
        job_state["__job_ids__"] = [item for item in known_ids if item != job_id][-99:] + [job_id]
        return job.public()

    def cancel(self, job_id: str) -> dict[str, Any]:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status not in {"queued", "running"}:
            return job.public()
        call_id = self._calls.get(job_id)
        if call_id:
            modal.FunctionCall.from_id(call_id).cancel(terminate_containers=True)
        with self._lock:
            job.status = "cancelled"
            job.updated_at = _utc_now()
            job.error = None
            job.log_tail.append("Modal GPU worker terminated.")
            del job.log_tail[:-250]
        state = job.public()
        if call_id:
            state["_modal_call_id"] = call_id
        job_state[job_id] = state
        return job.public()


def modal_gpu_hardware() -> dict[str, Any]:
    """Describe the configured worker without allocating it for inspection."""

    return {
        "summary": f"Modal {GPU} GPU worker configured",
        "devices": [
            {
                "id": "cuda:0",
                "kind": "cuda",
                "index": 0,
                "label": f"Modal {GPU}",
                "status": "available",
                "memory": None,
                "details": "Allocated only while a reconstruction is running",
                "supports_gaussian_splatting": True,
            }
        ],
        "accelerator_count": 1,
        "platform": {"system": "Linux", "machine": "x86_64", "python": "3.12", "torch": None, "cuda": "12.8"},
    }


@app.function(
    image=control_image,
    cpu=1,
    memory=2048,
    timeout=24 * 60 * 60,
    startup_timeout=5 * 60,
    max_containers=1,
    scaledown_window=2 * 60,
    volumes={WORKSPACE_MOUNT: workspace},
    secrets=[workspace_secret],
)
@modal.concurrent(max_inputs=50)
@modal.asgi_app(label="gtsfm-workspace")
def gtsfm_workspace():
    """Serve control-plane HTTP traffic on CPU and dispatch pipelines to GPU."""

    import hmac

    from fastapi import Request
    from fastapi.responses import JSONResponse

    # Function secrets are available here, after Modal has hydrated the
    # function. Checking at module import time causes every worker to crash
    # before Modal gets a chance to inject the value.
    workspace_api_key = os.environ.get("GTSFM_API_KEY", "")
    if not workspace_api_key:
        raise RuntimeError("The Modal workspace secret GTSFM_API_KEY was not injected")

    from visualization.app import create_app

    manager = ModalGpuJobManager(Path(WORKSPACE_MOUNT) / "results")
    web_app = create_app(
        f"{WORKSPACE_MOUNT}/results",
        job_manager=manager,
        hardware_provider=modal_gpu_hardware,
    )

    @web_app.middleware("http")
    async def protect_internet_workspace(request: Request, call_next):
        provided = request.headers.get("Authorization", "")
        if request.url.path.startswith(("/api/", "/data/")) and not hmac.compare_digest(
            provided, f"Bearer {workspace_api_key}"
        ):
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)

    return web_app
