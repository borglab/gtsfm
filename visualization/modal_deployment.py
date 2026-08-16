"""Background deployment support for a GTSFM workspace on Modal."""

from __future__ import annotations

import hashlib
import hmac
import os
import signal
import subprocess
import sys
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SUPPORTED_MODAL_GPUS = {
    "T4",
    "L4",
    "A10",
    "L40S",
    "A100-40GB",
    "A100-80GB",
    "RTX-PRO-6000",
    "H100",
    "H200",
    "B200",
}
DEFAULT_MODAL_RUNTIME_IMAGE = "docker.io/su071301/gtsfm-modal-runtime:firstclass"


def modal_workspace_api_key(token_id: str, token_secret: str) -> str:
    """Derive a stable app-specific key without exposing the Modal credentials."""

    digest = hmac.new(
        token_secret.encode("utf-8"),
        f"gtsfm-modal-workspace-v1:{token_id}".encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"gtsfm_{digest}"


def find_source_root() -> Path:
    """Find the checkout required to build the Modal image."""

    candidates = [
        Path(os.environ.get("GTSFM_SOURCE_ROOT", "")),
        Path.cwd(),
        Path(__file__).resolve().parents[1],
    ]
    for candidate in candidates:
        if not str(candidate):
            continue
        resolved = candidate.expanduser().resolve()
        if (resolved / "pyproject.toml").is_file() and (resolved / "visualization" / "modal_app.py").is_file():
            return resolved
    raise ValueError(
        "Modal deployment requires a GTSFM source checkout. Start `gtsfm run` from the repository directory."
    )


@dataclass
class ModalDeployment:
    id: str
    gpu: str
    cpu: float = 8
    memory_mb: int = 65536
    status: str = "queued"
    stage: str = "Waiting to deploy"
    phase: str = "queued"
    image_source: str = "source"
    runtime_image: str = ""
    log_tail: list[str] = field(default_factory=list)
    endpoint: str = ""
    api_key: str = ""
    error: str = ""
    cancel_requested: bool = field(default=False, repr=False, compare=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False, compare=False)
    token_id: str = field(default="", repr=False, compare=False)
    token_secret: str = field(default="", repr=False, compare=False)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "gpu": self.gpu,
            "cpu": self.cpu,
            "memory_mb": self.memory_mb,
            "status": self.status,
            "stage": self.stage,
            "phase": self.phase,
            "image_source": self.image_source,
            "runtime_image": self.runtime_image,
            "log_tail": list(self.log_tail),
            "endpoint": self.endpoint,
            "api_key": self.api_key if self.status == "completed" else "",
            "error": self.error,
        }


class ModalDeploymentManager:
    """Run Modal's image build/deploy command without blocking the workspace API."""

    def __init__(
        self,
        discover: Callable[[str, str], dict[str, str]],
        runtime_image: str | None = None,
    ) -> None:
        self._discover = discover
        self._runtime_image = (
            os.environ.get("GTSFM_MODAL_RUNTIME_IMAGE", DEFAULT_MODAL_RUNTIME_IMAGE)
            if runtime_image is None
            else runtime_image
        ).strip()
        self._deployments: dict[str, ModalDeployment] = {}
        self._lock = threading.RLock()

    def start(
        self,
        token_id: str,
        token_secret: str,
        gpu: str,
        *,
        cpu: float = 8,
        memory_mb: int = 65536,
    ) -> dict[str, Any]:
        if not token_id.startswith("ak-") or not token_secret.startswith("as-"):
            raise ValueError("Enter a valid Modal token ID and token secret")
        if gpu not in SUPPORTED_MODAL_GPUS:
            raise ValueError(f"Unsupported Modal GPU: {gpu}")
        if not 1 <= cpu <= 32:
            raise ValueError("Modal CPU allocation must be between 1 and 32 cores")
        if not 4096 <= memory_mb <= 524288:
            raise ValueError("Modal memory allocation must be between 4 GB and 512 GB")
        source_root = find_source_root()
        deployment = ModalDeployment(
            id=uuid.uuid4().hex[:12],
            gpu=gpu,
            cpu=cpu,
            memory_mb=memory_mb,
            image_source="prebuilt" if self._runtime_image else "source",
            runtime_image=self._runtime_image,
            token_id=token_id,
            token_secret=token_secret,
        )
        with self._lock:
            self._deployments[deployment.id] = deployment
        thread = threading.Thread(
            target=self._run,
            args=(deployment, source_root),
            daemon=True,
            name=f"gtsfm-modal-deploy-{deployment.id}",
        )
        thread.start()
        return deployment.public()

    def get(self, deployment_id: str) -> dict[str, Any] | None:
        with self._lock:
            deployment = self._deployments.get(deployment_id)
            return deployment.public() if deployment else None

    def cancel(self, deployment_id: str) -> dict[str, Any]:
        """Request cancellation and terminate the active Modal CLI process."""

        with self._lock:
            deployment = self._deployments.get(deployment_id)
            if deployment is None:
                raise KeyError(deployment_id)
            if deployment.status in {"completed", "failed", "cancelled"}:
                return deployment.public()
            deployment.cancel_requested = True
            deployment.status = "cancelling"
            deployment.stage = "Stopping Modal workspace setup"
            process = deployment.process
        if process is not None:
            self._terminate_process(process)
        return deployment.public()

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        """Terminate the CLI process tree, then force-kill it if it does not exit."""

        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
        except OSError:
            try:
                process.terminate()
            except OSError:
                pass

        def force_kill() -> None:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    if os.name == "posix":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                except (OSError, ProcessLookupError):
                    pass

        threading.Thread(target=force_kill, daemon=True, name="gtsfm-modal-cancel").start()

    def _mark_cancelled(self, deployment: ModalDeployment) -> None:
        with self._lock:
            deployment.status = "cancelled"
            deployment.phase = "cancelled"
            deployment.stage = "Modal workspace setup stopped"
            deployment.error = ""

    def _update(
        self,
        deployment: ModalDeployment,
        *,
        stage: str | None = None,
        phase: str | None = None,
        line: str | None = None,
    ) -> None:
        with self._lock:
            if stage:
                deployment.stage = stage
            if phase:
                deployment.phase = phase
            if line:
                deployment.log_tail.append(line)
                del deployment.log_tail[:-120]

    def _run(self, deployment: ModalDeployment, source_root: Path) -> None:
        env = os.environ.copy()
        env.update(
            {
                "MODAL_TOKEN_ID": deployment.token_id,
                "MODAL_TOKEN_SECRET": deployment.token_secret,
                "GTSFM_MODAL_GPU": deployment.gpu,
                "GTSFM_MODAL_CPU": str(deployment.cpu),
                "GTSFM_MODAL_MEMORY_MB": str(deployment.memory_mb),
                "GTSFM_API_KEY": modal_workspace_api_key(deployment.token_id, deployment.token_secret),
                "GTSFM_SOURCE_ROOT": str(source_root),
            }
        )
        if deployment.runtime_image:
            env["GTSFM_MODAL_RUNTIME_IMAGE"] = deployment.runtime_image
        command = [
            sys.executable,
            "-m",
            "modal",
            "deploy",
            str(source_root / "visualization" / "modal_app.py"),
            "--name",
            "gtsfm-studio",
            "--strategy",
            "rolling",
        ]

        def run_command() -> int:
            process = subprocess.Popen(
                command,
                cwd=source_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=os.name == "posix",
            )
            with self._lock:
                deployment.process = process
                cancel_requested = deployment.cancel_requested
            if cancel_requested:
                self._terminate_process(process)
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = raw_line.rstrip()
                if line:
                    lowered = line.lower()
                    if deployment.phase == "building" and any(
                        marker in lowered
                        for marker in ("created objects", "deploying app", "app deployed", "created web function")
                    ):
                        self._update(
                            deployment,
                            phase="deploying",
                            stage="Deploying the Modal CPU control service",
                        )
                    self._update(deployment, line=line)
            return process.wait()

        try:
            with self._lock:
                if deployment.cancel_requested:
                    self._mark_cancelled(deployment)
                    return
                deployment.status = "running"
                deployment.phase = "building"
                deployment.stage = (
                    "Preparing the prebuilt CUDA runtime"
                    if deployment.runtime_image
                    else "Building the CUDA workspace image"
                )
            return_code = run_command()
            if deployment.cancel_requested:
                self._mark_cancelled(deployment)
                return
            unavailable_output = "\n".join(deployment.log_tail).lower()
            prebuilt_unavailable = deployment.runtime_image and any(
                marker in unavailable_output
                for marker in (
                    "manifest unknown",
                    "manifest not found",
                    "failed to pull",
                    "no matching manifest",
                    "not found: ghcr.io",
                )
            )
            if return_code != 0 and prebuilt_unavailable:
                self._update(
                    deployment,
                    phase="building",
                    stage="Prebuilt runtime unavailable; building from source",
                    line="Prebuilt GTSFM runtime is not published yet. Falling back to the cached source build.",
                )
                with self._lock:
                    deployment.image_source = "source"
                    deployment.runtime_image = ""
                env.pop("GTSFM_MODAL_RUNTIME_IMAGE", None)
                return_code = run_command()
            if deployment.cancel_requested:
                self._mark_cancelled(deployment)
                return
            if return_code != 0:
                raise RuntimeError(f"Modal deployment exited with code {return_code}")
            self._update(
                deployment,
                phase="verifying",
                stage="Verifying the deployed Modal workspace",
            )
            if deployment.cancel_requested:
                self._mark_cancelled(deployment)
                return
            discovered = self._discover(deployment.token_id, deployment.token_secret)
            with self._lock:
                if deployment.cancel_requested:
                    self._mark_cancelled(deployment)
                    return
                deployment.endpoint = discovered["endpoint"]
                deployment.api_key = modal_workspace_api_key(deployment.token_id, deployment.token_secret)
                deployment.status = "completed"
                deployment.phase = "ready"
                deployment.stage = "Ready"
        except Exception as exc:
            with self._lock:
                if deployment.cancel_requested:
                    self._mark_cancelled(deployment)
                    return
                deployment.status = "failed"
                deployment.stage = {
                    "building": "CUDA workspace image setup failed",
                    "deploying": "Modal workspace deployment failed",
                    "verifying": "Modal workspace verification failed",
                }.get(deployment.phase, "Modal workspace setup failed")
                deployment.error = str(exc)
        finally:
            with self._lock:
                deployment.process = None
                deployment.token_id = ""
                deployment.token_secret = ""
