"""Remote whole-job execution through Ray's native cluster launcher and Jobs API.

Ray is installed once, separately from the configured GTSFM Python environment.
The local provider requires a direct Linux SSH endpoint; Windows users can expose
stock OpenSSH in WSL over Tailscale. No Studio server is started on the remote machine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import AbstractContextManager, aclosing, suppress
from pathlib import Path
from typing import Any, AsyncIterator, Mapping, Sequence


def _remote_path(path: str) -> str:
    """Quote a remote Linux path, preserving expansion of the remote home directory."""
    if not path or "\x00" in path or "\n" in path:
        raise ValueError("Enter a valid remote Linux path")
    if path == "~":
        return '"$HOME"'
    return '"$HOME"/' + shlex.quote(path[2:]) if path.startswith("~/") else shlex.quote(path)


class RayWorkspace(AbstractContextManager):
    """Own submitted Ray jobs and a native dashboard transport, leaving Ray available."""

    def __init__(
        self,
        spec: Mapping[str, Any],
        runtime_root: Path | None = None,
        *,
        local_ray_binary: str | None = None,
    ) -> None:
        """Prepare a stable local-provider configuration without connecting remotely.

        Args:
            spec: Saved SSH settings, including the isolated remote Ray executable.
            runtime_root: Local directory for reusable Ray launcher configurations.
            local_ray_binary: Optional explicit local Ray CLI executable.
        """
        self.spec = dict(spec)
        if spec.get("ssh_authentication") == "key":
            # Stock Ray copies auth.ssh_private_key onto the head node, even
            # without workers. Keep identity handling inside local OpenSSH.
            raise ValueError("Use local OpenSSH configuration/SSH agent so private keys remain local.")
        host = str(spec.get("ssh_host") or "").strip()
        user = str(spec.get("ssh_username") or "").strip()
        if not re.fullmatch(r"[A-Za-z0-9_.:-]+", host) or host.startswith("-"):
            raise ValueError("Enter a direct Linux SSH hostname or IP address")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", user) or user.startswith("-"):
            raise ValueError("Enter the Linux SSH username")
        if spec.get("ssh_wsl_distribution"):
            raise ValueError("Ray requires direct Linux SSH; use the WSL Tailscale SSH host")
        if int(spec.get("ssh_port") or 22) != 22:
            raise ValueError("Configure the SSH port in an OpenSSH rule matching the destination IP")
        remote_ray = str(spec.get("ssh_ray_binary") or "~/.local/bin/ray")
        environment_ray = Path(sys.executable).with_name("ray")
        self.local_ray_binary = local_ray_binary or (
            str(environment_ray) if environment_ray.is_file() else shutil.which("ray") or ""
        )
        if not self.local_ray_binary:
            raise ValueError("Install the optional SSH execution dependencies (Ray 2.59.0) first")
        identity = hashlib.sha256(f"{host}\0{user}".encode()).hexdigest()[:12]
        self.cluster_name = f"gtsfm-{identity}"
        # Separate from Ray's ordinary default ports. A collision fails startup;
        # never stop another runtime to free a port.
        self.gcs_port = 20000 + int(identity[:6], 16) % 10000
        self.dashboard_port = self.gcs_port + 10000
        root = runtime_root or Path.home() / ".cache" / "gtsfm" / "ray"
        self.config_path = root / f"{self.cluster_name}.yaml"
        auth = {"ssh_user": user}
        ray_path = _remote_path(remote_ray)
        ray_temp = _remote_path(f"/tmp/gtsfm-ray-{identity}")
        start = (
            f"{ray_path} start --head --port={self.gcs_port} "
            f"--dashboard-host=127.0.0.1 --dashboard-port={self.dashboard_port} "
            f"--include-dashboard=true --object-store-memory=134217728 --temp-dir={ray_temp} "
            "--disable-usage-stats --autoscaling-config=~/ray_bootstrap_config.yaml"
        )
        self.configuration = {
            "cluster_name": self.cluster_name,
            "provider": {"type": "local", "head_ip": host, "worker_ips": []},
            "auth": auth,
            "min_workers": 0,
            "max_workers": 0,
            "file_mounts": {},
            "cluster_synced_files": [],
            "file_mounts_sync_continuously": False,
            "initialization_commands": [],
            "setup_commands": [],
            "head_setup_commands": [],
            "worker_setup_commands": [],
            "head_start_ray_commands": [start],
            "worker_start_ray_commands": [],
        }
        self.endpoint = ""
        self.client: Any = None
        self.tunnel: subprocess.Popen | None = None
        self.stderr_tail = ""
        self._log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        self._lock = threading.RLock()
        self._closed = False
        self._close_complete = threading.Event()
        self._closing_thread: int | None = None
        self._processes: set[subprocess.Popen] = set()
        self._owned_jobs: set[str] = set()

    def __enter__(self) -> RayWorkspace:
        """Start or reuse Ray and connect the native dashboard tunnel."""
        try:
            with self._lock:
                self._ensure_open()
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                # JSON is YAML 1.2; Ray's canonical safe_load accepts this format.
                content = json.dumps(self.configuration, indent=2)
                if not self.config_path.exists() or self.config_path.read_text() != content:
                    with tempfile.NamedTemporaryFile(mode="w", dir=self.config_path.parent, delete=False) as stream:
                        temporary = Path(stream.name)
                        stream.write(content)
                    try:
                        temporary.chmod(0o600)
                        temporary.replace(self.config_path)
                    finally:
                        temporary.unlink(missing_ok=True)
            self._run_cli(
                "up",
                str(self.config_path),
                "-y",
                "--no-restart",
                "--disable-usage-stats",
                "--use-normal-shells",
                timeout=180,
            )
            # A local-provider record survives machine reboots, while --no-restart
            # skips its start commands. Native CLI status/start recovers that case
            # without stopping a live cluster or touching another runtime.
            health = f"{_remote_path(str(self.spec.get('ssh_ray_binary') or '~/.local/bin/ray'))} status "
            health += (
                f"--address=127.0.0.1:{self.gcs_port} >/dev/null 2>&1 || "
                f"{self.configuration['head_start_ray_commands'][0]}"
            )
            self._run_cli("exec", str(self.config_path), health, timeout=120)
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                local_port = probe.getsockname()[1]
            self.endpoint = f"http://127.0.0.1:{local_port}"
            self.tunnel = self._start_cli(
                "dashboard", str(self.config_path), "--port", str(local_port), "--remote-port", str(self.dashboard_port)
            )
            self._connect(timeout=40)
            return self
        except BaseException:
            self.close()
            raise

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        """Release owned jobs and transport when leaving the context."""
        self.close()

    def _connect(self, timeout: float) -> None:
        """Wait for the native dashboard and lazily import the optional Jobs SDK."""
        from ray.job_submission import JobSubmissionClient

        deadline = time.monotonic() + timeout
        error: Exception | None = None
        while time.monotonic() < deadline:
            with self._lock:
                self._ensure_open()
                if self.tunnel is None or self.tunnel.poll() is not None:
                    break
            try:
                client = JobSubmissionClient(self.endpoint)
                with self._lock:
                    self._ensure_open()
                    self.client = client
                return
            except (OSError, RuntimeError, ConnectionError) as exc:
                error = exc
                time.sleep(0.2)
        raise RuntimeError(f"Ray dashboard did not become ready: {error or self._diagnostics()}")

    def submit(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        submission_id: str | None = None,
        num_gpus: float = 1,
        num_cpus: float = 1,
        memory_bytes: int | None = None,
        cwd: str | None = None,
    ) -> str:
        """Submit an ordinary command supervised by Ray, using the project Python.

        Args:
            command: Executable and arguments; no shell command fragments.
            env: Environment additions for the entrypoint.
            submission_id: Stable caller ID; Ray rejects duplicate submission.
            num_gpus: Logical GPU reservation, also setting GPU visibility.
            num_cpus: Logical CPU reservation.
            memory_bytes: Optional logical Ray memory reservation in bytes.
            cwd: Remote working directory, defaulting to the configured checkout.

        Returns:
            Ray submission ID.
        """
        if not command or isinstance(command, (str, bytes)):
            raise ValueError("A remote command is required")
        entrypoint = (
            f"cd {_remote_path(cwd or str(self.spec.get('ssh_workspace') or '.'))} && "
            'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" && exec '
            + " ".join(
                [
                    _remote_path(str(command[0])),
                    *(
                        _remote_path(str(arg)) if str(arg).startswith("~/") else shlex.quote(str(arg))
                        for arg in command[1:]
                    ),
                ]
            )
        )
        with self._lock:
            self._ensure_ready()
            job_id = submission_id or f"gtsfm-{uuid.uuid4().hex}"
            # Record before the request: a lost response must still be cancelled
            # by close; never automatically replay a possibly accepted job.
            self._owned_jobs.add(job_id)
            return self.client.submit_job(
                entrypoint=entrypoint,
                submission_id=job_id,
                runtime_env={"env_vars": dict(env or {})},
                entrypoint_num_gpus=num_gpus,
                entrypoint_num_cpus=num_cpus,
                entrypoint_memory=memory_bytes,
            )

    def get(self, job_id: str) -> Any:
        """Return Ray's native JobDetails for a submission."""
        with self._lock:
            self._ensure_ready()
            return self.client.get_job_info(job_id)

    def logs(self, job_id: str) -> str:
        """Return the complete entrypoint log through the stable Jobs SDK."""
        with self._lock:
            self._ensure_ready()
            return self.client.get_job_logs(job_id)

    async def tail_logs(self, job_id: str) -> AsyncIterator[str]:
        """Yield native SDK log chunks without holding a lock across network awaits.

        Args:
            job_id: Ray submission ID whose stdout and stderr should be followed.

        Yields:
            Successive native log chunks. Transport closure during workspace
            shutdown ends the stream; unexpected failures while open propagate.
        """
        with self._lock:
            self._ensure_ready()
            client = self.client
        try:
            async with aclosing(client.tail_job_logs(job_id)) as stream:
                async for chunk in stream:
                    with self._lock:
                        closed = self._closed
                    if closed:
                        return
                    yield chunk
        except Exception:
            with self._lock:
                closed = self._closed
            if not closed:
                raise

    def cancel(self, job_id: str) -> bool:
        """Ask Ray to terminate a job owned by this workspace."""
        with self._lock:
            self._ensure_ready()
            if job_id not in self._owned_jobs:
                raise ValueError("This workspace does not own that Ray job")
            return self.client.stop_job(job_id)

    def run_command(
        self,
        command: Sequence[str],
        *,
        env: Mapping[str, str] | None = None,
        cwd: str | None = None,
        timeout: float = 60,
        num_gpus: float = 0,
    ) -> str:
        """Run a short inspection command as a supervised Ray job and return its log."""
        job_id = self.submit(command, env=env, cwd=cwd, num_gpus=num_gpus, num_cpus=0)
        deadline = time.monotonic() + timeout
        try:
            while time.monotonic() < deadline:
                status = str(self.get(job_id).status)
                if status in {"SUCCEEDED", "FAILED", "STOPPED"}:
                    output = self.logs(job_id)
                    if status != "SUCCEEDED":
                        raise RuntimeError(f"Remote command {status.lower()}: {output[-4000:]}")
                    return output
                time.sleep(0.2)
            raise TimeoutError("Remote Ray command timed out")
        finally:
            with self._lock:
                if not self._closed:
                    with suppress(Exception):
                        self.cancel(job_id)

    def upload_directory(self, local: Path, remote: str) -> None:
        """Copy a directory's contents using native ray rsync_up."""
        source = Path(local).resolve()
        if not source.is_dir():
            raise ValueError("Choose an existing local directory to upload")
        # Native rsync does not create a missing parent hierarchy.
        self.run_command(["mkdir", "-p", remote], cwd="~")
        self._run_cli("rsync-up", str(self.config_path), str(source) + "/", remote.rstrip("/") + "/", timeout=600)

    def download_directory(self, remote: str, local: Path) -> None:
        """Copy remote directory contents using native ray rsync_down."""
        target = Path(local).resolve()
        target.mkdir(parents=True, exist_ok=True)
        self._run_cli("rsync-down", str(self.config_path), remote.rstrip("/") + "/", str(target) + "/", timeout=600)

    def _ensure_open(self) -> None:
        """Reject operations after cancellation or shutdown."""
        if self._closed:
            raise RuntimeError("Ray workspace is closed")

    def _ensure_ready(self) -> None:
        """Require an open, connected workspace."""
        self._ensure_open()
        if self.client is None:
            raise RuntimeError("Ray workspace has not connected")

    def _start_cli(self, *args: str) -> subprocess.Popen:
        """Start a tracked native CLI process so close can interrupt startup too."""
        with self._lock:
            self._ensure_open()
            process = subprocess.Popen(
                [self.local_ray_binary, *args],
                stdin=subprocess.DEVNULL,
                stdout=self._log,
                stderr=self._log,
                start_new_session=True,
                env={**os.environ, "RAY_USAGE_STATS_ENABLED": "0"},
            )
            self._processes.add(process)
            return process

    def _run_cli(self, *args: str, timeout: float) -> None:
        """Wait for native Ray CLI completion, retaining actionable stderr."""
        process = self._start_cli(*args)
        try:
            result = process.wait(timeout=timeout)
            with self._lock:
                self._ensure_open()
            if result:
                raise RuntimeError(f"Ray {args[0]} failed (exit {result}): {self._diagnostics()}")
        except subprocess.TimeoutExpired as exc:
            self._stop_process(process)
            raise TimeoutError(f"Ray {args[0]} timed out: {self._diagnostics()}") from exc
        finally:
            with self._lock:
                self._processes.discard(process)

    def _diagnostics(self) -> str:
        """Read the bounded native launcher and tunnel log."""
        with self._lock:
            if not self._log.closed:
                self._log.seek(0)
                return self._log.read()[-8000:]
            return self.stderr_tail

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        """Reap the native CLI and its local SSH subprocess group."""
        if process.poll() is not None:
            return
        with suppress(ProcessLookupError):
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                if os.name == "posix":
                    os.killpg(process.pid, signal.SIGKILL)
                else:
                    process.kill()
            process.wait(timeout=5)

    def close(self) -> None:
        """Stop owned jobs and transport, waiting for concurrent cleanup to finish.

        The public Ray Jobs SDK supplies no request timeout parameter. A wedged
        dashboard can therefore delay stop requests; this adapter retains that
        native limitation rather than changing private SDK transport behavior.
        """
        with self._lock:
            if self._closed:
                if self._closing_thread == threading.get_ident():
                    return
                wait_for_close = True
                client, jobs, processes = None, (), ()
            else:
                self._closed = True
                self._closing_thread = threading.get_ident()
                wait_for_close = False
                client, jobs = self.client, tuple(self._owned_jobs)
                processes = tuple(self._processes)
        if wait_for_close:
            self._close_complete.wait()
            return
        if self._close_complete.is_set():
            return
        try:
            # Keep the dashboard transport until all stop requests have been sent.
            if client is not None:
                for job_id in jobs:
                    try:
                        client.stop_job(job_id)
                    except Exception as exc:
                        self.stderr_tail += f"\nCould not stop Ray job {job_id}: {exc}"
            for process in processes:
                try:
                    self._stop_process(process)
                except OSError as exc:
                    self.stderr_tail += f"\nCould not close native Ray transport: {exc}"
            with self._lock:
                self.stderr_tail = self._diagnostics() + self.stderr_tail
                self._log.close()
                self._processes.clear()
        finally:
            self._close_complete.set()
