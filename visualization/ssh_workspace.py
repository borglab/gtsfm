"""Own a temporary remote Studio service and an OpenSSH loopback tunnel.

This reuses Studio's existing HTTP job protocol. SSH identities, host aliases,
known_hosts, jump hosts and agents are handled by the user's OpenSSH client.
The remote checkout and Python environment are configured once, not per run.
"""

from __future__ import annotations

import base64
import json
import queue
import re
import secrets
import shlex
import socket
import subprocess
import tempfile
import threading
import time
from contextlib import AbstractContextManager, suppress
from pathlib import Path
from typing import Any, Mapping

# The launcher owns the service through stdin. Closing SSH delivers EOF and
# shuts the service down; the service lifespan cancels its remaining jobs.
_BOOTSTRAP = """
import json, os, signal, socket, subprocess, sys, tempfile
from pathlib import Path
options = json.loads(sys.stdin.readline())
root = Path(options["workspace"]).expanduser().resolve()
os.chdir(root)
sock = socket.socket()
sock.bind(("127.0.0.1", 0))
sock.listen(128)
port = sock.getsockname()[1]
env = dict(os.environ, GTSFM_API_KEY=options["api_key"])
script = ("from visualization.app import create_app; import uvicorn; "
          "uvicorn.run(create_app('results'), fd=" + str(sock.fileno()) +
          ", access_log=False, log_level='warning')")
with tempfile.TemporaryFile(mode="w+") as log:
    process = subprocess.Popen([sys.executable, "-u", "-c", script], env=env, stdout=log, stderr=log,
                               pass_fds=(sock.fileno(),), start_new_session=True)
    sock.close()
    for signum in (signal.SIGHUP, signal.SIGTERM):
        signal.signal(signum, lambda *_: sys.exit(0))
    print(json.dumps({"port": port}), flush=True)
    try:
        sys.stdin.read()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        log.seek(0)
        sys.stderr.write(log.read()[-8000:])
"""


def ssh_command(spec: Mapping[str, Any]) -> list[str]:
    """Build a noninteractive SSH command without overriding the user's identity defaults."""
    host = str(spec.get("ssh_host") or "").strip()
    user = str(spec.get("ssh_username") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:@\[\]-]+", host) or host.startswith("-"):
        raise ValueError("Enter an SSH host or host alias")
    if user and (not re.fullmatch(r"[A-Za-z0-9_.-]+", user) or user.startswith("-")):
        raise ValueError("Invalid SSH username")
    port = int(spec.get("ssh_port") or 22)
    if not 1 <= port <= 65535:
        raise ValueError("SSH port must be between 1 and 65535")
    args = [
        "ssh",
        "-T",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-p",
        str(port),
    ]
    if user:
        args += ["-l", user]
    if spec.get("ssh_authentication") == "key":
        identity = str(spec.get("ssh_private_key") or "").strip()
        if not identity or not Path(identity).expanduser().is_file():
            raise ValueError("Choose an existing SSH private key file")
        args += ["-i", str(Path(identity).expanduser()), "-o", "IdentitiesOnly=yes"]
    return [*args, host]


class SSHWorkspace(AbstractContextManager):
    """A single owner for the launcher, tunnel, and temporary API credential."""

    def __init__(self, spec: Mapping[str, Any]):
        self.spec = spec
        self.command = ssh_command(spec)
        if not str(spec.get("ssh_workspace") or "").strip():
            raise ValueError("Enter the remote GTSFM checkout directory")
        self.api_key = secrets.token_urlsafe(32)
        self.endpoint = ""
        self.launcher: subprocess.Popen | None = None
        self.tunnel: subprocess.Popen | None = None
        self.log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        self.stderr_tail = ""
        self._closed = False
        self._lifecycle_lock = threading.RLock()

    def __enter__(self):
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("SSH workspace was closed before startup")
            return self._open()

    def _open(self):
        try:
            python = str(self.spec.get("ssh_python") or ".venv/bin/python")
            # Resolve relative interpreters inside the configured checkout. Pass
            # workspace via stdin so tilde expansion happens in remote Python.
            workspace = str(self.spec["ssh_workspace"])
            if not python.startswith(("/", "~")) and "/" in python:
                python = workspace.rstrip("/") + "/" + python
            encoded = base64.b64encode(_BOOTSTRAP.encode()).decode()
            code = f"import base64;exec(base64.b64decode('{encoded}'))"
            executable = '"$HOME"/' + shlex.quote(python[2:]) if python.startswith("~/") else shlex.quote(python)
            linux_command = f"exec {executable} -u -c {shlex.quote(code)}"
            distribution = str(self.spec.get("ssh_wsl_distribution") or "").strip()
            if distribution:
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", distribution):
                    raise ValueError("Invalid WSL distribution name")
                # Windows OpenSSH's command shell passes this quoted argv to WSL.
                remote_command = subprocess.list2cmdline(
                    ["wsl.exe", "-d", distribution, "--", "sh", "-c", linux_command]
                )
            else:
                # expanduser is for the *remote* shell; do not expand local HOME.
                remote_command = linux_command
            self.launcher = subprocess.Popen(
                [*self.command, remote_command],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
            )
            self.launcher.stdin.write(json.dumps({"workspace": workspace, "api_key": self.api_key}) + "\n")
            self.launcher.stdin.flush()
            lines: queue.Queue = queue.Queue()
            threading.Thread(target=lambda: lines.put(self.launcher.stdout.readline()), daemon=True).start()
            try:
                handshake = json.loads(lines.get(timeout=30))
            except (queue.Empty, ValueError) as exc:
                self.log.seek(0)
                detail = self.log.read()[-1500:].strip()
                raise ValueError(
                    f"Remote startup failed; check the checkout, Python environment, and SSH access. {detail}"
                ) from exc
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                local_port = probe.getsockname()[1]
            forwarding = [
                "-o",
                "ExitOnForwardFailure=yes",
                "-N",
                "-L",
                f"127.0.0.1:{local_port}:127.0.0.1:{int(handshake['port'])}",
            ]
            self.tunnel = subprocess.Popen([*self.command[:-1], *forwarding, self.command[-1]], stderr=self.log)
            self.endpoint = f"http://127.0.0.1:{local_port}"
            return self
        except BaseException:
            self.close()
            raise

    def ready(self, read_json, timeout: float = 40) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.launcher.poll() is not None or self.tunnel.poll() is not None:
                break
            try:
                return read_json(f"{self.endpoint}/api/hardware", self.api_key)
            except (OSError, ValueError):
                time.sleep(0.25)
        raise ValueError(
            "Remote service did not start; verify GTSFM and Studio dependencies in the selected Python environment"
        )

    def close(self) -> None:
        with self._lifecycle_lock:
            self._close()

    def _close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self.launcher is not None:
                if self.launcher.stdin:
                    with suppress(BrokenPipeError):
                        self.launcher.stdin.close()
                self._reap(self.launcher, 20)
        finally:
            try:
                if self.tunnel is not None and self.tunnel.poll() is None:
                    self.tunnel.terminate()
                    self._reap(self.tunnel, 5)
            finally:
                self.log.seek(0)
                self.stderr_tail = self.log.read()[-8000:]
                self.log.close()

    @staticmethod
    def _reap(process: subprocess.Popen, timeout: float) -> None:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)

    def __exit__(self, *args):
        self.close()
