"""Locally saved execution settings; pipeline and dataset settings stay on the run form."""

from __future__ import annotations

import errno
import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class ExecutionSettings(BaseModel):
    """The same editable settings are saved, displayed, and submitted to the runner.

    Store SSH identity *paths*, never private key material or service credentials.
    YAML constructor paths are deliberately outside the normal saved-config workflow.
    """

    model_config = ConfigDict(extra="forbid")

    execution_target: Literal["local", "remote"] = "local"
    hardware: str = "cpu"
    remote_connection: Literal["api", "ssh"] = "ssh"
    remote_provider: str = "modal"
    remote_endpoint: str = ""
    remote_hardware: str = "cpu"
    modal_gpu: str = "L40S"
    ssh_host: str = ""
    ssh_port: int = Field(default=22, ge=1, le=65535)
    ssh_username: str = ""
    ssh_authentication: Literal["agent", "key"] = "agent"
    ssh_private_key: str = ""
    ssh_workspace: str = ""
    ssh_python: str = ".venv/bin/python"
    ssh_wsl_distribution: str = ""
    num_workers: int = Field(default=1, ge=1, le=1024)
    threads_per_worker: int = Field(default=1, ge=1, le=1024)
    worker_memory_limit: str = "32GB"
    dashboard_port: str = ""
    input_worker: str = ""
    dask_tmpdir: str = ""
    dask_worker_hosts: str = ""
    dask_scheduler_host: str = ""
    dask_scheduler_port: int = Field(default=0, ge=0, le=65535)
    dask_worker_python: str = ""
    num_retry_cluster_connection: int | None = Field(default=None, ge=0)


class ExecutionConfiguration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=80)
    settings: ExecutionSettings


class ExecutionStore:
    """One local file per Studio user, with atomic replacement and serialized edits."""

    def __init__(self, path: Path | None = None):
        config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        self.path = path or config_home / "gtsfm" / "execution-configurations.json"
        self._lock = threading.RLock()

    def list(self) -> list[dict]:
        with self._locked():
            return self._read()

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        # Report corruption rather than replacing the user's saved settings.
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [item.model_dump() for item in TypeAdapter(list[ExecutionConfiguration]).validate_python(data)]

    def save(self, configuration: ExecutionConfiguration) -> dict:
        item = configuration.model_copy(update={"name": configuration.name.strip()})
        if not item.name:
            raise ValueError("Enter a configuration name")
        with self._locked():
            items = [row for row in self._read() if row["name"] != item.name]
            items.append(item.model_dump())
            self._write(items)
        return item.model_dump()

    def delete(self, name: str) -> None:
        with self._locked():
            self._write([row for row in self._read() if row["name"] != name])

    @contextmanager
    def _locked(self):
        """Serialize complete transactions across Studio instances and processes.

        Keep the sidecar lock file in place: deleting it can split simultaneous
        callers between an old locked inode and a newly created unlocked one.
        """
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            lock_path = self.path.with_name(f".{self.path.name}.lock")
            descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
            with os.fdopen(descriptor, "r+b") as lock_file:
                if os.name == "nt":
                    import msvcrt

                    # Byte-range locks may extend beyond the end of the file.
                    while True:
                        lock_file.seek(0)
                        try:
                            msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
                            break
                        except OSError as exc:
                            if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                                raise
                            time.sleep(0.05)
                    try:
                        yield
                    finally:
                        lock_file.seek(0)
                        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                    try:
                        yield
                    finally:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write(self, items: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8") as stream:
                os.chmod(temporary, 0o600)
                json.dump(sorted(items, key=lambda row: row["name"].casefold()), stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            temporary.unlink(missing_ok=True)
