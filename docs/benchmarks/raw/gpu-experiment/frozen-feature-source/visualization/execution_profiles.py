"""Validated, host-local Dask execution profiles for GTSFM Studio."""

from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

PROFILE_SCHEMA_VERSION = 1
PROFILE_PATH = Path.home() / ".gtsfm" / "studio-execution-profiles.json"
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MEMORY = re.compile(
    r"^(?P<amount>[0-9]+(?:\.[0-9]+)?)\s*(?P<unit>B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)$", re.IGNORECASE
)
_INTEGER = re.compile(r"^[0-9]+$")
_GPU_ID = re.compile(r"^(?:cuda|rocm|nvidia):[0-9]+$|^mps$")
_MEMORY_FACTORS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}
_PROFILE_FIELDS = {
    "name",
    "execution_mode",
    "workers",
    "threads_per_worker",
    "host_memory_limit",
    "selected_gpus",
    "scratch_directory",
    "schema_version",
    "host_memory_bytes",
}


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be an integer")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _INTEGER.fullmatch(value.strip()):
        return int(value)
    raise ValueError(f"{field} must be an integer")


def _memory_bytes(value: object) -> int:
    if not isinstance(value, str):
        raise ValueError("host_memory_limit must be a value such as 32GB")
    match = _MEMORY.fullmatch(value.strip())
    if not match:
        raise ValueError("host_memory_limit must use an explicit unit, for example 32GB")
    parsed = int(float(match.group("amount")) * _MEMORY_FACTORS[match.group("unit").lower()])
    if parsed <= 0:
        raise ValueError("host_memory_limit must be greater than zero")
    return parsed


def _normalise_name(value: object) -> str:
    name = str(value or "").strip()
    if not _PROFILE_NAME.fullmatch(name):
        raise ValueError("Profile name must be 1-64 letters, numbers, dots, underscores, or hyphens")
    return name


def validate_profile(payload: Mapping[str, Any], *, name: str | None = None) -> dict[str, Any]:
    """Validate and normalize one canonical profile record."""

    if not isinstance(payload, Mapping):
        raise ValueError("Execution profile must be an object")
    unknown = set(payload) - _PROFILE_FIELDS
    if unknown:
        raise ValueError("Unknown execution profile fields: " + ", ".join(sorted(map(str, unknown))))
    if "schema_version" in payload and _strict_int(payload["schema_version"], "schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(f"Unsupported execution profile schema version: {payload['schema_version']}")
    profile_name = _normalise_name(name if name is not None else payload.get("name"))
    if payload.get("name") not in (None, profile_name):
        raise ValueError("Profile name does not match its storage key")
    raw_mode = payload.get("execution_mode")
    if raw_mode not in (None, "local"):
        raise ValueError("execution_mode currently supports local Studio execution only; remote profiles are not supported")
    mode = "local"
    workers = _strict_int(payload.get("workers", 1), "workers")
    threads = _strict_int(payload.get("threads_per_worker", 1), "threads_per_worker")
    if not 1 <= workers <= 256:
        raise ValueError("workers must be between 1 and 256")
    if not 1 <= threads <= 256:
        raise ValueError("threads_per_worker must be between 1 and 256")
    memory = str(payload.get("host_memory_limit") or "32GB").strip()
    memory_bytes = _memory_bytes(memory)
    if "host_memory_bytes" in payload and _strict_int(payload["host_memory_bytes"], "host_memory_bytes") != memory_bytes:
        raise ValueError("host_memory_bytes does not match host_memory_limit")
    scratch = str(payload.get("scratch_directory") or "").strip()
    if scratch:
        scratch_path = Path(scratch).expanduser()
        if not scratch_path.is_absolute():
            raise ValueError("scratch_directory must be an absolute path")
        if scratch_path.exists() and not scratch_path.is_dir():
            raise ValueError("scratch_directory must refer to a directory, not a regular file")
        if scratch_path.exists() and not os.access(scratch_path, os.W_OK):
            raise ValueError("scratch_directory is not writable")
        if not scratch_path.exists():
            parent = scratch_path.parent
            while not parent.exists() and parent != parent.parent:
                parent = parent.parent
            if not os.access(parent, os.W_OK):
                raise ValueError("scratch_directory parent is not writable")
        scratch = str(scratch_path)
    selected = payload.get("selected_gpus", [])
    if selected in (None, ""):
        selected = []
    if not isinstance(selected, list) or not all(isinstance(item, str) and item.strip() for item in selected):
        raise ValueError("selected_gpus must be a list of device IDs")
    selected = [item.strip() for item in selected]
    if len(set(selected)) != len(selected):
        raise ValueError("selected_gpus must not contain duplicate device IDs")
    invalid_gpus = [item for item in selected if not _GPU_ID.fullmatch(item)]
    if invalid_gpus:
        raise ValueError("Unsupported GPU device ID: " + ", ".join(invalid_gpus))
    if len(selected) > 1:
        raise ValueError("Multiple selected GPUs are not supported by the Studio launcher yet")
    if selected and workers != 1:
        raise ValueError("A selected GPU profile must use exactly one worker")
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "name": profile_name,
        "execution_mode": mode,
        "workers": workers,
        "threads_per_worker": threads,
        "host_memory_limit": memory,
        "host_memory_bytes": memory_bytes,
        "selected_gpus": selected,
        "scratch_directory": scratch,
    }


class ExecutionProfileStore:
    """Atomic JSON persistence for named profiles in the user's home directory."""

    _locks: dict[Path, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, path: Path | str = PROFILE_PATH) -> None:
        self.path = Path(path).expanduser()
        with self._locks_guard:
            self._lock = self._locks.setdefault(self.path, threading.RLock())

    def _read(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": PROFILE_SCHEMA_VERSION, "profiles": {}}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read execution profiles at {self.path}: {exc}") from exc
        if not isinstance(raw, dict) or set(raw) - {"schema_version", "profiles"}:
            raise ValueError(f"Execution profile file is invalid: {self.path}")
        if _strict_int(raw.get("schema_version"), "schema_version") != PROFILE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported execution profile schema version: {raw.get('schema_version')}")
        if not isinstance(raw.get("profiles"), dict):
            raise ValueError(f"Execution profile file is invalid: {self.path}")
        return raw

    def list(self) -> list[dict[str, Any]]:
        with self._lock, self._file_lock():
            profiles = self._read()["profiles"]
        return sorted(
            (validate_profile(payload, name=name) for name, payload in profiles.items()),
            key=lambda item: item["name"],
        )

    def get(self, name: str) -> dict[str, Any] | None:
        profile_name = _normalise_name(name)
        with self._lock, self._file_lock():
            profiles = self._read()["profiles"]
        if profile_name not in profiles:
            return None
        return validate_profile(profiles[profile_name], name=profile_name)

    @contextmanager
    def _file_lock(self):
        """Serialize read-modify-write across Studio processes on POSIX hosts."""
        lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_file:
            try:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError) as exc:
                raise RuntimeError("Execution profile storage requires a filesystem with advisory file locks") from exc
            try:
                yield
            finally:
                import fcntl
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _write(self, document: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=self.path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(document, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, self.path)
        finally:
            Path(temporary_name).unlink(missing_ok=True)

    def save(self, payload: Mapping[str, Any], *, name: str | None = None) -> dict[str, Any]:
        profile = validate_profile(payload, name=name)
        with self._lock, self._file_lock():
            document = self._read()
            # Validate every existing entry before preserving it; corrupted state must fail closed.
            for stored_name, stored_payload in document["profiles"].items():
                validate_profile(stored_payload, name=stored_name)
            document["profiles"][profile["name"]] = profile
            self._write(document)
            return profile

    def delete(self, name: str) -> bool:
        profile_name = _normalise_name(name)
        with self._lock, self._file_lock():
            document = self._read()
            for stored_name, stored_payload in document["profiles"].items():
                validate_profile(stored_payload, name=stored_name)
            if profile_name not in document["profiles"]:
                return False
            del document["profiles"][profile_name]
            self._write(document)
            return True


def apply_profile(spec: Mapping[str, Any], store: ExecutionProfileStore) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Resolve a profile authoritatively, rejecting conflicting controls."""

    resolved = dict(spec)
    name = str(spec.get("execution_profile") or spec.get("execution_profile_name") or "").strip()
    if not name:
        return resolved, None
    profile = store.get(name)
    if profile is None:
        raise ValueError(f"Execution profile not found: {name}")
    expected = {
        "execution_target": profile["execution_mode"],
        "num_workers": profile["workers"],
        "threads_per_worker": profile["threads_per_worker"],
        "worker_memory_limit": profile["host_memory_limit"],
        "dask_tmpdir": profile["scratch_directory"],
        "hardware": profile["selected_gpus"][0] if profile["selected_gpus"] else "cpu",
    }
    defaults = {"num_workers": 1, "threads_per_worker": 1, "worker_memory_limit": "32GB", "dask_tmpdir": "", "hardware": "cpu"}
    for field, value in expected.items():
        supplied = spec.get(field)
        if supplied in (None, ""):
            continue
        if field in {"num_workers", "threads_per_worker"}:
            supplied = _strict_int(supplied, field)
        if field in defaults and type(supplied) is type(defaults[field]) and supplied == defaults[field]:
            continue
        if supplied != value:
            raise ValueError(f"Execution profile {name} conflicts with {field}; update the profile or switch to manual mode")
    if str(spec.get("cluster_config") or "").strip() or str(spec.get("input_worker") or "").strip():
        raise ValueError("cluster_config and input_worker cannot be combined with a saved execution profile")
    for field, value in expected.items():
        resolved[field] = value
    resolved["execution_profile_name"] = profile["name"]
    resolved["resolved_execution_profile"] = profile
    return resolved, profile
