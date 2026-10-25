"""Durable Studio job records and conservative recovery of older run logs."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)
JOB_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")
SECRET_FIELDS = {
    "api_key",
    "modal_api_key",
    "modal_token_id",
    "modal_token_secret",
    "modal_token_command",
    "ssh_private_key",
}


def public_spec(spec: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in spec.items() if key not in SECRET_FIELDS}


def write_record(directory: Path, record: dict[str, Any]) -> None:
    """Atomically replace a private JSON record; never serialize process handles or credentials."""
    directory.mkdir(parents=True, exist_ok=True)
    record = {**record, "spec": public_spec(record["spec"])}
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=directory, suffix=".tmp", delete=False
        ) as file:
            temporary = Path(file.name)
            json.dump({"version": 1, "job": record}, file, ensure_ascii=False)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(directory / f'{record["id"]}.json')
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_records(directory: Path) -> Iterator[dict[str, Any]]:
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("version") != 1:
                raise ValueError("Unsupported history version")
            record = payload["job"]
            if not JOB_ID.fullmatch(record["id"]) or record["id"] != path.stem:
                raise ValueError("Invalid history job ID")
            if not isinstance(record["spec"], dict) or not isinstance(record["command"], list):
                raise ValueError("Invalid saved settings or command")
            if not all(isinstance(arg, str) for arg in record["command"]):
                raise ValueError("Invalid saved command")
            if not isinstance(record.get("log_tail", []), list) or not all(
                isinstance(line, str) for line in record.get("log_tail", [])
            ):
                raise ValueError("Invalid saved log tail")
            if record.get("pid") is not None and (type(record["pid"]) is not int or record["pid"] <= 0):
                raise ValueError("Invalid saved process ID")
            if record.get("remote") is not None and not isinstance(record["remote"], dict):
                raise ValueError("Invalid remote metadata")
            for name in ("name", "output_root", "live_root"):
                if not isinstance(record[name], str):
                    raise ValueError(f"Invalid {name}")
            for name in ("created_at", "updated_at"):
                record[name] = datetime.fromisoformat(record[name]).astimezone(timezone.utc).isoformat()
            if record["status"] not in {"queued", "running", "failed", "completed", "cancelled", "interrupted"}:
                raise ValueError("Invalid status")
            record["spec"] = public_spec(record["spec"])
            yield record
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            logger.warning("Skipping unreadable Studio job history: %s", path)


def log_excerpt(path: Path) -> tuple[str, list[str]]:
    """Read a bounded head/tail even for multi-gigabyte logs."""
    try:
        with path.open("rb") as file:
            head = file.read(64 * 1024).decode("utf-8", errors="replace")
            file.seek(0, os.SEEK_END)
            offset = max(0, file.tell() - 128 * 1024)
            file.seek(offset)
            tail = file.read().decode("utf-8", errors="replace").splitlines()
            if offset:
                tail = tail[1:]
            return head, tail[-250:]
    except OSError:
        return "", []


def recover_legacy_runs(results_root: Path, known_ids: set[str]) -> Iterator[dict[str, Any]]:
    """Recover history from logs, without inventing launch settings or exit codes."""
    runtime_root = results_root / ".gtsfm"
    run_roots = {}
    for path in (results_root / "runs").glob("*"):
        match = re.fullmatch(r"(.+)-([0-9a-f]{12})", path.name)
        if match and path.is_dir():
            run_roots[match[2]] = (match[1], path)
    for log in runtime_root.glob("*.log"):
        job_id = log.stem
        if job_id in known_ids or not re.fullmatch(r"[0-9a-f]{12}", job_id):
            continue
        head, tail = log_excerpt(log)
        if not head:
            continue
        try:
            modified = log.stat().st_mtime
        except OSError:
            continue
        created = datetime.fromtimestamp(modified, timezone.utc)
        timestamp = re.search(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", head, re.MULTILINE)
        if timestamp:
            # GTSFM's DaskAwareFormatter records UTC without an offset suffix.
            created = datetime.fromisoformat(timestamp[0]).replace(tzinfo=timezone.utc)
        name, output_root = run_roots.get(
            job_id, (f"Recovered run {job_id}", results_root / "runs" / f"recovered-{job_id}")
        )
        spec: dict[str, Any] = {"name": name}
        config = re.search(r"Config File: ([\w.-]+)", head)
        if config:
            spec["config_name"] = config[1]
        status = "interrupted"
        # Only terminal log evidence establishes completion/failure. Sparse output alone
        # does not mean a later splatting stage succeeded.
        for line in tail:
            if "Shutting down Dask client" in line or line.startswith("Saved Gaussian splats to "):
                status = "completed"
            if "Traceback (most recent call last)" in line or re.search(r"\*\*\* (?:SIG|Check failure)", line):
                status = "failed"
        yield {
            "id": job_id,
            "name": name,
            "status": status,
            "created_at": created.isoformat(),
            "updated_at": datetime.fromtimestamp(modified, timezone.utc).isoformat(),
            "output_root": str(output_root),
            "live_root": str(runtime_root / "live" / job_id),
            "spec": spec,
            "command": [],
            "log_tail": tail,
            "retry_available": False,
            "history_note": "Recovered from an older run log. The original command and full settings were not saved, so Retry is unavailable.",
        }
