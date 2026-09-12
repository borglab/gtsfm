"""Collect and publish Dask activity for Studio."""

import json
import os
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from dask.distributed import Client

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()


def _scheduler_task_counts(dask_scheduler: Any) -> dict[str, int]:
    """Return authoritative current task states from the Dask scheduler."""

    return dict(Counter(task.state for task in dask_scheduler.tasks.values()))


def collect(client: Client) -> dict[str, object]:
    """Collect a compact, browser-friendly snapshot from Dask worker heartbeats."""

    scheduler = client.scheduler_info()
    workers = list(scheduler.get("workers", {}).values())
    task_counts: Counter[str] = Counter()
    memory_bytes = 0
    memory_limit_bytes = 0
    cpu_percent = 0.0
    threads = 0

    for worker in workers:
        metrics = worker.get("metrics", {})
        for state, count in metrics.get("task_counts", {}).items():
            task_counts[str(state)] += int(count)
        memory_bytes += int(metrics.get("memory", 0) or 0)
        memory_limit_bytes += int(worker.get("memory_limit", 0) or 0)
        cpu_percent += float(metrics.get("cpu", 0) or 0)
        threads += int(worker.get("nthreads", 0) or 0)

    # Worker heartbeat counts can lag behind short tasks. Ask the scheduler for
    # its current state so the UI responds as soon as a task starts or finishes.
    try:
        scheduler_task_counts = Counter(client.run_on_scheduler(_scheduler_task_counts))
        if scheduler_task_counts:
            task_counts = scheduler_task_counts
    except Exception as exc:
        logger.debug("Unable to inspect current Dask scheduler tasks: %s", exc)

    running_tasks = task_counts["executing"] + task_counts["processing"]
    completed_tasks = task_counts["memory"]
    pending_tasks = sum(
        task_counts[state] for state in ("ready", "waiting", "queued", "no-worker", "constrained", "fetch", "flight")
    )
    return {
        "workers": len(workers),
        "threads": threads,
        "running_tasks": running_tasks,
        "completed_tasks": completed_tasks,
        "pending_tasks": pending_tasks,
        "failed_tasks": task_counts["error"] + task_counts["erred"],
        "memory_bytes": memory_bytes,
        "memory_limit_bytes": memory_limit_bytes,
        "cpu_percent": cpu_percent,
        "dashboard_url": client.dashboard_link,
        "updated_at": time.time(),
    }


def _write(path: Path, payload: dict[str, object]) -> None:
    """Atomically publish a Dask snapshot without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            json.dump(payload, file)
        os.replace(temporary_name, path)
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def publish(client: Client, stop: threading.Event, path: Path) -> None:
    """Publish Dask status once per second while a Studio-managed run is active."""

    while not stop.is_set():
        try:
            _write(path, collect(client))
        except Exception as exc:  # observability must never interrupt reconstruction
            logger.debug("Unable to publish Dask workspace stats: %s", exc)
        stop.wait(1.0)
