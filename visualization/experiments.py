"""Portable experiment records built around the existing GTSFM runner and exports."""

from __future__ import annotations

import hashlib
import json
import math
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping, NotRequired, TypedDict

import yaml

from visualization.datasets import dataset_manifest
from visualization.job_history import public_spec
from visualization.source_identity import source_identity

SCHEMA = "gtsfm.experiment.v1"
STAGE_RANK = {
    stage: rank for rank, stage in enumerate(("merged_retriangulated", "merged", "ba_output", "vggt", "anysplat"))
}


class ExperimentRecord(TypedDict):
    """Serializable submission snapshot and observed execution metadata."""

    runtime: dict[str, Any]
    schema: str
    id: str
    name: str
    created_at: str
    settings: dict[str, Any]
    source: dict[str, Any] | None
    submission_source: dict[str, Any] | None
    dataset: dict[str, Any]
    command: list[str]
    execution_id: NotRequired[str]


def _write_record(output: Path, record: ExperimentRecord) -> None:
    """Write an experiment snapshot with atomic replacement.

    Args:
        output: Experiment directory.
        record: Submission and execution metadata.
    """
    with NamedTemporaryFile(mode="w", encoding="utf-8", dir=output, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.close()
            temporary.replace(output / "experiment.json")
        finally:
            temporary.unlink(missing_ok=True)


def _read_record(output: Path, *, expected_id: str) -> dict[str, Any] | None:
    """Find a readable and matching experiment record across result layouts.

    Args:
        output: Experiment directory.
        expected_id: Job id that owns the record.

    Returns:
        Valid record, or None while a record is missing or unreadable.
    """
    for path in (output / "experiment.json", output / "results" / "experiment.json"):
        if not path.is_file() or path.is_symlink():
            continue
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if (
            isinstance(record, dict)
            and record.get("schema") == SCHEMA
            and record.get("id") == expected_id
            and isinstance(record.get("settings"), dict)
            and all(
                record.get(key) is None or isinstance(record[key], dict) for key in ("source", "dataset", "runtime")
            )
        ):
            return record
    return None


def capture_experiment(job: Any) -> ExperimentRecord | None:
    """Record inputs and source once before computation begins.

    Args:
        job: Managed run with public settings and its own output directory.

    Returns:
        The written record, or None if the job has been cancelled.
    """
    output = Path(job.output_root)
    output.mkdir(parents=True, exist_ok=True)
    files = []
    roots = [("dataset", job.spec.get("dataset_dir")), ("images", job.spec.get("images_dir"))]
    for role, value in roots:
        if not value:
            continue
        root = Path(value).expanduser().resolve()
        if not root.is_dir() or output.is_relative_to(root):
            continue
        for path, size, sha256 in dataset_manifest(root):
            if job.status == "cancelled":
                return None
            files.append(
                {
                    "role": role,
                    "path": path,
                    "bytes": size,
                    "sha256": sha256,
                }
            )
    packages = {}
    for name in ("gtsam", "torch", "numpy", "dask", "distributed", "modal"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            pass
    if job.status == "cancelled":
        return None
    submitting = job.spec.get("execution_target") == "remote"
    identity = source_identity()
    payload: ExperimentRecord = {
        "runtime": {"python": platform.python_version(), "packages": packages},
        "schema": SCHEMA,
        "id": job.id,
        "name": job.name,
        "created_at": job.created_at,
        "settings": public_spec(dict(job.spec)),
        "source": None if submitting else identity,
        "submission_source": identity if submitting else None,
        "dataset": {
            "sample_id": job.spec.get("sample_id", ""),
            "files": files,
            "manifest_sha256": (
                hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest() if files else None
            ),
        },
        "command": job.command,
    }
    _write_record(output, payload)
    return payload


def record_remote_execution(output: Path, submission: ExperimentRecord, remote_id: str) -> None:
    """Combine the submission record with the actual renderer's execution evidence.

    Args:
        output: Experiment directory containing the extracted remote results.
        submission: Original local snapshot that owns the experiment.
        remote_id: Job id used by the execution workspace.
    """
    executed = _read_record(output, expected_id=remote_id) or {}
    dataset = executed.get("dataset") or {}
    submission["execution_id"] = remote_id
    submission["source"] = executed.get("source")
    if dataset.get("files"):
        submission["dataset"] = dataset
    if executed.get("runtime"):
        submission["runtime"] = executed["runtime"]
    _write_record(output, submission)


def portable_settings(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Strip account/host identity from an experiment imported on another machine.

    Args:
        settings: Recorded scientific and execution settings.

    Returns:
        Dataset/pipeline settings ready for deliberate compute selection.
    """
    cleaned = public_spec(dict(settings))
    for key in list(cleaned):
        if key.startswith(("ssh_", "modal_", "remote_", "dask_")) or key in {
            "cluster_config",
            "execution_target",
            "hardware",
            "input_worker",
            "dashboard_port",
            "dataset_dir",
            "images_dir",
        }:
            cleaned.pop(key)
    options = cleaned.get("loader_options")
    if isinstance(options, dict):
        cleaned["loader_options"] = {
            key: value
            for key, value in options.items()
            if not (
                key.endswith(("_path", "_fpath"))
                and isinstance(value, str)
                and (Path(value).is_absolute() or value.startswith("~") or len(value) > 1 and value[1] == ":")
            )
        }
    return cleaned


def validate_import(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a shared experiment without executing it.

    Args:
        payload: Uploaded experiment JSON.

    Returns:
        Safe pipeline settings; applying them does not start a run.
    """
    if payload.get("schema") != SCHEMA or not isinstance(payload.get("settings"), dict):
        raise ValueError("Choose a GTSfM experiment JSON file.")
    settings = portable_settings(payload["settings"])
    for key in ("name", "sample_id", "loader", "config_name", "splat_implementation"):
        if key in settings and not isinstance(settings[key], str):
            raise ValueError(f"The experiment's {key} must be text.")
    for key in ("loader_options", "hydra_overrides", "gaussian_splatting_overrides"):
        value = settings.get(key, {})
        if not isinstance(value, dict) or any(
            not isinstance(item, (str, int, float, bool, type(None))) for item in value.values()
        ):
            raise ValueError(f"The experiment's {key} must contain scalar settings.")
    overrides = settings.get("advanced_overrides", [])
    if (
        not isinstance(overrides, (str, list))
        or isinstance(overrides, list)
        and any(not isinstance(item, str) for item in overrides)
    ):
        raise ValueError("The experiment's advanced overrides must be lines of text.")
    return settings


def _metric_scalars(values: Mapping[str, Any], prefix: str = "") -> dict[str, float | int]:
    """Flatten scalar measurements and summaries, excluding raw sample arrays.

    Args:
        values: A measured metric group.
        prefix: Parent field path for nested summaries.

    Returns:
        Finite scalar observations keyed by their full field path.
    """
    result = {}
    for name, value in values.items():
        key = f"{prefix}/{name}" if prefix else name
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
            result[key] = value
        elif isinstance(value, dict) and len(key.split("/")) < 12:
            result.update(_metric_scalars(value, key))
    return result


def _json_configuration(value: Any) -> Any:
    """Represent YAML nonfinite numbers explicitly in downloadable JSON.

    Args:
        value: Resolved pipeline configuration.

    Returns:
        JSON-compatible configuration with nonfinite values written as text.
    """
    if isinstance(value, dict):
        return {str(key): _json_configuration(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_configuration(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def experiment_details(job: Any, scenes: list[dict[str, Any]]) -> dict[str, Any]:
    """Join one experiment's history, geometry, metrics and configuration.

    Args:
        job: Managed or recovered job.
        scenes: Available result catalog entries below its output directory.

    Returns:
        A single experiment view with portable provenance and honest missing states.
    """
    root = Path(job.output_root)
    provenance = _read_record(root, expected_id=job.id)
    if provenance is None:
        provenance = {
            "schema": SCHEMA,
            "id": job.id,
            "name": job.name,
            "settings": public_spec(dict(job.spec)),
            "source": None,
            "dataset": None,
            "note": "This older run did not record a source or dataset fingerprint.",
        }
    metrics: dict[str, Any] = {}
    for path in sorted(root.rglob("*metrics*.json")):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            data = json.loads(path.read_text())
            for group, values in data.items():
                if isinstance(values, dict):
                    parent = path.relative_to(root).parent.as_posix()
                    label = group if parent == "." else f"{parent}/{group}"
                    metrics[label] = _metric_scalars(values)
        except (OSError, ValueError, AttributeError):
            pass
    configuration = None
    for path in sorted(root.rglob("config.yaml"), key=lambda p: len(p.parts)):
        if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
            continue
        try:
            configuration = _json_configuration(yaml.safe_load(path.read_text()))
            break
        except (OSError, ValueError, yaml.YAMLError):
            pass
    preferred = min(
        scenes,
        key=lambda item: (
            len(Path(item["rel_path"]).parts),
            STAGE_RANK.get(Path(item["rel_path"]).name, len(STAGE_RANK)),
        ),
        default=None,
    )
    return {
        "job": job.public(),
        "provenance": provenance,
        "configuration": configuration,
        "metrics": metrics,
        "results": scenes,
        "preferred_result": preferred,
    }
