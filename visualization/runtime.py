"""Runtime discovery and managed pipeline jobs for the browser workspace."""

from __future__ import annotations

import http.client
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import ssl
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urljoin, urlparse

import certifi
import yaml

import gtsfm

PACKAGE_ROOT = Path(gtsfm.__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT / "configs"
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
_REMOTE_REQUEST_TIMEOUT_SECONDS = 10 * 60
_RUN_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
_OPTIONAL_SUBMODULES = {
    "submodule-anysplat": {
        "directory": "AnySplat",
        "marker": Path("src/model/model/anysplat.py"),
        "url": "https://github.com/InternRobotics/AnySplat.git",
    },
    "submodule-fastvggt": {
        "directory": "FastVGGT",
        "marker": Path("README.md"),
        "url": "https://github.com/mystorm16/FastVGGT.git",
    },
    "submodule-mast3r": {
        "directory": "mast3r",
        "marker": Path("mast3r/model.py"),
        "url": "https://github.com/naver/mast3r.git",
    },
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _format_bytes(value: int | None) -> str | None:
    if not value:
        return None
    units = ("B", "KB", "MB", "GB", "TB")
    amount = float(value)
    for unit in units:
        if amount < 1024 or unit == units[-1]:
            return f"{amount:.1f} {unit}"
        amount /= 1024
    return None


def _machine_memory() -> int | None:
    try:
        if sys.platform == "darwin":
            return int(
                subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True, stderr=subprocess.DEVNULL).strip()
            )
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return int(pages * page_size)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _cpu_name() -> str:
    name = platform.processor().strip()
    if sys.platform == "darwin" and platform.machine() == "arm64" and name in {"", "arm"}:
        return "Apple Silicon"
    if name:
        return name
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        except subprocess.SubprocessError:
            pass
    return platform.machine() or "CPU"


@lru_cache(maxsize=1)
def detect_hardware() -> dict[str, Any]:
    """Return compute devices that the installed runtime can actually address."""

    devices: list[dict[str, Any]] = [
        {
            "id": "cpu",
            "kind": "cpu",
            "label": _cpu_name(),
            "status": "available",
            "memory": _format_bytes(_machine_memory()),
            "details": f"{os.cpu_count() or 1} logical cores",
            "supports_gaussian_splatting": False,
        }
    ]
    torch_version = None
    cuda_version = None
    accelerator_ids: set[str] = set()
    try:
        import torch

        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            backend = "rocm" if getattr(torch.version, "hip", None) else "cuda"
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                devices.append(
                    {
                        "id": f"{backend}:{index}",
                        "kind": backend,
                        "index": index,
                        "label": props.name,
                        "status": "available",
                        "memory": _format_bytes(props.total_memory),
                        "details": f"PyTorch {backend.upper()} device {index}",
                        "supports_gaussian_splatting": backend == "cuda",
                    }
                )
                accelerator_ids.add(f"{backend}:{index}")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            devices.append(
                {
                    "id": "mps",
                    "kind": "mps",
                    "label": "Apple Metal (MPS)",
                    "status": "available",
                    "memory": "shared",
                    "details": "Apple GPU through PyTorch MPS",
                    "supports_gaussian_splatting": False,
                }
            )
            accelerator_ids.add("mps")
    except (ImportError, RuntimeError):
        pass

    if sys.platform == "darwin" and platform.machine() == "arm64" and "mps" not in accelerator_ids:
        devices.append(
            {
                "id": "mps",
                "kind": "mps",
                "label": "Apple GPU (MPS)",
                "status": "PyTorch MPS unavailable",
                "memory": "shared",
                "details": "Detected Apple GPU; install an MPS-enabled PyTorch build to use it",
                "supports_gaussian_splatting": False,
            }
        )

    if not any(device["kind"] in {"cuda", "rocm"} for device in devices):
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=3,
            )
            for line in output.splitlines():
                index, name, memory_mb = (part.strip() for part in line.split(",", 2))
                devices.append(
                    {
                        "id": f"nvidia:{index}",
                        "kind": "nvidia",
                        "index": int(index),
                        "label": name,
                        "status": "PyTorch CUDA unavailable",
                        "memory": _format_bytes(int(memory_mb) * 1024 * 1024),
                        "details": "NVIDIA driver detected; install the CUDA PyTorch build to use it",
                        "supports_gaussian_splatting": False,
                    }
                )
        except (FileNotFoundError, ValueError, subprocess.SubprocessError):
            pass

    accelerators = sum(device["kind"] != "cpu" for device in devices)
    summary = f"{len(devices)} compute device{'s' if len(devices) != 1 else ''} detected"
    return {
        "summary": summary,
        "devices": devices,
        "accelerator_count": accelerators,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "torch": torch_version,
            "cuda": cuda_version,
        },
    }


def setup_status(results_root: Path) -> dict[str, Any]:
    """Inspect the local installation and report actionable setup checks."""

    checks: list[dict[str, Any]] = []

    def add_check(
        check_id: str,
        label: str,
        state: str,
        detail: str,
        *,
        required: bool = True,
        category: str = "runtime",
        action: dict[str, str | bool] | None = None,
    ) -> None:
        check: dict[str, Any] = {
            "id": check_id,
            "label": label,
            "state": state,
            "detail": detail,
            "required": required,
            "category": category,
        }
        if action is not None:
            check["action"] = action
        checks.append(check)

    python_supported = sys.version_info[:2] == (3, 12)
    add_check(
        "python",
        "Python runtime",
        "ready" if python_supported else "error",
        f"Python {platform.python_version()}" if python_supported else "GTSFM currently requires Python 3.12",
    )

    required_modules = {
        "gtsam": "GTSAM",
        "torch": "PyTorch",
        "fastapi": "FastAPI",
        "spz": "SPZ exporter",
    }
    missing_modules = [label for module, label in required_modules.items() if importlib.util.find_spec(module) is None]
    add_check(
        "dependencies",
        "Core packages",
        "ready" if not missing_modules else "error",
        "Required Python packages are installed" if not missing_modules else f"Missing: {', '.join(missing_modules)}",
    )

    workspace_ready = results_root.is_dir() and os.access(results_root, os.W_OK)
    add_check(
        "workspace",
        "Results workspace",
        "ready" if workspace_ready else "error",
        "Ready to store runs and uploaded datasets" if workspace_ready else "The results folder is not writable",
        category="system",
    )

    thirdparty_root = PACKAGE_ROOT.parent / "thirdparty"
    submodules = (
        ("vggt", "VGGT", Path("vggt/vggt/models/vggt.py"), True),
        ("lightglue", "LightGlue", Path("LightGlue/lightglue/lightglue.py"), True),
        ("anysplat", "AnySplat", Path("AnySplat/src/model/model/anysplat.py"), False),
        ("fastvggt", "FastVGGT", Path("FastVGGT/README.md"), False),
        ("mast3r", "MASt3R", Path("mast3r/mast3r/model.py"), False),
    )
    for check_id, label, marker, required in submodules:
        present = (thirdparty_root / marker).exists()
        action = None
        if not required and not present:
            git_available = shutil.which("git") is not None
            action = {
                "label": "Download",
                "enabled": git_available,
                "reason": (
                    "Download this extension from its official Git repository"
                    if git_available
                    else "Git is required to download this extension"
                ),
            }
        add_check(
            f"submodule-{check_id}",
            label,
            "ready" if present else ("error" if required else "optional"),
            (
                "Submodule installed"
                if present
                else ("Required submodule is missing" if required else "Optional extension is not installed")
            ),
            required=required,
            category="submodules",
            action=action,
        )

    graphviz_available = shutil.which("dot") is not None
    graphviz_command = _graphviz_install_command()
    add_check(
        "graphviz",
        "Graphviz",
        "ready" if graphviz_available else "optional",
        "Process-graph export available" if graphviz_available else "Optional process diagrams are disabled",
        required=False,
        category="system",
        action=(
            None
            if graphviz_available
            else {
                "label": "Install",
                "enabled": graphviz_command is not None,
                "reason": (
                    "Install Graphviz with the detected system package manager"
                    if graphviz_command
                    else "No supported system package manager was detected"
                ),
            }
        ),
    )

    hardware = detect_hardware()
    usable_devices = [device for device in hardware["devices"] if device.get("status") == "available"]
    accelerators = [device for device in usable_devices if device.get("kind") != "cpu"]
    add_check(
        "compute",
        "Compute devices",
        "ready" if usable_devices else "error",
        f"{len(usable_devices)} usable device{'s' if len(usable_devices) != 1 else ''}"
        + (
            f" · {len(accelerators)} accelerator{'s' if len(accelerators) != 1 else ''}"
            if accelerators
            else " · CPU mode"
        ),
        category="hardware",
    )

    cuda_splat_ready = any(device.get("supports_gaussian_splatting") for device in usable_devices)
    gsplat_installed = importlib.util.find_spec("gsplat") is not None
    add_check(
        "gaussian-splatting",
        "Gaussian splatting",
        "ready" if cuda_splat_ready and gsplat_installed else "optional",
        (
            "CUDA and gsplat are ready"
            if cuda_splat_ready and gsplat_installed
            else "Requires an NVIDIA CUDA environment; reconstruction is still available"
        ),
        required=False,
        category="hardware",
        action=(
            None
            if cuda_splat_ready and gsplat_installed
            else {
                "label": "Needs CUDA",
                "enabled": False,
                "reason": "An NVIDIA CUDA-capable GPU and driver cannot be installed by the workspace",
            }
        ),
    )

    error_count = sum(check["state"] == "error" for check in checks)
    warning_count = sum(check["state"] == "warning" for check in checks)
    optional_count = sum(check["state"] == "optional" for check in checks)
    overall = "error" if error_count else "warning" if warning_count else "ready"
    summary = "Setup needs attention" if error_count else "Ready to run"
    return {
        "status": overall,
        "summary": summary,
        "checked_at": _utc_now(),
        "counts": {
            "ready": sum(check["state"] == "ready" for check in checks),
            "warning": warning_count,
            "error": error_count,
            "optional": optional_count,
        },
        "items": checks,
    }


def _graphviz_install_command() -> list[str] | None:
    """Return a non-interactive Graphviz install command when one is available."""

    brew = shutil.which("brew")
    if brew:
        return [brew, "install", "graphviz"]
    apt_get = shutil.which("apt-get")
    if apt_get and hasattr(os, "geteuid") and os.geteuid() == 0:
        return [apt_get, "install", "-y", "graphviz"]
    return None


def _run_setup_command(command: list[str], *, cwd: Path | None = None, timeout: int = 1800) -> None:
    """Run one allow-listed setup command and surface a short actionable error."""

    try:
        completed = subprocess.run(
            command,
            cwd=str(cwd) if cwd else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("Setup timed out before it completed") from exc
    if completed.returncode != 0:
        output = (completed.stderr or completed.stdout or "Setup command failed").strip()
        raise RuntimeError(output[-2000:])


def install_optional_setup(check_id: str, results_root: Path) -> dict[str, Any]:
    """Install one allow-listed optional component, then return fresh setup state."""

    thirdparty_root = PACKAGE_ROOT.parent / "thirdparty"
    component = _OPTIONAL_SUBMODULES.get(check_id)
    if component is not None:
        git = shutil.which("git")
        if git is None:
            raise RuntimeError("Git is required to download this extension")
        destination = thirdparty_root / str(component["directory"])
        marker = destination / Path(component["marker"])
        if not marker.exists():
            if destination.exists():
                raise RuntimeError(f"{destination} exists but is incomplete; repair or remove it before retrying")
            thirdparty_root.mkdir(parents=True, exist_ok=True)
            temporary = thirdparty_root / f".{destination.name}-{uuid.uuid4().hex}.download"
            try:
                _run_setup_command(
                    [git, "clone", "--depth", "1", "--recurse-submodules", str(component["url"]), str(temporary)],
                    cwd=thirdparty_root,
                )
                temporary_marker = temporary / Path(component["marker"])
                if not temporary_marker.exists():
                    raise RuntimeError("The downloaded extension is missing its expected files")
                temporary.replace(destination)
            finally:
                shutil.rmtree(temporary, ignore_errors=True)
    elif check_id == "graphviz":
        command = _graphviz_install_command()
        if command is None:
            raise RuntimeError("No supported non-interactive package manager is available for Graphviz")
        if shutil.which("dot") is None:
            _run_setup_command(command)
    elif check_id == "gaussian-splatting":
        raise ValueError("Gaussian splatting requires NVIDIA CUDA hardware and drivers")
    else:
        raise ValueError("This setup check does not have an automatic action")

    status = setup_status(results_root)
    item = next((candidate for candidate in status["items"] if candidate["id"] == check_id), None)
    return {"item": item, "setup": status}


_CATALOG_YAML_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+\.yaml$")


def _yaml_catalog_paths(folder: Path) -> list[Path]:
    """Return canonical catalog YAMLs, excluding cloud/conflict copies.

    macOS cloud storage can create numbered copies such as ``vggt 2.yaml``.
    Those files are not GTSFM configurations and may be dataless placeholders,
    so attempting to read them can block the configuration endpoint indefinitely.
    """

    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.glob("*.yaml")
        if not path.name.startswith("_") and _CATALOG_YAML_NAME_PATTERN.fullmatch(path.name)
    )


def _yaml_stem_options(folder: Path) -> list[str]:
    if not folder.exists():
        return []
    return [path.stem for path in _yaml_catalog_paths(folder)]


def _display_name(value: str) -> str:
    """Format configuration identifiers for the user-facing catalog."""
    acronyms = {"api", "ba", "colmap", "gpu", "gs", "mvs", "sift", "vggt"}
    names = {"anysplat": "AnySplat", "megaloc": "MegaLoc"}
    return " ".join(
        names.get(token, token.upper() if token in acronyms else token.title()) for token in value.split("_")
    )


@lru_cache(maxsize=1)
def configuration_schema() -> dict[str, Any]:
    """Describe supported form fields and choices without importing the full runner."""

    root_configs = _yaml_stem_options(CONFIG_ROOT)
    excluded = {"cluster", "cluster_gpu", "local_scheduler_postgres_remote_cluster"}
    root_configs = [name for name in root_configs if name not in excluded]
    models: list[dict[str, Any]] = []
    for name in root_configs:
        data = yaml.safe_load((CONFIG_ROOT / f"{name}.yaml").read_text(encoding="utf-8")) or {}
        cluster_config = data.get("cluster_optimizer") or {}
        target = str(cluster_config.get("_target_") or "")
        if target.endswith(".Cacher"):
            cluster_config = cluster_config.get("optimizer") or {}
            target = str(cluster_config.get("_target_") or "")
        is_multiview = target.endswith(".Multiview") or target.endswith(".ClusterMVO")
        is_vggt = target.endswith(".ClusterVGGT")
        models.append(
            {
                "id": name,
                "label": _display_name(name),
                "capabilities": {
                    "iterative_splat": is_multiview or is_vggt,
                    "mvs": is_multiview,
                    "share_intrinsics": is_multiview,
                },
            }
        )
    loader_options: dict[str, list[dict[str, Any]]] = {}
    standard_loader_fields = {"_target_", "dataset_dir", "images_dir", "max_resolution", "input_worker"}
    for loader_path in _yaml_catalog_paths(CONFIG_ROOT / "loader"):
        data = yaml.safe_load(loader_path.read_text(encoding="utf-8")) or {}
        options: list[dict[str, Any]] = []
        for name, value in data.items():
            if name in standard_loader_fields:
                continue
            required = value == "???"
            if isinstance(value, bool):
                field_type = "boolean"
            elif isinstance(value, int):
                field_type = "integer"
            elif isinstance(value, float):
                field_type = "number"
            else:
                field_type = "string"
            options.append(
                {
                    "name": name,
                    "label": _display_name(name),
                    "type": field_type,
                    "required": required,
                    "default": None if required else value,
                }
            )
        loader_options[loader_path.stem] = options
    return {
        "models": models,
        "loaders": _yaml_stem_options(CONFIG_ROOT / "loader"),
        "loader_options": loader_options,
        "graph_partitioners": _yaml_stem_options(CONFIG_ROOT / "graph_partitioner"),
        "global_descriptors": _yaml_stem_options(CONFIG_ROOT / "global_descriptor"),
        "retrievers": _yaml_stem_options(CONFIG_ROOT / "retriever"),
        "correspondence_generators": _yaml_stem_options(CONFIG_ROOT / "correspondence"),
        "verifiers": _yaml_stem_options(CONFIG_ROOT / "verifier"),
        "log_levels": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        "gaussian_splatting_models": _yaml_stem_options(CONFIG_ROOT / "gaussian_splatting"),
        "splat_implementations": [
            {
                "id": "none",
                "label": "No splats",
                "description": "Run reconstruction only.",
                "live": False,
            },
            {
                "id": "gsplat",
                "label": "Optimized Gaussian splats",
                "description": "Iteratively optimize gsplat Gaussians and show live previews.",
                "live": True,
            },
            {
                "id": "anysplat",
                "label": "AnySplat",
                "description": "Generate splats with the feed-forward AnySplat model.",
                "live": False,
            },
        ],
        "defaults": {
            "config_name": "vggt",
            "loader": "olsson",
            "num_workers": 1,
            "threads_per_worker": 1,
            "worker_memory_limit": "32GB",
            "max_resolution": None,
            "splat_implementation": "gsplat",
            "run_gs": True,
            "run_mvs": False,
            "gs_max_steps": 7000,
            "live_preview_interval": 250,
            "hardware": "cpu",
        },
        "remote": {
            "supported": True,
            "description": "Connect to another GTSFM workspace API with its URL and API key.",
        },
    }


def _normalise_run_name(value: object) -> str:
    name = _RUN_NAME_PATTERN.sub("-", str(value or "run").strip()).strip("-._")
    return name[:64] or "run"


def _validate_choice(value: object, allowed: list[str], field_name: str) -> str:
    text = str(value or "")
    if text not in allowed:
        raise ValueError(f"Unknown {field_name}: {text}")
    return text


def build_runner_args(spec: Mapping[str, Any], output_root: Path) -> tuple[list[str], dict[str, str]]:
    """Validate a UI run request and translate it to the existing runner interface."""

    schema = configuration_schema()
    config_name = _validate_choice(spec.get("config_name"), [x["id"] for x in schema["models"]], "model")
    splat_implementation = str(spec.get("splat_implementation") or ("gsplat" if spec.get("run_gs") else "none"))
    _validate_choice(
        splat_implementation,
        [item["id"] for item in schema["splat_implementations"]],
        "splat implementation",
    )
    if splat_implementation == "anysplat":
        config_name = "anysplat"
    capabilities = {item["id"]: item["capabilities"] for item in schema["models"]}.get(config_name, {})
    if splat_implementation == "gsplat" and not capabilities.get("iterative_splat"):
        raise ValueError(f"The {config_name} model does not support iterative Gaussian splatting")
    loader = _validate_choice(spec.get("loader"), schema["loaders"], "loader")
    dataset_dir = Path(str(spec.get("dataset_dir") or "")).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")

    args = [
        "--config_name",
        config_name,
        "--loader",
        loader,
        "--dataset_dir",
        str(dataset_dir),
        "--output_root",
        str(output_root),
        "--num_workers",
        str(max(1, int(spec.get("num_workers", 1)))),
        "--threads_per_worker",
        str(max(1, int(spec.get("threads_per_worker", 1)))),
        "--worker_memory_limit",
        str(spec.get("worker_memory_limit") or "32GB"),
    ]

    images_dir = str(spec.get("images_dir") or "").strip()
    if images_dir:
        resolved_images = Path(images_dir).expanduser().resolve()
        if not resolved_images.is_dir():
            raise ValueError(f"Images directory does not exist: {resolved_images}")
        args.extend(["--images_dir", str(resolved_images)])

    input_worker = str(spec.get("input_worker") or "").strip()
    if input_worker:
        args.extend(["--input_worker", input_worker])

    max_resolution = spec.get("max_resolution")
    if max_resolution not in (None, ""):
        args.extend(["--max_resolution", str(max(1, int(max_resolution)))])

    graph_partitioner = str(spec.get("graph_partitioner") or "").strip()
    if graph_partitioner:
        graph_partitioner = _validate_choice(graph_partitioner, schema["graph_partitioners"], "graph partitioner")
        args.extend(["--graph_partitioner", graph_partitioner])

    loader_option_schema = {item["name"]: item for item in schema["loader_options"].get(loader, [])}
    loader_option_values = spec.get("loader_options") or {}
    if not isinstance(loader_option_values, Mapping):
        raise ValueError("Loader options must be an object")
    unknown_loader_options = set(loader_option_values) - set(loader_option_schema)
    if unknown_loader_options:
        raise ValueError(f"Unknown options for the {loader} loader: {', '.join(sorted(unknown_loader_options))}")
    for name, descriptor in loader_option_schema.items():
        value = loader_option_values.get(name)
        if value in (None, ""):
            if descriptor["required"]:
                raise ValueError(f"{descriptor['label']} is required for the {loader} loader")
            continue
        if descriptor["type"] == "boolean":
            encoded = "true" if value is True or str(value).lower() == "true" else "false"
        elif descriptor["type"] == "integer":
            encoded = str(int(value))
        elif descriptor["type"] == "number":
            encoded = str(float(value))
        else:
            encoded = json.dumps(str(value))
        args.append(f"loader.{name}={encoded}")

    optional_choices = {
        "global_descriptor_config_name": schema["global_descriptors"],
        "retriever_config_name": schema["retrievers"],
        "correspondence_generator_config_name": schema["correspondence_generators"],
        "verifier_config_name": schema["verifiers"],
        "log": schema["log_levels"],
    }
    for field_name, choices in optional_choices.items():
        value = str(spec.get(field_name) or "").strip()
        if value:
            _validate_choice(value, choices, field_name.replace("_", " "))
            args.extend([f"--{field_name}", value])

    for field_name in ("max_frame_lookahead", "num_matched", "num_retry_cluster_connection"):
        value = spec.get(field_name)
        if value not in (None, ""):
            args.extend([f"--{field_name}", str(max(0, int(value)))])
    for field_name in ("cluster_config", "dashboard_port", "dask_tmpdir"):
        value = str(spec.get(field_name) or "").strip()
        if value:
            args.extend([f"--{field_name}", value])
    if bool(spec.get("share_intrinsics")):
        if not capabilities.get("share_intrinsics"):
            raise ValueError(f"The {config_name} model does not expose shared-intrinsics bundle adjustment")
        args.append("--share_intrinsics")

    if bool(spec.get("run_mvs")):
        if not capabilities.get("mvs"):
            raise ValueError(f"The {config_name} model does not support the dense MVS stage")
        args.append("--run_mvs")
    if splat_implementation == "gsplat":
        args.extend(
            [
                "--run_gs",
                "--gaussian_splatting_config_name",
                _validate_choice(
                    spec.get("gaussian_splatting_config_name") or "base_gs",
                    schema["gaussian_splatting_models"],
                    "Gaussian splatting model",
                ),
                "--gs_max_steps",
                str(max(1, int(spec.get("gs_max_steps", 7000)))),
            ]
        )

    advanced_overrides = spec.get("advanced_overrides") or []
    if isinstance(advanced_overrides, str):
        advanced_overrides = [line.strip() for line in advanced_overrides.splitlines() if line.strip()]
    if not isinstance(advanced_overrides, list) or not all(isinstance(item, str) for item in advanced_overrides):
        raise ValueError("Advanced overrides must be one Hydra key=value expression per line")
    if any(item.startswith("--") or "\x00" in item for item in advanced_overrides):
        raise ValueError("Advanced overrides must use Hydra key=value syntax, not command-line flags")
    args.extend(advanced_overrides)

    hardware = str(spec.get("hardware") or "cpu")
    available = {item["id"]: item for item in detect_hardware()["devices"]}
    if hardware not in available:
        raise ValueError(f"Selected hardware is not available: {hardware}")
    if available[hardware]["status"] != "available":
        raise ValueError(f"Selected hardware is detected but unavailable: {available[hardware]['status']}")
    if splat_implementation in {"gsplat", "anysplat"} and not available[hardware]["supports_gaussian_splatting"]:
        raise ValueError("Gaussian splatting currently requires a CUDA-capable NVIDIA GPU")

    env: dict[str, str] = {"GTSFM_SELECTED_DEVICE": hardware}
    if hardware == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    elif hardware.startswith(("cuda:", "rocm:")):
        env["CUDA_VISIBLE_DEVICES"] = hardware.split(":", 1)[1]
    return args, env


@dataclass
class ManagedJob:
    id: str
    name: str
    status: str
    created_at: str
    updated_at: str
    output_root: str
    live_root: str
    spec: dict[str, Any]
    command: list[str]
    return_code: int | None = None
    error: str | None = None
    log_tail: list[str] = field(default_factory=list)
    pid: int | None = None
    remote: dict[str, Any] | None = None
    remote_api_key: str | None = field(default=None, repr=False, compare=False)
    remote_cancel_requested: bool = field(default=False, repr=False, compare=False)
    remote_cancel_dispatched: bool = field(default=False, repr=False, compare=False)
    process: subprocess.Popen[str] | None = field(default=None, repr=False, compare=False)

    def public(self) -> dict[str, Any]:
        payload = {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "output_root": self.output_root,
            "spec": self.spec,
            "command": self.command,
            "return_code": self.return_code,
            "error": self.error,
            "log_tail": list(self.log_tail),
            "pid": self.pid,
            "remote": self.remote,
        }
        payload["has_live_preview"] = (Path(self.live_root) / "live_splats.ply").exists()
        payload["has_final_splat"] = any(Path(self.output_root).rglob("gaussian_splats.ply"))
        return payload


class JobManager:
    """Own subprocesses started by one browser workspace process."""

    def __init__(self, results_root: Path) -> None:
        self.results_root = results_root.resolve()
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.runtime_root = self.results_root / ".gtsfm"
        self.runtime_root.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, ManagedJob] = {}
        self._lock = threading.RLock()

    def list(self) -> list[dict[str, Any]]:
        with self._lock:
            return [job.public() for job in sorted(self._jobs.values(), key=lambda item: item.created_at, reverse=True)]

    def get(self, job_id: str) -> ManagedJob | None:
        with self._lock:
            return self._jobs.get(job_id)

    def start(self, spec: Mapping[str, Any], *, job_id: str | None = None) -> dict[str, Any]:
        """Start a pipeline, optionally using a caller-provided durable job ID."""

        job_id = job_id or uuid.uuid4().hex[:12]
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", job_id):
            raise ValueError("Job ID contains unsupported characters")
        name = _normalise_run_name(spec.get("name"))
        run_root = self.results_root / "runs" / f"{name}-{job_id}"
        # Keep transient previews outside the results tree discovered by the viewer.
        live_root = self.runtime_root / "live" / job_id
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
        is_remote = str(spec.get("execution_target") or "local") == "remote"
        if is_remote:
            endpoint = str(spec.get("remote_endpoint") or "").rstrip("/")
            parsed_endpoint = urlparse(endpoint)
            if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
                raise ValueError("Enter a valid http(s) URL for the remote GTSFM workspace")
            api_key = str(spec.get("api_key") or "")
            if not api_key:
                raise ValueError("An API key is required for a remote run")
            args: list[str] = []
            device_env: dict[str, str] = {}
            command = ["remote", f"{endpoint}/api/jobs"]
        else:
            args, device_env = build_runner_args(spec, run_root)
            command = [sys.executable, "-m", "gtsfm.runner", *args]
        now = _utc_now()
        job = ManagedJob(
            id=job_id,
            name=name,
            status="queued",
            created_at=now,
            updated_at=now,
            output_root=str(run_root),
            live_root=str(live_root),
            spec=public_spec,
            command=command,
            remote={"endpoint": endpoint} if is_remote else None,
            remote_api_key=api_key if is_remote else None,
        )
        with self._lock:
            self._jobs[job_id] = job

        live_root.mkdir(parents=True, exist_ok=True)
        if is_remote:
            thread = threading.Thread(target=self._run_remote, args=(job,), daemon=True, name=f"gtsfm-remote-{job_id}")
            thread.start()
            return job.public()

        env = os.environ.copy()
        env.update(device_env)
        env["GTSFM_LIVE_DIR"] = str(live_root)
        env["GTSFM_LIVE_PREVIEW_INTERVAL"] = str(max(1, int(spec.get("live_preview_interval", 250))))
        thread = threading.Thread(target=self._run, args=(job, env), daemon=True, name=f"gtsfm-job-{job_id}")
        thread.start()
        return job.public()

    @staticmethod
    def _remote_json(url: str, api_key: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url,
            data=body,
            method="POST" if payload is not None else "GET",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        # Allocating a Modal GPU and loading the CUDA runtime can take more than
        # two minutes on the first request. Keep the socket open through that
        # cold start instead of launching a second competing verification.
        with urllib.request.urlopen(request, timeout=_REMOTE_REQUEST_TIMEOUT_SECONDS, context=_SSL_CONTEXT) as response:
            result = json.loads(response.read().decode("utf-8"))
        if not isinstance(result, dict):
            raise ValueError("Remote workspace returned an invalid response")
        return result

    @staticmethod
    def _remote_download(url: str, api_key: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"Authorization": f"Bearer {api_key}"})
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f"{destination.suffix}.part")
        try:
            with (
                urllib.request.urlopen(request, timeout=300, context=_SSL_CONTEXT) as response,
                temporary.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _remote_upload_directory(endpoint: str, api_key: str, directory: Path, runtime_root: Path) -> dict[str, Any]:
        """Stream a local directory as a tar archive to a remote workspace."""

        source = directory.expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Dataset directory does not exist: {source}")
        runtime_root.mkdir(parents=True, exist_ok=True)
        archive_file = tempfile.NamedTemporaryFile(
            prefix="gtsfm-transfer-", suffix=".tar", dir=runtime_root, delete=False
        )
        archive_path = Path(archive_file.name)
        archive_file.close()
        try:
            file_count = 0
            with tarfile.open(archive_path, mode="w") as archive:
                for path in sorted(source.rglob("*")):
                    if not path.is_file():
                        continue
                    try:
                        path.resolve().relative_to(source)
                    except ValueError:
                        continue
                    archive.add(path, arcname=path.relative_to(source).as_posix(), recursive=False)
                    file_count += 1
            if file_count == 0:
                raise ValueError(f"Dataset directory contains no transferable files: {source}")

            parsed = urlparse(endpoint)
            if parsed.scheme == "https":
                connection: http.client.HTTPConnection = http.client.HTTPSConnection(
                    parsed.hostname, parsed.port or 443, timeout=3600, context=_SSL_CONTEXT
                )
            elif parsed.scheme == "http":
                connection = http.client.HTTPConnection(parsed.hostname, parsed.port or 80, timeout=3600)
            else:
                raise ValueError("Remote workspace URL must use http or https")
            boundary = f"gtsfm-{uuid.uuid4().hex}"
            before = (
                f"--{boundary}\r\n"
                'Content-Disposition: form-data; name="archive"; filename="dataset.tar"\r\n'
                "Content-Type: application/x-tar\r\n\r\n"
            ).encode("utf-8")
            after = f"\r\n--{boundary}--\r\n".encode("utf-8")
            request_path = f"{parsed.path.rstrip('/')}/api/uploads/archive"
            connection.putrequest("POST", request_path)
            connection.putheader("Authorization", f"Bearer {api_key}")
            connection.putheader("Content-Type", f"multipart/form-data; boundary={boundary}")
            connection.putheader("Content-Length", str(len(before) + archive_path.stat().st_size + len(after)))
            connection.endheaders()
            connection.send(before)
            with archive_path.open("rb") as archive_stream:
                while chunk := archive_stream.read(1024 * 1024):
                    connection.send(chunk)
            connection.send(after)
            response = connection.getresponse()
            body = response.read()
            connection.close()
            try:
                result = json.loads(body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"Remote workspace returned an invalid upload response ({response.status})") from exc
            if response.status < 200 or response.status >= 300:
                message = result.get("error") if isinstance(result, dict) else None
                raise ValueError(str(message or f"Remote dataset upload failed with HTTP {response.status}"))
            if not isinstance(result, dict) or not result.get("path"):
                raise ValueError("Remote workspace did not return an uploaded dataset path")
            return result
        finally:
            archive_path.unlink(missing_ok=True)

    def _run_remote(self, job: ManagedJob) -> None:
        assert job.remote is not None and job.remote_api_key is not None
        endpoint = str(job.remote["endpoint"])
        try:
            with self._lock:
                if job.status == "cancelled":
                    return
            remote_spec = dict(job.spec)
            if remote_spec.get("sample_id"):
                remote_spec["dataset_dir"] = ""
                remote_spec["images_dir"] = ""
            else:
                for field_name in ("dataset_dir", "images_dir"):
                    with self._lock:
                        if job.status == "cancelled":
                            return
                    local_path = str(remote_spec.get(field_name) or "")
                    if not local_path:
                        continue
                    with self._lock:
                        job.log_tail.append(f"Uploading {field_name.replace('_', ' ')} to the remote workspace…")
                        job.status = "running"
                        job.updated_at = _utc_now()
                    uploaded = self._remote_upload_directory(
                        endpoint,
                        job.remote_api_key,
                        Path(local_path),
                        self.runtime_root / "transfers",
                    )
                    remote_spec[field_name] = str(uploaded["path"])
            with self._lock:
                if job.status == "cancelled":
                    return
                job.status = "running"
                job.updated_at = _utc_now()
                job.log_tail.append("Starting the remote GPU workspace; the first run may take several minutes…")
            remote_spec["execution_target"] = "local"
            created = self._remote_json(f"{endpoint}/api/jobs", job.remote_api_key, payload=remote_spec)
            remote_id = str(created["id"])
            with self._lock:
                cancelled = job.status == "cancelled"
                job.remote.update(
                    {
                        "job_id": remote_id,
                        "workspace_url": f"{endpoint}/?view=results",
                    }
                )
                if cancelled:
                    job.log_tail.append("Modal accepted the job. Sending the pending cancellation request…")
                if not cancelled:
                    job.status = str(created.get("status", "queued"))
                job.updated_at = _utc_now()
            if cancelled:
                self._dispatch_remote_cancel(job, background=False)
                return
            preview_version: object = None
            remote_status = str(created.get("status", "queued"))
            while remote_status in {"queued", "running"}:
                time.sleep(1)
                state = self._remote_json(f"{endpoint}/api/jobs/{remote_id}", job.remote_api_key)
                remote_status = str(state.get("status", "running"))
                try:
                    live = self._remote_json(f"{endpoint}/api/jobs/{remote_id}/live", job.remote_api_key)
                    live_path = Path(job.live_root) / "status.json"
                    live_path.parent.mkdir(parents=True, exist_ok=True)
                    live_temporary = live_path.with_suffix(".json.tmp")
                    live_temporary.write_text(json.dumps(live), encoding="utf-8")
                    live_temporary.replace(live_path)
                    next_version = live.get("preview_version")
                    preview_url = str(live.get("preview_url") or "")
                    if preview_url and next_version != preview_version:
                        self._remote_download(
                            urljoin(f"{endpoint}/", preview_url.lstrip("/")),
                            job.remote_api_key,
                            Path(job.live_root) / "live_splats.ply",
                        )
                        preview_version = next_version
                except (ValueError, urllib.error.URLError, TimeoutError):
                    pass
                with self._lock:
                    if job.status == "cancelled":
                        break
                    job.status = remote_status if remote_status in {"queued", "running"} else "running"
                    job.log_tail = list(state.get("log_tail") or [])[-250:]
                    job.error = state.get("error")
                    job.return_code = state.get("return_code")
                    job.updated_at = _utc_now()
            if job.status != "cancelled" and remote_status == "completed":
                if str(job.spec.get("splat_implementation") or "none") != "none":
                    self._remote_download(
                        f"{endpoint}/api/jobs/{remote_id}/splat?format=ply",
                        job.remote_api_key,
                        Path(job.output_root) / "gaussian_splats.ply",
                    )
                with self._lock:
                    job.status = "completed"
                    job.return_code = 0
                    job.updated_at = _utc_now()
            elif job.status != "cancelled":
                with self._lock:
                    job.status = remote_status
                    job.updated_at = _utc_now()
        except (KeyError, ValueError, urllib.error.URLError, TimeoutError) as exc:
            with self._lock:
                if job.status != "cancelled":
                    job.status = "failed"
                    job.error = f"Remote run failed: {exc}"
                    job.updated_at = _utc_now()

    def _run(self, job: ManagedJob, env: dict[str, str]) -> None:
        log_path = self.runtime_root / f"{job.id}.log"
        try:
            with self._lock:
                if job.status == "cancelled":
                    return
            process = subprocess.Popen(
                job.command,
                cwd=os.getcwd(),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            with self._lock:
                job.process = process
                job.pid = process.pid
                cancelled = job.status == "cancelled"
                if not cancelled:
                    job.status = "running"
                job.updated_at = _utc_now()
            if cancelled:
                self._terminate_process(process)
            with log_path.open("w", encoding="utf-8") as log_file:
                assert process.stdout is not None
                for line in process.stdout:
                    text = line.rstrip("\n")
                    log_file.write(line)
                    log_file.flush()
                    with self._lock:
                        job.log_tail.append(text)
                        del job.log_tail[:-250]
                        job.updated_at = _utc_now()
            return_code = process.wait()
            with self._lock:
                job.return_code = return_code
                if job.status != "cancelled":
                    job.status = "completed" if return_code == 0 else "failed"
                    if return_code != 0:
                        job.error = f"Pipeline exited with code {return_code}"
                job.updated_at = _utc_now()
        except Exception as exc:  # keep the web server alive if process launch fails
            with self._lock:
                if job.status != "cancelled":
                    job.status = "failed"
                    job.error = str(exc)
                    job.updated_at = _utc_now()

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        """Terminate a local pipeline process tree and force-kill it if needed."""

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
                except OSError:
                    pass

        threading.Thread(target=force_kill, daemon=True, name="gtsfm-job-cancel").start()

    def _dispatch_remote_cancel(self, job: ManagedJob, *, background: bool) -> bool:
        """Send one cancellation request to the remote workspace once its job ID exists."""

        with self._lock:
            remote = job.remote
            remote_id = str(remote.get("job_id") or "") if remote else ""
            if (
                not job.remote_cancel_requested
                or job.remote_cancel_dispatched
                or not remote
                or not remote_id
                or not job.remote_api_key
            ):
                return False
            job.remote_cancel_dispatched = True
            remote["cancel_status"] = "requested"
            endpoint = str(remote["endpoint"])
            api_key = job.remote_api_key

        def send_cancel() -> None:
            try:
                self._remote_json(f"{endpoint}/api/jobs/{remote_id}/cancel", api_key, payload={})
            except (urllib.error.URLError, ValueError, TimeoutError) as exc:
                with self._lock:
                    if job.remote is not None:
                        job.remote["cancel_status"] = "failed"
                    job.log_tail.append(f"Unable to confirm Modal cancellation: {exc}")
                    del job.log_tail[:-250]
                    job.updated_at = _utc_now()
            else:
                with self._lock:
                    if job.remote is not None:
                        job.remote["cancel_status"] = "confirmed"
                    job.log_tail.append("Modal job cancellation confirmed.")
                    del job.log_tail[:-250]
                    job.updated_at = _utc_now()

        if background:
            threading.Thread(
                target=send_cancel,
                daemon=True,
                name=f"gtsfm-remote-cancel-{job.id}",
            ).start()
        else:
            send_cancel()
        return True

    def cancel(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.status not in {"queued", "running", "cancelled"}:
                return job.public()
            if job.status != "cancelled":
                job.status = "cancelled"
                job.updated_at = _utc_now()
                job.log_tail.append("Cancellation requested.")
            process = job.process
            if job.remote is not None:
                job.remote_cancel_requested = True
                if job.remote.get("job_id"):
                    job.remote["cancel_status"] = "requested"
                else:
                    job.remote["cancel_status"] = "waiting_for_remote_job"
                    job.log_tail.append(
                        "Waiting for Modal to accept the starting job; it will be cancelled immediately afterward."
                    )
            del job.log_tail[:-250]
        if process is not None and process.poll() is None:
            self._terminate_process(process)
        elif job.remote is not None:
            # Do not make the browser wait on a network round-trip. If Modal is
            # still cold-starting, _run_remote dispatches this as soon as the
            # upstream job ID becomes available.
            self._dispatch_remote_cancel(job, background=True)
        return job.public()

    def live_root(self, job_id: str) -> Path:
        return self.runtime_root / "live" / job_id
