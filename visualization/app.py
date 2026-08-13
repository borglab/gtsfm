#!/usr/bin/env python3
"""FastAPI workspace server for GTSFM reconstruction and visualization."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import hmac
import json
import os
import shutil
import tarfile
import urllib.error
import uuid
from pathlib import Path, PurePosixPath
from typing import Annotated, Any
from urllib.parse import quote, urlparse

import uvicorn
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Request, UploadFile, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict
from starlette.websockets import WebSocketDisconnect

from visualization.runtime import (
    JobManager,
    configuration_schema,
    detect_hardware,
    install_optional_setup,
    setup_status,
)
from visualization.modal_deployment import ModalDeploymentManager, modal_remote_api_key
from visualization.samples import SampleDownloadError, prepare_sample, sample_catalog


PACKAGE_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = PACKAGE_ROOT / "static"
TEMPLATE_ROOT = PACKAGE_ROOT / "templates"
SPLAT_EXPORT_FORMATS = {"ply", "spz"}
IMAGE_SUFFIXES = {".avif", ".bmp", ".heic", ".heif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


class RemoteInspectRequest(BaseModel):
    """Credentials for inspecting an already-deployed remote workspace."""

    endpoint: str = ""
    api_key: str = ""
    remote_provider: str = "modal"


class ModalDiscoverRequest(BaseModel):
    """Modal account credentials used only for endpoint discovery."""

    token_id: str = ""
    token_secret: str = ""


class ModalDeployRequest(ModalDiscoverRequest):
    """Modal credentials and GPU choice for a managed deployment."""

    gpu: str = "L40S"
    cpu: float = 8
    memory_mb: int = 65536


class RunRequest(BaseModel):
    """Typed core fields plus provider/loader-specific run options."""

    model_config = ConfigDict(extra="allow")

    name: str = "my-scene"
    sample_id: str = ""
    dataset_dir: str = ""
    images_dir: str = ""
    loader: str
    config_name: str = "vggt"
    splat_implementation: str = "gsplat"
    execution_target: str = "local"
    hardware: str = "cpu"


def analyze_image_dataset(root: Path) -> dict[str, int | float]:
    """Read image headers and return lightweight inputs for runtime estimation."""

    image_count = 0
    image_bytes = 0
    total_pixels = 0
    max_width = 0
    max_height = 0
    if root.is_dir():
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            try:
                with Image.open(path) as image:
                    width, height = image.size
                if width <= 0 or height <= 0:
                    continue
                image_count += 1
                image_bytes += path.stat().st_size
                total_pixels += width * height
                max_width = max(max_width, width)
                max_height = max(max_height, height)
            except (OSError, UnidentifiedImageError):
                continue
    total_megapixels = total_pixels / 1_000_000
    return {
        "image_count": image_count,
        "image_bytes": image_bytes,
        "total_megapixels": round(total_megapixels, 2),
        "average_megapixels": round(total_megapixels / image_count, 2) if image_count else 0,
        "max_width": max_width,
        "max_height": max_height,
    }


def _extract_input_archive(archive_path: Path, destination: Path) -> tuple[int, int]:
    """Extract regular files from a tar archive without allowing path traversal or links."""

    file_count = 0
    total_bytes = 0
    with tarfile.open(archive_path, mode="r:*") as archive:
        for member in archive.getmembers():
            relative = PurePosixPath(member.name.replace("\\", "/"))
            if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise ValueError(f"Unsafe path in uploaded archive: {member.name}")
            target = destination.joinpath(*relative.parts)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                raise ValueError(f"Unsupported entry in uploaded archive: {member.name}")
            source = archive.extractfile(member)
            if source is None:
                raise ValueError(f"Unable to read uploaded archive entry: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("wb") as output:
                shutil.copyfileobj(source, output)
            file_count += 1
            total_bytes += target.stat().st_size
    return file_count, total_bytes


async def _discover_modal_endpoint(token_id: str, token_secret: str) -> dict[str, str]:
    """Find the deployed GTSFM web function in the token's default Modal environment."""

    from modal.client import _Client
    from modal.exception import AuthError, ConnectionError as ModalConnectionError, PermissionDeniedError
    from modal_proto import api_pb2

    client = None
    try:
        client = await _Client.from_credentials(token_id, token_secret)
        response = await client.stub.AppList(api_pb2.AppListRequest(environment_name=""))
        candidates: list[tuple[int, str, str, str]] = []
        live_states = {
            api_pb2.APP_STATE_DEPLOYED,
            api_pb2.APP_STATE_DETACHED,
            api_pb2.APP_STATE_EPHEMERAL,
            api_pb2.APP_STATE_INITIALIZING,
        }
        for deployed_app in response.apps:
            app_name = str(deployed_app.name or deployed_app.description or "")
            app_text = f"{deployed_app.name} {deployed_app.description}".lower()
            if "gtsfm" not in app_text or deployed_app.state not in live_states:
                continue
            layout = await client.stub.AppGetLayout(api_pb2.AppGetLayoutRequest(app_id=deployed_app.app_id))
            for item in layout.app_layout.objects:
                if not item.HasField("function_handle_metadata"):
                    continue
                metadata = item.function_handle_metadata
                url = str(metadata.web_url or "").rstrip("/")
                if not url.startswith("https://"):
                    continue
                function_name = str(metadata.function_name or "")
                function_text = function_name.lower()
                score = 10
                if any(word in function_text for word in ("workspace", "studio", "web", "app")):
                    score += 5
                if "gtsfm" in function_text:
                    score += 3
                candidates.append((score, url, app_name or "gtsfm", function_name or "web"))
        if not candidates:
            raise ValueError(
                "No deployed GTSFM web app was found in this Modal workspace. Deploy the GTSFM Modal app first, then try again."
            )
        _, endpoint, app_name, function_name = max(candidates, key=lambda item: item[0])
        return {"endpoint": endpoint, "app_name": app_name, "function_name": function_name}
    except (AuthError, PermissionDeniedError) as exc:
        raise ValueError("Modal rejected the token ID or secret.") from exc
    except ModalConnectionError as exc:
        raise ValueError("Unable to reach Modal while discovering the endpoint.") from exc
    finally:
        if client is not None:
            await client._close()


def _find_splats_file(scene_dir: Path, base_dir: Path) -> Path | None:
    current = scene_dir
    while True:
        candidate = current / "gaussian_splats.ply"
        if candidate.exists():
            try:
                candidate.relative_to(base_dir)
                return candidate
            except ValueError:
                return None
        if current == base_dir or current.parent == current:
            break
        current = current.parent
    return None


def find_scenes(base_dir: Path) -> list[dict[str, str | None]]:
    """Recursively find COLMAP reconstructions and standalone splat files."""

    scenes: list[dict[str, str | None]] = []
    splats_attached_to_scene: set[Path] = set()
    if not base_dir.exists():
        return scenes
    for points_file in base_dir.rglob("points3D.txt"):
        parent = points_file.parent
        images_file = parent / "images.txt"
        if not images_file.exists():
            continue
        rel = parent.relative_to(base_dir).as_posix()
        parts = rel.split("/")
        label = "/".join(parts[-3:]) if len(parts) >= 3 else rel
        splats_path = _find_splats_file(parent, base_dir)
        splats_url = None
        if splats_path is not None:
            splats_attached_to_scene.add(splats_path)
            splats_url = f"/data/{quote(splats_path.relative_to(base_dir).as_posix())}"
        scenes.append(
            {
                "kind": "scene",
                "label": label,
                "rel_path": rel,
                "points": f"/data/{quote(rel)}/points3D.txt",
                "images": f"/data/{quote(rel)}/images.txt",
                "splats": splats_url,
                "splat_rel_path": splats_path.relative_to(base_dir).as_posix() if splats_path else None,
            }
        )
    for splats_file in base_dir.rglob("gaussian_splats.ply"):
        if splats_file in splats_attached_to_scene or ".gtsfm" in splats_file.parts:
            continue
        rel_splat = splats_file.relative_to(base_dir).as_posix()
        scenes.append(
            {
                "kind": "splat",
                "label": rel_splat,
                "rel_path": rel_splat,
                "points": None,
                "images": None,
                "splats": f"/data/{quote(rel_splat)}",
                "splat_rel_path": rel_splat,
            }
        )
    scenes.sort(key=lambda item: item["rel_path"] or "")
    return scenes


def _download_filename(value: str) -> str:
    safe = "".join(character if character.isalnum() or character in {"-", "_", "."} else "-" for character in value)
    return safe.strip("-._") or "gaussian-splats"


def _export_splat(source: Path, export_root: Path, export_format: str, name: str) -> FileResponse:
    """Return a PLY directly or convert it to a cached SPZ export."""

    normalized_format = export_format.lower().lstrip(".")
    if normalized_format not in SPLAT_EXPORT_FORMATS:
        raise HTTPException(status_code=400, detail=f"Unsupported splat format: {export_format}")
    if not source.is_file() or source.suffix.lower() != ".ply":
        raise HTTPException(status_code=404, detail="Gaussian splat PLY was not found")

    filename = f"{_download_filename(name)}.{normalized_format}"
    if normalized_format == "ply":
        return FileResponse(source, media_type="application/octet-stream", filename=filename)

    source_stat = source.stat()
    cache_key = hashlib.sha256(
        f"{source.resolve()}:{source_stat.st_mtime_ns}:{source_stat.st_size}".encode("utf-8")
    ).hexdigest()[:20]
    export_root.mkdir(parents=True, exist_ok=True)
    destination = export_root / f"{cache_key}.spz"
    if not destination.exists():
        temporary = export_root / f".{cache_key}-{uuid.uuid4().hex}.spz"
        try:
            import spz

            unpack_options = spz.UnpackOptions()
            unpack_options.to_coord = spz.CoordinateSystem.RUB
            cloud = spz.load_splat_from_ply(str(source), unpack_options)
            if cloud.num_points <= 0:
                raise ValueError("The PLY contains no Gaussian splats")
            pack_options = spz.PackOptions()
            pack_options.from_coord = spz.CoordinateSystem.RUB
            if not spz.save_spz(cloud, pack_options, str(temporary)):
                raise ValueError("The SPZ encoder did not produce an export")
            temporary.replace(destination)
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            temporary.unlink(missing_ok=True)
            raise HTTPException(status_code=500, detail=f"Unable to create SPZ export: {exc}") from exc

    return FileResponse(destination, media_type="application/octet-stream", filename=filename)


def _live_state(manager: JobManager, resolved_base: Path, job_id: str) -> dict[str, object]:
    job = manager.get(job_id)
    if job is None:
        raise KeyError(job_id)
    status_path = manager.live_root(job_id) / "status.json"
    payload: dict[str, object] = {
        "job_id": job_id,
        "job_status": job.status,
        "stage": "pipeline",
        "progress": None,
    }
    if status_path.exists():
        try:
            loaded = json.loads(status_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
    dask_path = manager.live_root(job_id) / "dask.json"
    if dask_path.exists():
        try:
            dask_status = json.loads(dask_path.read_text(encoding="utf-8"))
            if isinstance(dask_status, dict):
                payload["dask"] = dask_status
        except (OSError, json.JSONDecodeError):
            pass
    preview_path = manager.live_root(job_id) / "live_splats.ply"
    if preview_path.exists():
        version = preview_path.stat().st_mtime_ns
        payload["preview_url"] = f"/api/jobs/{job_id}/live-splats?v={version}"
        payload["preview_version"] = version
    final_files = list(Path(job.output_root).rglob("gaussian_splats.ply"))
    if final_files:
        rel = final_files[0].relative_to(resolved_base).as_posix()
        payload["final_url"] = f"/data/{quote(rel)}"
    return payload


def _websocket_authorized(websocket: WebSocket) -> bool:
    expected = os.environ.get("GTSFM_API_KEY")
    if not expected:
        return True
    provided = websocket.query_params.get("token", "")
    return hmac.compare_digest(provided, expected)


def create_app(base_dir: Path | str = "results") -> FastAPI:
    """Create an isolated FastAPI workspace for ``base_dir``."""

    resolved_base = Path(base_dir).expanduser().resolve()
    manager = JobManager(resolved_base)
    modal_deployments = ModalDeploymentManager(
        lambda token_id, token_secret: asyncio.run(_discover_modal_endpoint(token_id, token_secret))
    )
    app = FastAPI(
        title="GTSFM Studio API",
        version="0.2.0",
        description="Local and remote Structure-from-Motion workspace API.",
    )
    app.state.gtsfm_jobs = manager
    app.state.modal_deployments = modal_deployments
    app.state.results_dir = resolved_base

    @app.middleware("http")
    async def add_response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(HTTPException)
    async def http_error(_request: Request, exc: HTTPException) -> JSONResponse:
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code, headers=exc.headers)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse({"error": "Invalid request", "details": exc.errors()}, status_code=422)

    def require_api_key(authorization: Annotated[str | None, Header()] = None) -> None:
        expected = os.environ.get("GTSFM_API_KEY")
        if expected and not hmac.compare_digest(authorization or "", f"Bearer {expected}"):
            raise HTTPException(status_code=401, detail="Unauthorized")

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((TEMPLATE_ROOT / "index.html").read_text(encoding="utf-8"))

    @app.get("/api/scenes")
    def list_scenes() -> dict[str, Any]:
        scenes = find_scenes(resolved_base)
        return {"base_dir": str(resolved_base), "count": len(scenes), "items": scenes}

    @app.get("/api/splats/export")
    def export_saved_splat(path: str, format: str = "ply") -> FileResponse:
        source = (resolved_base / path).resolve()
        try:
            source.relative_to(resolved_base)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Splat path is outside the results workspace") from exc
        return _export_splat(source, manager.runtime_root / "exports", format, source.parent.name)

    @app.get("/api/configuration")
    def get_configuration() -> dict[str, Any]:
        return configuration_schema()

    @app.get("/api/hardware")
    def get_hardware() -> dict[str, Any]:
        return detect_hardware()

    @app.get("/api/setup")
    def get_setup_status() -> dict[str, Any]:
        return setup_status(manager.results_root)

    @app.post("/api/setup/{check_id}/install", dependencies=[Depends(require_api_key)])
    def install_setup_check(check_id: str) -> dict[str, Any]:
        try:
            return install_optional_setup(check_id, manager.results_root)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    @app.get("/api/samples")
    def get_samples() -> dict[str, Any]:
        return {"items": sample_catalog(manager.runtime_root / "samples")}

    @app.post("/api/samples/{sample_id}/prepare", dependencies=[Depends(require_api_key)])
    def prepare_github_sample(sample_id: str) -> dict[str, Any]:
        try:
            prepared = prepare_sample(sample_id, manager.runtime_root / "samples")
            prepared["analysis"] = analyze_image_dataset(Path(prepared["path"]))
            return prepared
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Sample not found") from exc
        except SampleDownloadError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/uploads", dependencies=[Depends(require_api_key)])
    def upload_input_folder(
        files: Annotated[list[UploadFile], File()], manifest: Annotated[str, Form()] = "[]"
    ) -> dict[str, Any]:
        """Store a browser-selected input folder inside the managed workspace."""

        try:
            manifest_payload = json.loads(manifest)
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="The uploaded folder manifest is invalid") from exc
        if not files or not isinstance(manifest_payload, list) or len(files) != len(manifest_payload):
            raise HTTPException(status_code=400, detail="Drop a folder containing at least one file")

        relative_paths: list[PurePosixPath] = []
        for raw_path in manifest_payload:
            if not isinstance(raw_path, str):
                raise HTTPException(status_code=400, detail="The uploaded folder contains an invalid path")
            relative = PurePosixPath(raw_path.replace("\\", "/"))
            if relative.is_absolute() or not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
                raise HTTPException(status_code=400, detail=f"Unsafe path in uploaded folder: {raw_path}")
            relative_paths.append(relative)

        upload_id = uuid.uuid4().hex[:12]
        upload_root = manager.runtime_root / "uploads" / upload_id
        file_count = 0
        total_bytes = 0
        try:
            for uploaded_file, relative in zip(files, relative_paths, strict=True):
                destination = upload_root.joinpath(*relative.parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("wb") as output:
                    shutil.copyfileobj(uploaded_file.file, output)
                file_count += 1
                total_bytes += destination.stat().st_size
        except (OSError, ValueError) as exc:
            shutil.rmtree(upload_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Unable to store the input folder: {exc}") from exc
        finally:
            for uploaded_file in files:
                uploaded_file.file.close()

        common_root = relative_paths[0].parts[0]
        preserves_folder = all(len(path.parts) > 1 and path.parts[0] == common_root for path in relative_paths)
        dataset_root = upload_root / common_root if preserves_folder else upload_root
        return {
            "id": upload_id,
            "name": common_root if preserves_folder else "Selected files",
            "path": str(dataset_root),
            "file_count": file_count,
            "bytes": total_bytes,
            "analysis": analyze_image_dataset(dataset_root),
        }

    @app.post("/api/uploads/archive", dependencies=[Depends(require_api_key)])
    def upload_input_archive(archive: Annotated[UploadFile, File()]) -> dict[str, Any]:
        """Receive a streamed folder archive from another GTSFM workspace."""

        upload_id = uuid.uuid4().hex[:12]
        upload_root = manager.runtime_root / "uploads" / upload_id
        archive_path = manager.runtime_root / "uploads" / f"{upload_id}.tar"
        upload_root.mkdir(parents=True, exist_ok=True)
        try:
            with archive_path.open("wb") as output:
                shutil.copyfileobj(archive.file, output)
            file_count, total_bytes = _extract_input_archive(archive_path, upload_root)
            if not file_count:
                raise ValueError("The uploaded archive contains no files")
        except (OSError, tarfile.TarError, ValueError) as exc:
            shutil.rmtree(upload_root, ignore_errors=True)
            raise HTTPException(status_code=400, detail=f"Unable to store the input archive: {exc}") from exc
        finally:
            archive.file.close()
            archive_path.unlink(missing_ok=True)
        return {
            "id": upload_id,
            "name": Path(archive.filename or "dataset").stem,
            "path": str(upload_root),
            "file_count": file_count,
            "bytes": total_bytes,
            "analysis": analyze_image_dataset(upload_root),
        }

    @app.post("/api/remote/inspect")
    def inspect_remote(payload: RemoteInspectRequest) -> dict[str, Any]:
        endpoint = payload.endpoint.rstrip("/")
        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.scheme not in {"http", "https"} or not parsed_endpoint.netloc:
            raise HTTPException(status_code=400, detail="Enter a valid http(s) remote workspace URL")
        try:
            hardware = manager._remote_json(f"{endpoint}/api/hardware", payload.api_key)
            configuration = manager._remote_json(f"{endpoint}/api/configuration", payload.api_key)
            return {"hardware": hardware, "configuration": configuration}
        except (ValueError, urllib.error.URLError, TimeoutError) as exc:
            raise HTTPException(status_code=400, detail=f"Unable to inspect remote workspace: {exc}") from exc

    @app.post("/api/modal/discover", dependencies=[Depends(require_api_key)])
    async def discover_modal_endpoint(payload: ModalDiscoverRequest) -> dict[str, str]:
        if not payload.token_id.startswith("ak-") or not payload.token_secret.startswith("as-"):
            raise HTTPException(status_code=400, detail="Enter a valid Modal token ID and token secret")
        try:
            discovered = await _discover_modal_endpoint(payload.token_id, payload.token_secret)
            discovered["api_key"] = modal_remote_api_key(payload.token_id, payload.token_secret)
            return discovered
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/modal/deploy", status_code=202, dependencies=[Depends(require_api_key)])
    def deploy_modal_workspace(payload: ModalDeployRequest) -> dict[str, Any]:
        try:
            return modal_deployments.start(
                payload.token_id,
                payload.token_secret,
                payload.gpu,
                cpu=payload.cpu,
                memory_mb=payload.memory_mb,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/modal/deploy/{deployment_id}", dependencies=[Depends(require_api_key)])
    def modal_deployment_status(deployment_id: str) -> dict[str, Any]:
        deployment = modal_deployments.get(deployment_id)
        if deployment is None:
            raise HTTPException(status_code=404, detail="Modal deployment not found")
        return deployment

    @app.post("/api/modal/deploy/{deployment_id}/cancel", dependencies=[Depends(require_api_key)])
    def cancel_modal_deployment(deployment_id: str) -> dict[str, Any]:
        try:
            return modal_deployments.cancel(deployment_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Modal deployment not found") from exc

    @app.get("/api/jobs", dependencies=[Depends(require_api_key)])
    def list_jobs() -> dict[str, Any]:
        return {"items": manager.list()}

    @app.post("/api/jobs", status_code=202, dependencies=[Depends(require_api_key)])
    def start_job(payload: RunRequest) -> dict[str, Any]:
        spec = payload.model_dump()
        try:
            if spec.get("sample_id") and spec.get("execution_target") != "remote":
                spec["dataset_dir"] = prepare_sample(str(spec["sample_id"]), manager.runtime_root / "samples")["path"]
            return manager.start(spec)
        except (KeyError, SampleDownloadError, TypeError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/jobs/{job_id}/splat", dependencies=[Depends(require_api_key)])
    def download_job_splat(job_id: str, format: str = "ply") -> FileResponse:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        final_files = sorted(Path(job.output_root).rglob("gaussian_splats.ply"))
        if not final_files:
            raise HTTPException(status_code=404, detail="This run has no Gaussian splat result")
        return _export_splat(final_files[0], manager.runtime_root / "exports", format, job.name)

    @app.get("/api/jobs/{job_id}", dependencies=[Depends(require_api_key)])
    def get_job(job_id: str) -> dict[str, Any]:
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job.public()

    @app.post("/api/jobs/{job_id}/cancel", dependencies=[Depends(require_api_key)])
    def cancel_job(job_id: str) -> dict[str, Any]:
        try:
            return manager.cancel(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/api/jobs/{job_id}/live", dependencies=[Depends(require_api_key)])
    def get_live_state(job_id: str) -> dict[str, object]:
        try:
            return _live_state(manager, resolved_base, job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Job not found") from exc

    @app.get("/api/jobs/{job_id}/live-splats", dependencies=[Depends(require_api_key)])
    def get_live_splats(job_id: str) -> FileResponse:
        if manager.get(job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")
        preview_path = manager.live_root(job_id) / "live_splats.ply"
        if not preview_path.is_file():
            raise HTTPException(status_code=404, detail="Live splats are not ready")
        return FileResponse(preview_path, media_type="application/octet-stream")

    @app.websocket("/api/events/jobs")
    async def job_events(websocket: WebSocket) -> None:
        if not _websocket_authorized(websocket):
            await websocket.close(code=1008, reason="Unauthorized")
            return
        await websocket.accept()
        previous = ""
        try:
            while True:
                payload = {"items": manager.list()}
                fingerprint = json.dumps(payload, sort_keys=True, default=str)
                if fingerprint != previous:
                    await websocket.send_json(payload)
                    previous = fingerprint
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.5)
                except TimeoutError:
                    pass
        except WebSocketDisconnect:
            return

    @app.websocket("/api/events/jobs/{job_id}")
    async def live_job_events(websocket: WebSocket, job_id: str) -> None:
        if not _websocket_authorized(websocket):
            await websocket.close(code=1008, reason="Unauthorized")
            return
        if manager.get(job_id) is None:
            await websocket.close(code=1008, reason="Job not found")
            return
        await websocket.accept()
        previous = ""
        try:
            while True:
                job = manager.get(job_id)
                if job is None:
                    await websocket.close(code=1008, reason="Job not found")
                    return
                payload = {"job": job.public(), "live": _live_state(manager, resolved_base, job_id)}
                fingerprint = json.dumps(payload, sort_keys=True, default=str)
                if fingerprint != previous:
                    await websocket.send_json(payload)
                    previous = fingerprint
                try:
                    await asyncio.wait_for(websocket.receive_text(), timeout=0.35)
                except TimeoutError:
                    pass
        except WebSocketDisconnect:
            return

    @app.get("/data/{subpath:path}", include_in_schema=False)
    def serve_data(subpath: str) -> FileResponse:
        path = (resolved_base / subpath).resolve()
        try:
            path.relative_to(resolved_base)
        except ValueError as exc:
            raise HTTPException(status_code=403, detail="Path is outside the results workspace") from exc
        if not path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        return FileResponse(path)

    app.mount("/static", StaticFiles(directory=STATIC_ROOT), name="static")
    return app


app = create_app(Path(os.environ.get("RESULTS_DIR", "results")))


def main() -> None:
    parser = argparse.ArgumentParser(description="GTSFM browser workspace")
    parser.add_argument("--base", "-b", default="results", help="Results directory")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, default=5173)
    args = parser.parse_args()
    uvicorn.run(create_app(args.base), host=args.host, port=args.port, access_log=False, log_level="warning")


if __name__ == "__main__":
    main()
