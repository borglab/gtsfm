"""Check a real Studio run through its HTTP API, without submitting any computation.

Start the experiment in Studio, then pass its ID. This checks the returned
geometry, every source thumbnail/preview, provenance, result archive and record
round-trip. Browser rendering and native provider cleanup are separate checks.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image


class Studio:
    """Read a running Studio API using an optional host-local bearer token."""

    def __init__(self, url: str) -> None:
        self.url = url.rstrip("/")
        self.token = os.environ.get("GTSFM_STUDIO_API_KEY", "")

    def read(self, path: str, payload: dict | None = None) -> bytes:
        """Fetch one same-origin resource, or submit an experiment-record import.

        Args:
            path: Absolute API or data path on this Studio host.
            payload: Optional JSON request body.

        Returns:
            Response bytes.
        """
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Expected a same-origin Studio resource")
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode() if payload is not None else None,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.read()

    def json(self, path: str, payload: dict | None = None) -> Any:
        """Read a JSON response.

        Args:
            path: Same-origin resource path.
            payload: Optional JSON body.

        Returns:
            Parsed response.
        """
        return json.loads(self.read(path, payload))


def require(condition: Any, message: str) -> None:
    """Fail the verification if an observed condition is false.

    Args:
        condition: Observed result to check.
        message: Actionable failure description.
    """
    if not condition:
        raise RuntimeError(message)


def verify(studio: Studio, args: argparse.Namespace, report: dict) -> None:
    """Verify a completed real run and record its evidence.

    Args:
        studio: Running workspace API.
        args: CLI acceptance criteria.
        report: Evidence updated in place, including on failure.
    """
    prefix = f"/api/jobs/{args.job_id}"
    deadline = time.monotonic() + args.timeout
    while True:
        job = studio.json(prefix)
        report["status"] = job["status"]
        if job["status"] not in {"queued", "running"}:
            break
        require(time.monotonic() < deadline, "Run has not finished before the verification deadline")
        time.sleep(2)
    require(job["status"] == "completed", f"Run ended as {job['status']}: {job.get('error')}")
    details = studio.json(f"/api/experiments/{args.job_id}")
    (args.output.parent / f"{args.job_id}-experiment.json").write_text(json.dumps(details, indent=2))
    scene = details["preferred_result"]
    require(scene is not None, "No reconstruction returned to this Studio host")
    require(
        scene["camera_count"] == args.expected_cameras, "Reconstruction camera count differs from acceptance target"
    )
    require(scene["point_count"] > 0, "Reconstruction contains no points")
    report["geometry"] = {k: scene[k] for k in ("rel_path", "camera_count", "point_count")}
    resources = {}
    for kind in ("points", "images"):
        require(scene.get(kind), f"Missing {kind} resource")
        content = studio.read(scene[kind])
        require(content.strip(), f"Empty {kind} resource")
        resources[kind] = hashlib.sha256(content).hexdigest()
    provenance = details["provenance"]
    require((provenance.get("source") or {}).get("fingerprint"), "Execution source fingerprint is missing")
    dataset = provenance.get("dataset") or {}
    require(dataset.get("manifest_sha256") and dataset.get("files"), "Input content manifest is missing")
    require(details.get("configuration"), "Resolved configuration is missing")
    require(details.get("metrics"), "Measured metrics are missing")
    report["source"] = provenance["source"]
    report["dataset_manifest_sha256"] = dataset["manifest_sha256"]
    report["geometry_sha256"] = resources
    photos = studio.json("/api/photos?" + urllib.parse.urlencode({"scene": scene["rel_path"]}))["items"]
    require(len(photos) == args.expected_cameras, "Photo manifest differs from camera count")
    photo_evidence = []
    for photo in photos:
        item = {"name": photo["name"]}
        for kind, maximum in (("thumbnail", 256), ("preview", 1600)):
            require(photo.get(kind), f"Missing {kind} for {photo['name']}")
            data = studio.read(photo[kind])
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                require(image.format == "JPEG", f"Unexpected {kind} encoding")
                require(0 < max(image.size) <= maximum, f"{kind} exceeds {maximum}px")
                item[kind] = {"size": image.size, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        photo_evidence.append(item)
    report["photos"] = photo_evidence
    archive_data = studio.read(prefix + "/results?include_images=true")
    archive_path = args.output.parent / f"{args.job_id}-results.tar"
    archive_path.write_bytes(archive_data)
    with tarfile.open(fileobj=io.BytesIO(archive_data)) as archive:
        members = [m for m in archive if m.isfile()]
        require(any(Path(m.name).name == "points3D.txt" for m in members), "Result archive has no geometry")
        for kind, filename in (("points", "points3D.txt"), ("images", "images.txt")):
            hashes = {
                hashlib.sha256(archive.extractfile(member).read()).hexdigest()
                for member in members
                if Path(member.name).name == filename
            }
            require(resources[kind] in hashes, f"Downloaded archive differs from displayed {kind}")
        originals = [m for m in members if m.name.startswith("source-images/")]
        require(len(originals) >= args.expected_cameras, "Result archive is missing source images")
        if args.require_splats:
            require(any(Path(m.name).name == "gaussian_splats.ply" for m in members), "Archive is missing final splats")
    report["archive"] = {"bytes": len(archive_data), "sha256": hashlib.sha256(archive_data).hexdigest()}
    if args.require_splats:
        splats = studio.read(prefix + "/splat")
        if scene.get("splats"):
            require(
                splats == studio.read(scene["splats"]), "Splat download differs from the preferred result's splat view"
            )
        header = splats[:8192].decode("ascii", errors="replace")
        match = re.search(r"element vertex (\d+)", header)
        require(match and int(match[1]) > 0, "Downloaded splat file contains no vertices")
        report["splats"] = {"vertices": int(match[1]), "bytes": len(splats)}
    exported = studio.json(f"/api/experiments/{args.job_id}/export")
    before = {j["id"] for j in studio.json("/api/jobs")["items"]}
    imported = studio.json("/api/experiments/import", exported)["settings"]
    after = {j["id"] for j in studio.json("/api/jobs")["items"]}
    require(before == after, "Import coincided with a new job; repeat in an isolated workspace")
    forbidden = {"dataset_dir", "images_dir", "hardware", "execution_target", "api_key"}
    require(not forbidden.intersection(imported), "Import retained machine-local settings")
    require(not any(k.startswith(("ssh_", "modal_", "remote_")) for k in imported), "Import retained account settings")
    require(imported["config_name"] == job["spec"]["config_name"], "Import changed the pipeline")
    report["record_round_trip"] = {"new_jobs": 0, "pipeline_preserved": True, "machine_settings_removed": True}


def main() -> None:
    """Parse acceptance criteria and save a report even when verification fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-url", default="http://127.0.0.1:5173")
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--expected-cameras", type=int, required=True)
    parser.add_argument("--require-splats", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", args.job_id):
        parser.error("Invalid Studio job ID")
    if args.expected_cameras < 1 or args.timeout <= 0:
        parser.error("Expected cameras and timeout must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"job_id": args.job_id, "checked_at": datetime.now(timezone.utc).isoformat(), "passed": False}
    try:
        verify(Studio(args.studio_url), args, report)
        report["passed"] = True
        print(f"PASS {args.job_id}: geometry, photos, provenance, archive and record round-trip")
    except Exception as exc:
        report["error"] = str(exc)
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
