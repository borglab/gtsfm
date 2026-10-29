"""Exercise a locally saved Studio SSH configuration against real remote computation.

Run against a disposable Studio results workspace. No requests or pipeline outputs
are mocked. This is an output-equivalence smoke test, not a performance benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def request(url: str, payload: dict | None = None) -> dict:
    token = os.environ.get("GTSFM_STUDIO_API_KEY", "")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode() if payload is not None else None, headers=headers
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio-url", default="http://127.0.0.1:5173")
    parser.add_argument("--configuration", required=True, help="Name saved in Studio on this machine")
    parser.add_argument("--dataset", type=Path, required=True, help="Local input directory to stage over SSH")
    parser.add_argument("--output", type=Path, required=True, help="Evidence JSON outside the source tree")
    parser.add_argument("--reference", type=Path, help="A previous report from this script for exact export comparison")
    parser.add_argument("--config-name", default="vggt_sift_frontend_megaloc")
    parser.add_argument("--loader", default="olsson")
    parser.add_argument("--max-resolution", type=int, default=518)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("repeats and timeout must be positive")
    url = args.studio_url.rstrip("/")
    configurations = request(f"{url}/api/execution-configurations")["items"]
    settings = next((item["settings"] for item in configurations if item["name"] == args.configuration), None)
    if not settings or settings["execution_target"] != "remote" or settings["remote_connection"] != "ssh":
        parser.error("Select an existing SSH execution configuration")
    dataset = args.dataset.expanduser().resolve()
    if not dataset.is_dir():
        parser.error("dataset must be a local directory")
    input_hashes = {str(p.relative_to(dataset)): sha256(p) for p in sorted(dataset.rglob("*")) if p.is_file()}
    reference = json.loads(args.reference.read_text())["trials"][0]["exports"] if args.reference else None
    report = {"configuration": args.configuration, "inputs": input_hashes, "trials": [], "passed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    active = None
    try:
        for index in range(args.repeats):
            spec = dict(
                settings,
                name=f"ssh-verification-{index + 1}",
                dataset_dir=str(dataset),
                config_name=args.config_name,
                loader=args.loader,
                max_resolution=args.max_resolution,
                hardware=settings["remote_hardware"],
                splat_implementation="none",
            )
            started = time.monotonic()
            active = request(f"{url}/api/jobs", spec)["id"]
            deadline = started + args.timeout
            while True:
                state = request(f"{url}/api/jobs/{active}")
                if state["status"] not in {"queued", "running"}:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"Run {active} exceeded the verification deadline")
                time.sleep(0.5)
            if state["status"] != "completed":
                raise RuntimeError(f"Run {active}: {state['status']}: {state.get('error')}")
            output = Path(state["output_root"])
            exports = {
                str(p.relative_to(output)): sha256(p)
                for p in sorted(output.rglob("*"))
                if p.is_file() and p.name in {"cameras.txt", "images.txt", "points3D.txt", "similarity_named_pairs.txt"}
            }
            if not any(name.endswith("points3D.txt") for name in exports):
                raise RuntimeError("The remote job produced no local reconstruction")
            if reference is None:
                reference = exports
            if exports != reference:
                raise RuntimeError("Reconstruction exports differ from the reference")
            report["trials"].append(
                {"id": active, "request": spec, "exports": exports, "elapsed_seconds": time.monotonic() - started}
            )
            print(f"Run {index + 1}: complete; {len(exports)} export hashes match", flush=True)
            active = None
        after = {str(p.relative_to(dataset)): sha256(p) for p in sorted(dataset.rglob("*")) if p.is_file()}
        if after != input_hashes:
            raise RuntimeError("Input files changed during verification")
        report["passed"] = True
    finally:
        if active:
            try:
                request(f"{url}/api/jobs/{active}/cancel", {})
            except (OSError, ValueError) as exc:
                report["cancel_error"] = str(exc)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
