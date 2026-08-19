#!/usr/bin/env python3
"""Lightweight Flask server for the GTSFM pipeline trace visualizer.

Sibling to ./viz — completely separate code path. Scans benchmark_results/ (or
custom --base) for pipeline_trace/manifest.json files, serves the manifest +
per-stage COLMAP files to a Babylon.js frontend that animates the pipeline.

Run: ./pipeline-viz [--base benchmark_results] [--port 5174]
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

from flask import Flask, abort, jsonify, render_template, send_file
from werkzeug.utils import safe_join

app = Flask(__name__, static_folder="static", template_folder="templates")
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-store"
    return response


BASE_DIR = Path(os.environ.get("PIPELINE_VIZ_BASE", "results")).resolve()


def _run_entry(rel: str, label: str, num_stages: int, kind: str) -> dict:
    return {
        "id": rel,
        "label": label,
        "num_stages": num_stages,
        "kind": kind,
        "manifest_url": f"/api/runs/{quote(rel)}/manifest",
        "data_url_prefix": f"/data/{quote(rel)}",
    }


def find_runs(base_dir: Path) -> list[dict]:
    """Find STATIC reconstructions and (optionally) pipeline traces under base_dir.

    - A pipeline trace is a dir with `manifest.json` AND >=1 `stage_*/` subfolder — animated.
    - A static reconstruction is any dir with `points3D.txt` that isn't a trace stage — shown
      as a single-stage run. This is the modern viewer's primary use here (no replay needed).
    """
    runs: list[dict] = []
    seen: set[str] = set()

    # 1) Pipeline traces (if any exist under base_dir).
    for manifest_path in base_dir.rglob("manifest.json"):
        trace_dir = manifest_path.parent
        if not any(trace_dir.glob("stage_*")):
            continue
        rel = trace_dir.relative_to(base_dir).as_posix()
        parts = rel.split("/")
        label = "/".join([p for p in parts if p != "pipeline_trace"]) or rel
        try:
            with manifest_path.open() as f:
                num_stages = len(json.load(f).get("stages", []))
        except Exception:
            num_stages = 0
        seen.add(rel)
        runs.append(_run_entry(rel, label, num_stages, "trace"))

    # 2) Static reconstructions (points3D.txt). Skip trace stage_* dirs and already-listed dirs.
    for pts in base_dir.rglob("points3D.txt"):
        recon_dir = pts.parent
        if recon_dir.name.startswith("stage_") and (recon_dir.parent / "manifest.json").exists():
            continue  # a stage of a pipeline trace, not a standalone reconstruction
        rel = recon_dir.relative_to(base_dir).as_posix()
        if rel in seen:
            continue
        seen.add(rel)
        runs.append(_run_entry(rel, rel, 1, "static"))

    runs.sort(key=lambda r: r["id"])
    return runs


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/runs")
def list_runs():
    runs = find_runs(BASE_DIR)
    return jsonify({"base_dir": str(BASE_DIR), "count": len(runs), "items": runs})


@app.get("/api/runs/<path:run_id>/manifest")
def get_manifest(run_id):
    run_dir = safe_join(str(BASE_DIR), run_id)
    if run_dir is None:
        abort(404)
    run_path = Path(run_dir).resolve()
    try:
        run_path.relative_to(BASE_DIR)
    except ValueError:
        abort(403)
    manifest_path = run_path / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open() as f:
            return jsonify(json.load(f))
    # Static reconstruction → synthesize a single-stage manifest pointing at the dir root.
    if (run_path / "points3D.txt").exists():
        return jsonify({"stages": [{"stage_name": "reconstruction", "subdir": ""}]})
    abort(404)


@app.get("/data/<path:subpath>")
def serve_data(subpath):
    abs_path = safe_join(str(BASE_DIR), subpath)
    if abs_path is None:
        abort(404)
    p = Path(abs_path).resolve()
    try:
        p.relative_to(BASE_DIR)
    except ValueError:
        abort(403)
    if not p.exists() or not p.is_file():
        abort(404)
    return send_file(str(p), as_attachment=False)


def main():
    parser = argparse.ArgumentParser(description="GTSFM pipeline-viz server", add_help=False)
    parser.add_argument("--base", "-b", default="results",
                        help="Base folder to scan for reconstructions / pipeline traces (default: results)")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", "-p", type=int, default=5174,
                        help="Port (default: 5174 — sibling to ./viz on 5173).")
    parser.add_argument("--help", action="help", default=argparse.SUPPRESS)
    args = parser.parse_args()

    global BASE_DIR
    BASE_DIR = Path(args.base).resolve()
    print(f"[pipeline-viz] Serving reconstructions from: {BASE_DIR}")
    if not BASE_DIR.exists():
        print(f"[pipeline-viz] WARNING: base dir does not exist; create {BASE_DIR}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
