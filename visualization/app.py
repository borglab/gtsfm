#!/usr/bin/env python3
import argparse
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


def find_scenes(base_dir: Path) -> list[dict[str, str]]:
    """Recursively find all ba_output folders with COLMAP outputs.

    Args:
        base_dir: base directory to scan.

    Returns:
        List of scenes found, each as a dictionary with keys:
            - label: human-readable label for the scene
            - rel_path: relative path from base_dir
            - points: URL path to points3D.txt
            - images: URL path to images.txt
    """
    scenes = []
    for points_file in base_dir.rglob("points3D.txt"):
        if points_file.name != "points3D.txt":
            continue
        parent = points_file.parent
        images_file = parent / "images.txt"
        if images_file.exists():
            # scene_dir is the directory containing points3D.txt/images.txt
            scene_dir = parent
            rel = scene_dir.relative_to(base_dir).as_posix()
            # nice label: last 2 parts (…/sequence/ba_output)
            parts = rel.split("/")
            label = "/".join(parts[-3:]) if len(parts) >= 3 else rel
            scenes.append(
                {
                    "label": label,
                    "rel_path": rel,  # served via /data/<rel_path>/*
                    "points": f"/data/{quote(rel)}/points3D.txt",
                    "images": f"/data/{quote(rel)}/images.txt",
                }
            )
    # Sort for stability
    scenes.sort(key=lambda x: x["rel_path"])
    return scenes


BASE_DIR = Path(os.environ.get("RESULTS_DIR", "results")).resolve()


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/scenes")
def list_scenes():
    # Re-scan on each request (cheap for modest trees; or cache if huge)
    scenes = find_scenes(BASE_DIR)
    return jsonify({"base_dir": str(BASE_DIR), "count": len(scenes), "items": scenes})


@app.get("/data/<path:subpath>")
def serve_data(subpath):
    # Prevent directory traversal outside BASE_DIR
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
    # We disable the default help action to free up the `-h` flag.
    parser = argparse.ArgumentParser(description="GTSFM viz server", add_help=False)

    # Define the command-line arguments.
    parser.add_argument("--base", "-b", default="results", help="Base folder to scan (default: results)")
    parser.add_argument("--host", "-h", default="127.0.0.1", help="Host to run the server on.")
    parser.add_argument("--port", "-p", type=int, default=5173, help="Port to run the server on.")

    # Add a custom help argument that uses --help.
    parser.add_argument("--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit.")

    args = parser.parse_args()

    global BASE_DIR
    BASE_DIR = Path(args.base).resolve()

    # This print statement is now removed from the viz script and handled here.
    print(f"[viz] Serving results from: {BASE_DIR}")
    if not BASE_DIR.exists():
        print(f"[viz] WARNING: base dir does not exist; create {BASE_DIR}")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
