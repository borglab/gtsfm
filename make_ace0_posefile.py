#!/usr/bin/env python3
"""
make_ace0_posefile.py
Create an ACE0-compatible pose file from COLMAP's images.txt.

Inputs:
  - images.txt from COLMAP (found under result/images.txt)
  - optional: path to your images directory to sanity-check filenames

Output:
  - default: lines follow CE0's benchmark_poses format
  - with --format=matrix: each line is <image_name> followed by 16 row-major floats of 4x4 c2w

Notes:
  - COLMAP stores world->cam:  X_cam = R * X_world + t
  - ACE0 / Nerfstudio expect cam->world: T_cw = [R^T | -R^T t; 0 0 0 1]
  - Quaternion order in images.txt is: qw qx qy qz (scalar-first)
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

def qvec2rotmat(q: Tuple[float, float, float, float]) -> np.ndarray:
    """Convert COLMAP quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)

@dataclass
class ColmapImageEntry:
    """Minimal data parsed from COLMAP's images.txt for each image."""
    name: str
    qvec: Tuple[float, float, float, float]
    tvec: np.ndarray  # world->cam translation
    camera_id: int


def parse_images_txt(images_txt_path: str):
    """Yield ColmapImageEntry objects for each image in images.txt."""
    with open(images_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # Expected minimal format:
            # IMAGE_ID qw qx qy qz tx ty tz CAMERA_ID NAME [POINTS2D...]
            if len(parts) < 10:
                continue
            try:
                qw, qx, qy, qz = map(float, parts[1:5])
                tx, ty, tz     = map(float, parts[5:8])
                camera_id = int(parts[8])
            except ValueError:
                continue
            name = parts[9]

            qvec = (qw, qx, qy, qz)
            tvec = np.array([tx, ty, tz], dtype=np.float64)
            yield ColmapImageEntry(name=name, qvec=qvec, tvec=tvec, camera_id=camera_id)


def _tail_digits_metadata(filename: str) -> Optional[Tuple[str, int, str, int]]:
    """
    Return (key, value, ext, width) if basename ends with digits, else None.
    key: zero-padded digits with extension (e.g., '000012.jpg')
    value: integer value of the digits
    ext: lowercase extension (e.g., '.jpg')
    width: length of the digit run (for re-padding)
    """
    base = os.path.basename(filename)
    stem, ext = os.path.splitext(base.lower())
    match = re.search(r"(\d+)$", stem)
    if not match:
        return None
    digits = match.group(1)
    key = f"{digits}{ext}"
    return key, int(digits), ext, len(digits)


class ImageNameResolver:
    """Helps map names from images.txt to files present in images_dir."""

    def __init__(self, images_dir: str):
        self.images_dir = images_dir
        self._rel_index: Dict[str, str] = {}
        self._basename_index: Dict[str, List[str]] = defaultdict(list)
        self._digits_index: Dict[str, List[str]] = defaultdict(list)
        self.total_files = 0
        for root, _, files in os.walk(images_dir):
            for fname in files:
                abs_path = os.path.join(root, fname)
                rel_path = os.path.relpath(abs_path, images_dir)
                rel_path = os.path.normpath(rel_path).replace("\\", "/")
                key = rel_path.lower()
                if key not in self._rel_index:
                    self._rel_index[key] = rel_path
                base_key = os.path.basename(rel_path).lower()
                self._basename_index[base_key].append(rel_path)
                digits_meta = _tail_digits_metadata(rel_path)
                if digits_meta:
                    digits_key, _, _, _ = digits_meta
                    self._digits_index[digits_key].append(rel_path)
                self.total_files += 1

    @staticmethod
    def _unique_match(index: Dict[str, List[str]], key: Optional[str]) -> Optional[str]:
        if key is None:
            return None
        matches = index.get(key)
        if matches and len(matches) == 1:
            return matches[0]
        return None

    def resolve(self, requested: str) -> Optional[str]:
        normalized = os.path.normpath(requested.replace("\\", "/"))
        rel_match = self._rel_index.get(normalized.lower())
        if rel_match:
            return rel_match
        base_key = os.path.basename(normalized).lower()
        match = self._unique_match(self._basename_index, base_key)
        if match:
            return match
        digits_meta = _tail_digits_metadata(normalized)
        if not digits_meta:
            return None
        digits_key, digits_value, ext, width = digits_meta
        match = self._unique_match(self._digits_index, digits_key)
        if match:
            return match
        candidates = []
        for delta in (-1, 1):
            shifted = digits_value + delta
            if shifted < 0:
                continue
            shifted_key = f"{shifted:0{width}d}{ext}"
            match = self._unique_match(self._digits_index, shifted_key)
            if match:
                candidates.append(match)
        if len(candidates) == 1:
            return candidates[0]
        return None


def _looks_like_windows_abs(path: str) -> bool:
    return len(path) > 1 and path[1] == ":" and path[0].isalpha()


def _candidate_image_path(images_dir: str, name: str) -> str:
    expanded = os.path.expanduser(name)
    if os.path.isabs(expanded) or _looks_like_windows_abs(expanded) or expanded.startswith("\\\\"):
        return os.path.normpath(expanded)
    sanitized = expanded.replace("\\", "/").lstrip("/")
    candidate = os.path.join(images_dir, sanitized)
    return os.path.normpath(candidate)


def parse_cameras_txt(cameras_txt_path: str) -> Dict[int, float]:
    """Return mapping from camera_id to focal length (fx) parsed from cameras.txt."""
    fx_lookup: Dict[int, float] = {}
    with open(cameras_txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                camera_id = int(parts[0])
                fx = float(parts[4])
            except ValueError:
                continue
            fx_lookup[camera_id] = fx
    return fx_lookup

def main():
    ap = argparse.ArgumentParser(description="Create ACE0 pose file from COLMAP images.txt")
    ap.add_argument("--images-txt", required=True, help="Path to COLMAP images.txt (e.g., ~/tnt/result/images.txt)")
    ap.add_argument("--output", required=True, help="Path to write ACE0 pose file (e.g., ~/tnt/ace0_poses.txt)")
    ap.add_argument("--images-dir", default=None, help="Optional: path to images folder to sanity-check filenames")
    ap.add_argument("--fail-on-missing", action="store_true",
                    help="If set, raise error when an image listed in images.txt is not found in images-dir")
    ap.add_argument("--cameras-txt", default=None,
                    help="Path to COLMAP cameras.txt (required when --format=benchmark).")
    ap.add_argument("--format", choices=["matrix", "benchmark"], default="benchmark",
                    help="Output format: CE0 benchmark 10-field rows (benchmark) or 4x4 matrix per line (matrix).")
    ap.add_argument("--confidence", type=int, default=1000,
                    help="Confidence value to emit per pose in benchmark format (default: 1000).")
    args = ap.parse_args()

    images_dir = args.images_dir
    resolver: Optional[ImageNameResolver] = None
    if images_dir is not None:
        images_dir = os.path.abspath(os.path.expanduser(images_dir))
        if not os.path.isdir(images_dir):
            print(f"[WARN] images-dir does not exist: {images_dir}", file=sys.stderr)
            images_dir = None
        else:
            resolver = ImageNameResolver(images_dir)
            if resolver.total_files == 0:
                print(f"[WARN] images-dir contains no files: {images_dir}", file=sys.stderr)
                resolver = None

    output_mode = args.format
    camera_fx_lookup: Dict[int, float] = {}
    if output_mode == "benchmark":
        cameras_txt = args.cameras_txt
        if cameras_txt is None:
            inferred = os.path.join(os.path.dirname(os.path.abspath(os.path.expanduser(args.images_txt))), "cameras.txt")
            if os.path.isfile(inferred):
                cameras_txt = inferred
        if cameras_txt is None:
            raise ValueError("--cameras-txt must be provided when --format=benchmark (and no cameras.txt was inferred).")
        cameras_txt = os.path.abspath(os.path.expanduser(cameras_txt))
        if not os.path.isfile(cameras_txt):
            raise FileNotFoundError(f"cameras.txt not found: {cameras_txt}")
        camera_fx_lookup = parse_cameras_txt(cameras_txt)
        if not camera_fx_lookup:
            raise ValueError(f"No valid camera entries parsed from {cameras_txt}")

    count = 0
    remapped = 0
    benchmark_confidence = args.confidence
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as out:
        for entry in parse_images_txt(os.path.abspath(os.path.expanduser(args.images_txt))):
            name = entry.name
            output_name = name
            write_name = name
            if images_dir is not None:
                candidate_path = _candidate_image_path(images_dir, name)
                exists = os.path.exists(candidate_path)
                resolved_path = candidate_path
                if not exists and resolver is not None:
                    resolved_rel = resolver.resolve(name)
                    if resolved_rel is not None:
                        output_name = resolved_rel
                        candidate_path = os.path.join(images_dir, resolved_rel)
                        exists = os.path.exists(candidate_path)
                        resolved_path = candidate_path
                        if output_name != name:
                            remapped += 1
                if not exists:
                    msg = f"[WARN] Image listed in images.txt not found under images-dir: {name}"
                    if args.fail_on_missing:
                        raise FileNotFoundError(msg)
                    else:
                        print(msg, file=sys.stderr)
                else:
                    write_name = os.path.abspath(resolved_path)

            R = qvec2rotmat(entry.qvec)
            Rcw = R.T
            tcw = -Rcw @ entry.tvec

            if output_mode == "matrix":
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = Rcw
                T[:3, 3]  = tcw
                flat = " ".join(f"{v:.8f}" for v in T.reshape(-1))
                out.write(f"{write_name} {flat}\n")
            else:
                qw, qx, qy, qz = entry.qvec
                tx, ty, tz = entry.tvec
                fx = camera_fx_lookup.get(entry.camera_id)
                if fx is None:
                    raise KeyError(f"Camera ID {entry.camera_id} not found in cameras.txt; required for benchmark output.")
                out.write(
                    f"{write_name} "
                    f"{qw:.8f} {qx:.8f} {qy:.8f} {qz:.8f} "
                    f"{tx:.8f} {ty:.8f} {tz:.8f} "
                    f"{fx:.8f} {benchmark_confidence}\n"
                )
            count += 1

    print(f"[OK] Wrote {count} poses to: {args.output}")
    if images_dir is not None and remapped > 0:
        print(f"[INFO] Remapped {remapped} filenames based on files in images-dir", file=sys.stderr)

if __name__ == "__main__":
    main()
