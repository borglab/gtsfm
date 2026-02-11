"""
Run VGGT tracking + bundle adjustment for each cluster reconstruction.

This script mirrors the directory structure of an input results tree (e.g.
`.../2-reconstruction/vggt_cluster_run/results`) and re-runs VGGT tracking/BA
per `vggt` folder. Outputs are written in COLMAP format under a user-specified
output root with the same relative layout.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VGGT tracking + BA for each cluster reconstruction.")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help=(
            "Path to a results root or run root. If this path contains a "
            "`results/` subdir, that subdir is scanned for vggt reconstructions."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(Path.cwd()),
        help=(
            "Base output directory. Results are written under "
            "`<output_root>/results/...` unless output_root itself is named `results`."
        ),
    )
    parser.add_argument(
        "--images_root",
        type=str,
        default=None,
        help="Root directory for images referenced in images.txt (required if names are relative).",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        default=True,
        help="Recompute even if output already exists.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="pipeline/2-reconstruction/vggt/weights/model.pt",
        help="Path to the VGGT model checkpoint.",
    )
    parser.add_argument("--use_ba", action="store_true", default=True, help="Use BA for reconstruction.")
    parser.add_argument(
        "--no_use_ba",
        action="store_false",
        dest="use_ba",
        help="Disable BA (still runs VGGT feed-forward reconstruction).",
    )
    parser.add_argument(
        "--ba_tracker",
        type=str,
        choices=["vggt", "vggsfm"],
        default="vggt",
        help="Tracker used for BA (vggt or vggsfm).",
    )
    parser.add_argument("--img_load_resolution", type=int, default=1024, help="Square load resolution for VGGT input.")
    parser.add_argument("--vggt_fixed_resolution", type=int, default=518, help="VGGT internal inference resolution.")
    parser.add_argument(
        "--conf_thres_value", type=float, default=5.0, help="Confidence threshold value for depth filtering."
    )
    return parser.parse_args()


def _resolve_results_root(path: Path) -> Path:
    if (path / "results").is_dir():
        return path / "results"
    return path


def _resolve_output_results_root(path: Path) -> Path:
    if path.name == "results":
        return path
    return path / "results"


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _has_colmap_files(path: Path) -> bool:
    has_cameras = (path / "cameras.txt").exists() or (path / "cameras.bin").exists()
    has_images = (path / "images.txt").exists() or (path / "images.bin").exists()
    has_points = (path / "points3D.txt").exists() or (path / "points3D.bin").exists()
    return has_cameras and has_images and has_points


def _iter_vggt_dirs(results_root: Path, output_results_root: Path) -> list[Path]:
    vggt_dirs: list[Path] = []
    for dirpath, _, _ in os.walk(results_root):
        path = Path(dirpath)
        if _is_relative_to(path, output_results_root):
            continue
        if path.name != "vggt":
            continue
        if _has_colmap_files(path):
            vggt_dirs.append(path)
    return vggt_dirs


def _log(log_path: Path, message: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"{message}\n")
    print(message)


def _ensure_ba_module() -> None:
    try:
        importlib.import_module("ba")
        return
    except ModuleNotFoundError:
        candidate_eval_dirs: list[Path] = []
        search_roots = [Path(__file__).resolve()] + list(Path(__file__).resolve().parents)
        search_roots += [Path.cwd().resolve()] + list(Path.cwd().resolve().parents)
        for root in search_roots:
            thirdparty_eval = root / "thirdparty" / "vggt" / "evaluation"
            if (thirdparty_eval / "ba.py").exists():
                candidate_eval_dirs.append(thirdparty_eval)
            pipeline_eval = root / "pipeline" / "2-reconstruction" / "vggt" / "evaluation"
            if (pipeline_eval / "ba.py").exists():
                candidate_eval_dirs.append(pipeline_eval)

        lightglue_root = None
        for root in search_roots:
            candidate = root / "thirdparty" / "LightGlue"
            if candidate.exists():
                lightglue_root = candidate
                break
        if lightglue_root is not None:
            sys.path.insert(0, str(lightglue_root))

        for eval_dir in candidate_eval_dirs:
            sys.path.insert(0, str(eval_dir))
            sys.path.insert(0, str(eval_dir.parent))
            try:
                importlib.import_module("ba")
                return
            except ModuleNotFoundError:
                continue
        raise


def _parse_colmap_images_txt(images_txt_path: Path) -> list[str]:
    image_names: list[str] = []
    with open(images_txt_path, "r", encoding="utf-8") as images_file:
        lines = images_file.readlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not line or line.startswith("#"):
            idx += 1
            continue
        parts = line.split()
        if len(parts) < 10:
            raise ValueError(f"Invalid images.txt line in {images_txt_path}: {line}")
        image_names.append(" ".join(parts[9:]))
        idx += 2
    return image_names


def _resolve_image_paths(image_names: list[str], images_root: str | None) -> list[str]:
    resolved_paths = []
    for name in image_names:
        if os.path.isabs(name):
            resolved_paths.append(name)
        else:
            if images_root is None:
                raise ValueError("images_root is required when image filenames are relative.")
            resolved_paths.append(os.path.join(images_root, name))
    return resolved_paths


def _setup_vggt(args: argparse.Namespace):
    _ensure_ba_module()
    test_module = importlib.import_module("evaluation.test_co3d_cluster")
    setup_model = getattr(test_module, "setup_model")
    run_vggt_reconstruction = getattr(test_module, "run_vggt_reconstruction")
    model, device, dtype = setup_model(args)
    return run_vggt_reconstruction, model, device, dtype


def main() -> None:
    args = _parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    input_results_root = _resolve_results_root(input_root)
    output_results_root = _resolve_output_results_root(output_root)

    log_path = output_root / "cluster_ba.log"
    vggt_dirs = _iter_vggt_dirs(input_results_root, output_results_root)
    if not vggt_dirs:
        _log(log_path, f"No vggt reconstructions found under {input_results_root}.")
        return

    run_vggt_reconstruction, model, device, dtype = _setup_vggt(args)

    for vggt_dir in vggt_dirs:
        rel_path = vggt_dir.relative_to(input_results_root)
        output_dir = output_results_root / rel_path

        if args.skip_existing and _has_colmap_files(output_dir):
            _log(log_path, f"Skipping {vggt_dir}: output already exists at {output_dir}.")
            continue

        images_txt_path = vggt_dir / "images.txt"
        if not images_txt_path.exists():
            _log(log_path, f"Skipping {vggt_dir}: missing images.txt.")
            continue

        image_names = _parse_colmap_images_txt(images_txt_path)
        try:
            image_paths = _resolve_image_paths(image_names, args.images_root)
        except Exception as exc:
            _log(log_path, f"Skipping {vggt_dir}: {exc!r}")
            continue

        missing = [path for path in image_paths if not os.path.exists(path)]
        if missing:
            _log(log_path, f"Skipping {vggt_dir}: missing {len(missing)} images.")
            continue

        _log(log_path, f"Running VGGT+BA for {vggt_dir} -> {output_dir}")
        try:
            run_vggt_reconstruction(
                args,
                model,
                device,
                dtype,
                image_paths,
                str(output_dir),
                image_name_list=image_names,
            )
        except Exception as exc:
            _log(log_path, f"Failed {vggt_dir}: {exc!r}")


if __name__ == "__main__":
    main()
