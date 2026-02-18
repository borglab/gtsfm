"""
Align and merge cluster reconstructions using Sim(3) estimated from overlapping cameras.

Input:
  - COLMAP text reconstructions stored under `<input_root>/results/.../<input_model_name>`
    (e.g. `.../results/C_1/C_1_2/vggt`).
  - Cluster tree pickle from the partition stage.

Output:
  - COLMAP text reconstructions written under `<output_root>/results/.../<output_model_name>`
    (default `merged`) for every non-leaf cluster node.
  - COLMAP text reconstructions for the original (pre-merge) clusters written under
    `<output_root>/results/.../<original_model_name>` (default `<input_model_name>_original`).
"""

from __future__ import annotations

import argparse
import pickle
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from gtsam import Pose3, Similarity3, SfmTrack

import gtsfm.utils.logger as logger_utils
import gtsfm.common.types as gtsfm_types
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.outputs import cluster_label
from gtsfm.utils.tree import PostOrderIter, Tree

logger = logger_utils.get_logger()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align and merge cluster reconstructions using Sim(3).")
    parser.add_argument("--cluster_tree_path", type=str, required=True, help="Path to cluster_tree.pkl")
    parser.add_argument(
        "--input_root",
        type=str,
        required=True,
        help="Root containing `results/` from the reconstruction stage.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=None,
        help="Root to write merged outputs (defaults to input_root).",
    )
    parser.add_argument(
        "--input_model_name",
        type=str,
        default="vggt",
        help="Name of per-cluster COLMAP model folder to read.",
    )
    parser.add_argument(
        "--original_model_name",
        type=str,
        default=None,
        help="Name of per-cluster COLMAP model folder to write original (pre-merge) models. "
        "Defaults to '<input_model_name>_original'.",
    )
    parser.add_argument(
        "--output_model_name",
        type=str,
        default="merged",
        help="Name of per-cluster COLMAP model folder to write.",
    )
    parser.add_argument(
        "--min_common_cameras",
        type=int,
        default=2,
        help="Minimum number of overlapping cameras required to estimate Sim(3).",
    )
    parser.add_argument(
        "--drop_child_if_fail",
        action="store_true",
        default=True,
        help="Drop child if alignment or merging fails.",
    )
    parser.add_argument(
        "--no_drop_child_if_fail",
        action="store_false",
        dest="drop_child_if_fail",
        help="Keep child in place (identity Sim(3)) if alignment fails.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=False,
        help="Skip writing if merged output already exists (will be reused for parents).",
    )
    parser.add_argument(
        "--run_colmap_ba",
        action="store_true",
        default=False,
        help="Run COLMAP bundle_adjuster on each merged model before propagating to parents.",
    )
    parser.add_argument(
        "--colmap_path",
        type=str,
        default="colmap",
        help="Path to the COLMAP executable used for bundle_adjuster.",
    )
    parser.add_argument(
        "--colmap_ba_extra_args",
        type=str,
        default="",
        help="Extra arguments passed to COLMAP bundle_adjuster (single string, shell-like).",
    )
    parser.add_argument(
        "--convert_ba_to_txt",
        action="store_true",
        default=False,
        help="After COLMAP BA, run model_converter to produce a TXT model in a separate folder.",
    )
    return parser.parse_args()


def _load_cluster_tree(cluster_tree_path: str) -> Tree:
    with open(cluster_tree_path, "rb") as f:
        cluster_tree = pickle.load(f)
    if cluster_tree is None:
        raise ValueError(f"cluster_tree.pkl was empty or invalid: {cluster_tree_path}")
    return cluster_tree


def _cluster_dir(root: Path, path: Tuple[int, ...]) -> Path:
    cluster_dir = root / "results"
    if path:
        for depth in range(len(path)):
            cluster_dir = cluster_dir / cluster_label(path[: depth + 1])
    return cluster_dir


def _model_dir(root: Path, path: Tuple[int, ...], model_name: str) -> Path:
    return _cluster_dir(root, path) / model_name


def _read_scene(model_dir: Path, name_to_idx: Optional[Dict[str, int]] = None) -> Optional[GtsfmData]:
    if not model_dir.exists():
        return None
    if not (model_dir / "images.txt").exists() and not (model_dir / "images.bin").exists():
        return None
    try:
        scene = GtsfmData.read_colmap(str(model_dir))
        if name_to_idx is not None:
            scene = _remap_scene_to_global_indices(scene, name_to_idx)
        return scene
    except Exception as exc:
        logger.warning("Failed to read COLMAP model at %s: %s", model_dir, exc)
        return None


def _pose_map_by_name(scene: GtsfmData) -> Dict[str, Pose3]:
    pose_map: Dict[str, Pose3] = {}
    for idx, pose in scene.poses().items():
        name = scene.get_image_info(idx).name
        if name is None:
            continue
        pose_map[name] = pose
    return pose_map


def _parse_colmap_image_names(images_txt: Path) -> set[str]:
    names: set[str] = set()
    if not images_txt.exists():
        return names
    lines = [line.strip() for line in images_txt.read_text().splitlines() if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("#"):
            i += 1
            continue
        parts = lines[i].split()
        if len(parts) >= 10:
            names.add(parts[9])
        i += 2
    return names


def _build_global_name_map(input_root: Path, model_name: str) -> Dict[str, int]:
    base = input_root / "results"
    names: set[str] = set()
    for images_txt in base.rglob(f"{model_name}/images.txt"):
        names.update(_parse_colmap_image_names(images_txt))
    if not names:
        raise ValueError(f"No image names found under {base} for model {model_name}.")
    sorted_names = sorted(names)
    return {name: idx for idx, name in enumerate(sorted_names)}


def _remap_scene_to_global_indices(scene: GtsfmData, name_to_idx: Dict[str, int]) -> GtsfmData:
    remapped = GtsfmData(number_images=len(name_to_idx))

    # Remap cameras and image info by global image name.
    for idx, pose in scene.poses().items():
        info = scene.get_image_info(idx)
        name = info.name
        if name is None or name not in name_to_idx:
            continue
        new_idx = name_to_idx[name]
        camera = scene.get_camera(idx)
        if camera is None:
            continue
        calibration = camera.calibration()
        camera_type = gtsfm_types.get_camera_class_for_calibration(calibration)
        remapped.add_camera(new_idx, camera_type(pose, calibration))  # type: ignore
        remapped.set_image_info(new_idx, name=name, shape=info.shape)

    # Remap tracks by global image name.
    for track in scene.tracks():
        new_track = SfmTrack(track.point3())
        for k in range(track.numberMeasurements()):
            i, uv = track.measurement(k)
            name = scene.get_image_info(i).name
            if name is None or name not in name_to_idx:
                continue
            new_track.addMeasurement(name_to_idx[name], uv)
        if new_track.numberMeasurements() > 0:
            new_track.r = getattr(track, "r", 0)
            new_track.g = getattr(track, "g", 0)
            new_track.b = getattr(track, "b", 0)
            remapped.add_track(new_track)

    if scene.has_gaussian_splats():
        remapped.set_gaussian_splats(scene.get_gaussian_splats())

    return remapped


def _sim3_from_common_names(
    a_scene: GtsfmData, b_scene: GtsfmData, min_common: int
) -> Similarity3:
    a_map = _pose_map_by_name(a_scene)
    b_map = _pose_map_by_name(b_scene)
    common_names = [name for name in a_map if name in b_map]
    if len(common_names) < min_common:
        raise ValueError(f"Found only {len(common_names)} overlapping cameras (need {min_common}).")
    pose_pairs = [(a_map[name], b_map[name]) for name in common_names]
    return Similarity3.Align(pose_pairs)


def _align_and_merge(
    parent: GtsfmData,
    child: GtsfmData,
    min_common: int,
    drop_child_if_fail: bool,
) -> GtsfmData:
    try:
        aSb = _sim3_from_common_names(parent, child, min_common=min_common)
    except Exception as exc:
        if drop_child_if_fail:
            logger.warning("Dropping child due to alignment failure: %s", exc)
            return parent
        logger.warning("Alignment failed; using identity Sim(3): %s", exc)
        aSb = Similarity3()

    try:
        return parent.merged_with(child, aSb)
    except Exception as exc:
        logger.warning("Failed to merge child: %s", exc)
        return parent


def _iter_path_tree(cluster_tree: Tree) -> Iterable[Tree[Tuple[int, ...]]]:
    path_tree: Tree[Tuple[int, ...]] = cluster_tree.map_with_path(lambda path, _: path)
    return PostOrderIter(path_tree)


def _run_colmap_bundle_adjuster(
    input_model_dir: Path, output_model_dir: Path, colmap_path: str, extra_args: str
) -> bool:
    if not input_model_dir.exists():
        logger.warning("COLMAP BA skipped: model dir missing at %s", input_model_dir)
        return False
    cmd = [
        colmap_path,
        "bundle_adjuster",
        "--input_path",
        str(input_model_dir),
        "--output_path",
        str(output_model_dir),
    ]
    if extra_args:
        cmd.extend(shlex.split(extra_args))
    try:
        logger.info("Running COLMAP BA: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        logger.warning("COLMAP BA failed for %s: %s", input_model_dir, exc)
        return False


def _run_colmap_model_converter(input_model_dir: Path, output_model_dir: Path, colmap_path: str) -> bool:
    if not input_model_dir.exists():
        logger.warning("COLMAP model conversion skipped: model dir missing at %s", input_model_dir)
        return False
    cmd = [
        colmap_path,
        "model_converter",
        "--input_path",
        str(input_model_dir),
        "--output_path",
        str(output_model_dir),
        "--output_type",
        "TXT",
    ]
    try:
        logger.info("Running COLMAP model_converter: %s", " ".join(cmd))
        subprocess.run(cmd, check=True)
        return True
    except Exception as exc:
        logger.warning("COLMAP model conversion failed for %s: %s", input_model_dir, exc)
        return False


def main() -> None:
    args = _parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) if args.output_root is not None else input_root
    original_model_name = args.original_model_name or f"{args.input_model_name}_original"
    cluster_tree = _load_cluster_tree(args.cluster_tree_path)
    name_to_idx = _build_global_name_map(input_root, args.input_model_name)

    merged_cache: Dict[Tuple[int, ...], Optional[GtsfmData]] = {}

    for node in _iter_path_tree(cluster_tree):
        path = node.value
        input_dir = _model_dir(input_root, path, args.input_model_name)
        current = _read_scene(input_dir, name_to_idx)
        if current is None:
            logger.warning("Skipping %s: missing model at %s", path, input_dir)
            merged_cache[path] = None
            continue

        original_dir = _model_dir(output_root, path, original_model_name)
        if not args.skip_existing or not (original_dir / "cameras.txt").exists():
            original_dir.mkdir(parents=True, exist_ok=True)
            current.export_as_colmap_text(original_dir)
            logger.info("Wrote original model for %s to %s", path, original_dir)

        pre_ba_dir = _model_dir(output_root, path, f"{args.output_model_name}_pre_ba")
        ba_dir = _model_dir(output_root, path, f"{args.output_model_name}_colmap_ba")
        ba_txt_dir = _model_dir(output_root, path, f"{args.output_model_name}_colmap_ba_txt")

        if node.is_leaf():
            merged_cache[path] = current
            continue

        if args.skip_existing:
            cached = None
            if args.run_colmap_ba and (ba_dir / "cameras.txt").exists():
                cached = _read_scene(ba_dir, name_to_idx)
            elif (pre_ba_dir / "cameras.txt").exists():
                cached = _read_scene(pre_ba_dir, name_to_idx)
            if cached is not None:
                merged_cache[path] = cached
                logger.info("Reusing existing merged output at %s", ba_dir if args.run_colmap_ba else pre_ba_dir)
                continue

        merged = current
        for child in node.children:
            child_path = child.value
            child_scene = merged_cache.get(child_path)
            if child_scene is None:
                if args.run_colmap_ba:
                    child_scene = _read_scene(
                        _model_dir(output_root, child_path, f"{args.output_model_name}_colmap_ba"),
                        name_to_idx,
                    )
                if child_scene is None:
                    child_scene = _read_scene(
                        _model_dir(output_root, child_path, f"{args.output_model_name}_pre_ba"),
                        name_to_idx,
                    )
                if child_scene is None and args.convert_ba_to_txt:
                    child_scene = _read_scene(
                        _model_dir(output_root, child_path, f"{args.output_model_name}_colmap_ba_txt"),
                        name_to_idx,
                    )
                if child_scene is None:
                    child_scene = _read_scene(_model_dir(input_root, child_path, args.input_model_name), name_to_idx)
            if child_scene is None:
                logger.warning("Missing child model for %s -> %s", path, child_path)
                continue
            merged = _align_and_merge(
                merged,
                child_scene,
                min_common=args.min_common_cameras,
                drop_child_if_fail=args.drop_child_if_fail,
            )

        pre_ba_dir.mkdir(parents=True, exist_ok=True)
        merged.export_as_colmap_text(pre_ba_dir)
        if args.run_colmap_ba:
            ba_dir.mkdir(parents=True, exist_ok=True)
            if _run_colmap_bundle_adjuster(pre_ba_dir, ba_dir, args.colmap_path, args.colmap_ba_extra_args):
                merged_ba = _read_scene(ba_dir, name_to_idx)
                if merged_ba is not None:
                    merged = merged_ba
                else:
                    logger.warning("COLMAP BA completed but failed to reload model at %s", ba_dir)
                if args.convert_ba_to_txt:
                    ba_txt_dir.mkdir(parents=True, exist_ok=True)
                    _run_colmap_model_converter(ba_dir, ba_txt_dir, args.colmap_path)
        merged_cache[path] = merged
        logger.info(
            "Wrote merged model for %s to %s",
            path,
            ba_dir if args.run_colmap_ba else pre_ba_dir,
        )


if __name__ == "__main__":
    main()
