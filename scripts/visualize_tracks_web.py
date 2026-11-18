"""Generate a static web UI to visualize track reprojections per cluster.

The UI loads COLMAP text outputs (vggt / vggt_pre_ba) and lets you browse clusters,
see per-image tracks, and inspect details with zoomed pixel crops.

Example commands:
  python scripts/visualize_tracks_web.py \
      --results_root outputs/trevi_debug/results \
      --dataset_dir data/trevi \
      --serve --port 8000 \
      --gt_colmap_dir colmap/sparse/0

  python scripts/visualize_tracks_web.py \
      --results_root outputs/trevi_debug/results \
      --dataset_dir data/trevi \
      --gt_colmap_dir colmap/sparse/0 \
      --baseline data/trevi/sparse/0 \
      --baseline_label "glomap" \
      --serve --port 8000
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.server
import json
import logging
import os
import pickle
import re
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse
import itertools
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
from collections import Counter

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.utils.align import sim3_from_Pose3_maps_robust
from gtsfm.visualization.track_viz_utils import collect_reprojection_pairs


logger = logging.getLogger(__name__)

RECON_OPTIONS = ("vggt", "vggt_pre_ba", "merged")
COLMAP_REQUIRED_TXT = ("cameras.txt", "images.txt", "points3D.txt")
COLMAP_REQUIRED_BIN = ("cameras.bin", "images.bin", "points3D.bin")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a static tracks web viewer for cluster reconstructions.")
    parser.add_argument(
        "--results_root",
        type=str,
        required=True,
        help="Root directory containing cluster results (e.g. outputs/trevi_debug/results).",
    )
    parser.add_argument(
        "--cluster_tree_pkl",
        type=str,
        default=None,
        help="Path to cluster_tree.pkl. Defaults to <results_root>/cluster_tree.pkl.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Dataset root containing images.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional images directory override. Defaults to <dataset_dir>/images.",
    )
    parser.add_argument(
        "--colmap_files_subdir",
        type=str,
        default=None,
        help="COLMAP model directory relative to --dataset_dir, or an absolute path. "
        "Image indices are mapped back to the original filenames in that model.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for the web viewer. Defaults to <results_root>/tracks_web_viz.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Number of worker threads for loading COLMAP reconstructions.",
    )
    parser.add_argument(
        "--gt_colmap_dir",
        type=str,
        default=None,
        help="Subdirectory within dataset_dir containing ground truth COLMAP files "
        "(cameras.txt/bin, images.txt/bin, points3D.txt/bin). E.g. 'sparse/0'. "
        "Auto-detects from dataset_dir if not specified.",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        action="append",
        default=[],
        help="Path to a COLMAP model directory (txt or bin) to visualize as a baseline. "
        "Can be specified up to 2 times. Use --baseline_label to give each a name.",
    )
    parser.add_argument(
        "--baseline_label",
        type=str,
        action="append",
        default=[],
        help="Label for each --baseline (in order). Defaults to 'Baseline 1', 'Baseline 2'.",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Serve the output directory with a local HTTP server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to use when --serve is enabled.",
    )
    return parser.parse_args()


def _node_results_dir(results_root: Path, path_str: str) -> Path:
    """Map a cluster-tree path string to the corresponding results subdirectory."""
    if path_str == "root":
        return results_root
    indices = [int(p) for p in path_str.split(".") if p]
    parts: list[str] = []
    for depth in range(1, len(indices) + 1):
        partial = indices[:depth]
        name = "C_" + "_".join(str(i) for i in partial)
        parts.append(name)
    return results_root.joinpath(*parts)


def _partition_cut(node: ClusterTree) -> Dict[str, List[int]] | None:
    """Camera indices on each side of a binary partition.

    ``left`` and ``right`` are exclusive to one child. ``overlap`` holds cameras borrowed into both.
    """
    if len(node.children) != 2:
        return None
    left_keys = set(node.children[0].all_keys())
    right_keys = set(node.children[1].all_keys())
    left_only = sorted(left_keys - right_keys)
    right_only = sorted(right_keys - left_keys)
    overlap = sorted(left_keys & right_keys)
    if not left_only and not right_only and not overlap:
        return None
    return {"left": left_only, "right": right_only, "overlap": overlap}


def _collect_cluster_paths(node: ClusterTree, path: Tuple[int, ...] = ()) -> List[Dict[str, Any]]:
    """Return a flat list of cluster nodes with path strings and depths."""
    path_str = "root" if len(path) == 0 else ".".join(str(p) for p in path)
    nodes = [{"path": path_str, "depth": len(path), "cut": _partition_cut(node)}]
    for child_idx, child in enumerate(node.children, start=1):
        nodes.extend(_collect_cluster_paths(child, path + (child_idx,)))
    return nodes


def _has_colmap_files(dir_path: Path) -> bool:
    """Check whether dir_path contains a complete COLMAP model (text or binary)."""
    has_txt = all((dir_path / name).exists() for name in COLMAP_REQUIRED_TXT)
    has_bin = all((dir_path / name).exists() for name in COLMAP_REQUIRED_BIN)
    return has_txt or has_bin


def _normalize_shape(height_width: Tuple[int, int] | None, fallback: Tuple[int, int]) -> Tuple[int, int]:
    if height_width is None:
        return fallback
    h, w = int(height_width[0]), int(height_width[1])
    if h <= 0 or w <= 0:
        return fallback
    return (h, w)


def _fallback_shape_from_camera(gtsfm_data: GtsfmData, image_idx: int) -> Tuple[int, int]:
    cam = gtsfm_data.get_camera(image_idx)
    if cam is None:
        return (1, 1)
    calib = cam.calibration()
    cx = getattr(calib, "px", None)
    cy = getattr(calib, "py", None)
    if callable(cx):
        cx = cx()
    if callable(cy):
        cy = cy()
    if isinstance(cx, (int, float)) and isinstance(cy, (int, float)):
        width = max(int(round(cx * 2)), 1)
        height = max(int(round(cy * 2)), 1)
        return (height, width)
    return (1, 1)


_PLACEHOLDER_NAME = re.compile(r"^image_(\d+)\.jpg$")


def _loader_index(index: int, stored_name: str | None) -> int:
    """Recover the loader index encoded in ``image_NNNNNN.jpg`` after COLMAP reorders by name."""
    if stored_name:
        match = _PLACEHOLDER_NAME.match(Path(stored_name).name)
        if match:
            return int(match.group(1))
    return index


def _image_name(index: int, stored_name: str | None, index_to_filename: Dict[int, str] | None) -> str:
    """Prefer the COLMAP loader filename when a reconstruction stored a placeholder name."""
    source_index = _loader_index(index, stored_name)
    if index_to_filename and source_index in index_to_filename and (
        not stored_name or _PLACEHOLDER_NAME.match(Path(stored_name).name) is not None
    ):
        return index_to_filename[source_index]
    if stored_name:
        return stored_name
    return f"image_{index:06d}.jpg"


def _build_image_entries(
    gtsfm_data: GtsfmData,
    images_dir: Path,
    *,
    include_file_path: bool,
    index_to_filename: Dict[int, str] | None = None,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    num_tracks = gtsfm_data.number_tracks()
    allowed_track_indices = set(range(num_tracks))
    track_lengths: Dict[int, int] = {}
    for track_idx in range(num_tracks):
        track_lengths[track_idx] = int(gtsfm_data.get_track(track_idx).numberMeasurements())

    for image_idx in sorted(gtsfm_data.get_valid_camera_indices()):
        info = gtsfm_data.get_image_info(image_idx)
        name = _image_name(image_idx, info.name, index_to_filename)
        file_path = (images_dir / name).resolve()
        if not file_path.exists():
            logger.warning("Missing image file for %s (expected %s)", name, file_path)
        fallback_shape = _fallback_shape_from_camera(gtsfm_data, image_idx)
        height, width = _normalize_shape(info.shape, fallback_shape)

        pairs = collect_reprojection_pairs(gtsfm_data, image_idx, allowed_track_indices)
        tracks: List[Dict[str, Any]] = []
        for track_idx, uv_measured, uv_reproj in pairs:
            reproj_error = float(((uv_measured[0] - uv_reproj[0]) ** 2 + (uv_measured[1] - uv_reproj[1]) ** 2) ** 0.5)
            tracks.append(
                {
                    "track_id": int(track_idx),
                    "meas": [float(uv_measured[0]), float(uv_measured[1])],
                    "reproj": [float(uv_reproj[0]), float(uv_reproj[1])],
                    "track_len": int(track_lengths.get(int(track_idx), 0)),
                    "reproj_err": reproj_error,
                }
            )

        entry: Dict[str, Any] = {
            "image_id": int(image_idx),
            "name": name,
            "width": int(width),
            "height": int(height),
            "tracks": tracks,
        }
        if include_file_path:
            entry["file_path"] = name
        entries.append(entry)
    return entries


def _load_recon_images(
    recon_dir: Path,
    images_dir: Path,
    cluster_path: str,
    recon: str,
    index_to_filename: Dict[int, str] | None = None,
) -> List[Dict[str, Any]]:
    if not _has_colmap_files(recon_dir):
        return []
    logger.info("Loading COLMAP data for %s (%s)", cluster_path, recon)
    try:
        gtsfm_data = GtsfmData.read_colmap(str(recon_dir))
    except Exception as exc:
        logger.exception("Skipping %s/%s due to error: %s", cluster_path, recon, exc)
        return []
    return _build_image_entries(
        gtsfm_data, images_dir, include_file_path=True, index_to_filename=index_to_filename
    )


def _build_data_payload(
    cluster_nodes: Iterable[Dict[str, Any]],
    results_root: Path,
    images_dir: Path,
    num_workers: int,
    gt_data: GtsfmData | None = None,
    baseline_data: List[Tuple[str, GtsfmData]] | None = None,
    index_to_filename: Dict[int, str] | None = None,
) -> Dict[str, Any]:
    clusters: List[Dict[str, Any]] = []
    images_by_cluster: Dict[str, Dict[str, Any]] = {}
    node_dirs: Dict[str, Path] = {}

    tasks: List[Tuple[str, str, Path]] = []
    for node in cluster_nodes:
        path_str = node["path"]
        node_dir = _node_results_dir(results_root, path_str)
        node_dirs[path_str] = node_dir
        label = "root" if path_str == "root" else f"C_{path_str.replace('.', '_')}"
        cluster_entry = {
            "id": path_str,
            "path": path_str,
            "label": label,
            "depth": node["depth"],
            "cut": node.get("cut"),
        }
        clusters.append(cluster_entry)
        images_by_cluster[path_str] = {}
        for recon in RECON_OPTIONS:
            tasks.append((path_str, recon, node_dir / recon))

    max_workers = max(1, num_workers)
    logger.info("Loading reconstructions with %d worker threads.", max_workers)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_load_recon_images, recon_dir, images_dir, path_str, recon, index_to_filename): (
                path_str,
                recon,
            )
            for path_str, recon, recon_dir in tasks
        }
        for future in concurrent.futures.as_completed(future_map):
            path_str, recon = future_map[future]
            images = future.result()
            if images:
                images_by_cluster[path_str][recon] = images

    # Build pose data for each cluster.
    poses_by_cluster: Dict[str, Any] = {}
    for cluster in clusters:
        path_str = cluster["id"]
        node_dir = node_dirs[path_str]
        try:
            pose_data = _build_pose_data_for_cluster(
                gt_data, node_dir, baseline_data=baseline_data, index_to_filename=index_to_filename
            )
            if pose_data:
                poses_by_cluster[path_str] = pose_data
                logger.info("Built pose data for cluster %s: %s", path_str, sorted(k for k in pose_data if k not in ("edges", "arrow_scale")))
            else:
                logger.warning("No pose data returned for cluster %s (node_dir=%s)", path_str, node_dir)
        except Exception as exc:
            logger.warning("Pose data generation failed for %s: %s", path_str, exc, exc_info=True)

    result: Dict[str, Any] = {
        "clusters": clusters,
        "imagesByCluster": images_by_cluster,
        "posesByCluster": poses_by_cluster,
        "defaultRecon": "vggt",
        "baseImageUrl": None,
    }
    return result


def _load_index_to_filename(
    dataset_dir: Path, images_dir: Path, colmap_files_subdir: str | None
) -> Dict[int, str] | None:
    """Map loader indices to original filenames using the same COLMAP model the run used."""
    if not colmap_files_subdir:
        return None
    from gtsfm.loader.colmap_loader import ColmapLoader

    loader = ColmapLoader(
        dataset_dir=str(dataset_dir),
        images_dir=str(images_dir),
        colmap_files_subdir=colmap_files_subdir,
        use_gt_intrinsics=False,
        use_gt_extrinsics=True,
    )
    mapping = {idx: loader.get_image_fname(idx) for idx in range(len(loader))}
    logger.info("Mapped %d image indices to filenames from %s", len(mapping), colmap_files_subdir)
    return mapping


def _find_gt_colmap_dir(dataset_dir: Path, gt_colmap_subdir: str | None) -> Path | None:
    """Find the ground truth COLMAP directory, trying common locations.

    Args:
        dataset_dir: Root dataset directory.
        gt_colmap_subdir: Optional subdirectory relative to dataset_dir (e.g. "sparse/0").
    """
    if gt_colmap_subdir is not None:
        candidate = dataset_dir / gt_colmap_subdir
        if _has_colmap_files(candidate):
            return candidate
        logger.warning("Specified --gt_colmap_dir %s does not contain COLMAP files.", candidate)
        return None
    for candidate in [dataset_dir, dataset_dir / "sparse" / "0", dataset_dir / "colmap_ground_truth"]:
        if _has_colmap_files(candidate):
            logger.info("Auto-detected GT COLMAP dir: %s", candidate)
            return candidate
    return None


def _name_pose_map(data: GtsfmData, index_to_filename: Dict[int, str] | None = None) -> Dict[str, Any]:
    """Build a mapping from image name to (local_index, Pose3, rotation_matrix)."""
    result: Dict[str, Any] = {}
    for idx in data.get_valid_camera_indices():
        info = data.get_image_info(idx)
        cam = data.get_camera(idx)
        name = _image_name(idx, info.name, index_to_filename)
        if cam is not None and name:
            pose = cam.pose()
            result[name] = {
                "idx": _loader_index(idx, info.name),
                "pose": pose,
                "R": pose.rotation().matrix(),
                "t": pose.translation(),
            }
    return result


def _fit_plane_basis(translations: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit a 2D plane to a set of 3D points via PCA. Returns (basis1, basis2, centroid)."""
    centroid = translations.mean(axis=0)
    centered = translations - centroid
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    return Vt[0], Vt[1], centroid


def _project_poses_to_2d(
    name_pose_map: Dict[str, Any],
    basis1: np.ndarray,
    basis2: np.ndarray,
    centroid: np.ndarray,
) -> List[Dict[str, Any]]:
    """Project 3D camera poses onto a 2D plane."""
    entries: List[Dict[str, Any]] = []
    for name in sorted(name_pose_map.keys()):
        info = name_pose_map[name]
        t = info["t"]
        R = info["R"]
        d = t - centroid
        u = float(np.dot(d, basis1))
        v = float(np.dot(d, basis2))
        # Camera x-axis and z-axis in world frame, projected onto plane.
        x_axis_3d = R[:, 0]
        z_axis_3d = R[:, 2]
        ax = [float(np.dot(x_axis_3d, basis1)), float(np.dot(x_axis_3d, basis2))]
        az = [float(np.dot(z_axis_3d, basis1)), float(np.dot(z_axis_3d, basis2))]
        # Normalize to unit direction.
        ax_len = (ax[0] ** 2 + ax[1] ** 2) ** 0.5
        az_len = (az[0] ** 2 + az[1] ** 2) ** 0.5
        if ax_len > 1e-8:
            ax = [ax[0] / ax_len, ax[1] / ax_len]
        if az_len > 1e-8:
            az = [az[0] / az_len, az[1] / az_len]
        entries.append({"name": name, "idx": info["idx"], "x": u, "y": v, "ax": ax, "az": az})
    return entries


def _rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle between two 3x3 rotation matrices, in degrees."""
    R_err = R1 @ R2.T
    cos_angle = (np.trace(R_err) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _translation_angle_deg(t1: np.ndarray, t2: np.ndarray) -> float:
    """Angular error between two translation direction vectors, in degrees."""
    n1 = np.linalg.norm(t1)
    n2 = np.linalg.norm(t2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 180.0
    cos_angle = np.dot(t1, t2) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))


def _compute_covisibility_edges(
    data: GtsfmData, index_to_filename: Dict[int, str] | None = None
) -> List[Dict[str, Any]]:
    """Compute covisibility edges (pairs of cameras sharing tracks) with shared track counts."""
    id_to_name: Dict[int, str] = {}
    for idx in data.get_valid_camera_indices():
        info = data.get_image_info(idx)
        name = _image_name(idx, info.name, index_to_filename)
        if name:
            id_to_name[idx] = name

    edge_counts: Counter = Counter()
    for track_idx in range(data.number_tracks()):
        track = data.get_track(track_idx)
        names = set()
        for m in range(track.numberMeasurements()):
            cid = track.measurement(m)[0]
            if cid in id_to_name:
                names.add(id_to_name[cid])
        for n1, n2 in itertools.combinations(sorted(names), 2):
            edge_counts[(n1, n2)] += 1
    return [{"i": n1, "j": n2, "n": cnt} for (n1, n2), cnt in edge_counts.items()]


def _build_pose_data_for_cluster(
    gt_data: GtsfmData | None,
    cluster_dir: Path,
    baseline_data: List[Tuple[str, GtsfmData]] | None = None,
    index_to_filename: Dict[int, str] | None = None,
) -> Dict[str, Any] | None:
    """Build 2D-projected camera pose data for one cluster node."""
    # Load estimated reconstructions.
    recon_data: Dict[str, GtsfmData] = {}
    for recon_name in RECON_OPTIONS:
        rdir = cluster_dir / recon_name
        if _has_colmap_files(rdir):
            try:
                recon_data[recon_name] = GtsfmData.read_colmap(str(rdir))
            except Exception as exc:
                logger.warning("Could not load %s for poses: %s", rdir, exc)

    if not recon_data and gt_data is None:
        return None

    gt_names = _name_pose_map(gt_data) if gt_data else {}
    est_names: Dict[str, Dict[str, Any]] = {}
    for rn, rd in recon_data.items():
        est_names[rn] = _name_pose_map(rd, index_to_filename)

    # Add baseline reconstructions (only cameras that overlap with this cluster's reconstructions).
    if baseline_data:
        cluster_cam_names: set[str] = set()
        for en in est_names.values():
            cluster_cam_names.update(en.keys())
        # Build basename -> cluster name lookup for matching across naming conventions.
        cluster_basename_to_name = {}
        for n in cluster_cam_names:
            cluster_basename_to_name[Path(n).name] = n
        for bl_key, bl_gtsfm in baseline_data:
            bl_names = _name_pose_map(bl_gtsfm)
            # Match by basename, then remap to cluster's naming convention.
            bl_filtered: Dict[str, Any] = {}
            for bl_name, bl_info in bl_names.items():
                base = Path(bl_name).name
                cluster_name = cluster_basename_to_name.get(base)
                if cluster_name is not None:
                    bl_filtered[cluster_name] = bl_info
            logger.info("Baseline %s: %d total cameras, %d matched to cluster.", bl_key, len(bl_names), len(bl_filtered))
            if bl_filtered:
                est_names[bl_key] = bl_filtered

    # Align each estimated reconstruction to GT via Sim(3).
    aligned_names: Dict[str, Dict[str, Any]] = {}
    for rn, en in est_names.items():
        if gt_names:
            common = sorted(set(gt_names) & set(en))
            if len(common) >= 2:
                gt_dict = {i: gt_names[n]["pose"] for i, n in enumerate(common)}
                est_dict = {i: en[n]["pose"] for i, n in enumerate(common)}
                try:
                    gt_S_est = sim3_from_Pose3_maps_robust(gt_dict, est_dict)
                except Exception as exc:
                    logger.warning("Sim(3) alignment failed for %s: %s", rn, exc)
                    aligned_names[rn] = en
                    continue
                # Apply transform to all estimated poses.
                aligned: Dict[str, Any] = {}
                for name, info in en.items():
                    new_pose = gt_S_est.transformFrom(info["pose"])
                    aligned[name] = {
                        "idx": info["idx"],
                        "pose": new_pose,
                        "R": new_pose.rotation().matrix(),
                        "t": new_pose.translation(),
                    }
                aligned_names[rn] = aligned
            else:
                logger.warning("Only %d common cameras for %s alignment, skipping.", len(common), rn)
                aligned_names[rn] = en
        else:
            aligned_names[rn] = en

    # Choose the primary set for plane fitting (GT if available, else vggt).
    primary = gt_names if gt_names else aligned_names.get("vggt", next(iter(aligned_names.values()), {}))
    if not primary:
        return None

    translations = np.array([info["t"] for info in primary.values()])
    basis1, basis2, centroid = _fit_plane_basis(translations)

    result: Dict[str, Any] = {}
    # Restrict GT to cameras that appear in at least one core reconstruction (not baselines).
    all_est_names = set()
    for rn, an in aligned_names.items():
        if rn in RECON_OPTIONS:
            all_est_names.update(an.keys())

    if gt_names:
        relevant_gt = {n: v for n, v in gt_names.items() if n in all_est_names} if all_est_names else gt_names
        result["gt"] = _project_poses_to_2d(relevant_gt, basis1, basis2, centroid)

    for rn, an in aligned_names.items():
        result[rn] = _project_poses_to_2d(an, basis1, basis2, centroid)

    # Compute arrow scale from core sets only (exclude baselines).
    all_pts = []
    for core_key in ["gt", "vggt", "vggt_pre_ba", "merged"]:
        if core_key in result and isinstance(result[core_key], list):
            for e in result[core_key]:
                all_pts.append([e["x"], e["y"]])
    if len(all_pts) > 1:
        pts = np.array(all_pts)
        extent = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min(), 1e-6)
        result["arrow_scale"] = float(extent * 0.04)
    else:
        result["arrow_scale"] = 1.0

    # Helper: add relative pose errors (estimated vs GT) to a list of edges.
    def _add_errors_to_edges(edges: List[Dict[str, Any]]) -> None:
        if not gt_names:
            return
        for edge in edges:
            ni, nj = edge["i"], edge["j"]
            if ni not in gt_names or nj not in gt_names:
                continue
            gt_pose_i = gt_names[ni]["pose"]
            gt_pose_j = gt_names[nj]["pose"]
            gt_rel = gt_pose_i.between(gt_pose_j)
            gt_R_rel = gt_rel.rotation().matrix()
            gt_t_rel = gt_rel.translation()
            for rn, an in aligned_names.items():
                if ni not in an or nj not in an:
                    continue
                est_rel = an[ni]["pose"].between(an[nj]["pose"])
                rot_err = _rotation_angle_deg(gt_R_rel, est_rel.rotation().matrix())
                trans_err = _translation_angle_deg(gt_t_rel, est_rel.translation())
                edge[rn] = {"rot_err": round(rot_err, 2), "trans_err": round(trans_err, 2)}

    def _build_all_pairs(covis_edges: List[Dict[str, Any]], cam_names: set) -> List[Dict[str, Any]]:
        """Extend covisibility edges with all remaining camera pairs."""
        covis_set = {(e["i"], e["j"]) for e in covis_edges}
        extra: List[Dict[str, Any]] = []
        for n1, n2 in itertools.combinations(sorted(cam_names), 2):
            if (n1, n2) not in covis_set:
                extra.append({"i": n1, "j": n2, "n": 0})
        _add_errors_to_edges(extra)
        return covis_edges + extra

    # Compute separate edge sets for vggt and merged.
    # vggt edges.
    vggt_ref = recon_data.get("vggt")
    vggt_covis = _compute_covisibility_edges(vggt_ref, index_to_filename) if vggt_ref else []
    _add_errors_to_edges(vggt_covis)
    vggt_cam_names = set(aligned_names.get("vggt", {}).keys())
    if gt_names:
        vggt_cam_names &= set(gt_names.keys()) | vggt_cam_names
    result["edges"] = vggt_covis
    result["all_edges"] = _build_all_pairs(vggt_covis, vggt_cam_names)

    # merged edges (if merged reconstruction exists).
    merged_ref = recon_data.get("merged")
    if merged_ref:
        merged_covis = _compute_covisibility_edges(merged_ref, index_to_filename)
        _add_errors_to_edges(merged_covis)
        merged_cam_names = set(aligned_names.get("merged", {}).keys())
        if gt_names:
            merged_cam_names &= set(gt_names.keys()) | merged_cam_names
        result["merged_edges"] = merged_covis
        result["merged_all_edges"] = _build_all_pairs(merged_covis, merged_cam_names)

    return result


def _write_html(output_dir: Path, title: str, *, serve_mode: bool) -> None:
    html = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>__TITLE__</title>
    <style>
      :root {{
        color-scheme: light dark;
        --bg: #0d1117;
        --bg-soft: #0f141c;
        --panel: #141923;
        --panel-strong: #181f2c;
        --text: #e6e6e6;
        --muted: #9aa4b2;
        --accent: #7aa2ff;
        --border: #263041;
        --highlight: #ffd166;
      }}
      * {{
        box-sizing: border-box;
      }}
      body {{
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: radial-gradient(circle at top, #101826 0%, #0b0f16 45%, #07090d 100%);
        color: var(--text);
      }}
      header {{
        position: sticky;
        top: 0;
        z-index: 5;
        display: flex;
        flex-wrap: wrap;
        gap: 12px 18px;
        align-items: center;
        padding: 14px 18px;
        border-bottom: 1px solid var(--border);
        background: rgba(20, 25, 35, 0.95);
        backdrop-filter: blur(8px);
      }}
      header h1 {{
        font-size: 16px;
        font-weight: 600;
        margin: 0 12px 0 0;
      }}
      select, label, input {{
        font-size: 13px;
        color: var(--text);
      }}
      select, input[type="text"] {{
        background: var(--panel-strong);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 8px;
      }}
      input[type="range"] {{
        accent-color: var(--accent);
      }}
      .layout {{
        display: grid;
        grid-template-columns: 1.15fr 1.85fr;
        height: calc(100vh - 64px);
      }}
      .left {{
        border-right: 1px solid var(--border);
        overflow: auto;
        padding: 16px;
        background: var(--bg-soft);
      }}
      .right {{
        display: flex;
        flex-direction: column;
        gap: 12px;
        padding: 16px;
        overflow-y: auto;
      }}
      .panel {{
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 12px;
      }}
      .sidebar-header {{
        display: flex;
        flex-direction: column;
        gap: 8px;
        margin-bottom: 12px;
      }}
      .thumb-card {{
        display: grid;
        grid-template-columns: 220px 1fr;
        gap: 12px;
        padding: 12px;
        border: 1px solid var(--border);
        border-radius: 10px;
        margin-bottom: 12px;
        background: var(--panel);
        cursor: pointer;
        transition: border-color 0.15s ease, transform 0.15s ease;
      }}
      .thumb-card:hover {{
        border-color: var(--accent);
        transform: translateY(-1px);
      }}
      .thumb-card.selected {{
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent) inset;
      }}
      canvas {{
        background: #0b0d12;
        border-radius: 8px;
        max-width: 100%;
      }}
      .meta {{
        display: flex;
        flex-direction: column;
        gap: 6px;
      }}
      .muted {{
        color: var(--muted);
      }}
      .controls {{
        display: flex;
        gap: 16px;
        align-items: center;
        flex-wrap: wrap;
      }}
      .zoom-pane {{
        display: grid;
        grid-template-columns: auto 1fr;
        gap: 10px;
        align-items: center;
      }}
      .zoom-canvas {{
        width: 240px;
        height: 240px;
        border: 1px solid var(--border);
      }}
      .status {{
        font-size: 13px;
        color: var(--muted);
      }}
      .track-pill {{
        padding: 2px 8px;
        border-radius: 999px;
        border: 1px solid var(--border);
        font-size: 12px;
        display: inline-block;
      }}
      .kpi {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        font-size: 12px;
      }}
      .kpi span {{
        background: #1a2332;
        border: 1px solid var(--border);
        border-radius: 999px;
        padding: 2px 8px;
      }}
      .button-row {{
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
      }}
      button {{
        background: #1a2332;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 6px 10px;
        cursor: pointer;
      }}
      button.active {{
        border-color: var(--accent);
        box-shadow: 0 0 0 1px var(--accent) inset;
      }}
      .legend-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-right: 10px;
      }}
      .dot {{
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
      }}
      .dot-reproj {{
        background: #f25f5c;
      }}
      .dot-meas {{
        border: 2px solid #00ff80;
        background: transparent;
      }}
      .track-patches-grid {{
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
        gap: 10px;
        max-height: 55vh;
        overflow: auto;
      }}
      .track-patch-card {{
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 8px;
        background: #101723;
      }}
      .track-patch-canvas {{
        width: 100%;
        aspect-ratio: 1 / 1;
        border: 1px solid var(--border);
      }}
      .hist-canvas {{
        width: 100%;
        height: 60px;
        border-radius: 6px;
        border: 1px solid var(--border);
        margin-top: 4px;
      }}
      .main-hist-panel {{
        display: none;
      }}
      .main-hist-controls {{
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
        margin-bottom: 8px;
      }}
      .main-hist-controls label {{
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 12px;
        color: var(--muted);
      }}
      #mainHistDiv {{
        height: 220px;
        width: 100%;
      }}
      .pose-panel {{
        display: none;
      }}
      .pose-controls {{
        display: flex;
        align-items: center;
        gap: 14px;
        flex-wrap: wrap;
        margin-bottom: 8px;
      }}
      .pose-controls label {{
        display: flex;
        align-items: center;
        gap: 5px;
        font-size: 12px;
        color: var(--muted);
      }}
      #poseSvg {{
        width: 100%;
        height: 450px;
        background: #080b11;
        border-radius: 8px;
        border: 1px solid var(--border);
        display: block;
      }}
      .pose-legend {{
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
        margin-top: 6px;
        font-size: 11px;
      }}
      .pose-legend-item {{
        display: flex;
        align-items: center;
        gap: 5px;
        color: var(--muted);
      }}
      .pose-legend-swatch {{
        width: 10px;
        height: 10px;
        border-radius: 2px;
      }}
      .auc-panel {{
        display: none;
      }}
      .auc-panel table {{
        border-collapse: collapse;
        width: 100%;
        font-size: 12px;
      }}
      .auc-panel th, .auc-panel td {{
        padding: 4px 10px;
        text-align: right;
        border-bottom: 1px solid var(--border);
      }}
      .auc-panel th {{
        text-align: right;
        color: var(--muted);
        font-weight: 600;
      }}
      .auc-panel th:first-child, .auc-panel td:first-child {{
        text-align: left;
      }}
    </style>
  </head>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js" charset="utf-8"></script>
  <script src="https://d3js.org/d3.v7.min.js" charset="utf-8"></script>
  <body>
    <header>
      <h1>Tracks Viewer</h1>
      <label>
        Cluster
        <select id="clusterSelect"></select>
      </label>
      <label>
        Recon
        <select id="reconSelect">
          <option value="vggt">vggt</option>
          <option value="vggt_pre_ba">vggt_pre_ba</option>
          <option value="merged">merged</option>
        </select>
      </label>
      <label>
        <input type="checkbox" id="lineToggle" checked />
        Show reprojection line
      </label>
      <label>
        Crop size
        <input type="range" id="cropSize" min="30" max="60" value="40" />
      </label>
      <label>
        Patch size (YxY)
        <input type="number" id="patchSizeInput" min="16" max="2048" step="16" value="128" style="width:90px" />
      </label>
      <label>
        Max tracks / patch
        <input type="number" id="maxPerPatchInput" min="1" max="100" step="1" value="1" style="width:70px" />
      </label>
      <span class="status" id="statusText"></span>
    </header>
    <div class="layout">
      <div class="left">
        <div class="sidebar-header">
          <input type="text" id="imageFilter" placeholder="Filter images by name or id..." />
          <div class="kpi">
            <span id="imageCount">0 images</span>
            <span id="trackCount">0 tracks</span>
          </div>
        </div>
        <div id="thumbList"></div>
      </div>
      <div class="right">
        <div class="panel controls">
          <span class="track-pill" id="trackBadge">Track: none</span>
          <span class="muted" id="imageBadge"></span>
          <span class="muted">Click any track in thumbnail or main image to highlight everywhere.</span>
          <div class="legend-chip"><span class="dot dot-reproj"></span><span class="muted">solid dot = reprojection</span></div>
          <div class="legend-chip"><span class="dot dot-meas"></span><span class="muted">green circle = measured point</span></div>
        </div>
        <div class="panel">
          <div class="button-row">
            <button id="modeAllBtn" class="active">Show all tracks</button>
            <button id="modeBestBtn">Best / patch</button>
            <button id="modeWorstBtn">Worst / patch</button>
            <button id="modeRandomBtn">Random / patch</button>
          </div>
        </div>
        <div class="panel">
          <canvas id="mainCanvas"></canvas>
        </div>
        <div class="panel zoom-pane">
          <canvas id="zoomCanvas" class="zoom-canvas" width="240" height="240"></canvas>
          <div class="muted">Hover the main image to inspect pixels.</div>
        </div>
        <div class="panel">
          <div class="muted" id="patchesInfo">Select a track to view patches across images.</div>
          <div id="trackPatchesGrid" class="track-patches-grid"></div>
        </div>
        <div class="panel main-hist-panel" id="mainHistPanel">
          <div class="main-hist-controls">
            <span style="font-size:12px;color:var(--muted);font-weight:500;">Reprojection error distribution</span>
            <label>Bins <input type="number" id="histBinsInput" min="5" max="500" step="5" value="30" style="width:60px;" /></label>
            <label><input type="checkbox" id="histLogYToggle" /> Log Y</label>
            <label><input type="checkbox" id="histShowCdfToggle" /> Show CDF</label>
          </div>
          <div id="mainHistDiv"></div>
        </div>
        <div class="panel pose-panel" id="posePanel">
          <div class="pose-controls">
            <span style="font-size:12px;color:var(--muted);font-weight:500;">Camera poses (2D plane projection)</span>
            <label><input type="checkbox" id="poseShowGt" checked /> GT</label>
            <label><input type="checkbox" id="poseShowPreBa" checked /> Pre-BA</label>
            <label><input type="checkbox" id="poseShowVggt" checked /> Post-BA</label>
            <label><input type="checkbox" id="poseShowMerged" checked /> Merged</label>
            <span id="poseBaselineControls"></span>
            <label><input type="checkbox" id="poseShowEdges" checked /> Edges</label>
            <label><input type="checkbox" id="poseShowCut" checked /> Partition</label>
            <label>Edge set
              <select id="poseEdgeSet" style="font-size:12px;">
                <option value="covis" selected>Covisibility</option>
                <option value="all">All pairs</option>
              </select>
            </label>
            <label><input type="checkbox" id="poseShowLabels" /> IDs</label>
            <label><input type="checkbox" id="poseFlip" /> Flip</label>
            <label>Edge color
              <select id="poseEdgeMetric" style="font-size:12px;">
                <option value="max" selected>Max(R,T)</option>
                <option value="rot">Rotation</option>
                <option value="trans">Translation</option>
                <option value="none">None</option>
              </select>
            </label>
          </div>
          <svg id="poseSvg"></svg>
          <div class="pose-legend">
            <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#4ade80;"></div> GT</div>
            <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#f59e0b;"></div> Pre-BA</div>
            <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#60a5fa;"></div> Post-BA</div>
            <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#c084fc;"></div> Merged</div>
            <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#ef4444;"></div> X axis</div>
            <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#3b82f6;"></div> Z axis (view dir)</div>
            <span id="poseCutLegend" style="display:contents;">
              <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#67e8f9;"></div> Left child</div>
              <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#fdba74;"></div> Right child</div>
              <div class="pose-legend-item"><div class="pose-legend-swatch" style="background:#c4b5fd;"></div> Borrowed</div>
            </span>
            <span id="poseBaselineLegend"></span>
            <div class="pose-legend-item" id="edgeColorLegend" style="display:none;gap:4px;">
              <span style="font-size:11px;color:var(--muted);">Edge err:</span>
              <div style="width:100px;height:10px;border-radius:2px;background:linear-gradient(to right,#22c55e,#eab308,#ef4444);"></div>
              <span style="font-size:10px;color:var(--muted);" id="edgeColorMax">0-10&deg;</span>
            </div>
          </div>
        </div>
        <div class="panel auc-panel" id="aucPanel">
          <div style="font-size:12px;color:var(--muted);font-weight:500;margin-bottom:8px;">Pose AUC (edge-based)</div>
          <div id="aucContent" style="font-family:monospace;font-size:12px;line-height:1.6;color:var(--fg);"></div>
        </div>
      </div>
    </div>
    <script>
      const serverMode = __SERVER_MODE__;
    </script>
    <script src="data.js"></script>
    <script>
      const data = window.TRACKS_DATA || { clusters: [], imagesByCluster: {}, defaultRecon: "vggt", baseImageUrl: null };
      // Baseline colors: magenta, cyan.
      const BASELINE_COLORS = ["#e879f9", "#22d3ee"];
      // Dynamically create baseline checkboxes and legend entries.
      function initBaselines() {
        const baselines = data.baselines || [];
        if (!baselines.length) return;
        const ctrlContainer = document.getElementById("poseBaselineControls");
        const legendContainer = document.getElementById("poseBaselineLegend");
        if (ctrlContainer.children.length) return;  // Already initialized.
        baselines.forEach((bl, i) => {
          const lbl = document.createElement("label");
          const cb = document.createElement("input");
          cb.type = "checkbox";
          cb.id = "poseShowBaseline_" + bl.key;
          cb.checked = true;
          lbl.appendChild(cb);
          lbl.appendChild(document.createTextNode(" " + bl.label));
          ctrlContainer.appendChild(lbl);
          // Legend swatch.
          const legendItem = document.createElement("div");
          legendItem.className = "pose-legend-item";
          const swatch = document.createElement("div");
          swatch.className = "pose-legend-swatch";
          swatch.style.background = BASELINE_COLORS[i] || "#a78bfa";
          legendItem.appendChild(swatch);
          legendItem.appendChild(document.createTextNode(" " + bl.label));
          legendContainer.appendChild(legendItem);
        });
      }
      // Run immediately for static mode; in serve mode it's called again after fetchClustersIfNeeded.
      initBaselines();
      const clusterSelect = document.getElementById("clusterSelect");
      const reconSelect = document.getElementById("reconSelect");
      const thumbList = document.getElementById("thumbList");
      const statusText = document.getElementById("statusText");
      const lineToggle = document.getElementById("lineToggle");
      const cropSize = document.getElementById("cropSize");
      const mainCanvas = document.getElementById("mainCanvas");
      const zoomCanvas = document.getElementById("zoomCanvas");
      const trackBadge = document.getElementById("trackBadge");
      const imageBadge = document.getElementById("imageBadge");
      const imageFilter = document.getElementById("imageFilter");
      const imageCount = document.getElementById("imageCount");
      const trackCount = document.getElementById("trackCount");
      const patchSizeInput = document.getElementById("patchSizeInput");
      const maxPerPatchInput = document.getElementById("maxPerPatchInput");
      const modeAllBtn = document.getElementById("modeAllBtn");
      const modeBestBtn = document.getElementById("modeBestBtn");
      const modeWorstBtn = document.getElementById("modeWorstBtn");
      const modeRandomBtn = document.getElementById("modeRandomBtn");
      const patchesInfo = document.getElementById("patchesInfo");
      const trackPatchesGrid = document.getElementById("trackPatchesGrid");
      const baseMainCanvas = document.createElement("canvas");
      const baseMainCtx = baseMainCanvas.getContext("2d");
      const imageCache = new Map();

      const state = {
        clusterId: null,
        recon: data.defaultRecon || "vggt",
        selectedImageId: null,
        selectedTrackId: null,
        showLine: true,
        cropSize: Number(cropSize.value),
        filterText: "",
        patchSize: Number(patchSizeInput.value),
        maxPerPatch: Number(maxPerPatchInput.value),
        patchMode: "all",
        shownTrackIds: null,
        loadingImages: false,
        poseFocusedName: null,
      };

      function hsvToRgb(h, s, v) {
        const i = Math.floor(h * 6);
        const f = h * 6 - i;
        const p = v * (1 - s);
        const q = v * (1 - f * s);
        const t = v * (1 - (1 - f) * s);
        let r, g, b;
        switch (i % 6) {
          case 0: r = v; g = t; b = p; break;
          case 1: r = q; g = v; b = p; break;
          case 2: r = p; g = v; b = t; break;
          case 3: r = p; g = q; b = v; break;
          case 4: r = t; g = p; b = v; break;
          case 5: r = v; g = p; b = q; break;
        }
        return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
      }

      function trackColor(trackId) {
        const hue = (trackId * 0.61803398875) % 1.0;
        return hsvToRgb(hue, 0.7, 0.95);
      }

      function setStatus(msg) {
        statusText.textContent = msg || "";
      }

      function applyFilter(images) {
        const term = state.filterText.trim().toLowerCase();
        if (!term) {
          return images;
        }
        return images.filter((img) => {
          const name = img.name ? img.name.toLowerCase() : "";
          return name.includes(term) || String(img.image_id).includes(term);
        });
      }

      function imageMeanReprojErr(imageData) {
        if (!imageData || !imageData.tracks || !imageData.tracks.length) return 0.0;
        let sum = 0.0;
        let count = 0;
        imageData.tracks.forEach((t) => {
          if (Number.isFinite(t.reproj_err)) {
            sum += t.reproj_err;
            count += 1;
          }
        });
        return count > 0 ? sum / count : 0.0;
      }

      function getEffectiveRecon() {
        const cluster = data.imagesByCluster[state.clusterId] || {};
        if (state.recon === "merged" && !cluster["merged"]) return "vggt";
        return state.recon;
      }

      function getClusterImages() {
        const cluster = data.imagesByCluster[state.clusterId] || {};
        return cluster[getEffectiveRecon()] || [];
      }

      function getSelectedImageData(images) {
        if (!images || !images.length || state.selectedImageId === null) {
          return null;
        }
        return images.find((img) => img.image_id === state.selectedImageId) || null;
      }

      function setPatchMode(mode) {
        state.patchMode = mode;
        modeAllBtn.classList.toggle("active", mode === "all");
        modeBestBtn.classList.toggle("active", mode === "best");
        modeWorstBtn.classList.toggle("active", mode === "worst");
        modeRandomBtn.classList.toggle("active", mode === "random");
      }

      function computePatchSelection(imageData, mode) {
        if (!imageData || mode === "all") {
          return null;
        }
        const patchSize = Math.max(1, Number(state.patchSize) || 128);
        const maxPerPatch = Math.max(1, Number(state.maxPerPatch) || 1);
        const buckets = new Map();
        imageData.tracks.forEach((track) => {
          const py = Math.floor(track.meas[1] / patchSize);
          const px = Math.floor(track.meas[0] / patchSize);
          const key = `${py}_${px}`;
          if (!buckets.has(key)) {
            buckets.set(key, []);
          }
          buckets.get(key).push(track);
        });

        const chosen = new Set();
        buckets.forEach((tracks) => {
          let ordered = tracks.slice();
          if (mode === "random") {
            for (let i = ordered.length - 1; i > 0; i--) {
              const j = Math.floor(Math.random() * (i + 1));
              const tmp = ordered[i];
              ordered[i] = ordered[j];
              ordered[j] = tmp;
            }
          } else if (mode === "best") {
            ordered.sort((a, b) => {
              if (b.track_len !== a.track_len) return b.track_len - a.track_len;
              return a.reproj_err - b.reproj_err;
            });
          } else if (mode === "worst") {
            ordered.sort((a, b) => {
              if (b.track_len !== a.track_len) return b.track_len - a.track_len;
              return b.reproj_err - a.reproj_err;
            });
          }
          ordered.slice(0, maxPerPatch).forEach((t) => chosen.add(t.track_id));
        });
        if (state.selectedTrackId !== null) {
          chosen.add(state.selectedTrackId);
        }
        return chosen;
      }

      async function fetchClustersIfNeeded() {
        if (!serverMode || data.clusters.length) return;
        const resp = await fetch("/api/clusters");
        const payload = await resp.json();
        data.clusters = payload.clusters || [];
        data.defaultRecon = payload.defaultRecon || "vggt";
        data.baseImageUrl = payload.baseImageUrl || null;
        if (payload.baselines) data.baselines = payload.baselines;
      }

      async function fetchImagesIfNeeded() {
        if (!serverMode || !state.clusterId) return;
        const cluster = data.imagesByCluster[state.clusterId] || {};
        const effectiveRecon = getEffectiveRecon();
        if (cluster[effectiveRecon]) return;
        state.loadingImages = true;
        setStatus(`Loading images for ${state.clusterId} (${state.recon})...`);
        const resp = await fetch(
          `/api/images?cluster=${encodeURIComponent(state.clusterId)}&recon=${encodeURIComponent(state.recon)}`
        );
        const payload = await resp.json();
        data.imagesByCluster[state.clusterId] = data.imagesByCluster[state.clusterId] || {};
        const images = payload.images || [];
        data.imagesByCluster[state.clusterId][state.recon] = images;
        // If merged returned empty, fallback: fetch vggt instead.
        if (state.recon === "merged" && !images.length && !cluster["vggt"]) {
          const resp2 = await fetch(
            `/api/images?cluster=${encodeURIComponent(state.clusterId)}&recon=vggt`
          );
          const payload2 = await resp2.json();
          data.imagesByCluster[state.clusterId]["vggt"] = payload2.images || [];
        }
        state.loadingImages = false;
      }

      async function fetchPosesIfNeeded() {
        if (!serverMode || !state.clusterId) return;
        if (!data.posesByCluster) data.posesByCluster = {};
        if (data.posesByCluster[state.clusterId]) return;
        try {
          const resp = await fetch(`/api/poses?cluster=${encodeURIComponent(state.clusterId)}`);
          const payload = await resp.json();
          if (payload && Object.keys(payload).length) {
            data.posesByCluster[state.clusterId] = payload;
          }
        } catch (e) {
          console.warn("Failed to fetch pose data:", e);
        }
      }

      function populateClusters() {
        const prevClusterId = state.clusterId;
        clusterSelect.innerHTML = "";
        data.clusters.forEach((cluster) => {
          const option = document.createElement("option");
          option.value = cluster.id;
          option.textContent = `${cluster.label}`;
          clusterSelect.appendChild(option);
        });
        const hasPrev = prevClusterId && data.clusters.some((cluster) => cluster.id === prevClusterId);
        state.clusterId = hasPrev ? prevClusterId : (data.clusters.length ? data.clusters[0].id : null);
        clusterSelect.value = state.clusterId;
      }

      function drawTracks(ctx, tracks, scaleX, scaleY, selectedTrackId, showLine, shownTrackIds) {
        tracks.forEach((track) => {
          if (shownTrackIds && !shownTrackIds.has(track.track_id) && track.track_id !== selectedTrackId) {
            return;
          }
          const [r, g, b] = trackColor(track.track_id);
          const isSelected = selectedTrackId !== null && track.track_id === selectedTrackId;
          const isDimmed = selectedTrackId !== null && !isSelected;
          const lineWidth = isSelected ? 2.5 : 1.2;
          const dotRadius = isSelected ? 4 : 2.2;
          const measX = track.meas[0] * scaleX;
          const measY = track.meas[1] * scaleY;
          const repX = track.reproj[0] * scaleX;
          const repY = track.reproj[1] * scaleY;

          if (showLine) {
            ctx.strokeStyle = isSelected
              ? "rgba(255, 209, 102, 0.95)"
              : (isDimmed ? "rgba(255,255,255,0.08)" : "rgba(255, 255, 255, 0.35)");
            ctx.lineWidth = lineWidth;
            ctx.beginPath();
            ctx.moveTo(measX, measY);
            ctx.lineTo(repX, repY);
            ctx.stroke();
          }

          const fillAlpha = isDimmed ? 0.18 : 0.95;
          ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${fillAlpha})`;
          ctx.beginPath();
          ctx.arc(repX, repY, dotRadius, 0, Math.PI * 2);
          ctx.fill();

          ctx.strokeStyle = isDimmed ? "rgba(0, 255, 128, 0.25)" : "rgba(0, 255, 128, 0.9)";
          ctx.lineWidth = Math.max(1, lineWidth);
          ctx.beginPath();
          ctx.arc(measX, measY, dotRadius, 0, Math.PI * 2);
          ctx.stroke();

          if (isSelected) {
            ctx.strokeStyle = "rgba(255, 209, 102, 1.0)";
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            ctx.arc(measX, measY, dotRadius + 3, 0, Math.PI * 2);
            ctx.stroke();
          }
        });
      }

      function imageSrc(imageData) {
        if (data.baseImageUrl) {
          return `${data.baseImageUrl}?name=${encodeURIComponent(imageData.name)}`;
        }
        return imageData.file_path;
      }

      function loadImageCached(src) {
        if (imageCache.has(src)) {
          return imageCache.get(src);
        }
        const promise = new Promise((resolve, reject) => {
          const img = new Image();
          img.onload = () => resolve(img);
          img.onerror = () => reject(new Error(`Failed to load image: ${src}`));
          img.src = src;
        });
        imageCache.set(src, promise);
        return promise;
      }

      function renderToColmapCanvas(img, targetWidth, targetHeight) {
        const tmpCanvas = document.createElement("canvas");
        tmpCanvas.width = targetWidth;
        tmpCanvas.height = targetHeight;
        const tmpCtx = tmpCanvas.getContext("2d");
        tmpCtx.clearRect(0, 0, targetWidth, targetHeight);
        // Normalize image into COLMAP frame so track (x,y) aligns with cropping coordinates.
        tmpCtx.drawImage(img, 0, 0, targetWidth, targetHeight);
        return tmpCanvas;
      }

      function drawSelectedTrackMarkers(ctx, measX, measY, reprojX, reprojY) {
        // Measured point: green cross.
        ctx.strokeStyle = "rgba(0, 255, 128, 1.0)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(measX - 4, measY);
        ctx.lineTo(measX + 4, measY);
        ctx.moveTo(measX, measY - 4);
        ctx.lineTo(measX, measY + 4);
        ctx.stroke();
        // Reprojected point: yellow cross.
        ctx.strokeStyle = "rgba(255, 220, 120, 1.0)";
        ctx.beginPath();
        ctx.moveTo(reprojX - 4, reprojY);
        ctx.lineTo(reprojX + 4, reprojY);
        ctx.moveTo(reprojX, reprojY - 4);
        ctx.lineTo(reprojX, reprojY + 4);
        ctx.stroke();
      }

      function renderThumbnail(imageData, container, onSelect) {
        const card = document.createElement("div");
        card.className = "thumb-card";
        if (imageData.image_id === state.selectedImageId) {
          card.classList.add("selected");
        }
        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        const targetW = 220;
        const scale = targetW / imageData.width;
        canvas.width = targetW;
        canvas.height = Math.round(imageData.height * scale);

        const img = new Image();
        img.onload = () => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          drawTracks(
            ctx,
            imageData.tracks,
            scale,
            scale,
            state.selectedTrackId,
            state.showLine,
            state.shownTrackIds
          );
        };
        img.src = imageSrc(imageData);

        card.addEventListener("click", () => onSelect(imageData));
        canvas.addEventListener("click", (evt) => {
          evt.stopPropagation();
          const rect = canvas.getBoundingClientRect();
          const x = evt.clientX - rect.left;
          const y = evt.clientY - rect.top;
          const trackId = pickTrack(imageData, x / scale, y / scale, state.shownTrackIds);
          if (trackId !== null) {
            state.selectedTrackId = trackId;
            const t = imageData.tracks.find((tr) => tr.track_id === trackId);
            if (t) {
              trackBadge.textContent = `Track: ${trackId} | len=${t.track_len} | err=${t.reproj_err.toFixed(2)}px`;
            } else {
              trackBadge.textContent = `Track: ${trackId}`;
            }
            renderAll();
          }
        });

        const meta = document.createElement("div");
        meta.className = "meta";
        meta.innerHTML = `<div><strong>${imageData.name}</strong></div>
          <div class="muted">ID: ${imageData.image_id}</div>
          <div class="muted">Tracks: ${imageData.tracks.length}</div>
          <div class="muted" style="font-size:10px;margin-top:2px;">Reproj error (px)</div>`;
        const histCanvas = document.createElement("canvas");
        histCanvas.className = "hist-canvas";
        meta.appendChild(histCanvas);
        card.appendChild(canvas);
        card.appendChild(meta);
        container.appendChild(card);
        // Render histogram after card is in DOM so clientWidth is available.
        requestAnimationFrame(() => renderHistogram(imageData, histCanvas));
      }

      function pickTrack(imageData, x, y, shownTrackIds) {
        let best = null;
        let bestDist = Infinity;
        const radius = 10;
        imageData.tracks.forEach((track) => {
          if (shownTrackIds && !shownTrackIds.has(track.track_id)) {
            return;
          }
          const dx = track.meas[0] - x;
          const dy = track.meas[1] - y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < radius && dist < bestDist) {
            bestDist = dist;
            best = track.track_id;
          }
        });
        return best;
      }

      function renderMainImage(imageData) {
        if (!imageData) {
          mainCanvas.width = 10;
          mainCanvas.height = 10;
          baseMainCanvas.width = 10;
          baseMainCanvas.height = 10;
          const ctx = mainCanvas.getContext("2d");
          ctx.clearRect(0, 0, 10, 10);
          baseMainCtx.clearRect(0, 0, 10, 10);
          return;
        }
        mainCanvas.width = imageData.width;
        mainCanvas.height = imageData.height;
        baseMainCanvas.width = imageData.width;
        baseMainCanvas.height = imageData.height;
        const ctx = mainCanvas.getContext("2d");
        const img = new Image();
        img.onload = () => {
          baseMainCtx.clearRect(0, 0, imageData.width, imageData.height);
          baseMainCtx.drawImage(img, 0, 0, imageData.width, imageData.height);
          ctx.drawImage(img, 0, 0, imageData.width, imageData.height);
          drawTracks(
            ctx,
            imageData.tracks,
            1,
            1,
            state.selectedTrackId,
            state.showLine,
            state.shownTrackIds
          );
        };
        img.src = imageSrc(imageData);
        imageBadge.textContent = `${imageData.name} (${imageData.width}x${imageData.height})`;
      }

      function renderHistogram(imageData, targetCanvas) {
        const ctx = targetCanvas.getContext("2d");
        const width = (targetCanvas.width = targetCanvas.clientWidth || 260);
        const height = (targetCanvas.height = targetCanvas.clientHeight || 60);
        ctx.clearRect(0, 0, width, height);
        if (!imageData || !imageData.tracks || !imageData.tracks.length) {
          ctx.fillStyle = "rgba(150,150,160,0.8)";
          ctx.font = "11px system-ui";
          ctx.fillText("No tracks for this image.", 8, height / 2);
          return;
        }
        const errors2 = imageData.tracks.map((t) => t.reproj_err);
        const nBins = 30;
        const minVal = 0.0;
        const sorted = errors2.slice().sort((a, b) => a - b);
        const maxRaw = sorted[sorted.length - 1];
        const p95 = sorted[Math.floor(0.95 * (sorted.length - 1))];
        const maxVal = Math.max(p95, maxRaw * 0.5, 1e-3);
        const binWidth = (maxVal - minVal) / nBins;
        const bins = new Array(nBins).fill(0);
        errors2.forEach((v) => {
          if (!Number.isFinite(v)) return;
          const clamped = Math.min(Math.max(v, minVal), maxVal - 1e-9);
          const idx = Math.floor((clamped - minVal) / binWidth);
          bins[idx] += 1;
        });
        const maxCount = Math.max(...bins, 1);

        // Background
        ctx.fillStyle = "rgba(20,24,36,1)";
        ctx.fillRect(0, 0, width, height);

        // Axes
        const padLeft = 32;
        const padBottom = 14;
        const plotW = width - padLeft - 4;
        const plotH = height - padBottom - 4;
        const originX = padLeft;
        const originY = height - padBottom;

        ctx.strokeStyle = "rgba(60,70,90,1)";
        ctx.lineWidth = 1;
        ctx.beginPath();
        // X axis
        ctx.moveTo(originX, originY + 0.5);
        ctx.lineTo(originX + plotW, originY + 0.5);
        // Y axis
        ctx.moveTo(originX + 0.5, originY);
        ctx.lineTo(originX + 0.5, originY - plotH);
        ctx.stroke();

        ctx.fillStyle = "rgba(120,130,150,0.8)";
        ctx.font = "9px system-ui";

        // X ticks and labels (0 to maxVal)
        const xTicks = 4;
        for (let i = 0; i <= xTicks; i++) {
          const t = i / xTicks;
          const val = minVal + t * (maxVal - minVal);
          const x = originX + t * plotW;
          ctx.strokeStyle = "rgba(70,80,95,1)";
          ctx.beginPath();
          ctx.moveTo(x, originY);
          ctx.lineTo(x, originY + 4);
          ctx.stroke();
          const label = val.toFixed(1);
          const textW = ctx.measureText(label).width;
          ctx.fillText(label, x - textW / 2, originY + 12);
        }

        // Y ticks and labels (0 to maxCount)
        const yTicks = 3;
        for (let i = 0; i <= yTicks; i++) {
          const t = i / yTicks;
          const countVal = t * maxCount;
          const y = originY - t * plotH;
          ctx.strokeStyle = "rgba(50,60,80,0.6)";
          ctx.beginPath();
          ctx.moveTo(originX - 3, y);
          ctx.lineTo(originX, y);
          ctx.stroke();
          const label = Math.round(countVal).toString();
          const textW = ctx.measureText(label).width;
          ctx.fillText(label, originX - 6 - textW, y + 3);
        }

        const barW = width / nBins;
        bins.forEach((count, i) => {
          const frac = count / maxCount;
          const barH = frac * plotH;
          const x = originX + i * (plotW / nBins);
          const y = originY - barH;
          ctx.fillStyle = "rgba(122,162,255,0.8)";
          ctx.fillRect(x, y, Math.max(1, (plotW / nBins) - 1), barH);
        });
      }

      async function renderSelectedTrackPatches(images) {
        trackPatchesGrid.innerHTML = "";
        if (state.selectedTrackId === null) {
          patchesInfo.textContent = "Select a track to view patches across images.";
          return;
        }
        const patchImages = images
          .map((img) => {
            const t = img.tracks.find((tr) => tr.track_id === state.selectedTrackId);
            return t ? { image: img, track: t } : null;
          })
          .filter((v) => v !== null);
        patchesInfo.textContent = `Track ${state.selectedTrackId} appears in ${patchImages.length} image(s).`;
        if (!patchImages.length) {
          return;
        }
        const patchSize = Math.max(30, Number(state.cropSize) || 40);
        const outSize = 180;
        for (const item of patchImages) {
          const card = document.createElement("div");
          card.className = "track-patch-card";
          const label = document.createElement("div");
          label.className = "muted";
          label.textContent = `${item.image.name} (ID ${item.image.image_id})`;
          const canvas = document.createElement("canvas");
          canvas.className = "track-patch-canvas";
          canvas.width = outSize;
          canvas.height = outSize;
          card.appendChild(label);
          card.appendChild(canvas);
          trackPatchesGrid.appendChild(card);

          try {
            const img = await loadImageCached(imageSrc(item.image));
            const ctx = canvas.getContext("2d");
            const normalizedCanvas = renderToColmapCanvas(img, item.image.width, item.image.height);
            const cx = Math.round(item.track.meas[0]);
            const cy = Math.round(item.track.meas[1]);
            const half = Math.floor(patchSize / 2);
            const sx = Math.max(0, cx - half);
            const sy = Math.max(0, cy - half);
            const sw = Math.min(patchSize, item.image.width - sx);
            const sh = Math.min(patchSize, item.image.height - sy);
            ctx.clearRect(0, 0, outSize, outSize);
            ctx.imageSmoothingEnabled = false;
            ctx.drawImage(normalizedCanvas, sx, sy, sw, sh, 0, 0, outSize, outSize);
            const measX = ((item.track.meas[0] - sx) / sw) * outSize;
            const measY = ((item.track.meas[1] - sy) / sh) * outSize;
            const reprojX = ((item.track.reproj[0] - sx) / sw) * outSize;
            const reprojY = ((item.track.reproj[1] - sy) / sh) * outSize;
            drawSelectedTrackMarkers(ctx, measX, measY, reprojX, reprojY);
          } catch (_) {
            // Keep card with label if image failed to load.
          }
        }
      }

      // ---- Camera pose visualization (D3) ----
      let poseZoomBehavior = null;

      function renderPoseVisualization() {
        const panel = document.getElementById("posePanel");
        const poseData = data.posesByCluster ? data.posesByCluster[state.clusterId] : null;
        if (!poseData) {
          panel.style.display = "none";
          return;
        }
        panel.style.display = "block";

        const showGt = document.getElementById("poseShowGt").checked;
        const showPreBa = document.getElementById("poseShowPreBa").checked;
        const showVggt = document.getElementById("poseShowVggt").checked;
        const showMerged = document.getElementById("poseShowMerged").checked;
        const showEdges = document.getElementById("poseShowEdges").checked;
        const showCut = document.getElementById("poseShowCut").checked;
        const showLabels = document.getElementById("poseShowLabels").checked;
        const flip = document.getElementById("poseFlip").checked ? -1 : 1;
        const edgeSetKey = document.getElementById("poseEdgeSet").value;
        // Pick edges from merged or vggt depending on the merged toggle.
        const useMergedEdges = showMerged && poseData.merged_edges;
        const covisEdges = useMergedEdges ? poseData.merged_edges : (poseData.edges || []);
        const allPairEdges = useMergedEdges ? (poseData.merged_all_edges || covisEdges) : (poseData.all_edges || covisEdges);
        const activeEdges = edgeSetKey === "all" ? allPairEdges : covisEdges;

        const svg = d3.select("#poseSvg");
        const rect = svg.node().getBoundingClientRect();
        const width = rect.width || 600;
        const height = rect.height || 450;
        svg.attr("viewBox", `0 0 ${width} ${height}`);

        // Collect all visible points for scaling.
        const allPts = [];
        const sets = [];
        if (showGt && poseData.gt) sets.push(poseData.gt);
        if (showVggt && poseData.vggt) sets.push(poseData.vggt);
        if (showPreBa && poseData.vggt_pre_ba) sets.push(poseData.vggt_pre_ba);
        // Always use core sets (gt, vggt, vggt_pre_ba) for stable extents.
        // Baselines are excluded so they don't blow up the scale.
        ["gt", "vggt", "vggt_pre_ba", "merged"].forEach((k) => {
          if (poseData[k]) poseData[k].forEach((c) => allPts.push(c));
        });
        if (!allPts.length) { panel.style.display = "none"; return; }

        const xExt = d3.extent(allPts, (d) => d.x);
        const yExt = d3.extent(allPts, (d) => d.y);
        const margin = 40;
        const dataW = (xExt[1] - xExt[0]) || 1;
        const dataH = (yExt[1] - yExt[0]) || 1;
        const plotW = width - 2 * margin;
        const plotH = height - 2 * margin;
        const scale = Math.min(plotW / dataW, plotH / dataH) * 0.85;
        const cxData = (xExt[0] + xExt[1]) / 2;
        const cyData = (yExt[0] + yExt[1]) / 2;

        const xS = (d) => margin + plotW / 2 + (d - cxData) * scale;
        const yS = (d) => margin + plotH / 2 - (d - cyData) * scale * flip;

        const arrowLen = (poseData.arrow_scale || 1) * scale;

        // Remove old content, keep one persistent <g>.
        svg.selectAll("*").remove();
        const g = svg.append("g");

        // Setup zoom.
        if (!poseZoomBehavior) {
          poseZoomBehavior = d3.zoom()
            .scaleExtent([0.2, 80])
            .on("zoom", (event) => {
              svg.select("g").attr("transform", event.transform);
            });
        }
        svg.call(poseZoomBehavior);

        // Build name->screen-position lookup for edges.
        const posLookup = {};
        const buildLookup = (arr, key) => {
          if (!arr) return;
          arr.forEach((c) => {
            if (!posLookup[c.name]) posLookup[c.name] = {};
            posLookup[c.name][key] = { sx: xS(c.x), sy: yS(c.y) };
          });
        };
        buildLookup(poseData.gt, "gt");
        buildLookup(poseData.vggt, "vggt");
        buildLookup(poseData.vggt_pre_ba, "vggt_pre_ba");
        buildLookup(poseData.merged, "merged");
        (data.baselines || []).forEach((bl) => buildLookup(poseData[bl.key], bl.key));

        // Edge error coloring.
        const edgeMetric = document.getElementById("poseEdgeMetric").value;
        const edgeLegend = document.getElementById("edgeColorLegend");
        const edgeColorMax = document.getElementById("edgeColorMax");
        // Use the currently selected reconstruction for error lookups.
        const errRecon = getEffectiveRecon();
        // Determine the error value for an edge given the metric.
        function edgeError(e) {
          const errs = e[errRecon];
          if (!errs) return null;
          if (edgeMetric === "rot") return errs.rot_err;
          if (edgeMetric === "trans") return errs.trans_err;
          if (edgeMetric === "max") return Math.max(errs.rot_err, errs.trans_err);
          return null;
        }
        // Build a color scale: green (0) -> yellow (mid) -> red (high).
        let errColorScale = null;
        let maxErrVal = 10;
        if (edgeMetric !== "none" && activeEdges.length) {
          const errVals = activeEdges.map(edgeError).filter((v) => v !== null && Number.isFinite(v));
          if (errVals.length) {
            maxErrVal = Math.max(d3.quantile(errVals.sort(d3.ascending), 0.95), 1);
            errColorScale = d3.scaleSequential(d3.interpolateRdYlGn).domain([maxErrVal, 0]);
            edgeLegend.style.display = "flex";
            const metricLabel = edgeMetric === "rot" ? "rot" : edgeMetric === "trans" ? "trans" : "max(R,T)";
            edgeColorMax.textContent = `0-${maxErrVal.toFixed(1)}\\u00b0 (${metricLabel})`;
          } else {
            edgeLegend.style.display = "none";
          }
        } else {
          edgeLegend.style.display = "none";
        }

        // Voronoi cells of the camera centers, shaded by partition membership.
        const cutLegend = document.getElementById("poseCutLegend");
        if (cutLegend) cutLegend.style.display = showCut ? "contents" : "none";
        if (showCut) {
          const cluster = (data.clusters || []).find((item) => item.id === state.clusterId);
          const cut = cluster && cluster.cut;
          const poseSource = (showMerged && poseData.merged) || (showVggt && poseData.vggt) || poseData.merged || poseData.vggt || poseData.gt;
          if (cut && poseSource && poseSource.length) {
            const leftIds = new Set(cut.left || []);
            const rightIds = new Set(cut.right || []);
            const overlapIds = new Set(cut.overlap || []);
            const seen = new Map();
            const sites = poseSource.map((cam) => {
              let sx = xS(cam.x);
              let sy = yS(cam.y);
              const key = sx.toFixed(2) + "," + sy.toFixed(2);
              const n = seen.get(key) || 0;
              seen.set(key, n + 1);
              if (n) {
                const ang = n * 2.399;
                sx += Math.cos(ang) * 0.4;
                sy += Math.sin(ang) * 0.4;
              }
              return { cam, sx, sy };
            });
            const delaunay = d3.Delaunay.from(sites, (d) => d.sx, (d) => d.sy);
            const voronoi = delaunay.voronoi([0, 0, width, height]);
            const cellG = g.append("g").attr("class", "partition-voronoi");
            sites.forEach((site, i) => {
              const idx = site.cam.idx;
              let fill = null;
              let label = null;
              if (overlapIds.has(idx)) {
                fill = "rgba(196, 181, 253, 0.34)";
                label = "borrowed";
              } else if (leftIds.has(idx)) {
                fill = "rgba(103, 232, 249, 0.28)";
                label = "left child";
              } else if (rightIds.has(idx)) {
                fill = "rgba(253, 186, 116, 0.28)";
                label = "right child";
              }
              if (!fill) return;
              const cell = voronoi.cellPolygon(i);
              if (!cell) return;
              cellG.append("path")
                .attr("d", "M" + cell.map((p) => p[0] + "," + p[1]).join("L") + "Z")
                .attr("fill", fill)
                .attr("stroke", "rgba(248, 250, 252, 0.18)")
                .attr("stroke-width", 0.6)
                .attr("pointer-events", "none")
                .append("title").text(idx + " " + label);
            });
          }
        }

        // Draw edges.
        if (showEdges && activeEdges.length) {
          const edgeG = g.append("g");
          activeEdges.forEach((e) => {
            const p1 = posLookup[e.i];
            const p2 = posLookup[e.j];
            if (!p1 || !p2) return;
            // Use vggt positions for edges.
            const s1 = p1.vggt || p1.gt || Object.values(p1)[0];
            const s2 = p2.vggt || p2.gt || Object.values(p2)[0];
            if (!s1 || !s2) return;
            const errVal = edgeError(e);
            const strokeColor = (errColorScale && errVal !== null) ? errColorScale(Math.min(errVal, maxErrVal)) : "rgba(80,100,140,0.25)";
            const strokeOpacity = (errColorScale && errVal !== null) ? 0.6 : 0.25;
            const line = edgeG.append("line")
              .attr("x1", s1.sx).attr("y1", s1.sy)
              .attr("x2", s2.sx).attr("y2", s2.sy)
              .attr("stroke", strokeColor)
              .attr("stroke-opacity", strokeOpacity)
              .attr("stroke-width", 0.8);
            const errLabel = errVal !== null ? ` | err=${errVal.toFixed(1)}\\u00b0` : "";
            line.append("title").text(`${e.i} <-> ${e.j}: ${e.n} tracks${errLabel}`);
          });
        }

        // Determine the selected image name for highlighting.
        let selectedName = null;
        if (state.selectedImageId !== null) {
          const images = getClusterImages();
          const selImg = images.find((im) => im.image_id === state.selectedImageId);
          if (selImg) selectedName = selImg.name;
        }

        // Draw camera sets in order: GT (back), baselines, pre-BA, then post-BA (front).
        const camSets = [];
        if (showGt && poseData.gt) {
          // Filter GT cameras: if merged is visible use merged camera set, otherwise use vggt.
          const refSet = (showMerged && poseData.merged) ? poseData.merged : poseData.vggt;
          if (refSet) {
            const refNames = new Set(refSet.map((c) => c.name));
            const filteredGt = poseData.gt.filter((c) => refNames.has(c.name));
            camSets.push({ key: "gt", cams: filteredGt, color: "#4ade80" });
          } else {
            camSets.push({ key: "gt", cams: poseData.gt, color: "#4ade80" });
          }
        }
        (data.baselines || []).forEach((bl, i) => {
          const cb = document.getElementById("poseShowBaseline_" + bl.key);
          if (cb && cb.checked && poseData[bl.key]) {
            camSets.push({ key: bl.key, cams: poseData[bl.key], color: BASELINE_COLORS[i] || "#a78bfa" });
          }
        });
        if (showPreBa && poseData.vggt_pre_ba) camSets.push({ key: "vggt_pre_ba", cams: poseData.vggt_pre_ba, color: "#f59e0b" });
        if (showVggt && poseData.vggt) camSets.push({ key: "vggt", cams: poseData.vggt, color: "#60a5fa" });
        if (showMerged && poseData.merged) camSets.push({ key: "merged", cams: poseData.merged, color: "#c084fc" });

        camSets.forEach((cs) => {
          const camG = g.append("g").attr("class", "cam-set-" + cs.key);
          cs.cams.forEach((c) => {
            const sx = xS(c.x);
            const sy = yS(c.y);
            const isSelected = (c.name === selectedName);

            // X axis arrow (red).
            camG.append("line")
              .attr("x1", sx).attr("y1", sy)
              .attr("x2", sx + c.ax[0] * arrowLen).attr("y2", sy - c.ax[1] * arrowLen * flip)
              .attr("stroke", "#ef4444").attr("stroke-width", 1.5)
              .attr("stroke-opacity", 0.8);

            // Z axis arrow (blue).
            camG.append("line")
              .attr("x1", sx).attr("y1", sy)
              .attr("x2", sx + c.az[0] * arrowLen).attr("y2", sy - c.az[1] * arrowLen * flip)
              .attr("stroke", "#3b82f6").attr("stroke-width", 1.5)
              .attr("stroke-opacity", 0.8);

            // Selected highlight ring.
            if (isSelected) {
              camG.append("circle")
                .attr("cx", sx).attr("cy", sy)
                .attr("r", 8)
                .attr("fill", "none")
                .attr("stroke", "#fff")
                .attr("stroke-width", 2)
                .attr("stroke-dasharray", "3,2")
                .attr("opacity", 0.9);
            }

            // Yellow highlight for pose-focused node (all recon types for same image).
            const isFocused = (c.name === state.poseFocusedName) && (c.name !== selectedName);
            if (isFocused) {
              camG.append("circle")
                .attr("cx", sx).attr("cy", sy)
                .attr("r", 6)
                .attr("fill", "none")
                .attr("stroke", "#facc15")
                .attr("stroke-width", 2.5);
            }

            // Camera dot.
            const dot = camG.append("circle")
              .attr("cx", sx).attr("cy", sy)
              .attr("r", isSelected ? 5 : 3.5)
              .attr("fill", cs.color)
              .attr("stroke", isSelected ? "#fff" : "#000")
              .attr("stroke-width", isSelected ? 1.5 : 0.5)
              .style("cursor", "pointer");
            dot.append("title").text(`${c.idx} [${cs.key}]`);
            dot.on("click", (event) => {
              event.stopPropagation();
              state.poseFocusedName = (state.poseFocusedName === c.name) ? null : c.name;
              renderPoseVisualization();
            });

            // Camera ID label: only for vggt when toggled on.
            if (showLabels && cs.key === "vggt") {
              camG.append("text")
                .attr("x", sx + 5).attr("y", sy - 5)
                .attr("fill", cs.color)
                .attr("font-size", "7px")
                .attr("font-family", "system-ui, sans-serif")
                .attr("opacity", 0.75)
                .text(c.idx);
            }
          });
        });
        renderAucPanel();
      }

      // ---------- Pose AUC panel ----------
      function poseAuc(errors, thresholds) {
        // Mirrors gtsfm/utils/metrics.py:pose_auc
        if (!errors.length) return thresholds.map(() => NaN);
        const sorted = Float64Array.from(errors).sort();
        const n = sorted.length;
        // Prepend 0 to both recall and errors arrays.
        const recall = new Float64Array(n + 1);
        const errs = new Float64Array(n + 1);
        errs[0] = 0;
        recall[0] = 0;
        for (let i = 0; i < n; i++) {
          errs[i + 1] = sorted[i];
          recall[i + 1] = (i + 1) / n;
        }
        return thresholds.map((t) => {
          // Find last_index: first index where errs > t.
          let last = errs.length;
          for (let i = 0; i < errs.length; i++) {
            if (errs[i] > t) { last = i; break; }
          }
          if (last <= 1) return 0;
          // Build truncated arrays: errs[0..last-1] + t, recall[0..last-1] + recall[last-1]
          const e = new Float64Array(last + 1);
          const r = new Float64Array(last + 1);
          for (let i = 0; i < last; i++) { e[i] = errs[i]; r[i] = recall[i]; }
          e[last] = t;
          r[last] = recall[last - 1];
          // Trapezoidal integration.
          let area = 0;
          for (let i = 1; i <= last; i++) {
            area += (e[i] - e[i - 1]) * (r[i] + r[i - 1]) / 2;
          }
          return area / t;
        });
      }

      function renderAucPanel() {
        const panel = document.getElementById("aucPanel");
        const content = document.getElementById("aucContent");
        const poseData = data.posesByCluster ? data.posesByCluster[state.clusterId] : null;
        if (!poseData) {
          panel.style.display = "none";
          return;
        }
        const edgeSetKey = document.getElementById("poseEdgeSet").value;
        const showMergedAuc = document.getElementById("poseShowMerged").checked;
        const useMergedEdgesAuc = showMergedAuc && poseData.merged_edges;
        const covisEdgesAuc = useMergedEdgesAuc ? poseData.merged_edges : (poseData.edges || []);
        const allPairEdgesAuc = useMergedEdgesAuc ? (poseData.merged_all_edges || covisEdgesAuc) : (poseData.all_edges || covisEdgesAuc);
        const activeEdges = edgeSetKey === "all" ? allPairEdgesAuc : covisEdgesAuc;
        if (!activeEdges.length) {
          panel.style.display = "none";
          return;
        }
        panel.style.display = "block";

        const edgeMetric = document.getElementById("poseEdgeMetric").value;
        if (edgeMetric === "none") {
          content.innerHTML = '<span style="color:var(--muted);">Select an edge metric to see AUC.</span>';
          return;
        }
        function getErr(e, reconKey) {
          const errs = e[reconKey];
          if (!errs) return null;
          if (edgeMetric === "rot") return errs.rot_err;
          if (edgeMetric === "trans") return errs.trans_err;
          return Math.max(errs.rot_err, errs.trans_err);
        }

        // Determine selected image name.
        let selectedName = null;
        if (state.selectedImageId !== null) {
          const images = getClusterImages();
          const selImg = images.find((im) => im.image_id === state.selectedImageId);
          if (selImg) selectedName = selImg.name;
        }

        const thresholds = [1, 2.5, 5, 10, 20];
        const metricLabel = edgeMetric === "rot" ? "Rotation" : edgeMetric === "trans" ? "Translation" : "Max(R,T)";

        // Collect errors and build rows for each reconstruction.
        const effRecon = getEffectiveRecon();
        const reconLabel = effRecon === "vggt" ? "Post-BA" : effRecon === "vggt_pre_ba" ? "Pre-BA" : "Merged";
        // Each entry: { key, label, edges } — edges to use for that recon's AUC.
        const reconEntries = [{ key: effRecon, label: reconLabel, edges: activeEdges }];
        // Always include merged separately using its own edge set.
        if (effRecon !== "merged" && poseData.merged) {
          const mergedCovis = poseData.merged_edges || [];
          const mergedAll = poseData.merged_all_edges || mergedCovis;
          const mergedEdges = edgeSetKey === "all" ? mergedAll : mergedCovis;
          reconEntries.push({ key: "merged", label: "Merged", edges: mergedEdges });
        }
        (data.baselines || []).forEach((bl) => reconEntries.push({ key: bl.key, label: bl.label, edges: activeEdges }));

        let rows = [];
        reconEntries.forEach((rk) => {
          const allErrs = [];
          const selErrs = [];
          const exclErrs = [];
          rk.edges.forEach((e) => {
            const v = getErr(e, rk.key);
            if (v === null || !Number.isFinite(v)) return;
            allErrs.push(v);
            if (selectedName) {
              const involves = (e.i === selectedName || e.j === selectedName);
              if (involves) selErrs.push(v);
              else exclErrs.push(v);
            }
          });
          if (!allErrs.length) return;
          rows.push({ label: `${rk.label} — all (${allErrs.length})`, aucs: poseAuc(allErrs, thresholds) });
          if (selectedName) {
            rows.push({ label: `${rk.label} — sel (${selErrs.length})`, aucs: poseAuc(selErrs, thresholds) });
            rows.push({ label: `${rk.label} — excl (${exclErrs.length})`, aucs: poseAuc(exclErrs, thresholds) });
          }
        });

        let html = `<div style="margin-bottom:6px;font-size:11px;color:var(--muted);">Metric: <b>${metricLabel}</b></div>`;
        html += '<table><thead><tr><th>Subset</th>';
        thresholds.forEach((t) => { html += `<th>@${t}&deg;</th>`; });
        html += '</tr></thead><tbody>';
        rows.forEach((r) => {
          html += `<tr><td style="font-weight:500;">${r.label}</td>`;
          r.aucs.forEach((a) => {
            const txt = Number.isNaN(a) ? '&mdash;' : (a * 100).toFixed(1);
            html += `<td>${txt}</td>`;
          });
          html += '</tr>';
        });
        html += '</tbody></table>';
        content.innerHTML = html;
      }

      function renderMainHistogram(imageData) {
        const panel = document.getElementById("mainHistPanel");
        if (!imageData || !imageData.tracks || !imageData.tracks.length) {
          panel.style.display = "none";
          return;
        }
        panel.style.display = "block";
        const nbins = Math.max(5, Number(document.getElementById("histBinsInput").value) || 30);
        const logY = document.getElementById("histLogYToggle").checked;
        const showCdf = document.getElementById("histShowCdfToggle").checked;

        const errors = imageData.tracks.map((t) => t.reproj_err).filter((v) => Number.isFinite(v));
        errors.sort((a, b) => a - b);

        const traces = [{
          x: errors,
          type: "histogram",
          nbinsx: nbins,
          name: "count",
          marker: { color: "rgba(122,162,255,0.75)", line: { color: "rgba(90,120,210,1)", width: 1 } },
          hovertemplate: "err: %{x:.3f} px<br>count: %{y}<extra></extra>",
          yaxis: "y",
        }];

        if (showCdf) {
          const n = errors.length;
          const cdfY = errors.map((_, i) => ((i + 1) / n) * 100);
          traces.push({
            x: errors,
            y: cdfY,
            type: "scatter",
            mode: "lines",
            name: "CDF %",
            line: { color: "rgba(255,209,102,0.85)", width: 2 },
            hovertemplate: "err: %{x:.3f} px<br>CDF: %{y:.1f}%<extra></extra>",
            yaxis: "y2",
          });
        }

        const layout = {
          paper_bgcolor: "rgba(0,0,0,0)",
          plot_bgcolor: "rgba(14,18,28,0.55)",
          font: { color: "#9aa4b2", size: 11 },
          margin: { l: 52, r: showCdf ? 52 : 16, t: 10, b: 48 },
          xaxis: { title: { text: "Reprojection error (px)", standoff: 8 }, color: "#9aa4b2", gridcolor: "#263041", zerolinecolor: "#3a4560" },
          yaxis: { title: { text: logY ? "Count (log)" : "Count", standoff: 6 }, type: logY ? "log" : "linear", color: "#9aa4b2", gridcolor: "#263041", zerolinecolor: "#3a4560" },
          bargap: 0.04,
          showlegend: showCdf,
          legend: { x: 0.65, y: 0.97, bgcolor: "rgba(14,18,28,0.7)", bordercolor: "#263041", borderwidth: 1, font: { size: 11 } },
        };

        if (showCdf) {
          layout.yaxis2 = { title: { text: "CDF (%)", standoff: 6 }, overlaying: "y", side: "right", range: [0, 100], color: "#ffd166", gridcolor: "rgba(0,0,0,0)", showgrid: false };
        }

        Plotly.react("mainHistDiv", traces, layout, {
          responsive: true,
          displayModeBar: true,
          modeBarButtonsToRemove: ["toImage", "sendDataToCloud"],
          displaylogo: false,
          scrollZoom: true,
        });
      }

      async function renderAll() {
        if (serverMode) {
          await fetchClustersIfNeeded();
          populateClusters();
          await fetchImagesIfNeeded();
          await fetchPosesIfNeeded();
        }
        const images = getClusterImages();
        let filtered = applyFilter(images);
        thumbList.innerHTML = "";
        if (!state.clusterId) {
          setStatus("No clusters available.");
          imageCount.textContent = "0 images";
          trackCount.textContent = "0 tracks";
          return;
        }
        if (!images.length) {
          setStatus("No COLMAP data for this cluster/recon.");
          imageCount.textContent = "0 images";
          trackCount.textContent = "0 tracks";
          renderMainImage(null);
          return;
        }
        // Sort by mean reprojection error descending so \"worst\" views float to top.
        filtered = filtered.slice().sort((a, b) => imageMeanReprojErr(b) - imageMeanReprojErr(a));

        setStatus(`${filtered.length} / ${images.length} images (sorted by mean reprojection error)`);
        const trackTotal = filtered.reduce((acc, img) => acc + img.tracks.length, 0);
        imageCount.textContent = `${filtered.length} images`;
        trackCount.textContent = `${trackTotal} tracks`;
        if (state.selectedImageId === null || !filtered.find((i) => i.image_id === state.selectedImageId)) {
          state.selectedImageId = filtered.length ? filtered[0].image_id : null;
        }
        const selected = filtered.find((i) => i.image_id === state.selectedImageId) || null;
        state.shownTrackIds = computePatchSelection(selected, state.patchMode);
        const shownTrackCount = state.shownTrackIds ? state.shownTrackIds.size : trackTotal;
        if (state.patchMode !== "all") {
          setStatus(
            `${filtered.length} / ${images.length} images, showing ${shownTrackCount} tracks via ${state.patchMode} mode`
          );
        }
        filtered.forEach((img) => {
          renderThumbnail(img, thumbList, (imageData) => {
            state.selectedImageId = imageData.image_id;
            renderAll();
          });
        });
        renderMainImage(selected);
        renderMainHistogram(selected);
        renderPoseVisualization();
        await renderSelectedTrackPatches(images);
      }

      function setupZoom() {
        const ctx = zoomCanvas.getContext("2d");
        mainCanvas.addEventListener("mousemove", (evt) => {
          const images = applyFilter(getClusterImages());
          const selectedImage = getSelectedImageData(images);
          const rect = mainCanvas.getBoundingClientRect();
          const x = evt.clientX - rect.left;
          const y = evt.clientY - rect.top;
          const scaleX = mainCanvas.width / rect.width;
          const scaleY = mainCanvas.height / rect.height;
          const cx = Math.round(x * scaleX);
          const cy = Math.round(y * scaleY);
          const size = state.cropSize;
          const half = Math.floor(size / 2);
          const sx = Math.max(0, cx - half);
          const sy = Math.max(0, cy - half);
          const sw = Math.min(size, mainCanvas.width - sx);
          const sh = Math.min(size, mainCanvas.height - sy);
          ctx.clearRect(0, 0, zoomCanvas.width, zoomCanvas.height);
          ctx.imageSmoothingEnabled = false;
          // Use clean source image for zoom (no overlay clutter), then draw only tiny selected marker.
          ctx.drawImage(baseMainCanvas, sx, sy, sw, sh, 0, 0, zoomCanvas.width, zoomCanvas.height);

          if (selectedImage && state.selectedTrackId !== null) {
            const selectedTrack = selectedImage.tracks.find((t) => t.track_id === state.selectedTrackId);
            if (selectedTrack) {
              const measIn = selectedTrack.meas[0] >= sx && selectedTrack.meas[0] <= (sx + sw)
                && selectedTrack.meas[1] >= sy && selectedTrack.meas[1] <= (sy + sh);
              const reprojIn = selectedTrack.reproj[0] >= sx && selectedTrack.reproj[0] <= (sx + sw)
                && selectedTrack.reproj[1] >= sy && selectedTrack.reproj[1] <= (sy + sh);
              if (measIn || reprojIn) {
                const measX = ((selectedTrack.meas[0] - sx) / sw) * zoomCanvas.width;
                const measY = ((selectedTrack.meas[1] - sy) / sh) * zoomCanvas.height;
                const reprojX = ((selectedTrack.reproj[0] - sx) / sw) * zoomCanvas.width;
                const reprojY = ((selectedTrack.reproj[1] - sy) / sh) * zoomCanvas.height;
                drawSelectedTrackMarkers(ctx, measX, measY, reprojX, reprojY);
              }
            }
          }
        });
      }

      function setupMainCanvasSelection() {
        mainCanvas.addEventListener("click", (evt) => {
          const images = applyFilter(getClusterImages());
          const selected = getSelectedImageData(images);
          if (!selected) {
            return;
          }
          const rect = mainCanvas.getBoundingClientRect();
          const x = (evt.clientX - rect.left) * (mainCanvas.width / rect.width);
          const y = (evt.clientY - rect.top) * (mainCanvas.height / rect.height);
          const trackId = pickTrack(selected, x, y, state.shownTrackIds);
          if (trackId === null) {
            return;
          }
          state.selectedTrackId = trackId;
          const t = selected.tracks.find((tr) => tr.track_id === trackId);
          if (t) {
            trackBadge.textContent = `Track: ${trackId} | len=${t.track_len} | err=${t.reproj_err.toFixed(2)}px`;
          } else {
            trackBadge.textContent = `Track: ${trackId}`;
          }
          renderAll();
        });
      }

      function updateUrlParams() {
        const params = new URLSearchParams(window.location.search);
        if (state.clusterId) params.set("cluster", state.clusterId); else params.delete("cluster");
        if (state.recon !== "vggt") params.set("recon", state.recon); else params.delete("recon");
        const newUrl = window.location.pathname + (params.toString() ? "?" + params.toString() : "");
        window.history.replaceState(null, "", newUrl);
      }

      async function init() {
        await fetchClustersIfNeeded();
        initBaselines();
        // Restore state from URL params.
        const urlParams = new URLSearchParams(window.location.search);
        const urlCluster = urlParams.get("cluster");
        const urlRecon = urlParams.get("recon");
        if (urlCluster && data.clusters.some((c) => c.id === urlCluster)) {
          state.clusterId = urlCluster;
        }
        if (urlRecon && ["vggt", "vggt_pre_ba", "merged"].includes(urlRecon)) {
          state.recon = urlRecon;
        }
        populateClusters();
        reconSelect.value = state.recon;
        clusterSelect.addEventListener("change", () => {
          state.clusterId = clusterSelect.value;
          state.selectedImageId = null;
          updateUrlParams();
          if (serverMode) {
            setStatus(`Loading images for ${state.clusterId} (${state.recon})...`);
          }
          renderAll();
        });
        reconSelect.addEventListener("change", () => {
          state.recon = reconSelect.value;
          state.selectedImageId = null;
          updateUrlParams();
          if (serverMode) {
            setStatus(`Loading images for ${state.clusterId} (${state.recon})...`);
          }
          renderAll();
        });
        lineToggle.addEventListener("change", () => {
          state.showLine = lineToggle.checked;
          renderAll();
        });
        cropSize.addEventListener("input", () => {
          state.cropSize = Number(cropSize.value);
        });
        imageFilter.addEventListener("input", () => {
          state.filterText = imageFilter.value;
          state.selectedImageId = null;
          renderAll();
        });
        patchSizeInput.addEventListener("change", () => {
          state.patchSize = Math.max(16, Number(patchSizeInput.value) || 128);
          renderAll();
        });
        maxPerPatchInput.addEventListener("change", () => {
          state.maxPerPatch = Math.max(1, Number(maxPerPatchInput.value) || 1);
          renderAll();
        });
        modeAllBtn.addEventListener("click", () => {
          setPatchMode("all");
          renderAll();
        });
        modeBestBtn.addEventListener("click", () => {
          setPatchMode("best");
          renderAll();
        });
        modeWorstBtn.addEventListener("click", () => {
          setPatchMode("worst");
          renderAll();
        });
        modeRandomBtn.addEventListener("click", () => {
          setPatchMode("random");
          renderAll();
        });
        document.addEventListener("keydown", (evt) => {
          if (evt.key !== "Escape") {
            return;
          }
          if (state.poseFocusedName) {
            state.poseFocusedName = null;
            renderPoseVisualization();
            return;
          }
          state.selectedTrackId = null;
          trackBadge.textContent = "Track: none";
          renderAll();
        });
        function replotHist() {
          const images = applyFilter(getClusterImages());
          renderMainHistogram(getSelectedImageData(images));
        }
        document.getElementById("histBinsInput").addEventListener("change", replotHist);
        document.getElementById("histLogYToggle").addEventListener("change", replotHist);
        document.getElementById("histShowCdfToggle").addEventListener("change", replotHist);
        // Pose visualization toggles.
        ["poseShowGt", "poseShowPreBa", "poseShowVggt", "poseShowMerged", "poseShowEdges", "poseShowCut", "poseShowLabels", "poseFlip"].forEach((id) => {
          document.getElementById(id).addEventListener("change", renderPoseVisualization);
        });
        document.getElementById("poseEdgeMetric").addEventListener("change", renderPoseVisualization);
        document.getElementById("poseEdgeSet").addEventListener("change", renderPoseVisualization);
        (data.baselines || []).forEach((bl) => {
          const cb = document.getElementById("poseShowBaseline_" + bl.key);
          if (cb) cb.addEventListener("change", renderPoseVisualization);
        });
        setupZoom();
        setupMainCanvasSelection();
        renderAll();
      }

      init();
    </script>
  </body>
</html>
"""
    html = html.replace("__TITLE__", title)
    html = html.replace("__SERVER_MODE__", "true" if serve_mode else "false")
    # The template originally used doubled braces for f-string safety; normalize CSS braces now.
    style_start = html.find("<style>")
    style_end = html.find("</style>", style_start)
    if style_start != -1 and style_end != -1:
        css_block = html[style_start:style_end]
        css_block = css_block.replace("{{", "{").replace("}}", "}")
        html = html[:style_start] + css_block + html[style_end:]
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

    results_root = Path(args.results_root).resolve()
    cluster_tree_pkl = (
        Path(args.cluster_tree_pkl).resolve()
        if args.cluster_tree_pkl is not None
        else results_root / "cluster_tree.pkl"
    )
    if not cluster_tree_pkl.exists():
        raise FileNotFoundError(f"cluster_tree_pkl does not exist: {cluster_tree_pkl}")

    dataset_dir = Path(args.dataset_dir).resolve()
    images_dir = Path(args.images_dir).resolve() if args.images_dir else dataset_dir / "images"
    index_to_filename = _load_index_to_filename(dataset_dir, images_dir, args.colmap_files_subdir)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else results_root / "tracks_web_viz"

    with cluster_tree_pkl.open("rb") as f:
        cluster_tree = pickle.load(f)
    if not isinstance(cluster_tree, ClusterTree):
        raise TypeError(f"Expected ClusterTree, got {type(cluster_tree)}")

    cluster_nodes = _collect_cluster_paths(cluster_tree)
    num_workers = args.num_workers or min(16, (os.cpu_count() or 4))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load ground truth poses if available.
    gt_dir = _find_gt_colmap_dir(dataset_dir, args.gt_colmap_dir)
    gt_data: GtsfmData | None = None
    if gt_dir is not None:
        try:
            gt_data = GtsfmData.read_colmap(str(gt_dir))
            logger.info("Loaded GT with %d cameras from %s", len(gt_data.get_valid_camera_indices()), gt_dir)
        except Exception as exc:
            logger.warning("Could not load GT COLMAP data from %s: %s", gt_dir, exc)

    # Load baseline reconstructions.
    baseline_data: List[Tuple[str, GtsfmData]] = []
    baseline_labels: List[Dict[str, str]] = []
    if len(args.baseline) > 2:
        logger.warning("At most 2 baselines are supported; ignoring extras.")
    for idx, bl_path_str in enumerate(args.baseline[:2]):
        bl_path = Path(bl_path_str).resolve()
        label = args.baseline_label[idx] if idx < len(args.baseline_label) else f"Baseline {idx + 1}"
        bl_key = f"baseline_{idx}"
        if not _has_colmap_files(bl_path):
            logger.warning("Baseline %d (%s) has no COLMAP files at %s, skipping.", idx, label, bl_path)
            continue
        try:
            bl_gtsfm = GtsfmData.read_colmap(str(bl_path))
            baseline_data.append((bl_key, bl_gtsfm))
            baseline_labels.append({"key": bl_key, "label": label})
            logger.info("Loaded baseline '%s' with %d cameras from %s", label, len(bl_gtsfm.get_valid_camera_indices()), bl_path)
        except Exception as exc:
            logger.warning("Could not load baseline %d from %s: %s", idx, bl_path, exc)

    if not args.serve:
        payload = _build_data_payload(
            cluster_nodes,
            results_root,
            images_dir,
            num_workers,
            gt_data=gt_data,
            baseline_data=baseline_data,
            index_to_filename=index_to_filename,
        )
        if baseline_labels:
            payload["baselines"] = baseline_labels
        data_js = "window.TRACKS_DATA = " + json.dumps(payload) + ";"
        (output_dir / "data.js").write_text(data_js, encoding="utf-8")
        _write_html(output_dir, "Tracks Viewer", serve_mode=False)
        logger.info("Wrote tracks web viewer to %s", output_dir / "index.html")
        return

    (output_dir / "data.js").write_text("window.TRACKS_DATA = null;", encoding="utf-8")
    _write_html(output_dir, "Tracks Viewer", serve_mode=True)
    logger.info("Wrote tracks web viewer to %s", output_dir / "index.html")

    cluster_entries = [
        {
            "id": node["path"],
            "path": node["path"],
            "label": "root" if node["path"] == "root" else f"C_{node['path'].replace('.', '_')}",
            "depth": node["depth"],
            "cut": node.get("cut"),
        }
        for node in cluster_nodes
    ]
    cluster_index = {entry["path"]: entry for entry in cluster_entries}
    cache: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}

    class TracksRequestHandler(http.server.BaseHTTPRequestHandler):
        def _send_json(self, payload: Dict[str, Any]) -> None:
            data_bytes = json.dumps(payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data_bytes)))
            self.end_headers()
            self.wfile.write(data_bytes)

        def _send_file(self, path: Path) -> None:
            if not path.exists():
                self.send_error(404, "File not found")
                return
            data = path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            if path.suffix.lower() == ".html":
                self.send_header("Content-Type", "text/html; charset=utf-8")
            elif path.suffix.lower() == ".js":
                self.send_header("Content-Type", "text/javascript; charset=utf-8")
            elif path.suffix.lower() in {".jpg", ".jpeg"}:
                self.send_header("Content-Type", "image/jpeg")
            elif path.suffix.lower() == ".png":
                self.send_header("Content-Type", "image/png")
            else:
                self.send_header("Content-Type", "application/octet-stream")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            path = parsed.path
            query = parse_qs(parsed.query)

            if path in {"/", "/index.html"}:
                return self._send_file(output_dir / "index.html")
            if path == "/data.js":
                return self._send_file(output_dir / "data.js")
            if path == "/api/clusters":
                resp = {
                    "clusters": cluster_entries,
                    "defaultRecon": "vggt",
                    "baseImageUrl": "/image",
                }
                if baseline_labels:
                    resp["baselines"] = baseline_labels
                return self._send_json(resp)
            if path == "/api/images":
                cluster = (query.get("cluster") or [""])[0]
                recon = (query.get("recon") or [""])[0]
                if cluster not in cluster_index or recon not in RECON_OPTIONS:
                    return self._send_json({"images": []})
                cache_key = (cluster, recon)
                if cache_key not in cache:
                    node_dir = _node_results_dir(results_root, cluster)
                    recon_dir = node_dir / recon
                    if not _has_colmap_files(recon_dir):
                        cache[cache_key] = []
                    else:
                        try:
                            gtsfm_data = GtsfmData.read_colmap(str(recon_dir))
                            cache[cache_key] = _build_image_entries(
                                gtsfm_data,
                                images_dir,
                                include_file_path=False,
                                index_to_filename=index_to_filename,
                            )
                        except Exception as exc:
                            logger.warning(
                                "Image data request failed for cluster %s, recon %s: %s",
                                cluster,
                                recon,
                                exc,
                            )
                            cache[cache_key] = []
                return self._send_json({"images": cache[cache_key]})
            if path == "/api/poses":
                cluster = (query.get("cluster") or [""])[0]
                if cluster not in cluster_index:
                    return self._send_json({})
                node_dir = _node_results_dir(results_root, cluster)
                try:
                    pose_data = _build_pose_data_for_cluster(
                        gt_data, node_dir, baseline_data=baseline_data, index_to_filename=index_to_filename
                    )
                    return self._send_json(pose_data or {})
                except Exception as exc:
                    logger.warning("Pose data request failed for %s: %s", cluster, exc)
                    return self._send_json({})
            if path == "/image":
                name = (query.get("name") or [""])[0]
                name = unquote(name)
                if not name:
                    self.send_error(404, "File not found")
                    return
                candidate = (images_dir / name).resolve()
                images_root = images_dir.resolve()
                if os.path.commonpath([str(candidate), str(images_root)]) != str(images_root):
                    self.send_error(403, "Forbidden")
                    return
                return self._send_file(candidate)

            # Fallback: serve static files from output_dir, with path traversal protection.
            candidate = (output_dir / path.lstrip("/")).resolve()
            if os.path.commonpath([str(candidate), str(output_dir.resolve())]) != str(output_dir.resolve()):
                self.send_error(403, "Forbidden")
                return
            return self._send_file(candidate)

    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), TracksRequestHandler) as httpd:
        logger.info("Serving %s at http://127.0.0.1:%d", output_dir, args.port)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
