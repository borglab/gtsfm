"""Utilities for merging cluster reconstructions and exporting results."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import gtsam
import numpy as np
from dask.distributed import Client, Future
from gtsam import Pose3, Similarity3, TrajectoryAlignerSim3, UnaryMeasurementPose3

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.utils import align as align_utils
from gtsfm.utils import data_utils
from gtsfm.utils.reprojection import compute_track_reprojection_errors
from gtsfm.utils.splat import GaussiansProtocol, merge_gaussian_splats
from gtsfm.utils.transform import camera_map_with_sim3, transform_gaussian_splats
from gtsfm.utils.tree import Tree
from gtsfm.utils.tree_dask import submit_tree_map

if TYPE_CHECKING:
    from gtsfm.scene_optimizer import ClusterExecutionHandles


logger = logger_utils.get_logger()

_SCENE_PLOTS_DIR_ATTR = "_gtsfm_plots_dir"
_SCENE_LABEL_ATTR = "_gtsfm_cluster_label"
# Sidecar attached to each scene: packed (camera, pixel) -> global-track-id index (see
# build_measurement_gid_arrays). Rides the GtsfmData through dask exactly like the plots/label attrs,
# so merges can recognize that two locally-triangulated tracks are the SAME global track even when the
# two reconstructions share no cameras (the identity was established once, globally, by the verified
# union-find; per-cluster slicing used to discard it).
_SCENE_GID_INDEX_ATTR = "_gtsfm_measurement_gid_index"


@dataclass
class MergingOptions:
    """Options controlling how cluster reconstructions are merged."""

    run_bundle_adjustment: bool = True
    merge_duplicate_tracks: bool = True
    drop_child_if_merging_fail: bool = True
    drop_outlier_after_camera_merging: bool = True
    drop_camera_with_no_track: bool = True
    keep_all_cameras: bool = False
    # ANNEX (prior-backed retention): capture the cameras each node's post-BA track filter drops and
    # carry them up the tree OUTSIDE the scenes — re-seated by the same Sim3 that seats their subtree,
    # never entering a Sim3 solve, duplicate-camera resolution, or BA. The root's annex is exported as
    # a separate posed-only model (results/annex_posed_only). Contrast keep_all_cameras=true, which
    # keeps trackless cameras INSIDE every intermediate scene, where stale copies win parent-vs-child
    # dedup and poison the merges (measured on ToL: core median 1.18m -> 18.35m).
    export_trackless_annex: bool = False
    # MERGE_GUARD Sim3 scale band — DISABLED by default (band = [0, inf)). VGGT scene scale is
    # arbitrary PER CLUSTER (per-batch normalization), so heterogeneous cluster extents produce
    # large-but-lawful solved scales: the old [0.25, 4] band executed 18 lawful Roman Forum children
    # (~1,200 cams incl. a 258-cam subtree at scale 4.06). Genuinely diverged seats are caught by
    # the structureless/0-track guards, the correspondence floor, and the post-merge BA + reproj
    # filters. Set a finite band only to re-enable the legacy behavior.
    sim3_scale_band_min: float = 0.0
    sim3_scale_band_max: float = float("inf")
    plot_reprojection_histograms: bool = True
    use_nonlinear_sim3_alignment: bool = False
    max_track_correspondences_for_sim3: int = 150
    scale_and_average_focal_length: bool = False
    pre_ba_max_reproj_error: float = 14.0
    pre_ba_min_track_length: int = 2
    post_ba_max_reproj_error: float = 3.0
    min_track_length: int = 2
    allow_post_ba_reproj_filtering: bool = True
    metric_constructed_only: bool = False
    # When set, the root merge captures the good-pose cameras that the post-BA track filter drops
    # (e.g. low VGGT-depth-confidence cams left track-less at construction) and hands their merged
    # poses to the post-merge retriangulation, which geometrically re-triangulates their global 2D
    # tracks against those poses. Cameras that still fail to gain a >=3-view track are cleanly dropped
    # by the retri's own final filter. Orthogonal to the per-cluster depth gate: an unrecovered camera
    # is inert (excluded from the retri BA, contributes zero factors), while a recovered good-pose camera
    # (>=15 inlier tracks) joins the retri BA and jointly refines shared structure as intended — all poses
    # pinned by use_pose_prior_all_cameras, so the existing reconstruction is bounded, not free to drift.
    recover_trackless_cameras_in_retriangulation: bool = False
    # Global-track-ID merge guard (active only when scenes carry the gid-index sidecar, i.e. the verified
    # pipeline): a child with >= guard_child_min_cams cameras whose ID-matched Sim3 correspondences come out
    # below min_sim3_correspondences_large_child has no reliable structural tie to the parent — its 7-DoF
    # seat would be under-constrained (ToL: an 83-cam child seated on 26 correspondences produced a rigid
    # 68-camera stray block 60-190m off). Such children are DROPPED (accuracy over camera count).
    guard_child_min_cams: int = 30
    min_sim3_correspondences_large_child: int = 50
    ba_options: BundleAdjustmentOptions = field(default_factory=BundleAdjustmentOptions)
    # Post-merge retriangulation BA mode. When retri_free_ba is True the retri BA drops the
    # calibration/pose priors — a fully-free final solve (GLOMAP-style): the priors that protect the
    # incremental merges throttle the full-scene BA, which otherwise locks in the accuracy (BM offline
    # replay: constructed AUC@10 0.770->0.806 on the same input; robust kernel/GNC inherited from
    # ba_options). retri_iterations>1 re-triangulates against the improved poses between BAs — only
    # worth it when the merged input is poor. Defaults preserve existing behavior.
    retri_free_ba: bool = False
    retri_iterations: int = 1


def _create_unary_measurements(scene: GtsfmData) -> list[UnaryMeasurementPose3]:
    # TODO(akshay-krishnan): investigate using a scene-dependent noise model
    # perhaps * np.exp(-len(scene.get_valid_camera_indices()) / 100.0)

    unary_measurements = []
    camera_reproj_errors = scene.get_scene_reprojection_errors_per_camera()
    num_good_measurements = {
        cam_id: (~np.isnan(errors) & (errors < 2.0)).sum() for cam_id, errors in camera_reproj_errors.items()
    }
    for i, camera in scene.get_camera_poses().items():
        if camera is None:
            continue
        noise_model = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([1e-2, 1e-2, 1e-2, 1e-1, 1e-1, 1e-1]) / np.sqrt(num_good_measurements.get(i, 0) + 1e-6)
        )
        unary_measurement = UnaryMeasurementPose3(i, camera, noise_model)
        unary_measurements.append(unary_measurement)
    return unary_measurements


def _measurement_key(cam_idx: int, uv: np.ndarray) -> tuple[int, int, int]:
    return cam_idx, math.floor(float(uv[0])), math.floor(float(uv[1]))


# ---------------------------------------------------------------------------
# Global-track-ID (gid) index: packed numpy, NOT python dicts — lean enough to ride each scene
# (KB-MB per cluster, ~20MB at the root union) with zero extra gather/broadcast. Keys use the SAME
# integer-floor quantization as _measurement_key, and measurements survive BA/filtering verbatim, so a
# track's global identity is recoverable at ANY pipeline stage from any one of its measurements.
# ---------------------------------------------------------------------------

_GID_CAM_SHIFT = 44
_GID_U_SHIFT = 22
_GID_COORD_MASK = (1 << 22) - 1


def _pack_measurement_keys(cams: np.ndarray, us: np.ndarray, vs: np.ndarray) -> np.ndarray:
    """Pack (camera, floor(u), floor(v)) into a single int64 (same quantization as _measurement_key)."""
    u = np.clip(np.floor(us).astype(np.int64), 0, _GID_COORD_MASK)
    v = np.clip(np.floor(vs).astype(np.int64), 0, _GID_COORD_MASK)
    return (cams.astype(np.int64) << _GID_CAM_SHIFT) | (u << _GID_U_SHIFT) | v


def build_measurement_gid_arrays(tracks_2d) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten global 2D tracks into parallel (packed_key, gid, cam) arrays (unsorted).

    Built ONCE in the main process from the global union-find tracks; per-cluster slices are cut with
    slice_gid_index. ~1.8M entries / ~20MB for 337k Tower-of-London tracks.
    """
    cams, us, vs, gids = [], [], [], []
    for gid, track in enumerate(tracks_2d):
        for m in track.measurements:
            cams.append(m.i)
            us.append(float(m.uv[0]))
            vs.append(float(m.uv[1]))
            gids.append(gid)
    cams_arr = np.asarray(cams, dtype=np.int64)
    keys = _pack_measurement_keys(cams_arr, np.asarray(us), np.asarray(vs))
    return keys, np.asarray(gids, dtype=np.int32), cams_arr.astype(np.int32)


def slice_gid_index(
    keys: np.ndarray, gids: np.ndarray, cams: np.ndarray, camera_set
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Cut the per-cluster gid index: entries whose camera is in ``camera_set``, sorted by key for lookup."""
    if keys is None or len(keys) == 0:
        return None
    mask = np.isin(cams, np.fromiter(camera_set, dtype=np.int32))
    if not mask.any():
        return None
    k, g = keys[mask], gids[mask]
    order = np.argsort(k, kind="stable")
    return k[order], g[order]


def _lookup_gid(index: tuple[np.ndarray, np.ndarray], cam_idx: int, uv: np.ndarray) -> int:
    """Return the global track id for one measurement, or -1 if unknown."""
    keys, gids = index
    key = int(
        _pack_measurement_keys(np.array([cam_idx]), np.array([float(uv[0])]), np.array([float(uv[1])]))[0]
    )
    pos = int(np.searchsorted(keys, key))
    if pos < len(keys) and int(keys[pos]) == key:
        return int(gids[pos])
    return -1


def _track_gid(track: gtsam.SfmTrack, index: Optional[tuple[np.ndarray, np.ndarray]]) -> int:
    """Global id of a 3D track = MAJORITY VOTE over ALL of its measurements found in the index (-1 if none).

    First-hit identity was dishonest: a single pixel-bin collision on the first indexed measurement
    assigned the whole track a foreign gid, minting phantom parent<->child "correspondences" between
    tracks that share no physical point (offline replay: zero-overlap children showed up with 150
    matches). Voting over every measurement makes rare collisions harmless — the true identity of the
    remaining measurements outvotes them — so anchor counts (and the MERGE_GUARD decisions built on
    them) are honest.
    """
    if index is None:
        return -1
    votes: dict[int, int] = {}
    for m_idx in range(track.numberMeasurements()):
        cam_idx, uv = track.measurement(m_idx)
        gid = _lookup_gid(index, cam_idx, uv)
        if gid >= 0:
            votes[gid] = votes.get(gid, 0) + 1
    if not votes:
        return -1
    return max(votes, key=votes.get)  # type: ignore[arg-type]


def _scene_gid_index(scene: Optional[GtsfmData]) -> Optional[tuple[np.ndarray, np.ndarray]]:
    return getattr(scene, _SCENE_GID_INDEX_ATTR, None) if scene is not None else None


def _union_gid_indices(*indices) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Union of packed indices (first occurrence of a key wins); propagated up the merge tree."""
    live = [ix for ix in indices if ix is not None and len(ix[0]) > 0]
    if not live:
        return None
    keys = np.concatenate([ix[0] for ix in live])
    gids = np.concatenate([ix[1] for ix in live])
    uniq_keys, first = np.unique(keys, return_index=True)
    return uniq_keys, gids[first]


def _prefilter_point_pairs(
    pairs: list[tuple[np.ndarray, np.ndarray]], spread: float
) -> tuple[list[tuple[np.ndarray, np.ndarray]], float, float]:
    """RANSAC-verify point pairs BEFORE the joint Sim3 solve; returns (inliers, pair_scale, inlier_ratio).

    A wrong pair under a tight noise sigma dominates the quadratic cost and can warp the solve; worse, a
    COHERENTLY wrong pair set (e.g. mismatched regions) implies a consistent wrong similarity that the
    solver will follow off a cliff (observed: children with 150 correspondences solving at scale 81-289).
    Fitting the pairs alone first both cleans them and measures the transform they actually imply, so the
    caller can refuse a seat with evidence instead of detonating.
    """
    if len(pairs) < 5:
        return pairs, 1.0, 1.0
    A = np.array([np.asarray(p) for p, _ in pairs])  # parent
    B = np.array([np.asarray(c) for _, c in pairs])  # child

    def _umeyama(src, dst):
        mu_s, mu_d = src.mean(0), dst.mean(0)
        Sc, Dc = src - mu_s, dst - mu_d
        U, S, Vt = np.linalg.svd(Dc.T @ Sc / len(src))
        d = np.sign(np.linalg.det(U @ Vt))
        R = U @ np.diag([1.0, 1.0, d]) @ Vt
        var = (Sc**2).sum() / len(src)
        if var < 1e-12:
            return None
        s = (S * [1.0, 1.0, d]).sum() / var
        return s, R, mu_d - s * R @ mu_s

    rng = np.random.default_rng(0)
    thresh = max(1e-6, 0.10 * spread)
    best_inl, best_count = None, -1
    for _ in range(256):
        idx = rng.choice(len(pairs), 3, replace=False)
        fit = _umeyama(A[idx], B[idx])  # parent -> child
        if fit is None:
            continue
        s, R, t = fit
        if not (1e-6 < s < 1e6):
            continue
        err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
        inl = err < thresh
        if inl.sum() > best_count:
            best_count, best_inl = int(inl.sum()), inl
    if best_inl is None or best_count < 5:
        return [], 1.0, 0.0
    fit = _umeyama(A[best_inl], B[best_inl])
    if fit is None:
        return [], 1.0, 0.0
    s, R, t = fit
    err = np.linalg.norm((s * (R @ A.T)).T + t - B, axis=1)
    inl = err < thresh
    inliers = [pairs[k] for k in range(len(pairs)) if inl[k]]
    # pair_scale is child->parent (matches the solved aSb convention used by the scale band).
    pair_scale = 1.0 / s if s > 1e-12 else float("inf")
    return inliers, float(pair_scale), float(inl.mean())


def _select_gid_point_correspondences(
    parent_scene: GtsfmData,
    child_scene: GtsfmData,
    parent_index: tuple[np.ndarray, np.ndarray],
    child_index: tuple[np.ndarray, np.ndarray],
    max_correspondences: int = 100,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sim3 point correspondences by GLOBAL TRACK IDENTITY — no shared camera required.

    Both reconstructions cut their tracks from the same global track set, so matching by gid recognizes
    the same physical point triangulated from DISJOINT camera sets (the cross-cluster edges the
    shared-camera matcher cannot see). Pairs are sorted longest-tracks-first and capped.
    """
    parent_by_gid: dict[int, int] = {}
    for t_idx, track in enumerate(parent_scene.tracks()):
        gid = _track_gid(track, parent_index)
        if gid >= 0 and gid not in parent_by_gid:
            parent_by_gid[gid] = t_idx

    pairs: list[tuple[int, int, int]] = []  # (combined_len, parent_idx, child_idx)
    for c_idx, child_track in enumerate(child_scene.tracks()):
        gid = _track_gid(child_track, child_index)
        p_idx = parent_by_gid.get(gid) if gid >= 0 else None
        if p_idx is not None:
            combined = parent_scene.get_track(p_idx).numberMeasurements() + child_track.numberMeasurements()
            pairs.append((combined, p_idx, c_idx))

    pairs.sort(key=lambda p: -p[0])
    parent_tracks = parent_scene.tracks()
    child_tracks = child_scene.tracks()
    return [
        (parent_tracks[p].point3(), child_tracks[c].point3()) for _, p, c in pairs[:max_correspondences]
    ]


def _track_max_reprojection_error(scene: GtsfmData, track: gtsam.SfmTrack) -> float:
    errors, _ = compute_track_reprojection_errors(scene.cameras(), track)
    if np.isinf(errors).any():
        return float("inf")
    finite_errors = errors[np.isfinite(errors)]
    if finite_errors.size == 0:  # all-NaN (e.g. every projection failed) — treat as unusable, not a crash
        return float("inf")
    return float(np.max(finite_errors))


def _select_overlapping_track_point_correspondences(
    parent_scene: GtsfmData,
    child_scene: GtsfmData,
    max_correspondences: int = 100,
    preferred_min_track_length: int = 3,
    preferred_max_reproj_error_px: float = 2.0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Find parent-child overlapping tracks and return selected 3D point correspondences.

    Overlap is established using shared 2D measurements in common cameras (quantized to integer
    pixel bins). Returned pairs are (parent_point3, child_point3), suitable for bSa estimation.
    """
    parent_tracks = parent_scene.tracks()
    child_tracks = child_scene.tracks()
    if len(parent_tracks) == 0 or len(child_tracks) == 0:
        return []

    parent_track_quality: list[tuple[bool, float, int]] = []
    for track in parent_tracks:
        max_err = _track_max_reprojection_error(parent_scene, track)
        track_len = track.numberMeasurements()
        is_preferred = track_len >= preferred_min_track_length and max_err < preferred_max_reproj_error_px
        parent_track_quality.append((is_preferred, max_err, track_len))

    child_track_quality: list[tuple[bool, float, int]] = []
    for track in child_tracks:
        max_err = _track_max_reprojection_error(child_scene, track)
        track_len = track.numberMeasurements()
        is_preferred = track_len >= preferred_min_track_length and max_err < preferred_max_reproj_error_px
        child_track_quality.append((is_preferred, max_err, track_len))

    parent_meas_to_tracks: dict[tuple[int, int, int], list[int]] = {}
    for track_idx, track in enumerate(parent_tracks):
        for m_idx in range(track.numberMeasurements()):
            cam_idx, uv = track.measurement(m_idx)
            key = _measurement_key(cam_idx, uv)
            parent_meas_to_tracks.setdefault(key, []).append(track_idx)

    candidate_pairs: list[tuple[int, int, int]] = []
    for child_idx, child_track in enumerate(child_tracks):
        overlap_count_per_parent: dict[int, int] = {}
        for m_idx in range(child_track.numberMeasurements()):
            cam_idx, uv = child_track.measurement(m_idx)
            key = _measurement_key(cam_idx, uv)
            for parent_idx in parent_meas_to_tracks.get(key, []):
                overlap_count_per_parent[parent_idx] = overlap_count_per_parent.get(parent_idx, 0) + 1
        if len(overlap_count_per_parent) == 0:
            continue
        best_parent_idx = max(overlap_count_per_parent.items(), key=lambda item: item[1])[0]
        shared_count = overlap_count_per_parent[best_parent_idx]
        candidate_pairs.append((best_parent_idx, child_idx, shared_count))

    if len(candidate_pairs) == 0:
        return []

    candidate_pairs.sort(
        key=lambda pair: (
            -pair[2],  # more overlapping measurements first
            -(parent_track_quality[pair[0]][2] + child_track_quality[pair[1]][2]),  # longer tracks first
            parent_track_quality[pair[0]][1] + child_track_quality[pair[1]][1],  # smaller max errors first
        )
    )

    used_parent_indices: set[int] = set()
    used_child_indices: set[int] = set()
    unique_pairs: list[tuple[int, int, int]] = []
    for parent_idx, child_idx, shared_count in candidate_pairs:
        if parent_idx in used_parent_indices or child_idx in used_child_indices:
            continue
        used_parent_indices.add(parent_idx)
        used_child_indices.add(child_idx)
        unique_pairs.append((parent_idx, child_idx, shared_count))

    # Prioritize pairs where both tracks satisfy length and reprojection quality.
    unique_pairs.sort(
        key=lambda pair: (
            0 if (parent_track_quality[pair[0]][0] and child_track_quality[pair[1]][0]) else 1,
            -pair[2],
            -(parent_track_quality[pair[0]][2] + child_track_quality[pair[1]][2]),
            parent_track_quality[pair[0]][1] + child_track_quality[pair[1]][1],
        )
    )

    selected_pairs = unique_pairs[:max_correspondences]
    point_correspondences: list[tuple[np.ndarray, np.ndarray]] = []
    for parent_idx, child_idx, _ in selected_pairs:
        point_correspondences.append((parent_tracks[parent_idx].point3(), child_tracks[child_idx].point3()))

    return point_correspondences


def merge_scenes_with_sim3_nonlinear(
    parent_scene: GtsfmData,
    children_scenes: list[GtsfmData],
    max_track_correspondences_for_sim3: int = 150,
    scale_and_average_focal_length_in_merging: bool = False,
    guard_child_min_cams: int = 30,
    min_sim3_correspondences_large_child: int = 50,
    sim3_scale_band: Tuple[float, float] = (0.0, float("inf")),
) -> GtsfmData:
    if len(children_scenes) == 0:
        return parent_scene

    parent_camera_ids = set(parent_scene.get_valid_camera_indices())
    parent_gid_index = _scene_gid_index(parent_scene)

    # Parent camera-spread radius: the scene-scale yardstick for the point-pair RANSAC threshold and the
    # point-factor sigma below.
    _centers = np.array(
        [np.array(parent_scene.get_camera(i).pose().translation()) for i in parent_camera_ids]
    )
    spread = float(np.median(np.linalg.norm(_centers - _centers.mean(axis=0), axis=1))) if len(_centers) >= 2 else 1.0

    valid_child_scenes: list[GtsfmData] = []
    overlapping_points: list[list[tuple[np.ndarray, np.ndarray]]] = []
    for i, child_scene in enumerate(children_scenes):
        child_camera_ids = set(child_scene.get_valid_camera_indices())
        common_camera_ids = parent_camera_ids & child_camera_ids
        child_gid_index = _scene_gid_index(child_scene)
        id_mode = parent_gid_index is not None and child_gid_index is not None

        if max_track_correspondences_for_sim3 > 0:
            if id_mode:
                # Match by GLOBAL TRACK IDENTITY: recognizes the same physical points triangulated from
                # DISJOINT camera sets (the cross-cluster overlap the shared-camera matcher cannot see).
                points = _select_gid_point_correspondences(
                    parent_scene,
                    child_scene,
                    parent_gid_index,
                    child_gid_index,
                    max_correspondences=max_track_correspondences_for_sim3,
                )
            else:
                points = _select_overlapping_track_point_correspondences(
                    parent_scene,
                    child_scene,
                    max_correspondences=max_track_correspondences_for_sim3,
                    preferred_min_track_length=3,
                    preferred_max_reproj_error_px=1.5,
                )
        else:
            points = []

        # RANSAC-verify the pairs BEFORE the joint solve (ID mode): clean outlier matches, and measure the
        # similarity the pair set actually implies. A coherently-wrong pair set (pair_scale far from 1)
        # means the correspondences cannot be trusted to seat this child — refuse with evidence, pre-solve.
        if id_mode and points:
            points, pair_scale, pair_inlier_ratio = _prefilter_point_pairs(points, spread)
            if points and not (sim3_scale_band[0] <= pair_scale <= sim3_scale_band[1]):
                logger.warning(
                    "MERGE_GUARD: dropping child %d pre-solve (cams=%d, pairs imply scale=%.3g "
                    "[inlier_ratio=%.2f], shared_cams=%d) — inconsistent correspondences.",
                    i,
                    len(child_camera_ids),
                    pair_scale,
                    pair_inlier_ratio,
                    len(common_camera_ids),
                )
                continue

        # MERGE_GUARD: a child with NO structure cannot be seated by anything — never merge it (a 36-cam
        # leaf held by 1 track and 0-track children previously rode in on pose priors and detonated merges).
        if child_scene.number_tracks() == 0:
            logger.warning(
                "MERGE_GUARD: dropping child %d (cams=%d, tracks=0) — structureless reconstruction.",
                i,
                len(child_camera_ids),
            )
            continue

        # MERGE_GUARD (ID mode only, where correspondence count is a complete anchor measure): a large
        # child without a reliable structural tie gets an under-constrained 7-DoF seat -> rigid stray
        # block (ToL: 83 cams seated on 26 correspondences drifted 60-190m). Accuracy > camera count.
        # OVERLAP ESCAPE: >=15 surviving shared cameras are a strong pose anchor on their own (measured:
        # a 50-cam child with 21 shared cams seated fine at 25 correspondences; 7-9 shared cams failed) —
        # BUT only for children with real structure: a 66-cam child holding ONE track escaped through the
        # camera clause and detonated the root merge (1.6e22 px max). Pose anchors can place a block; they
        # cannot vouch for a block whose own reconstruction is hollow.
        camera_escape = len(common_camera_ids) >= 15 and child_scene.number_tracks() >= 50
        if (
            id_mode
            and len(child_camera_ids) >= guard_child_min_cams
            and len(points) < min_sim3_correspondences_large_child
            and not camera_escape
        ):
            logger.warning(
                "MERGE_GUARD: dropping child %d (cams=%d, id_correspondences=%d < %d, shared_cams=%d) — "
                "under-constrained Sim3 seat.",
                i,
                len(child_camera_ids),
                len(points),
                min_sim3_correspondences_large_child,
                len(common_camera_ids),
            )
            continue

        # Admission needs SOME constraint: shared-camera pose factors or point correspondences. (With the
        # gid index, a child sharing ZERO cameras can still be seated on its point correspondences alone —
        # tree-level overlap that died in reconstruction no longer orphans an anchorable child.)
        if len(common_camera_ids) == 0 and len(points) < 10:
            logger.warning(
                "Child scene %d has insufficient overlap with parent (0 shared cams, %d correspondences), skipping",
                i,
                len(points),
            )
            continue

        valid_child_scenes.append(child_scene)
        overlapping_points.append(points)
        logger.info(
            "Using %d parent-child point correspondences for child %d Sim3 alignment%s.",
            len(points),
            i,
            " (global-track-id)" if id_mode else "",
        )

    if len(valid_child_scenes) == 0:
        return parent_scene

    aTi_measurements = _create_unary_measurements(parent_scene)
    bTi_measurements = [_create_unary_measurements(child_scene) for child_scene in valid_child_scenes]

    # Point-correspondence sigma. LEGACY (no gid sidecars) = fixed 1.0: points are effectively advisory
    # and seats are pose-anchored — the exact R3-baseline semantics that produced the most complete ToL
    # structure. ID MODE = 10% of the parent camera-spread radius (synthetic-validated stable incl.
    # clumped/low-count pairs): identity-matched anchors carry real weight; the RANSAC prefilter above
    # removes outlier pairs first.
    point3_sigma = max(1e-4, 0.10 * spread) if parent_gid_index is not None else 1.0

    try:
        # New GTSAM API supports adding parent-child 3D correspondences per child.
        aligner = TrajectoryAlignerSim3(
            aTi_measurements, bTi_measurements, [], False, overlapping_points, point3_sigma
        )
    except TypeError:
        logger.warning(
            "TrajectoryAlignerSim3 point-correspondence API unavailable, falling back to pose-only alignment."
        )
        aligner = TrajectoryAlignerSim3(aTi_measurements, bTi_measurements)
    result = aligner.solve()

    opt_aTi = {i: result.atPose3(i) for i in parent_scene.get_valid_camera_indices() if i in result.keys()}

    opt_aSb_list = []
    for i in range(len(valid_child_scenes)):
        opt_bSa = result.atSimilarity3(gtsam.Symbol("S", i).key())
        opt_aSb = opt_bSa.inverse()
        opt_aSb_list.append(opt_aSb)

    # MERGE_GUARD scale band: clusters are metric (global Fetzer focals), so a child's solved Sim3 scale
    # must be ~1. A wild scale means the solve diverged (few/degenerate constraints — e.g. a 9-cam child
    # on 12 correspondences detonated a merge to 5.7e8 px max reproj); merging it plants a garbage block
    # that BA can only "fix" by deletion. Drop such children post-solve.
    kept_children, kept_sim3 = [], []
    for i, (child_scene, opt_aSb) in enumerate(zip(valid_child_scenes, opt_aSb_list)):
        s = float(opt_aSb.scale())
        if not (sim3_scale_band[0] <= s <= sim3_scale_band[1]):
            logger.warning(
                "MERGE_GUARD: dropping child %d post-solve (cams=%d, solved Sim3 scale=%.3g outside "
                "[%g, %g]) — diverged seat.",
                i,
                len(child_scene.get_valid_camera_indices()),
                s,
                sim3_scale_band[0],
                sim3_scale_band[1],
            )
            continue
        kept_children.append(child_scene)
        kept_sim3.append(opt_aSb)
    valid_child_scenes, opt_aSb_list = kept_children, kept_sim3
    if len(valid_child_scenes) == 0:
        return parent_scene

    merged = parent_scene
    for i, aTi in opt_aTi.items():
        # Optionally scale-and-average intrinsics using child candidates in parent frame.
        if scale_and_average_focal_length_in_merging:
            valid_calibrations = []
            valid_focal_scale_factors = []
            for child_idx, child_scene in enumerate(valid_child_scenes):
                if i in child_scene.get_valid_camera_indices():
                    valid_calibrations.append(child_scene.get_camera(i).calibration())  # type: ignore
                    valid_focal_scale_factors.append(opt_aSb_list[child_idx].scale())
            valid_calibrations.append(parent_scene.get_camera(i).calibration())  # type: ignore
            valid_focal_scale_factors.append(1.0)
            updated_calibration = get_average_calibration(valid_calibrations, valid_focal_scale_factors)
        else:
            updated_calibration = parent_scene.get_camera(i).calibration()  # type: ignore
        new_camera = gtsfm_types.create_camera(aTi, updated_calibration)
        merged.update_camera(i, new_camera)

    for i, child_scene in enumerate(valid_child_scenes):
        opt_aSb = opt_aSb_list[i]
        merged = merged.merged_with(
            child_scene,
            opt_aSb,
            scale_focal_length=scale_and_average_focal_length_in_merging,
        )  # type: ignore
    return merged


def get_average_calibration(
    calibrations: list[gtsfm_types.CALIBRATION_TYPE],
    focal_scale_factors: list[float],
) -> gtsfm_types.CALIBRATION_TYPE:
    """Get the average calibration from the cameras."""
    assert len(calibrations) > 0, "At least one calibration is required to compute the average calibration."
    calibration_vectors = []
    calib_cls = type(calibrations[0])

    def to_cal_vector(c: gtsfm_types.CALIBRATION_TYPE, focal_scale_factor: float) -> np.ndarray:
        if isinstance(c, gtsam.Cal3Bundler):
            return np.array([c.fx() * focal_scale_factor, c.k1(), c.k2(), c.px(), c.py()])
        else:
            calib_vec = c.vector()
            calib_vec[:2] *= focal_scale_factor
            return calib_vec  # type: ignore

    for s, calibration in zip(focal_scale_factors, calibrations):
        calibration_vectors.append(to_cal_vector(calibration, s))
    average_calibration_vector = np.mean(calibration_vectors, axis=0)
    if calib_cls == gtsam.Cal3Bundler:
        # Cal3Bundler expects the parameters in the order: fx, k1, k2, px, py, no vector constructor.
        return calib_cls(*average_calibration_vector.tolist())  # type: ignore
    else:
        return calib_cls(average_calibration_vector)  # type: ignore


@dataclass(frozen=True)
class MergedNodeResult:
    """Results of merging child scenes with parent scenes in the reconstruction tree.

    Attributes:
        scene: The merged scene.
        metrics: The metrics for the merged scene.
    """

    scene: Optional[GtsfmData]
    pre_ba_scene: Optional[GtsfmData]
    metrics: GtsfmMetricsGroup
    pre_ba_metrics: GtsfmMetricsGroup
    # Good-pose cameras dropped by the post-BA track filter at this node, keyed by camera index ->
    # merged-frame Camera. Populated only at nodes where recover_trackless_cameras_in_retriangulation
    # is set; the root node's set is what the post-merge retriangulation tries to geometrically recover.
    trackless_cameras: Optional[dict] = None
    # Prior-backed ANNEX accumulated over this subtree (export_trackless_annex only): every camera a
    # track filter dropped anywhere below, carried in THIS node's frame (each child's annex re-seated
    # by the Sim3 relating the child scene to this merged result). Never merged into `scene`; the root's
    # annex is exported as a separate posed-only model.
    annex_cameras: Optional[dict] = None


@dataclass(frozen=True)
class MergedNodeSummary:
    """Lightweight summary of a merged node safe to return to the client."""

    merge_success: bool
    num_images: int
    num_tracks: int
    metrics: GtsfmMetricsGroup
    pre_ba_metrics: GtsfmMetricsGroup


def _sanitize_component(value: str, fallback: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").lower()
    return sanitized or fallback


def annotate_scene_with_metadata(
    scene: Optional[GtsfmData],
    plots_dir: str | Path | None,
    cluster_label: Optional[str],
    measurement_gid_index: Optional[tuple[np.ndarray, np.ndarray]] = None,
) -> Optional[GtsfmData]:
    """Attach plotting metadata (and the optional per-cluster gid-index sidecar) to a scene before merging."""
    if scene is None:
        return None
    if plots_dir is not None:
        setattr(scene, _SCENE_PLOTS_DIR_ATTR, Path(plots_dir))
    if cluster_label is not None:
        setattr(scene, _SCENE_LABEL_ATTR, cluster_label)
    if measurement_gid_index is not None:
        setattr(scene, _SCENE_GID_INDEX_ATTR, measurement_gid_index)
    return scene


def _propagate_scene_metadata(target: Optional[GtsfmData], *sources: Optional[GtsfmData]) -> None:
    """Ensure merged scenes keep the plots directory and label metadata."""
    if target is None:
        return
    for source in sources:
        if source is None:
            continue
        plots_dir = getattr(source, _SCENE_PLOTS_DIR_ATTR, None)
        label = getattr(source, _SCENE_LABEL_ATTR, None)
        if plots_dir is not None or label is not None:
            annotate_scene_with_metadata(target, plots_dir, label)
        if (
            getattr(target, _SCENE_PLOTS_DIR_ATTR, None) is not None
            and getattr(target, _SCENE_LABEL_ATTR, None) is not None
        ):
            break


def _compute_scene_reprojection_stats(
    scene: Optional[GtsfmData],
) -> Optional[tuple[np.ndarray, float, float, float, float]]:
    """Aggregate reprojection error stats for a scene."""
    if scene is None:
        return None
    cameras = scene.cameras()
    if len(cameras) == 0 or scene.number_tracks() == 0:
        return None

    error_blocks: list[np.ndarray] = []
    for track in scene.tracks():
        if track.numberMeasurements() == 0:
            continue
        errors, _ = compute_track_reprojection_errors(cameras, track)
        if errors.size == 0:
            continue
        valid_errors = errors[~np.isnan(errors)]
        if valid_errors.size > 0:
            error_blocks.append(valid_errors)

    if len(error_blocks) == 0:
        return None
    all_errors = np.concatenate(error_blocks)
    return (
        all_errors,
        float(np.mean(all_errors)),
        float(np.median(all_errors)),
        float(np.min(all_errors)),
        float(np.max(all_errors)),
    )


def _log_scene_reprojection_stats(scene: Optional[GtsfmData], label: str, *, plot_histograms: bool) -> None:
    """Log reprojection error statistics."""
    stats = _compute_scene_reprojection_stats(scene)
    if stats is None:
        logger.info("📏 %s reprojection error stats unavailable", label)
        return
    errors, mean, median, min_err, max_err = stats
    assert scene is not None
    logger.info(
        "📏 %s reprojection error (px): mean=%.2f median=%.2f min=%.2f max=%.2f, #cameras=%d, #tracks=%d, #images=%d",
        label,
        mean,
        median,
        min_err,
        max_err,
        len(scene.cameras()),
        scene.number_tracks(),
        scene.number_images(),
    )
    if plot_histograms:
        _plot_reprojection_error_distribution(errors, scene, label, mean, median, min_err, max_err)


def _plot_reprojection_error_distribution(
    errors: np.ndarray,
    scene: Optional[GtsfmData],
    label: str,
    mean: float,
    median: float,
    min_err: float,
    max_err: float,
) -> None:
    """Persist a histogram of reprojection errors for diagnostics."""
    if scene is None:
        return
    plots_dir: Optional[Path] = getattr(scene, _SCENE_PLOTS_DIR_ATTR, None)
    if plots_dir is None:
        return
    cluster_label = getattr(scene, _SCENE_LABEL_ATTR, None) or label
    sanitized_cluster = _sanitize_component(cluster_label, "scene")
    sanitized_context = _sanitize_component(label, "context")
    output_path = plots_dir / f"{sanitized_cluster}_{sanitized_context}_reprojection_error_hist.png"

    try:
        import matplotlib.pyplot as plt  # Local import, plotting only happens when needed

        plots_dir.mkdir(parents=True, exist_ok=True)
        bin_count = int(np.clip(np.sqrt(errors.size) * 2, 10, 120))
        fig, ax = plt.subplots(figsize=(6.4, 4.0))
        ax.hist(errors, bins=bin_count, color="#1f77b4", edgecolor="white")
        ax.set_title(f"{cluster_label} reprojection error distribution")
        ax.set_xlabel("Reprojection error (px)")
        ax.set_ylabel("Track count")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
        stats_text = "\n".join(
            [
                f"mean: {mean:.2f} px",
                f"median: {median:.2f} px",
                f"min: {min_err:.2f} px",
                f"max: {max_err:.2f} px",
            ]
        )
        ax.text(
            0.98,
            0.98,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover - plotting is best-effort
        logger.warning("⚠️ Failed to save reprojection error plot %s: %s", output_path, exc)


def _get_pose_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
    save_dir: Optional[str] = None,
    metric_constructed_only: bool = False,
) -> GtsfmMetricsGroup:
    """Compute pose metrics for a BA result after aligning with ground truth.

    Args:
        result_data: The GtsfmData object to compute pose metrics for.
        cameras_gt: The ground truth cameras.
        save_dir: The directory to save the pose metrics to.

    Returns:
        A GtsfmMetricsGroup object containing the pose metrics.
    """
    # Compute metrics for only those images that belong to the merged scene.
    image_idxs = list(result_data._image_info.keys())
    poses_gt: dict[int, Pose3] = {
        i: cameras_gt[i].pose() for i in image_idxs if i < len(cameras_gt) and cameras_gt[i] is not None
    }

    if len(poses_gt) == 0:
        return GtsfmMetricsGroup(name="ba_pose_error_metrics", metrics=[])

    aligned_result_data = result_data.align_via_sim3_and_transform(poses_gt)
    return metrics_utils.compute_ba_pose_metrics(
        gt_wTi=poses_gt,
        computed_wTi=aligned_result_data.get_camera_poses(),
        save_dir=save_dir,
        store_full_data=False,
        metric_constructed_only=metric_constructed_only,
    )


def _get_intrinsics_metrics(
    result_data: GtsfmData,
    cameras_gt: list[Optional[gtsfm_types.CAMERA_TYPE]],
) -> GtsfmMetricsGroup:
    """Compute intrinsics metrics for a merged result against ground truth cameras."""
    image_idxs = list(result_data._image_info.keys())
    gt_cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    computed_cameras: dict[int, gtsfm_types.CAMERA_TYPE] = {}
    for i in image_idxs:
        if i >= len(cameras_gt):
            continue
        gt_cam = cameras_gt[i]
        est_cam = result_data.get_camera(i)
        if gt_cam is not None and est_cam is not None:
            gt_cameras[i] = gt_cam
            computed_cameras[i] = est_cam
    if len(gt_cameras) == 0:
        return GtsfmMetricsGroup(name="intrinsics_metrics", metrics=[])
    return metrics_utils.compute_intrinsics_metrics(
        gt_cameras=gt_cameras,
        computed_cameras=computed_cameras,
        store_full_data=True,
    )


def compute_merging_metrics(
    merged_scene: Optional[GtsfmData],
    *,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    save_dir: Optional[str | Path] = None,
    store_full_data: bool = False,
    metric_constructed_only: bool = False,
    child_camera_counts: list[int] | None = None,
    child_camera_overlap_with_parent: list[int] | None = None,
    suffix: str = "",
) -> GtsfmMetricsGroup:
    """Build metrics describing a merged reconstruction at a tree node.

    Args:
        merged_scene: The merged scene.
        cameras_gt: The ground truth cameras.
        save_dir: The directory to save the metrics to.
        store_full_data: Whether to store full data.
        child_camera_counts: The number of cameras in each child.
        child_camera_overlap_with_parent: The number of cameras in the parent that are also in each child.

    Returns:
        A GtsfmMetricsGroup object containing the merging metrics.
    """
    child_camera_counts = child_camera_counts or []
    child_camera_overlap_with_parent = child_camera_overlap_with_parent or []
    child_camera_counts_arr = np.asarray(child_camera_counts, dtype=np.int32)
    child_camera_overlap_arr = np.asarray(child_camera_overlap_with_parent, dtype=np.int32)
    pose_save_dir = str(save_dir) if isinstance(save_dir, Path) else save_dir
    metrics = [
        GtsfmMetric("merge_success", 1 if merged_scene is not None else 0),
        GtsfmMetric("merge_child_count", len(child_camera_counts)),
        GtsfmMetric(
            "child_camera_counts",
            child_camera_counts_arr,
            store_full_data=True,
        ),
        GtsfmMetric(
            "child_camera_overlap_with_parent",
            child_camera_overlap_arr,
            store_full_data=True,
        ),
    ]
    if merged_scene is not None:
        metrics.extend(merged_scene.get_metrics(suffix="_merged", store_full_data=store_full_data))
    merging_metrics = GtsfmMetricsGroup(name=f"merging_metrics{suffix}", metrics=metrics)
    if cameras_gt is not None and merged_scene is not None:
        ba_pose_error_metrics = _get_pose_metrics(
            merged_scene,
            cameras_gt,
            save_dir=pose_save_dir,
            metric_constructed_only=metric_constructed_only,
        )
        merging_metrics.extend(ba_pose_error_metrics)
        merging_metrics.extend(_get_intrinsics_metrics(merged_scene, cameras_gt))
    return merging_metrics


def _run_export_task(payload: Tuple[Optional[Path], Future | MergedNodeResult]) -> None:
    """Persist a merged reconstruction to COLMAP text format.

    Args:
        payload: Tuple pairing the directory, and the future or result of the merged reconstruction.
    """
    merged_dir, merged_future = payload
    merged_result = merged_future.result() if isinstance(merged_future, Future) else merged_future
    merged_scene = merged_result.scene
    if merged_dir is None or merged_scene is None:
        return
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged_scene.export_as_colmap_text(merged_dir)
    if merged_scene.has_gaussian_splats():
        gaussian_splats = merged_scene.get_gaussian_splats()
        if isinstance(gaussian_splats, GaussiansProtocol):
            try:
                # Lazy import: cluster_anysplat transitively requires the GPU-only vggt package, which
                # merging must not hard-depend on (only splat-carrying scenes ever reach this branch).
                from gtsfm.cluster_optimizer.cluster_anysplat import save_splats

                save_splats(merged_scene, merged_dir)
            except Exception as exc:
                logger.warning("⚠️ Failed to export Gaussian splats: %s", exc)
    pre_ba_merged = merged_result.pre_ba_scene
    if pre_ba_merged is not None:
        pre_ba_dir = merged_dir.parent / "merged_pre_ba"
        pre_ba_dir.mkdir(parents=True, exist_ok=True)
        pre_ba_merged.export_as_colmap_text(pre_ba_dir)
        if pre_ba_merged.has_gaussian_splats():
            gaussian_splats = pre_ba_merged.get_gaussian_splats()
            if isinstance(gaussian_splats, GaussiansProtocol):
                try:
                    from gtsfm.cluster_optimizer.cluster_anysplat import save_splats

                    save_splats(pre_ba_merged, pre_ba_dir)
                except Exception as exc:
                    logger.warning("⚠️ Failed to export Gaussian splats: %s", exc)


def schedule_exports(
    client: Client, handles_tree: Tree[ClusterExecutionHandles], merged_future_tree: Tree[Future]
) -> Tree[Future]:
    """Schedule persistence of merged reconstructions for each cluster."""

    def _to_payload_with_future(
        value: tuple[ClusterExecutionHandles, Future], child_payloads: tuple[object, ...]
    ) -> tuple[Optional[Path], Future]:
        handle, merged_future = value
        is_leaf = len(child_payloads) == 0
        merged_dir = None if is_leaf else handle.output_paths.results / "merged"
        return merged_dir, merged_future

    export_payload_tree = Tree.zip(handles_tree, merged_future_tree).map_with_children(_to_payload_with_future)

    return submit_tree_map(client, export_payload_tree, _run_export_task, pure=False)


def _summarize_merged_result(result: MergedNodeResult) -> MergedNodeSummary:
    """Project a merged result down to the small client-facing payload."""
    scene = result.scene
    if scene is None:
        return MergedNodeSummary(
            merge_success=False,
            num_images=0,
            num_tracks=0,
            metrics=result.metrics,
            pre_ba_metrics=result.pre_ba_metrics,
        )

    return MergedNodeSummary(
        merge_success=True,
        num_images=scene.number_images(),
        num_tracks=scene.number_tracks(),
        metrics=result.metrics,
        pre_ba_metrics=result.pre_ba_metrics,
    )


def schedule_summaries(client: Client, merged_future_tree: Tree[Future]) -> Tree[Future]:
    """Schedule extraction of lightweight merge summaries for client-side reporting."""
    return merged_future_tree.map(
        lambda merged_future: client.submit(_summarize_merged_result, merged_future, pure=False)
    )


def _drop_outlier_tracks(scene: GtsfmData) -> GtsfmData:
    """Drop outlier tracks from a scene based on reprojection error.

    Args:
        scene: The scene to drop outlier tracks from.

    Returns:
        The scene with outlier tracks dropped.
    """
    if scene.number_tracks() == 0:
        return scene
    track_errors: list[float] = []
    tracks = scene.tracks()
    cameras = scene.cameras()
    for track in tracks:
        errors, _ = compute_track_reprojection_errors(cameras, track)
        mean_error = float(np.nanmean(errors)) if errors.size > 0 else np.nan
        track_errors.append(mean_error)
    finite_errors = np.array(track_errors, dtype=float)
    finite_errors = finite_errors[np.isfinite(finite_errors)]
    if finite_errors.size > 0:
        median_error = float(np.median(finite_errors))
        error_threshold = min(5.0, median_error * 5.0)
        retained_tracks = [
            track for track, err in zip(tracks, track_errors) if np.isfinite(err) and err <= error_threshold
        ]
        removed_count = scene.number_tracks() - len(retained_tracks)
        if removed_count > 0:
            logger.info(
                "🧹 Dropping %d tracks with reprojection error > %.2f px (median=%.2f).",
                removed_count,
                error_threshold,
                median_error,
            )
            filtered = GtsfmData.from_cameras_and_tracks(
                cameras,
                retained_tracks,
                number_images=scene.number_images(),
                image_info=scene._clone_image_info(),
                gaussian_splats=scene.get_gaussian_splats(),
            )
            _propagate_scene_metadata(filtered, scene)
            scene = filtered
    return scene


def filter_tracks_by_triangulation_angle(scene: GtsfmData, min_angle_deg: float) -> GtsfmData:
    """Drop tracks whose maximum pairwise triangulation angle is below ``min_angle_deg``.

    Low-parallax tracks are depth-unconstrained: the reprojection filters cannot see their depth
    error (it lies along the viewing ray), so they survive BA as fuzz/spray around the structure.
    Output-side filter only — intended for the final merged scene at export, never inside merging.
    """
    if min_angle_deg <= 0:
        return scene
    cameras = scene.cameras()
    centers = {cam_idx: np.asarray(camera.pose().translation(), dtype=float) for cam_idx, camera in cameras.items()}

    retained_tracks = []
    for track in scene.tracks():
        track_cams = {int(track.measurement(k)[0]) for k in range(track.numberMeasurements())}
        cam_centers = np.array([centers[i] for i in track_cams if i in centers])
        if len(cam_centers) < 2:
            continue  # <2 posed views: no triangulation support at all
        rays = cam_centers - np.asarray(track.point3(), dtype=float)
        norms = np.linalg.norm(rays, axis=1)
        rays = rays[norms > 1e-9] / norms[norms > 1e-9, None]
        if len(rays) < 2:
            continue
        if len(rays) > 25:  # cap the pairwise cost for long tracks (evenly spaced, deterministic)
            rays = rays[np.linspace(0, len(rays) - 1, 25).astype(int)]
        cos_matrix = np.clip(rays @ rays.T, -1.0, 1.0)
        np.fill_diagonal(cos_matrix, 1.0)
        if math.degrees(math.acos(float(cos_matrix.min()))) >= min_angle_deg:
            retained_tracks.append(track)

    removed = scene.number_tracks() - len(retained_tracks)
    logger.info(
        "🔭 Triangulation-angle filter (<%.1f deg): dropped %d of %d tracks (%.1f%%).",
        min_angle_deg,
        removed,
        scene.number_tracks(),
        100.0 * removed / max(scene.number_tracks(), 1),
    )
    if removed == 0:
        return scene
    filtered = GtsfmData.from_cameras_and_tracks(
        cameras,
        retained_tracks,
        number_images=scene.number_images(),
        image_info=scene._clone_image_info(),
        gaussian_splats=scene.get_gaussian_splats(),
    )
    _propagate_scene_metadata(filtered, scene)
    return filtered


def _align_and_merge_results(
    result1: GtsfmData,
    result2: GtsfmData,
    drop_if_merging_fails: bool = True,
    scale_focal_length_in_merging: bool = False,
) -> GtsfmData:
    """Align result2 to result1 and merge it with result1.

    Args:
        result1: The first result to merge.
        result2: The second result to merge.
        drop_if_merging_fails: Whether to drop result2 if merging fails.

    Returns:
        The merged result if merging succeeds, otherwise the first result.
    """
    try:
        _1S2 = align_utils.sim3_from_Pose3_maps(result1.poses(), result2.poses())
    except Exception as exc:
        if drop_if_merging_fails:
            logger.warning("⚠️ Dropping a node because the two results are not aligned: %s", exc)
            return result1
        else:
            _1S2 = Similarity3()
    try:
        merged = result1.merged_with(result2, _1S2, scale_focal_length=scale_focal_length_in_merging)
        return merged
    except Exception as exc:
        logger.warning("⚠️ Failed to merge results: %s", exc)
        return result1


def _annex_on_bailout(
    merged_scene: Optional[GtsfmData],
    child_results: tuple["MergedNodeResult", ...],
    options: "MergingOptions",
) -> Optional[dict]:
    """Carry the children's annexes even when a node bails out early (BA failure, no-track prune):
    the subtree's dropped cameras must not silently vanish on exactly the failure paths the annex
    exists for. No fresh drops here — the node produced none."""
    if not options.export_trackless_annex or merged_scene is None:
        return None
    try:
        return _carry_annex_cameras(merged_scene, child_results, {})
    except Exception as exc:
        logger.warning("🎒 Annex: bailout carry failed (%s) — subtree annex dropped at this node.", exc)
        return None


def _carry_annex_cameras(
    merged_scene: GtsfmData,
    child_results: tuple["MergedNodeResult", ...],
    fresh_drops: dict,
) -> Optional[dict]:
    """Accumulate the prior-backed annex at a merge node.

    Each child's annex is re-seated into this node's frame by the Sim3 relating the child scene's poses
    to the merged result (the same rail the gaussian-splat carry rides), then this node's own post-BA
    filter drops are added (already in-frame, freshest poses — they override child copies). Cameras that
    re-registered with tracks in the merged scene are purged: the tracked scene copy always wins. The
    annex never enters a scene, a Sim3 solve, duplicate resolution, or a BA — inert cargo by construction.
    """
    annex: dict = {}
    merged_poses = merged_scene.poses()
    for child in child_results:
        child_annex = child.annex_cameras
        if not child_annex or child.scene is None:
            continue
        child_poses = child.scene.poses()
        shared = sorted(merged_poses.keys() & child_poses.keys())
        # CARRY GUARDS (RF lesson: a Sim3 fit on few / near-collinear anchors solves a garbage
        # scale and launches the whole annex block off to numerical infinity — better to drop a
        # child's annex than export exploded poses):
        #   (a) >= 4 anchors;
        #   (b) anchors non-degenerate (second principal extent >= 5% of the first);
        #   (c) the solved transform must actually map the anchors onto their merged copies
        #       (median residual <= 25% of the merged camera spread).
        if len(shared) < 4:
            logger.info(
                "🎒 Annex: child shares only %d cam(s) with the merged node — dropping its %d annex cam(s).",
                len(shared),
                len(child_annex),
            )
            continue
        a_pts = np.array([np.array(merged_poses[i].translation()) for i in shared])
        b_pts = np.array([np.array(child_poses[i].translation()) for i in shared])
        sv = np.linalg.svd(a_pts - a_pts.mean(axis=0), compute_uv=False)
        if sv[0] <= 0 or sv[1] / sv[0] < 0.05:
            logger.info(
                "🎒 Annex: child's %d shared cams are near-collinear — dropping its %d annex cam(s).",
                len(shared),
                len(child_annex),
            )
            continue
        try:
            # Robust (hypothesis-sampled) fit: a plain least-squares Align is poisoned by stray
            # cameras with extreme coordinates in either scene (the RF annex displacement).
            aSb = align_utils.sim3_from_Pose3_maps_robust(merged_poses, child_poses)
        except Exception as exc:
            logger.warning(
                "🎒 Annex: seat transform failed for a child (%s) — dropping its %d annex cam(s).",
                exc,
                len(child_annex),
            )
            continue
        pred = np.array([np.array(aSb.transformFrom(gtsam.Point3(*p))) for p in b_pts])
        fit_res = np.linalg.norm(pred - a_pts, axis=1)
        all_pts = np.array([np.array(p.translation()) for p in merged_poses.values()])
        spread = float(np.median(np.linalg.norm(all_pts - all_pts.mean(axis=0), axis=1))) or 1.0
        if float(np.median(fit_res)) > 0.25 * spread:
            logger.warning(
                "🎒 Annex: carry fit does not reproduce its %d anchors (median residual %.3g vs "
                "spread %.3g) — dropping the child's %d annex cam(s).",
                len(shared),
                float(np.median(fit_res)),
                spread,
                len(child_annex),
            )
            continue
        annex.update(camera_map_with_sim3(aSb, child_annex))
    annex.update(fresh_drops)
    for i in merged_scene.get_valid_camera_indices():
        annex.pop(i, None)
    if annex:
        logger.info("🎒 Annex: carrying %d prior-backed camera pose(s) up the tree.", len(annex))
    return annex or None


def combine_results(
    current: GtsfmData | None,
    child_results: tuple[MergedNodeResult, ...],
    *,
    cameras_gt: Optional[list[Optional[gtsfm_types.CAMERA_TYPE]]] = None,
    options: MergingOptions | None = None,
    store_full_data: bool = False,
) -> MergedNodeResult:
    """Run the merging and parent BA pipeline using already-transformed children.

    Args:
        current: The current scene.
        child_results: The results of the child clusters.
        cameras_gt: The ground truth cameras.
        options: Merging configuration. Uses defaults if None.
        store_full_data: Whether to store full data for the merging metrics.

    Returns:
        A MergedNodeResult object containing the merged scene and its metrics.
    """
    if options is None:
        options = MergingOptions()

    child_scenes: tuple[Optional[GtsfmData], ...] = tuple[GtsfmData | None, ...](child.scene for child in child_results)

    # Union of the gid-index sidecars (this node's own + all children's) — used for gid-based duplicate
    # fusion below and propagated on the outgoing scenes so ancestors can ID-match against this subtree.
    node_gid_index = _union_gid_indices(_scene_gid_index(current), *[_scene_gid_index(c) for c in child_scenes])

    # Some stats for the merging metrics.
    parent_camera_set = set(current.get_valid_camera_indices()) if current is not None else set()
    child_camera_counts: list[int] = []
    child_camera_overlap_with_parent: list[int] = []
    for child_scene in child_scenes:
        child_cam_set = set(child_scene.get_valid_camera_indices()) if child_scene is not None else set()
        child_camera_counts.append(len(child_cam_set))
        child_camera_overlap_with_parent.append(len(child_cam_set & parent_camera_set))

    def _finalize_result(
        result_scene: Optional[GtsfmData],
        pre_ba_scene: Optional[GtsfmData],
        trackless_cameras: Optional[dict] = None,
        annex_cameras: Optional[dict] = None,
    ) -> MergedNodeResult:
        # Sidecar rides the outgoing scenes (same rail as the plots/label attrs) — zero extra shipping.
        for _scn in (result_scene, pre_ba_scene):
            if _scn is not None and node_gid_index is not None:
                setattr(_scn, _SCENE_GID_INDEX_ATTR, node_gid_index)
        return MergedNodeResult(
            scene=result_scene,
            pre_ba_scene=pre_ba_scene,
            trackless_cameras=trackless_cameras,
            annex_cameras=annex_cameras,
            metrics=compute_merging_metrics(
                result_scene,
                cameras_gt=cameras_gt,
                store_full_data=store_full_data,
                child_camera_counts=child_camera_counts,
                child_camera_overlap_with_parent=child_camera_overlap_with_parent,
                metric_constructed_only=options.metric_constructed_only,
            ),
            pre_ba_metrics=compute_merging_metrics(
                pre_ba_scene,
                cameras_gt=cameras_gt,
                store_full_data=store_full_data,
                child_camera_counts=child_camera_counts,
                child_camera_overlap_with_parent=child_camera_overlap_with_parent,
                suffix="_pre_ba",
                metric_constructed_only=options.metric_constructed_only,
            ),
        )

    if current is None:
        return _finalize_result(None, None)

    # Log reprojection stats for the current scene and all children.
    _log_scene_reprojection_stats(current, "Current Node", plot_histograms=options.plot_reprojection_histograms)
    valid_child_scenes = [c for c in child_scenes if c is not None]

    logger.info("🫱🏻‍🫲🏽 Merging with %d / %d valid children ", len(valid_child_scenes), len(child_scenes))

    for idx, child in enumerate(child_scenes):
        if child is not None:
            _log_scene_reprojection_stats(child, f"child #{idx}", plot_histograms=options.plot_reprojection_histograms)

    if len(valid_child_scenes) == 0:
        return _finalize_result(current, None)

    metadata_source = current

    # Initialize the merged scene: use the current scene.
    merged = current
    _log_scene_reprojection_stats(merged, "Current node", plot_histograms=options.plot_reprojection_histograms)

    if options.use_nonlinear_sim3_alignment:
        merged = merge_scenes_with_sim3_nonlinear(
            merged,
            valid_child_scenes,
            max_track_correspondences_for_sim3=options.max_track_correspondences_for_sim3,
            scale_and_average_focal_length_in_merging=options.scale_and_average_focal_length,
            guard_child_min_cams=options.guard_child_min_cams,
            min_sim3_correspondences_large_child=options.min_sim3_correspondences_large_child,
            sim3_scale_band=(options.sim3_scale_band_min, options.sim3_scale_band_max),
        )
        _log_scene_reprojection_stats(
            merged, "Merged with children (nonlinear alignment)", plot_histograms=options.plot_reprojection_histograms
        )
    else:
        # Merge all children into the merged scene.
        for i, child in enumerate(valid_child_scenes):
            merged = _align_and_merge_results(
                merged,
                child,
                drop_if_merging_fails=options.drop_child_if_merging_fail,
                scale_focal_length_in_merging=options.scale_and_average_focal_length,
            )
            _log_scene_reprojection_stats(
                merged, f"Merged with child #{i + 1}", plot_histograms=options.plot_reprojection_histograms
            )

    _propagate_scene_metadata(merged, metadata_source)

    if merged is None:
        return _finalize_result(None, None)

    if options.merge_duplicate_tracks and merged is not None and merged.number_tracks() > 0:
        original_track_count = merged.number_tracks()
        merged_tracks: list = []
        measurement_to_track: dict[tuple[int, int, int], int] = {}
        gid_to_track: dict[int, int] = {}

        for track in merged.tracks():
            measurements = [track.measurement(k) for k in range(track.numberMeasurements())]
            target_idx = None
            for cam_idx, uv in measurements:
                existing_idx = measurement_to_track.get(_measurement_key(cam_idx, uv))
                if existing_idx is not None:
                    target_idx = existing_idx
                    break
            # Same GLOBAL track triangulated by different clusters from DISJOINT cameras shares no
            # measurement key — fuse it by global identity instead (kills ghost copies / double walls).
            track_gid = _track_gid(track, node_gid_index)
            if target_idx is None and track_gid >= 0:
                target_idx = gid_to_track.get(track_gid)

            if target_idx is None:
                merged_tracks.append(track)
                target_idx = len(merged_tracks) - 1
            else:
                base_track = merged_tracks[target_idx]
                existing_cams = {base_track.measurement(m_idx)[0] for m_idx in range(base_track.numberMeasurements())}
                for cam_idx, uv in measurements:
                    if cam_idx in existing_cams:
                        continue
                    base_track.addMeasurement(cam_idx, uv)
                    existing_cams.add(cam_idx)

            base_track = merged_tracks[target_idx]
            for m_idx in range(base_track.numberMeasurements()):
                cam_idx, uv = base_track.measurement(m_idx)
                measurement_to_track[_measurement_key(cam_idx, uv)] = target_idx
            if track_gid >= 0:
                gid_to_track[track_gid] = target_idx

        if len(merged_tracks) < original_track_count:
            logger.info(
                "🪢 Merged %d duplicate tracks into %d unique tracks after camera alignment.",
                original_track_count,
                len(merged_tracks),
            )
        valid_tracks: list = []
        for track in merged_tracks:
            if track.hasUniqueCameras():
                valid_tracks.append(track)
        if len(valid_tracks) < len(merged_tracks):
            logger.info(
                "🧹 Discarding %d invalid tracks with repeated camera observations.",
                len(merged_tracks) - len(valid_tracks),
            )
        merged._tracks = valid_tracks

    if not options.run_bundle_adjustment:
        if options.drop_outlier_after_camera_merging:
            merged = _drop_outlier_tracks(merged)
        return _finalize_result(merged, None, annex_cameras=_annex_on_bailout(merged, child_results, options))

    # Log cameras that have no supporting track measurements before running BA.
    if options.drop_camera_with_no_track:
        merged, should_run_ba = data_utils.remove_cameras_with_no_tracks(merged, "parent BA")
        if not should_run_ba:
            return _finalize_result(merged, None, annex_cameras=_annex_on_bailout(merged, child_results, options))
    else:
        logger.info("📌 Retaining zero-track cameras before parent BA (drop disabled).")

    if options.pre_ba_max_reproj_error > 0.0:
        merged = merged.filter_landmark_measurements(options.pre_ba_max_reproj_error, options.pre_ba_min_track_length)
    try:
        optimizer = options.ba_options.to_optimizer(min_track_length=options.min_track_length)
        merged_with_ba, _ = optimizer.run_simple_ba(merged)
        _propagate_scene_metadata(merged_with_ba, merged)
        _log_scene_reprojection_stats(
            merged_with_ba,
            "merged result (with ba)",
            plot_histograms=options.plot_reprojection_histograms,
        )
        if options.drop_outlier_after_camera_merging:
            merged_with_ba = _drop_outlier_tracks(merged_with_ba)
        _log_scene_reprojection_stats(
            merged_with_ba,
            "merged result (drop_outlier_after_camera_merging)",
            plot_histograms=options.plot_reprojection_histograms,
        )

        trackless_cameras: Optional[dict[int, gtsfm_types.CAMERA_TYPE]] = None
        dropped_this_node: dict[int, gtsfm_types.CAMERA_TYPE] = {}
        if options.allow_post_ba_reproj_filtering:
            scene_before_filter = merged_with_ba
            cams_before_filter = set(scene_before_filter.get_valid_camera_indices())
            merged_with_ba = merged_with_ba.filter_landmark_measurements(
                options.post_ba_max_reproj_error,
                options.min_track_length,
                retain_cameras_without_tracks=options.keep_all_cameras,
            )
            if options.recover_trackless_cameras_in_retriangulation or options.export_trackless_annex:
                # Capture the good-pose cameras the track filter just dropped (e.g. low VGGT-depth-conf
                # cams) with their merged-frame poses, so the post-merge retriangulation can recover them
                # and/or the annex can carry them up the tree.
                dropped_ids = cams_before_filter - set(merged_with_ba.get_valid_camera_indices())
                dropped_this_node = {
                    i: scene_before_filter.get_camera(i)
                    for i in dropped_ids
                    if scene_before_filter.get_camera(i) is not None
                }
                if options.recover_trackless_cameras_in_retriangulation:
                    trackless_cameras = dropped_this_node
        annex_cameras: Optional[dict] = None
        if options.export_trackless_annex:
            annex_cameras = _carry_annex_cameras(merged_with_ba, child_results, dropped_this_node)
        _log_scene_reprojection_stats(
            merged_with_ba,
            "merged result (with ba + outlier filtering)",
            plot_histograms=options.plot_reprojection_histograms,
        )
        # TODO: the order here is different from the merging order above, we should fix this.
        if merged.has_gaussian_splats():
            logger.info("🫱🏻‍🫲🏽 Merging Gaussians")
            try:
                merged_with_ba.set_gaussian_splats(None)
                merged_gaussians = None
                if current is not None and current.has_gaussian_splats():
                    postba_S_current = align_utils.sim3_from_Pose3_maps(merged_with_ba.poses(), current.poses())
                    current_gaussians = current.get_gaussian_splats()
                    if current_gaussians is not None:
                        merged_gaussians = transform_gaussian_splats(current_gaussians, postba_S_current)

                for child in valid_child_scenes:
                    try:
                        postba_S_child = align_utils.sim3_from_Pose3_maps(merged_with_ba.poses(), child.poses())

                        if child.has_gaussian_splats():
                            transformed_other_gaussians = transform_gaussian_splats(
                                child.get_gaussian_splats(), postba_S_child  # type: ignore
                            )
                            if merged_gaussians is None:
                                merged_gaussians = transformed_other_gaussians
                            else:
                                merged_gaussians = merge_gaussian_splats(merged_gaussians, transformed_other_gaussians)
                    except Exception as e:
                        logger.warning("⚠️ Failed to align and merge gaussians: %s", e)
                merged_with_ba.set_gaussian_splats(merged_gaussians)
                return _finalize_result(merged_with_ba, merged, trackless_cameras, annex_cameras)

            except Exception as alignment_exc:
                logger.warning("⚠️ Failed to compute pre/post BA Sim(3): %s", alignment_exc)
                return _finalize_result(merged_with_ba, merged, trackless_cameras, annex_cameras)

        else:
            logger.info("✖️ No Gaussians to merge")
            return _finalize_result(merged_with_ba, merged, trackless_cameras, annex_cameras)
    except Exception as exc:
        logger.warning("⚠️ Failed to run bundle adjustment: %s", exc)
        return _finalize_result(merged, None, annex_cameras=_annex_on_bailout(merged, child_results, options))


__all__ = [
    "MergingOptions",
    "MergedNodeResult",
    "MergedNodeSummary",
    "combine_results",
    "schedule_exports",
    "schedule_summaries",
    "compute_merging_metrics",
    "annotate_scene_with_metadata",
]
