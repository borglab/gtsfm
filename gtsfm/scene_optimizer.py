"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TypeVar, cast

import matplotlib
import numpy as np
from dask.delayed import delayed
from dask.distributed import Client, Future, performance_report
from omegaconf import OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm import cluster_merging
from gtsfm.cluster_merging import MergingOptions
from gtsfm.cluster_optimizer import Base, save_metrics_reports
from gtsfm.cluster_optimizer.cluster_optimizer_base import ClusterContext
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.outputs import OutputPaths, cluster_label, prepare_output_paths
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.evaluation.retrieval_metrics import save_retrieval_two_view_metrics
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.graph_partitioner.single_partitioner import SinglePartitioner
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator
from gtsfm.ui.process_graph_generator import ProcessGraphGenerator
from gtsfm.utils.tree import PreOrderIter
from gtsfm.utils.tree_dask import submit_tree_map_with_children

# Set matplotlib backend to "Agg" (Anti-Grain Geometry) for headless rendering
# This must be called before importing pyplot or any other matplotlib modules
# "Agg" is a non-interactive backend that renders to files without requiring a display
matplotlib.use("Agg")

DEFAULT_OUTPUT_ROOT = str(Path(__file__).resolve().parent.parent)

logger = logger_utils.get_logger()
T = TypeVar("T")


@dataclass(frozen=True)
class ClusterExecutionHandles:
    """Futures tracking the execution of a single cluster optimization."""

    reconstruction: Future  # Optional[GtsfmData]
    metrics: Future  # list[GtsfmMetricsGroup]
    io_barrier: Future  # None
    output_paths: OutputPaths
    cluster_path: tuple[int, ...]
    label: str
    edge_count: int


def _identity(value: T) -> T:
    """Return value unchanged. Used to seed futures without extra scheduling."""
    return value


def _empty_cluster_handles(context: ClusterContext, edge_count: int) -> ClusterExecutionHandles:
    """Create placeholder futures for clusters that were skipped."""
    client = context.client
    reconstruction: Future = client.submit(_identity, None, pure=False)
    metrics: Future = client.submit(_identity, [], pure=False)
    io_barrier: Future = client.submit(_identity, None, pure=False)
    return ClusterExecutionHandles(
        reconstruction=reconstruction,
        metrics=metrics,
        io_barrier=io_barrier,
        output_paths=context.output_paths,
        cluster_path=context.cluster_path,
        label=context.label,
        edge_count=edge_count,
    )


def _collect_metric_results(*results: object) -> list[GtsfmMetricsGroup]:
    """Normalize metric outputs into a flat list."""
    collected: list[GtsfmMetricsGroup] = []
    for result in results:
        if result is None:
            continue
        if isinstance(result, (list, tuple)):
            for item in result:
                if item is not None:
                    collected.append(cast(GtsfmMetricsGroup, item))
        else:
            collected.append(cast(GtsfmMetricsGroup, result))
    return collected


def _finalize_io_tasks(*_args: object) -> None:
    """Barrier task used to depend on all I/O side effects."""
    return None


def _load_precomputed_frontend(path: str, num_images: int, expected_dims=None, max_resolution=None):
    """Load a precomputed frontend (scripts/convert_wilsonkl_frontend.py output).

    Returns (padded_keypoints_list, v_corr_idxs_dict) — the exact products of the global
    verification stage. The 1DSfM benchmark ships the frontend all published methods consumed
    (EGs/coords/tracks); consuming it skips retrieval + SIFT + matching + two-view entirely
    and makes comparisons to the published table rows share identical two-view inputs.
    """
    from gtsfm.common.keypoints import Keypoints

    z = np.load(path)
    n = int(z["n_images"])
    if n > num_images:
        raise ValueError(f"precomputed frontend covers {n} images but loader has {num_images}")
    if max_resolution is not None and "max_resolution" in z and int(z["max_resolution"]) != int(max_resolution):
        raise ValueError(
            f"precomputed frontend was converted at max_resolution={int(z['max_resolution'])} "
            f"but this run uses {int(max_resolution)} — reconvert."
        )
    # EXIF-rotation guard: the benchmark's coords are in Bundler's raw pixel frame; our loader
    # applies EXIF transpose. Drop images whose loader-frame dims disagree with the converter's
    # expectation (their keypoints would be transposed -> silent geometric poison otherwise).
    dropped_imgs = set()
    if expected_dims is not None and "img_dims" in z:
        img_dims = z["img_dims"]
        for i, (w, h) in expected_dims.items():
            if i < n and img_dims[i][0] > 0:
                if abs(int(img_dims[i][0]) - int(w)) > 2 or abs(int(img_dims[i][1]) - int(h)) > 2:
                    dropped_imgs.add(i)
        if dropped_imgs:
            logger.warning(
                "📦 Precomputed frontend: dropping %d image(s) whose loader dims mismatch the "
                "converted frame (EXIF-rotation / different source files).", len(dropped_imgs)
            )
    kp_offsets, kp_flat = z["kp_offsets"], z["kp_flat"]
    keypoints_list = []
    for i in range(num_images):
        if i < n and i not in dropped_imgs:
            a, b = int(kp_offsets[i]), int(kp_offsets[i + 1])
            keypoints_list.append(Keypoints(coordinates=kp_flat[a:b].astype(np.float64)))
        else:
            keypoints_list.append(Keypoints(coordinates=np.zeros((0, 2))))
    edges, corr_offsets, corr_flat = z["edges"], z["corr_offsets"], z["corr_flat"]
    v_corr_idxs_dict = {}
    for k in range(len(edges)):
        i1, i2 = int(edges[k][0]), int(edges[k][1])
        if i1 in dropped_imgs or i2 in dropped_imgs:
            continue
        a, b = int(corr_offsets[k]), int(corr_offsets[k + 1])
        v_corr_idxs_dict[(i1, i2)] = corr_flat[a:b]
    return keypoints_list, v_corr_idxs_dict


def _drop_unreportable_cameras(scene: GtsfmData, radius_mult: float = 15.0) -> tuple[GtsfmData, int]:
    """Export policy: a camera beyond ``radius_mult`` x the robust scene radius (95th-percentile
    center distance from the median center) — or at non-finite coordinates — is not a reportable
    pose and is dropped from the EXPORTED model, together with its track measurements (tracks keep
    their remaining measurements; sub-2-view leftovers are removed). Declared in the paper protocol;
    recall is unaffected (such cameras are never near their true location). Returns a filtered COPY
    and the number of cameras dropped; the input scene is never mutated.
    """
    ids = list(scene.get_valid_camera_indices())
    if len(ids) < 8:
        return scene, 0
    centers = {i: np.array(scene.get_camera(i).pose().translation()) for i in ids}
    C = np.array(list(centers.values()))
    ctr = np.median(C, axis=0)
    radius = float(np.percentile(np.linalg.norm(C - ctr, axis=1), 95)) or 1.0
    far = {
        i for i, c in centers.items()
        if not np.all(np.isfinite(c)) or float(np.linalg.norm(c - ctr)) > radius_mult * radius
    }
    if not far:
        return scene, 0
    from gtsam import SfmTrack as _SfmTrack

    out = GtsfmData(number_images=scene.number_images())
    out._image_info = scene._clone_image_info()
    for i in ids:
        if i not in far:
            out.add_camera(i, scene.get_camera(i))
    for j in range(scene.number_tracks()):
        t = scene.get_track(j)
        kept = [(int(t.measurement(k)[0]), t.measurement(k)[1]) for k in range(t.numberMeasurements())
                if int(t.measurement(k)[0]) not in far]
        if len(kept) < 2:
            continue
        tc = _SfmTrack(t.point3())
        tc.r, tc.g, tc.b = t.r, t.g, t.b
        for i, uv in kept:
            tc.addMeasurement(i, uv)
        out.add_track(tc)
    return out, len(far)


def _colorize_scene_tracks_in_place(scene: GtsfmData, loader) -> int:
    """Set each track's (r, g, b) by sampling the source image at its first measurement, if in bounds.

    Only measurement(0) is tried — a track whose first measurement rounds out of bounds, or whose
    first-measurement image fails to load / is not RGB, keeps its previous color (no fallback scan).
    Tracks are grouped by first-measurement camera so only one image is in memory at a time
    (memory O(1 image), one pass over the registered images). Track keypoints live at the loader
    resolution, so `loader.get_image(i)` samples in the right frame. Same single-sample scheme as
    the offline colorize tool the qualitative figures were made with.

    Returns the number of tracks colored. Never raises — a failed image load just leaves its
    tracks at their previous color.
    """
    from collections import defaultdict

    by_cam: dict[int, list] = defaultdict(list)
    for j in range(scene.number_tracks()):
        t = scene.get_track(j)
        if t.numberMeasurements() > 0:
            i, _ = t.measurement(0)
            by_cam[int(i)].append(t)

    n_colored = 0
    for i in sorted(by_cam):
        try:
            arr = loader.get_image(i).value_array
        except Exception:
            continue
        if arr is None or arr.ndim != 3 or arr.shape[2] < 3:
            continue
        h, w = arr.shape[:2]
        for t in by_cam[i]:
            _, uv = t.measurement(0)
            u, v = int(round(float(uv[0]))), int(round(float(uv[1])))
            if 0 <= u < w and 0 <= v < h:
                t.r, t.g, t.b = float(arr[v, u, 0]), float(arr[v, u, 1]), float(arr[v, u, 2])
                n_colored += 1
    return n_colored


def _run_post_merge_retriangulation(
    scene: GtsfmData,
    tracks_2d: list,
    options: MergingOptions,
    trackless_cameras: Optional[dict] = None,
) -> tuple[GtsfmData, dict]:
    """Re-triangulate verified multiview 2D tracks against the merged cameras, then bundle-adjust.

    Mirrors the cluster-level retri stage (`_run_cluster_ba`): builds a fresh sparse structure from
    the verified correspondences (discarding the VGGT-depth points), runs BA -- which jointly refines
    structure AND the merged (VGGT-predicted -> per-cluster-BA'd -> merged -> merge-BA'd) poses -- then
    filters by reprojection error. Designed to run as a single dask task on a worker.

    If `trackless_cameras` is provided (good-pose cameras the merge dropped for lack of a VGGT-depth
    track), their merged poses are injected so the geometric retriangulation -- which is depth-confidence
    INDEPENDENT -- can re-triangulate their existing global 2D tracks and recover them. Cameras that
    still fail to gain a >=3-view track are excluded from BA and dropped by the final filter.

    Returns:
        (refined scene, dropped cameras, root_frame_ids): `dropped` maps camera index -> Camera for
        every camera that entered retri but is absent from the refined scene, at its last-held pose —
        the retri-stage contribution to the prior-backed annex. `root_frame_ids` are the dropped
        cameras that never entered the final BA graph: their poses are still in the INPUT (root-merge)
        gauge and need the caller's root->final re-seat; the rest are already in the final frame.
    """
    from gtsfm.bundle.bundle_adjustment import multi_view_retriangulate_from_2d_tracks

    if trackless_cameras:
        injected = []
        for i, cam in trackless_cameras.items():
            if cam is not None and scene.get_camera(i) is None:
                scene.add_camera(i, cam)  # local op on this worker's copy of the merged scene
                injected.append(i)
        logger.info(
            "♻️  Post-merge retri: injected %d trackless good-pose camera(s) for geometric recovery: %s",
            len(injected),
            sorted(injected),
        )

    # Last-seen camera per index (post-BA where available): the pose a camera holds if a later
    # filter drops it — captured for the annex instead of being lost. `never_baed` tracks cameras
    # that never actually entered the FINAL retri BA graph (run_simple_ba returns excluded cameras
    # at their unoptimized input poses): their captured poses are still in the INPUT (root-merge)
    # gauge and must ride the root->final re-seat, unlike genuinely BA'd-then-dropped cameras.
    last_seen: dict = {i: scene.get_camera(i) for i in scene.get_valid_camera_indices()}
    never_baed: set = set(last_seen)

    def _dropped_vs(final_scene: GtsfmData) -> tuple[dict, set]:
        final_ids = set(final_scene.get_valid_camera_indices())
        dropped = {i: cam for i, cam in last_seen.items() if i not in final_ids and cam is not None}
        return dropped, never_baed & set(dropped)

    ba_options = options.ba_options
    if options.retri_free_ba:
        # Fully-free final solve (GLOMAP-style): the calibration/pose priors that protect the
        # incremental merges throttle the full-scene BA — dropped here. Robustness (robust kernel /
        # GNC) is inherited from ba_options unchanged.
        from dataclasses import replace as _dc_replace

        ba_options = _dc_replace(
            ba_options,
            use_calibration_prior=False,
            use_pose_prior_all_cameras=False,
        )
        logger.info("🔓 Post-merge retri BA running FREE (no calib/pose priors).")

    refined = scene
    for retri_iter in range(max(1, options.retri_iterations)):
        retri = multi_view_retriangulate_from_2d_tracks(refined, tracks_2d)
        if retri.number_tracks() == 0:
            logger.warning("Post-merge retriangulation produced no tracks; keeping previous scene.")
            kept = refined if retri_iter > 0 else scene
            dropped, root_frame_ids = _dropped_vs(kept)
            return kept, dropped, root_frame_ids
        optimizer = ba_options.to_optimizer(min_track_length=options.min_track_length)
        refined, _ = optimizer.run_simple_ba(retri)
        baed_this_iter = set()
        for i in refined.get_valid_camera_indices():
            cam = refined.get_camera(i)
            if cam is not None:
                last_seen[i] = cam
            if len(refined.get_measurements_for_camera(i)) >= ba_options.min_tracks_per_camera:
                baed_this_iter.add(i)
        # Only the LAST BA's gauge is the final frame — judge against this iteration alone.
        never_baed = set(last_seen) - baed_this_iter
        refined = refined.filter_landmark_measurements(
            options.post_ba_max_reproj_error,
            options.min_track_length,
            # When recovering trackless cameras, force-drop any that still fail to triangulate (decoupled
            # from keep_all_cameras) so unrecovered cams don't linger as pose-only and skew the AUC denom.
            retain_cameras_without_tracks=(False if trackless_cameras else options.keep_all_cameras),
        )
        logger.info(
            "🔁 Retri iteration %d/%d: %d tracks after BA + filter.",
            retri_iter + 1,
            max(1, options.retri_iterations),
            refined.number_tracks(),
        )
    # Carry the merged scene's full image set (incl. unregistered images) so the retri pose metrics
    # use the SAME all-images GT denominator as the merged metrics; otherwise the retri "overall" AUC
    # is computed over only its registered cameras and collapses into "constructed-only".
    refined._image_info = scene._clone_image_info()

    if trackless_cameras:
        recovered_set = set(refined.get_valid_camera_indices())
        for i in sorted(trackless_cameras):
            touching = [t for t in tracks_2d if any(m.i == i for m in t.measurements)]
            max_views = max((t.number_measurements() for t in touching), default=0)
            n_retri = len(refined.get_measurements_for_camera(i))  # 0 if dropped; <15 keeps the merged pose
            logger.info(
                "♻️  Retri recovery [cam %d]: recovered=%s, retri_tracks=%d (global tracks touching=%d, "
                "max track views=%d) [<15 retri_tracks => kept merged pose, excluded from retri BA]",
                i,
                i in recovered_set,
                n_retri,
                len(touching),
                max_views,
            )

    dropped, root_frame_ids = _dropped_vs(refined)
    return refined, dropped, root_frame_ids


# ---------------------------------------------------------------------------
# Post-root-merge BOUNDARY RECOVERY (offline-validated exp5 recipe).
#
# The root merge can only keep cameras some cluster reconstructed AND some Sim3 seat carried in; island
# cameras (clusters the MERGE_GUARD dropped, cams the per-cluster BA lost) end up with global-track
# measurements but no pose. The recovery: (a) read the merged scene's own 3D as {gid: xyz} via the
# majority-vote track identity, (b) DLT-triangulate every "boundary" global track that has >=3 posed
# views (<3px mean reproj), (c) RANSAC-DLT resect unposed cameras against the UNION cluster-3D ∪
# fresh-3D, iterating b<->c so newly posed cams enable new triangulations, (d) one BA over the
# augmented scene. The union in (c) is essential: offline, an island camera saw 508 anchors in
# cluster-3D vs 95 in fresh-only triangulation — the union is what made resection work.
# ---------------------------------------------------------------------------


def _dlt_triangulate(
    measurements: list,
    posed: dict,
    intrinsics: dict,
) -> tuple[Optional[np.ndarray], Optional[float]]:
    """DLT multiview triangulation from posed cams; returns (X, mean reproj px) or (None, None).

    Faithful port of the offline-validated triangulate(): ``measurements`` is [(cam, u, v), ...],
    ``posed[i] = (R_w2c, t_w2c)`` (COLMAP convention), ``intrinsics[i] = (f, k1, k2, px, py)`` with only
    f/px/py used (undistorted pinhole model, same as the offline replay).
    """
    A = []
    for i, u, v in measurements:
        Rm, tr = posed[i]
        f, _, _, px, py = intrinsics[i]
        xn, yn = (u - px) / f, (v - py) / f
        P = np.hstack([Rm, tr[:, None]])
        A.append(xn * P[2] - P[0])
        A.append(yn * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.array(A))
    Xh = Vt[-1]
    if abs(Xh[3]) < 1e-12:
        return None, None
    X = Xh[:3] / Xh[3]
    errs = []
    for i, u, v in measurements:
        Rm, tr = posed[i]
        f, _, _, px, py = intrinsics[i]
        p = Rm @ X + tr
        if p[2] <= 1e-9:  # behind a camera -> reject
            return None, None
        errs.append(np.hypot(p[0] / p[2] * f + px - u, p[1] / p[2] * f + py - v))
    return X, float(np.mean(errs))


def _ransac_dlt_resect(
    observations: list,
    intrinsics_entry: tuple,
    struct: dict,
    iters: int = 800,
    min_inl: int = 10,
    px_thresh: float = 4.0,
) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
    """RANSAC-DLT resection of one camera against {gid: xyz}; returns (R_w2c, t_w2c, center, n_inl) or None.

    Faithful port of the offline-validated resect(): ``observations`` is this camera's global-track
    measurements [(gid, u, v), ...]; only observations whose gid is in ``struct`` participate.
    """
    P3, P2 = [], []
    for t, uu, vv in observations:
        X3 = struct.get(int(t))
        if X3 is not None:
            P3.append(X3)
            P2.append((uu, vv))
    if len(P3) < min_inl:
        return None
    f, _, _, px0, py0 = intrinsics_entry
    X = np.array(P3)
    x = (np.array(P2) - [px0, py0]) / f
    rng = np.random.default_rng(0)
    best, best_inl = -1, None

    def dlt(Xs, xs):
        A = []
        for (a, b, c), (xn, yn) in zip(Xs, xs):
            A.append([a, b, c, 1, 0, 0, 0, 0, -xn * a, -xn * b, -xn * c, -xn])
            A.append([0, 0, 0, 0, a, b, c, 1, -yn * a, -yn * b, -yn * c, -yn])
        _, _, Vt = np.linalg.svd(np.array(A))
        P = Vt[-1].reshape(3, 4)
        U, S, Vt2 = np.linalg.svd(P[:, :3])
        d = np.sign(np.linalg.det(U @ Vt2))
        Rm = U @ np.diag([1, 1, d]) @ Vt2
        sc = (S * [1, 1, d]).mean()
        if abs(sc) < 1e-12:
            return None
        t = P[:, 3] / sc
        if np.median((Rm @ X.T).T[:, 2] + t[2]) < 0:  # cheirality: majority of points must be in front
            Rm, t = -Rm, -t
        return Rm, t

    for _ in range(iters):
        idx = rng.choice(len(X), 6, replace=False)
        fit = dlt(X[idx], x[idx])
        if fit is None:
            continue
        Rm, t = fit
        pr = (Rm @ X.T).T + t
        e = np.linalg.norm(pr[:, :2] / np.maximum(pr[:, 2:3], 1e-9) - x, axis=1)
        inl = (pr[:, 2] > 1e-9) & (e < px_thresh / f)
        if inl.sum() > best:
            best, best_inl = int(inl.sum()), inl
    if best < min_inl:
        return None
    fit = dlt(X[best_inl], x[best_inl])
    if fit is None:
        return None
    Rm, t = fit
    return Rm, t, -Rm.T @ t, best


def _calibration_to_tuple(cal) -> tuple[float, float, float, float, float]:
    """(f, k1, k2, px, py) from a gtsam calibration (k1/k2 default 0 for non-Bundler models)."""
    k1 = float(cal.k1()) if hasattr(cal, "k1") else 0.0
    k2 = float(cal.k2()) if hasattr(cal, "k2") else 0.0
    return float(cal.fx()), k1, k2, float(cal.px()), float(cal.py())


def _run_boundary_recovery(
    scene: GtsfmData,
    tracks_2d: list,
    options: MergingOptions,
    fallback_intrinsics: Optional[dict] = None,
) -> GtsfmData:
    """Boundary triangulation + island-camera resection against the merged root scene, then one BA.

    Runs as a single dask task on a worker (mirrors _run_post_merge_retriangulation). ``scene`` is the
    worker-local copy of the root merged scene (mutated in place before BA); ``fallback_intrinsics``
    maps camera index -> (f, k1, k2, px, py) for cameras NOT in the merged scene (global-Fetzer focals
    with loader-initial fallback — the same precedence the offline replay used).
    """
    from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack

    from gtsfm import cluster_merging as _cm

    gid_index = _cm._scene_gid_index(scene)
    if gid_index is None:
        logger.warning(
            "🧭 Boundary recovery: merged scene carries no gid-index sidecar "
            "(enable_gid_merge_anchoring off?); skipping."
        )
        return scene
    fallback_intrinsics = fallback_intrinsics or {}

    # Posed cameras of the merged scene, in the COLMAP (R_w2c, t_w2c) convention the DLT helpers use.
    posed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    intr: dict[int, tuple[float, float, float, float, float]] = {}
    for i in scene.get_valid_camera_indices():
        cam = scene.get_camera(i)
        if cam is None:
            continue
        pose = cam.pose()  # gtsam camera pose = cam-to-world
        R_w2c = np.asarray(pose.rotation().matrix()).T
        posed[i] = (R_w2c, -R_w2c @ np.asarray(pose.translation()))
        intr[i] = _calibration_to_tuple(cam.calibration())

    # (a) gid -> 3D from the merged scene's own tracks (majority-vote identity; first track per gid wins).
    struct: dict[int, np.ndarray] = {}
    for track in scene.tracks():
        gid = _cm._track_gid(track, gid_index)
        if gid >= 0 and gid not in struct:
            struct[gid] = np.asarray(track.point3())
    n_cluster3d = len(struct)

    # Per-camera observation index over the global 2D tracks (for resection).
    cam_obs: dict[int, list[tuple[int, float, float]]] = {}
    for tid, tr in enumerate(tracks_2d):
        for m in tr.measurements:
            cam_obs.setdefault(int(m.i), []).append((tid, float(m.uv[0]), float(m.uv[1])))
    unposed_candidates = sorted(i for i in cam_obs if i not in posed)
    logger.info(
        "🧭 Boundary recovery: %d posed cams, %d cluster-3D gids, %d unposed cameras with "
        "global-track measurements (%d with intrinsics).",
        len(posed),
        n_cluster3d,
        len(unposed_candidates),
        sum(1 for i in unposed_candidates if i in fallback_intrinsics),
    )

    # (b)<->(c): triangulate boundary tracks / resect unposed cameras, iterate (newly posed cams enable
    # new triangulations). Cluster 3D wins on gid collision: gids already in `struct` are never
    # re-triangulated, so fresh points only AUGMENT the merged structure (the essential union).
    boundary_pts: dict[int, np.ndarray] = {}
    recovered: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, int]] = {}
    for it in range(3):
        new_pts = 0
        for tid, tr in enumerate(tracks_2d):
            if tid in struct or tr.number_measurements() < 3:
                continue
            mp, seen = [], set()
            for m in tr.measurements:
                i = int(m.i)
                if i in posed and i in intr and i not in seen:
                    mp.append((i, float(m.uv[0]), float(m.uv[1])))
                    seen.add(i)
            if len(mp) < 3:
                continue
            X, err = _dlt_triangulate(mp, posed, intr)
            if X is not None and err is not None and err < 3.0:
                boundary_pts[tid] = X
                struct[tid] = X
                new_pts += 1
        newly = 0
        for ci in unposed_candidates:
            if ci in posed:
                continue
            k_entry = fallback_intrinsics.get(ci)
            if k_entry is None:
                continue
            r = _ransac_dlt_resect(cam_obs[ci], k_entry, struct)
            if r is not None:
                Rm, t, C, ninl = r
                posed[ci] = (Rm, t)
                intr[ci] = k_entry
                recovered[ci] = (Rm, t, C, ninl)
                newly += 1
        logger.info(
            "🧭 Boundary recovery iter %d: +%d triangulated (total boundary %d), "
            "+%d cameras resected (total recovered %d, posed %d).",
            it,
            new_pts,
            len(boundary_pts),
            newly,
            len(recovered),
            len(posed),
        )
        if newly == 0:
            break

    if not recovered and not boundary_pts:
        logger.info("🧭 Boundary recovery: nothing to recover (0 boundary tracks, 0 cameras); scene unchanged.")
        return scene

    # (d) Add recovered cameras (COLMAP w2c -> gtsam cam-to-world) + boundary tracks restricted to
    # posed cameras, then ONE BA over the augmented scene and the standard post-BA reproj filter.
    for ci in sorted(recovered):
        Rm, t, C, ninl = recovered[ci]
        f, k1, k2, px, py = intr[ci]
        pose_c2w = Pose3(Rot3(Rm.T), C)  # COLMAP w2c -> gtsam camera pose (cam-to-world)
        scene.add_camera(ci, PinholeCameraCal3Bundler(pose_c2w, Cal3Bundler(f, k1, k2, px, py)))
        logger.info("🧭 Boundary recovery: resected camera %d with %d RANSAC inliers.", ci, ninl)

    final_cams = set(scene.get_valid_camera_indices())
    added_tracks = 0
    for gid in sorted(boundary_pts):
        track = SfmTrack(boundary_pts[gid])
        seen = set()
        for m in tracks_2d[gid].measurements:
            i = int(m.i)
            if i in final_cams and i not in seen:
                track.addMeasurement(i, np.array([float(m.uv[0]), float(m.uv[1])]))
                seen.add(i)
        if track.numberMeasurements() >= 3 and scene.add_track(track):
            added_tracks += 1
    logger.info(
        "🧭 Boundary recovery: triangulated %d boundary tracks (added %d), resected %d cameras; "
        "scene now %d cams / %d tracks. Running BA...",
        len(boundary_pts),
        added_tracks,
        len(recovered),
        len(final_cams),
        scene.number_tracks(),
    )

    try:
        optimizer = options.ba_options.to_optimizer(min_track_length=options.min_track_length)
        refined, _ = optimizer.run_simple_ba(scene)
        refined = refined.filter_landmark_measurements(
            options.post_ba_max_reproj_error,
            options.min_track_length,
            retain_cameras_without_tracks=options.keep_all_cameras,
        )
    except Exception as exc:
        logger.warning("🧭 Boundary recovery BA failed (%s); returning un-refined augmented scene.", exc)
        refined = scene
    # Same all-images GT denominator as the merged metrics (see _run_post_merge_retriangulation).
    refined._image_info = scene._clone_image_info()
    logger.info(
        "🧭 Boundary recovery final: %d cameras / %d tracks (was %d cams pre-recovery).",
        len(refined.get_valid_camera_indices()),
        refined.number_tracks(),
        len(final_cams) - len(recovered),
    )
    return refined


class SceneOptimizer:
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(
        self,
        loader: LoaderBase,
        image_pairs_generator: ImagePairsGenerator,
        cluster_optimizer: Base,
        graph_partitioner: GraphPartitionerBase = SinglePartitioner(),
        output_root: str = DEFAULT_OUTPUT_ROOT,
        output_worker: Optional[str] = None,
        merging_options: MergingOptions | None = None,
        # --- Bridge params ---
        bridge_min_similarity: float = 0.0,
        bridge_top_k: int = 10,
        bridge_min_component_size: int = 3,
        # --- Verified-viewgraph pipeline: verified-graph partition + post-merge retriangulation ---
        use_verified_pipeline: bool = False,
        # Attach global-track-id sidecars so merges anchor on identity-matched correspondences (opt-in
        # experiment; off = the R3-baseline legacy merge, which produced the most complete ToL structure).
        enable_gid_merge_anchoring: bool = False,
        # Post-merge retriangulation + BA (structure refinement). Off while re-establishing the
        # structure-complete baseline — fewer moving parts; flip back on afterwards.
        run_post_merge_retriangulation: bool = True,
        # Post-root-merge boundary recovery (offline-validated): triangulate global boundary tracks
        # against the merged cameras, RANSAC-DLT resect cameras the merge never placed against the
        # cluster-3D ∪ fresh-3D union, iterate, then one BA. Requires the gid sidecar
        # (enable_gid_merge_anchoring) to read the merged scene's 3D by global identity.
        enable_boundary_recovery: bool = False,
        # Export-time low-parallax track filter (0 = off). Tracks whose max pairwise triangulation
        # angle is below this are depth-unconstrained: they pass the reprojection filters (reproj
        # error is blind to depth error along the ray) yet scatter as fuzz/spray around the
        # structure. ToL 120/0.15 audit: 16% of merged tracks sat below 1.5deg (COLMAP's default
        # cutoff). Output-side only — merges, poses, and BA never see it.
        min_triangulation_angle_deg: float = 0.0,
        # Emit results/final_reconstruction: the post-retri scene, angle-filtered (per
        # min_triangulation_angle_deg), tracks COLORIZED from the source images, with the annex
        # cameras included as posed-only frustums — one drag-and-drop folder for viz/analysis,
        # no local post-processing. Export-side only; never touches the solve.
        export_final_reconstruction: bool = True,
        # Path to a precomputed-frontend npz (scripts/convert_wilsonkl_frontend.py). When set, the
        # global frontend (SIFT/matching/two-view) is skipped and these keypoints + verified
        # correspondences are used instead — the standard 1DSfM protocol, where all published
        # methods consume the benchmark's released view graph and tracks.
        precomputed_frontend_path: Optional[str] = None,
        # Path to a pair-list file (e.g. a 1DSfM EGs.txt; any lines starting with two integer image
        # ids). Skips retrieval and runs OUR full frontend (SIFT/matching/two-view) on exactly these
        # pairs — the accuracy of an own-frontend run at a fraction of the matching cost, since
        # every pair is known-productive. Ignored when precomputed_frontend_path is set.
        precomputed_pairs_path: Optional[str] = None,
    ) -> None:
        self.loader = loader
        self.image_pairs_generator = image_pairs_generator
        self.graph_partitioner = graph_partitioner
        self.cluster_optimizer = cluster_optimizer
        self._merging_options = merging_options or MergingOptions()
        self._bridge_min_similarity = bridge_min_similarity
        self._bridge_top_k = bridge_top_k
        self._bridge_min_component_size = bridge_min_component_size
        self._use_verified_pipeline = use_verified_pipeline
        self._enable_gid_merge_anchoring = enable_gid_merge_anchoring
        self._run_post_merge_retriangulation = run_post_merge_retriangulation
        self._enable_boundary_recovery = enable_boundary_recovery
        self._min_triangulation_angle_deg = min_triangulation_angle_deg
        self._export_final_reconstruction = export_final_reconstruction
        self._precomputed_frontend_path = precomputed_frontend_path
        self._precomputed_pairs_path = precomputed_pairs_path
        # Propagate metric_constructed_only to the cluster optimizer if it supports it.
        if hasattr(self.cluster_optimizer, "_metric_constructed_only"):
            setattr(self.cluster_optimizer, "_metric_constructed_only", self._merging_options.metric_constructed_only)
        elif hasattr(self.cluster_optimizer, "_optimizer") and hasattr(
            getattr(self.cluster_optimizer, "_optimizer"), "_metric_constructed_only"
        ):
            setattr(
                self.cluster_optimizer._optimizer,
                "_metric_constructed_only",
                self._merging_options.metric_constructed_only,
            )
        self._config_snapshot = None
        self.output_root = Path(output_root)
        if output_worker is not None:
            self.cluster_optimizer._output_worker = output_worker
        logger.info(f"Results, plots, and metrics will be saved at {self.output_root}")

    def __repr__(self) -> str:
        """Returns string representation of class."""
        return f"""
        {self.image_pairs_generator}
        {self.graph_partitioner}
        {self.cluster_optimizer}
        """

    def _schedule_single_cluster(self, context: ClusterContext) -> ClusterExecutionHandles:
        """Schedule the optimizer for a single cluster and return futures tracking its execution."""
        if len(context.visibility_graph) == 0:
            logger.warning("Skipping cluster %s as it has no edges.", context.label)
            return _empty_cluster_handles(context, 0)

        logger.info(
            "Creating computation graph for cluster %s with %d visibility edges.",
            context.label,
            len(context.visibility_graph),
        )

        computation = self.cluster_optimizer.create_computation_graph(
            context=context,
        )
        if computation is None or computation.sfm_result is None:
            logger.warning("Cluster optimizer produced no result for cluster %s.", context.label)
            return _empty_cluster_handles(context, len(context.visibility_graph))

        io_graph = delayed(_finalize_io_tasks, pure=False)(*computation.io_tasks)
        metrics_graph = delayed(_collect_metric_results, pure=False)(*computation.metric_tasks)
        annotated_reconstruction = delayed(cluster_merging.annotate_scene_with_metadata, pure=False)(
            computation.sfm_result,
            context.output_paths.plots,
            context.label,
            measurement_gid_index=context.measurement_gid_index,
        )

        io_future: Future = context.client.compute(io_graph)  # type: ignore
        metrics_future: Future = context.client.compute(metrics_graph)  # type: ignore
        reconstruction_future: Future = context.client.compute(annotated_reconstruction)  # type: ignore

        return ClusterExecutionHandles(
            reconstruction=reconstruction_future,
            metrics=metrics_future,
            io_barrier=io_future,
            output_paths=context.output_paths,
            cluster_path=context.cluster_path,
            label=context.label,
            edge_count=len(context.visibility_graph),
        )

    def run(self, client: Client) -> None:
        """Run the SceneOptimizer."""
        start_time = time.time()
        base_metrics_groups = []

        # Process Graph Generation: Visualize the process graph, which is a flow of data across GTSFM's modules.
        process_graph_generator = ProcessGraphGenerator()
        base_output_paths = prepare_output_paths(self.output_root, None)
        config_snapshot = self._config_snapshot
        if config_snapshot is not None:
            config_path = base_output_paths.results / "config.yaml"
            OmegaConf.save(config=config_snapshot, f=str(config_path))
            logger.info("📦 Saved final config snapshot to %s", config_path)
        process_graph_generator.save_graph(str(base_output_paths.plots / "process_graph_output.svg"))

        logger.info("🔥 GTSFM: Running image pair retrieval...")
        retriever_metrics, visibility_graph, similarity_matrix = self._run_retriever(client, base_output_paths)
        base_metrics_groups.append(retriever_metrics)
        image_future_map = self.loader.get_image_futures(client)
        one_view_data_dict = self.loader.get_one_view_data_dict()

        # Optional verified-viewgraph pipeline: globally verify the retrieval graph, then
        # (a) partition on the verified subgraph and (b) keep global 2D tracks for post-merge retriangulation.
        global_tracks_2d: Optional[list] = None
        global_refined_intrinsics: Optional[dict] = None
        global_v_corr_idxs_dict: Optional[dict] = None
        global_keypoints: Optional[list] = None
        gid_index_arrays: Optional[tuple] = None  # packed (keys, gids, cams) from the global 2D tracks
        if self._use_verified_pipeline:
            from gtsfm.cluster_optimizer.cluster_mvo import ClusterMVO, _pad_keypoints_list
            from gtsfm.two_view_estimator import create_v_corr_idxs_futures
            from gtsfm.multi_view_optimizer import get_2d_tracks
            from gtsfm.products.visibility_graph import visibility_graph_keys
            from gtsfm.utils.graph import get_nodes_in_largest_connected_component

            logger.info("🔎 GTSFM: Global two-view verification over %d retrieval edges...", len(visibility_graph))
            num_images = len(self.loader)
            retrieval_edge_count = len(visibility_graph)
            # Pass image futures directly; Dask materializes them as a dependency (no nested gather).
            image_futures = [image_future_map[idx] for idx in range(num_images)]

            # Reuse the per-cluster frontend chain over the FULL retrieval graph (populates per-pair
            # caches, so subsequent per-cluster frontends cache-hit). _run_two_view_estimation already
            # filters to result.valid().
            corr_gen = self.cluster_optimizer.correspondence_generator
            if self._precomputed_frontend_path:
                # Benchmark-released frontend (e.g. 1DSfM EGs/coords/tracks via
                # scripts/convert_wilsonkl_frontend.py): the exact two-view inputs the published
                # table rows consumed. Skips SIFT + matching + two-view entirely. NOTE: keypoints
                # must be in the SAME loader frame (max_resolution) as this run's config.
                logger.info(
                    "📦 Loading precomputed frontend from %s (SIFT/matching/two-view skipped).",
                    self._precomputed_frontend_path,
                )
                _expected_dims = {
                    int(_i): (2.0 * _ovd.intrinsics.px(), 2.0 * _ovd.intrinsics.py())
                    for _i, _ovd in one_view_data_dict.items()
                    if _ovd.intrinsics is not None
                }
                padded_keypoints_list, v_corr_idxs_dict = _load_precomputed_frontend(
                    self._precomputed_frontend_path,
                    num_images,
                    expected_dims=_expected_dims,
                    max_resolution=getattr(self.loader, "_max_resolution", None),
                )
            elif getattr(corr_gen, "produces_verified_correspondences", False):
                # COLMAP-DB frontend: read the already-verified keypoints + matches straight from the
                # database in the main process. No Dask two-view estimation and — crucially — no
                # client.gather of all per-edge results, the step that OOM-killed workers at scale.
                logger.info("🗄️  Reading verified correspondences from the COLMAP database (Dask frontend skipped).")
                keypoints_list, v_corr_idxs_dict = corr_gen.generate_correspondences(
                    client, image_futures, list(visibility_graph)
                )
                padded_keypoints_list = _pad_keypoints_list(keypoints_list, num_images)
            else:
                # Global two-view over the retrieval graph, returning only v_corr_idxs (each heavy
                # per-pair TwoViewResult is dropped the instant it is reduced). run_2view runs
                # identically either way (TwoViewEstimatorCacher / DB writes unaffected). Dispatch on
                # worker count:
                relative_pose_priors = self.loader.get_relative_pose_priors(visibility_graph) or {}
                gt_scene_mesh = self.loader.get_gt_scene_trimesh()
                try:
                    num_workers = len(client.scheduler_info()["workers"])
                except Exception:
                    num_workers = 1

                if num_workers <= 1:
                    # Single worker (or unknown): run INLINE in the main process. With no worker to
                    # lose there is zero scheduler<->worker comm-failure surface — the arrangement that
                    # otherwise died with "lost dependencies" over multi-hour runs when one monolithic
                    # task on one worker was dropped.
                    logger.info("🔵 [frontend] 1 worker → running the global frontend inline (in-process).")
                    images = client.gather(image_futures)
                    keypoints_list, putative_corr_idxs_dict, _ = ClusterMVO._run_correspondence_generator(
                        self.cluster_optimizer.correspondence_generator, list(visibility_graph), images
                    )
                    padded_keypoints_list = _pad_keypoints_list(keypoints_list, num_images)
                    v_corr_idxs_dict, _ = ClusterMVO._run_two_view_v_corr_idxs(
                        self.cluster_optimizer.two_view_estimator,
                        padded_keypoints_list,
                        putative_corr_idxs_dict,
                        relative_pose_priors,
                        gt_scene_mesh,
                        one_view_data_dict,
                    )
                else:
                    # Multiple workers: fan the frontend out across the pool. Correspondence generation
                    # (per-image detection + per-pair matching) and two-view (chunked) both run in
                    # parallel; only lean v_corr_idxs is gathered. Chunking (many small tasks, not one
                    # monolithic task) means a dropped worker recomputes only its chunk, not the whole
                    # multi-hour frontend.
                    logger.info("🔵 [frontend] %d workers → running the global frontend in parallel.", num_workers)
                    keypoints_list, putative_corr_idxs_dict = corr_gen.generate_correspondences(
                        client, image_futures, list(visibility_graph)
                    )
                    padded_keypoints_list = _pad_keypoints_list(keypoints_list, num_images)
                    v_corr_idxs_dict = create_v_corr_idxs_futures(
                        client,
                        self.cluster_optimizer.two_view_estimator,
                        padded_keypoints_list,
                        putative_corr_idxs_dict,
                        relative_pose_priors,
                        gt_scene_mesh,
                        one_view_data_dict,
                    )

            verified_graph = sorted(v_corr_idxs_dict.keys())
            all_nodes = visibility_graph_keys(verified_graph)
            largest_cc = set(get_nodes_in_largest_connected_component(verified_graph)) if verified_graph else set()
            logger.info(
                "🔎 Verified graph: %d/%d edges; nodes=%d, largest_cc=%d, dropped_by_partition=%d",
                len(verified_graph),
                retrieval_edge_count,
                len(all_nodes),
                len(largest_cc),
                len(all_nodes) - len(largest_cc),
            )
            visibility_graph = verified_graph

            # Build global 2D tracks (verified correspondences -> union-find tracks) for post-merge retriangulation.
            global_tracks_2d = get_2d_tracks(v_corr_idxs_dict, padded_keypoints_list)
            logger.info("🔎 Built %d global 2D tracks from verified correspondences.", len(global_tracks_2d))

            # Persist the global tracks for OFFLINE merge/BA replay (~25MB npz). The COLMAP text exports
            # are complete per-node BA problems, but global track IDENTITY (which cluster tracks are the
            # same physical point) only exists here — dumping it enables desk-side experiments (merge
            # replays, gid-correspondence debugging, cluster-Sim3-averaging prototypes) without PACE runs.
            try:
                _cams, _us, _vs, _tids = [], [], [], []
                for _tid, _tr in enumerate(global_tracks_2d):
                    for _m in _tr.measurements:
                        _cams.append(_m.i)
                        _us.append(float(_m.uv[0]))
                        _vs.append(float(_m.uv[1]))
                        _tids.append(_tid)
                np.savez_compressed(
                    base_output_paths.results / "global_tracks_2d.npz",
                    cam=np.asarray(_cams, dtype=np.int32),
                    u=np.asarray(_us, dtype=np.float32),
                    v=np.asarray(_vs, dtype=np.float32),
                    track_id=np.asarray(_tids, dtype=np.int32),
                )
                logger.info("💾 Saved global tracks for offline replay -> %s", base_output_paths.results / "global_tracks_2d.npz")
                del _cams, _us, _vs, _tids
            except Exception as _exc:
                logger.warning("Failed to save global_tracks_2d.npz (offline replay dump): %s", _exc)

            # Packed (camera, pixel) -> global-track-id arrays. Sliced per cluster below (mirroring the
            # per-node context packaging — no global broadcast) so merges can match tracks by GLOBAL
            # IDENTITY: same physical point triangulated by two clusters from disjoint cameras.
            # Opt-in: without the sidecars the merge runs the legacy (R3-baseline) path.
            if self._enable_gid_merge_anchoring:
                gid_index_arrays = cluster_merging.build_measurement_gid_arrays(global_tracks_2d)
                logger.info(
                    "🔗 Built global-track-id index: %d measurement keys over %d tracks (%.1f MB packed).",
                    len(gid_index_arrays[0]),
                    len(global_tracks_2d),
                    sum(a.nbytes for a in gid_index_arrays) / 1e6,
                )

            # Reuse the global verified correspondences + keypoints per cluster (skip the per-cluster
            # frontend). Plumbed via ClusterContext; clusters subset by their edges and build tracks_2d
            # eagerly in the main process (no scatter, no per-cluster two-view re-estimation).
            if getattr(self.cluster_optimizer, "reuses_global_correspondences", False):
                global_v_corr_idxs_dict = v_corr_idxs_dict
                global_keypoints = padded_keypoints_list

            # Global Fetzer calibration: estimate every camera's focal ONCE over the full verified view
            # graph (heuristic init, never VGGT focals), supplied to clusters via ClusterContext. Replaces
            # the per-cluster calibration, which falls back to VGGT focals for sparsely-connected cameras.
            if getattr(self.cluster_optimizer, "uses_global_view_graph_calibration", False):
                if getattr(self.cluster_optimizer, "calibration_source", "fetzer") == "exif":
                    # EXIF passthrough: hand the loader/EXIF intrinsics to every cluster verbatim.
                    # ToL gold audit: EXIF 1.91% median focal error (unbiased) vs scipy Fetzer 5.26%
                    # (−4% bias) — and the 1DSfM reference pipeline itself calibrated from EXIF.
                    # Same frozen-K coherence as Fetzer (bit-identical per camera across clusters).
                    #
                    # EXIF-LESS cameras are SKIPPED (not served): the loader substitutes the
                    # 1.2 x maxdim heuristic for missing EXIF, and passing that through would PIN a
                    # frequently-wrong focal at sigma=5px (measured: ~19-40% of 1DSfM cameras lack
                    # EXIF; the heuristic is ~17% off on average). Cameras absent from this dict
                    # fall back in the cluster build to the geometry model's predicted focal
                    # (VGGT-Omega: ~1-2% median error) — trust EXIF where it exists, the learned
                    # prior where it doesn't. NOTE: calibration content is not part of the cluster
                    # cache keys — apply to fresh scenes or clear the cluster cache when toggling.
                    _heur_factor = float(getattr(self.loader, "_default_focal_length_factor", 1.2))

                    def _is_heuristic_cal(cal) -> bool:
                        expected = _heur_factor * max(2.0 * cal.px(), 2.0 * cal.py())
                        return expected > 0 and abs(cal.fx() - expected) / expected < 1e-6

                    global_refined_intrinsics = {
                        idx: ovd.intrinsics
                        for idx, ovd in one_view_data_dict.items()
                        if ovd.intrinsics is not None and not _is_heuristic_cal(ovd.intrinsics)
                    }
                    n_total = sum(1 for ovd in one_view_data_dict.values() if ovd.intrinsics is not None)
                    logger.info(
                        "🔭 Global calibration: EXIF passthrough for %d cameras; %d EXIF-less camera(s) "
                        "fall back to the model-predicted focals (Fetzer skipped).",
                        len(global_refined_intrinsics),
                        n_total - len(global_refined_intrinsics),
                    )
                else:
                    from gtsfm.view_graph_estimator.view_graph_calibration import (
                        compute_global_view_graph_intrinsics,
                    )

                    global_refined_intrinsics = client.gather(
                        client.compute(
                            delayed(compute_global_view_graph_intrinsics)(
                                v_corr_idxs_dict, padded_keypoints_list, one_view_data_dict
                            )
                        )
                    )
                    logger.info(
                        "🔭 Global Fetzer calibration: refined %d focals over the full verified view graph.",
                        len(global_refined_intrinsics),
                    )

        # Bridge reconnection: add cross-component edges to reconnect island components.
        if similarity_matrix is not None and self._bridge_min_similarity > 0:
            from gtsfm.utils.viewgraph_reconnector import reconnect_visibility_graph

            bridge_result = reconnect_visibility_graph(
                visibility_graph=visibility_graph,
                similarity_matrix=similarity_matrix,
                min_bridge_similarity=self._bridge_min_similarity,
                top_k_per_component=self._bridge_top_k,
                min_component_size=self._bridge_min_component_size,
            )
            if bridge_result.bridge_edges:
                logger.info(
                    "🌉 Bridge reconnection: added %d edges, components %d -> %d " "(reconnected %d, unreachable %d)",
                    len(bridge_result.bridge_edges),
                    bridge_result.num_components_before,
                    bridge_result.num_components_after,
                    bridge_result.components_reconnected,
                    bridge_result.components_unreachable,
                )
                visibility_graph = bridge_result.reconnected_graph
            del similarity_matrix

        # Graph partitioning: Divide the visibility graph into clusters (runs eagerly, no delayed/futures).
        logger.info("🔥 GTSFM: Partitioning the view graph...")
        assert self.graph_partitioner is not None, "Graph partitioner is not set up!"
        cluster_tree = self.graph_partitioner.run(visibility_graph)
        self.graph_partitioner.log_partition_details(cluster_tree, base_output_paths)

        # --- Offline-replay telemetry (all small; enables desk-side merge/BA/prior experiments on a
        # downloaded results folder without re-running the pipeline) ---
        try:
            import pickle as _pickle

            with open(base_output_paths.results / "cluster_tree.pkl", "wb") as _f:
                _pickle.dump(cluster_tree, _f)  # exact tree structure (which node merges into which)
        except Exception as _exc:
            logger.warning("Failed to save cluster_tree.pkl: %s", _exc)
        try:
            if self._use_verified_pipeline and v_corr_idxs_dict:
                _i = np.array([e[0] for e in v_corr_idxs_dict], dtype=np.int32)
                _j = np.array([e[1] for e in v_corr_idxs_dict], dtype=np.int32)
                _n = np.array([len(v) for v in v_corr_idxs_dict.values()], dtype=np.int32)
                np.savez_compressed(
                    base_output_paths.results / "verified_graph.npz", i=_i, j=_j, num_inliers=_n
                )  # the verified view graph: per-edge inlier counts (edge-addition / connectivity studies)
        except Exception as _exc:
            logger.warning("Failed to save verified_graph.npz: %s", _exc)
        try:
            import json as _json

            _cal = {}
            for _idx, _ovd in one_view_data_dict.items():
                _entry = {}
                if _ovd.intrinsics is not None:
                    _c = _ovd.intrinsics
                    _entry["initial"] = [_c.fx(), _c.k1(), _c.k2(), _c.px(), _c.py()]
                if global_refined_intrinsics and _idx in global_refined_intrinsics:
                    _c = global_refined_intrinsics[_idx]
                    _entry["fetzer"] = [_c.fx(), _c.k1(), _c.k2(), _c.px(), _c.py()]
                if _entry:
                    _cal[int(_idx)] = _entry
            with open(base_output_paths.results / "calibrations.json", "w") as _f:
                _json.dump(_cal, _f)  # loader-initial vs global-Fetzer focals (focal/prior experiments)
        except Exception as _exc:
            logger.warning("Failed to save calibrations.json: %s", _exc)
        save_retrieval_two_view_metrics(base_output_paths)

        logger.info("🔥 GTSFM: Scheduling cluster optimizations...")
        merged_scene: Optional[cluster_merging.MergedNodeSummary] = None

        with performance_report(filename="dask_reports/scene-optimizer.html"):
            if cluster_tree is None:
                logger.warning("No clusters generated by partitioner; skipping reconstruction and merge.")
            else:
                num_images = len(self.loader)

                def to_context(path: tuple[int, ...], visibility_graph: VisibilityGraph) -> ClusterContext:
                    output_paths = base_output_paths if len(path) == 0 else prepare_output_paths(self.output_root, path)
                    return ClusterContext(
                        client=client,
                        loader=self.loader,
                        num_images=num_images,
                        output_paths=output_paths,
                        image_future_map=image_future_map,
                        one_view_data_dict=one_view_data_dict,
                        cluster_path=path,
                        label=cluster_label(path),
                        visibility_graph=visibility_graph,
                        global_refined_intrinsics=global_refined_intrinsics,
                        global_v_corr_idxs_dict=global_v_corr_idxs_dict,
                        global_keypoints=global_keypoints,
                        # Per-cluster slice of the gid index (only this node's cameras) — lean per-node
                        # packaging, mirrors the context pattern; None disables ID-matching gracefully.
                        measurement_gid_index=(
                            cluster_merging.slice_gid_index(
                                *gid_index_arrays, {k for edge in visibility_graph for k in edge}
                            )
                            if gid_index_arrays is not None
                            else None
                        ),
                    )

                context_tree = cluster_tree.map_with_path(to_context)

                # Runs reconstruction on each node of the VisibilityGraph (with context) tree.
                # Returns handles to various outputs: reconstruction, metrics, io_barrier etc.
                handles_tree = context_tree.map(self._schedule_single_cluster)

                # Get the reconstruction handle and run merging to get a tree of merged result handles.
                reconstruction_tree = handles_tree.map(lambda handle: handle.reconstruction)

                cameras_gt = self.loader.get_gt_cameras()

                def merge_fn(
                    reconstruction: object, child_results: tuple[cluster_merging.MergedNodeResult, ...]
                ) -> cluster_merging.MergedNodeResult:
                    return cluster_merging.combine_results(
                        cast(Optional[GtsfmData], reconstruction),
                        child_results,
                        cameras_gt=cameras_gt,
                        options=self._merging_options,
                    )

                merged_future_tree = submit_tree_map_with_children(client, reconstruction_tree, merge_fn)
                export_tree = cluster_merging.schedule_exports(client, handles_tree, merged_future_tree)
                summary_tree = cluster_merging.schedule_summaries(client, merged_future_tree)
                root_merge_summary: Optional[cluster_merging.MergedNodeSummary] = None
                root_merged_result: Optional[cluster_merging.MergedNodeResult] = None
                for handle_node, summary_node, export_node, merged_node in zip(
                    PreOrderIter(handles_tree),
                    PreOrderIter(summary_tree),
                    PreOrderIter(export_tree),
                    PreOrderIter(merged_future_tree),
                ):
                    handle = handle_node.value
                    summary_future = summary_node.value
                    export_future = export_node.value

                    metrics_groups = list(handle.metrics.result())
                    handle.io_barrier.result()
                    export_future.result()
                    if handle.cluster_path == ():
                        merged_summary = summary_future.result()
                        base_metrics_groups.extend(metrics_groups)
                        base_metrics_groups.append(merged_summary.metrics)
                        base_metrics_groups.append(merged_summary.pre_ba_metrics)
                        root_merge_summary = merged_summary
                        if self._use_verified_pipeline:
                            # Materialize the full root reconstruction (idempotent; already computed for the summary).
                            root_merged_result = merged_node.value.result()
                    else:
                        merged_summary = summary_future.result()
                        metrics_groups.append(merged_summary.metrics)
                        metrics_groups.append(merged_summary.pre_ba_metrics)
                        save_metrics_reports(metrics_groups, str(handle.output_paths.metrics))
                if root_merge_summary is not None:
                    logger.info("🔥 GTSFM: Running cluster optimization and merging...")
                    merged_scene = root_merge_summary

                # Post-merge retriangulation (structure refinement): rebuild sparse structure from the
                # globally-verified tracks against the merged poses, then BA. Written as a separate output.
                retri_scene: Optional[GtsfmData] = None
                retri_dropped: dict = {}
                retri_root_frame_ids: set = set()
                if (
                    self._use_verified_pipeline
                    and self._run_post_merge_retriangulation
                    and global_tracks_2d
                    and root_merged_result is not None
                    and root_merged_result.scene is not None
                ):
                    logger.info(
                        "🔧 GTSFM: Post-merge retriangulation on %d global 2D tracks vs merged poses...",
                        len(global_tracks_2d),
                    )
                    refined: GtsfmData
                    refined, retri_dropped, retri_root_frame_ids = client.submit(
                        _run_post_merge_retriangulation,
                        root_merged_result.scene,
                        global_tracks_2d,
                        self._merging_options,
                        root_merged_result.trackless_cameras,
                        pure=False,
                    ).result()
                    # Reportable-pose policy (method-level): cameras beyond 15x the robust scene
                    # radius are not exported — in any tier.
                    refined, n_unreportable = _drop_unreportable_cameras(refined)
                    if n_unreportable:
                        logger.warning(
                            "🚷 Reportable-pose policy: dropped %d camera(s) beyond 15x the robust "
                            "scene radius from the exports.", n_unreportable,
                        )
                    retri_scene = refined
                    retri_dir = base_output_paths.results / "merged_retriangulated"
                    retri_dir.mkdir(parents=True, exist_ok=True)
                    refined.export_as_colmap_text(retri_dir)
                    logger.info(
                        "🔧 Retriangulated scene: %d images, %d tracks → %s",
                        refined.number_images(),
                        refined.number_tracks(),
                        retri_dir,
                    )
                    base_metrics_groups.append(
                        cluster_merging.compute_merging_metrics(
                            refined,
                            cameras_gt=cameras_gt,
                            metric_constructed_only=self._merging_options.metric_constructed_only,
                            suffix="_retriangulated",
                        )
                    )

                # Prior-backed ANNEX export: every camera a track filter dropped anywhere in the tree
                # (carried up outside the scenes) plus the retri stage's own drops, re-seated onto the
                # final scene and written as a posed-only model. Zero effect on the core solve — the
                # annex never entered a Sim3 seat, a duplicate resolution, or a BA.
                annex: dict = {}  # populated below; also consumed by the final_reconstruction export
                if (
                    self._use_verified_pipeline
                    and self._merging_options.export_trackless_annex
                    and root_merged_result is not None
                    and root_merged_result.scene is not None
                ):
                    from gtsfm.utils import align as align_utils
                    from gtsfm.utils.transform import camera_map_with_sim3

                    final_scene = retri_scene if retri_scene is not None else root_merged_result.scene
                    tree_annex = dict(root_merged_result.annex_cameras or {})
                    # Retri drops that never entered the final BA graph are still in the root-merge
                    # gauge — fold them into the tree annex (fresher poses win) so they ride the
                    # root->final re-seat below; BA'd-then-dropped cameras are already final-frame.
                    tree_annex.update({i: retri_dropped[i] for i in retri_root_frame_ids if i in retri_dropped})
                    final_frame_drops = {
                        i: c for i, c in retri_dropped.items() if i not in retri_root_frame_ids
                    }
                    if tree_annex and retri_scene is not None:
                        # The retri BA can drift the gauge relative to the root-merge frame the tree
                        # annex was captured in — absorb it with one Sim3 over the shared cameras.
                        try:
                            # Robust fit: the root scene can hold extreme-coordinate strays that
                            # poison a least-squares Align (RF: the whole annex displaced by one
                            # bad fSr).
                            fSr = align_utils.sim3_from_Pose3_maps_robust(
                                final_scene.poses(), root_merged_result.scene.poses()
                            )
                            tree_annex = camera_map_with_sim3(fSr, tree_annex)
                        except Exception as exc:
                            logger.warning(
                                "🎒 Annex: root→final re-seat failed (%s); exporting root-frame poses.", exc
                            )
                    annex.update(tree_annex)
                    annex.update(final_frame_drops)
                    for i in final_scene.get_valid_camera_indices():
                        annex.pop(i, None)
                    # Radius guard (last line of defense): a degenerate carry fit anywhere in the
                    # tree can launch an annex block to numerically absurd coordinates. Anything
                    # beyond 15x the core's robust radius is not a reportable pose — drop it.
                    core_centers = np.array(
                        [np.array(final_scene.get_camera(i).pose().translation())
                         for i in final_scene.get_valid_camera_indices()]
                    )
                    ctr = np.median(core_centers, axis=0)
                    radius = float(np.percentile(np.linalg.norm(core_centers - ctr, axis=1), 95)) or 1.0
                    far = [
                        i for i, c in annex.items()
                        if c is None or not np.all(np.isfinite(np.array(c.pose().translation())))
                        or float(np.linalg.norm(np.array(c.pose().translation()) - ctr)) > 15.0 * radius
                    ]
                    for i in far:
                        annex.pop(i, None)
                    if far:
                        logger.warning(
                            "🎒 Annex: dropped %d camera(s) beyond 15x the core radius "
                            "(degenerate carry fits upstream); %d remain.",
                            len(far),
                            len(annex),
                        )
                    if annex:
                        annex_scene = GtsfmData(number_images=final_scene.number_images())
                        annex_scene._image_info = final_scene._clone_image_info()
                        for i, cam in annex.items():
                            if cam is not None:
                                annex_scene.add_camera(i, cam)
                        annex_dir = base_output_paths.results / "annex_posed_only"
                        annex_dir.mkdir(parents=True, exist_ok=True)
                        annex_scene.export_as_colmap_text(annex_dir)
                        logger.info(
                            "🎒 Annex export: %d prior-backed camera pose(s) (zero tracks by design) → %s",
                            len(annex),
                            annex_dir,
                        )
                    else:
                        logger.info("🎒 Annex export enabled, but no camera was dropped anywhere in the tree.")

                # Post-root-merge boundary recovery: DLT-triangulate boundary global tracks against the
                # merged cameras, RANSAC-DLT resect unposed cameras against cluster-3D ∪ fresh-3D,
                # iterate, then one BA. Runs as a single dask task on a worker (like the retri stage).
                if (
                    self._use_verified_pipeline
                    and self._enable_boundary_recovery
                    and global_tracks_2d
                    and root_merged_result is not None
                    and root_merged_result.scene is not None
                ):
                    # Intrinsics for cameras NOT in the merged scene: global-Fetzer focals, falling
                    # back to the loader-initial intrinsics (same precedence as the offline replay).
                    boundary_intrinsics: dict[int, tuple] = {}
                    for _idx, _ovd in one_view_data_dict.items():
                        _cal = None
                        if global_refined_intrinsics and _idx in global_refined_intrinsics:
                            _cal = global_refined_intrinsics[_idx]
                        elif _ovd.intrinsics is not None:
                            _cal = _ovd.intrinsics
                        if _cal is not None:
                            boundary_intrinsics[int(_idx)] = _calibration_to_tuple(_cal)
                    logger.info(
                        "🧭 GTSFM: Boundary recovery on %d global 2D tracks vs the merged root scene...",
                        len(global_tracks_2d),
                    )
                    recovered_scene: GtsfmData = client.submit(
                        _run_boundary_recovery,
                        root_merged_result.scene,
                        global_tracks_2d,
                        self._merging_options,
                        boundary_intrinsics,
                        pure=False,
                    ).result()
                    recovery_dir = base_output_paths.results / "merged_boundary_recovered"
                    recovery_dir.mkdir(parents=True, exist_ok=True)
                    recovered_scene.export_as_colmap_text(recovery_dir)
                    logger.info(
                        "🧭 Boundary-recovered scene: %d images, %d cameras, %d tracks → %s",
                        recovered_scene.number_images(),
                        len(recovered_scene.get_valid_camera_indices()),
                        recovered_scene.number_tracks(),
                        recovery_dir,
                    )
                    base_metrics_groups.append(
                        cluster_merging.compute_merging_metrics(
                            recovered_scene,
                            cameras_gt=cameras_gt,
                            metric_constructed_only=self._merging_options.metric_constructed_only,
                            suffix="_boundary_recovered",
                        )
                    )

                # Export-time low-parallax cleanup: write an angle-filtered copy of the final merged
                # scene alongside merged/ (the unfiltered export stays for A/B). Poses/merges untouched.
                if (
                    self._use_verified_pipeline
                    and self._min_triangulation_angle_deg > 0
                    and root_merged_result is not None
                    and root_merged_result.scene is not None
                ):
                    angle_filtered = cluster_merging.filter_tracks_by_triangulation_angle(
                        root_merged_result.scene, self._min_triangulation_angle_deg
                    )
                    angle_dir = base_output_paths.results / "merged_anglefiltered"
                    angle_dir.mkdir(parents=True, exist_ok=True)
                    angle_filtered.export_as_colmap_text(angle_dir)
                    logger.info(
                        "🔭 Angle-filtered scene (>=%.1f deg): %d tracks → %s",
                        self._min_triangulation_angle_deg,
                        angle_filtered.number_tracks(),
                        angle_dir,
                    )
                    base_metrics_groups.append(
                        cluster_merging.compute_merging_metrics(
                            angle_filtered,
                            cameras_gt=cameras_gt,
                            metric_constructed_only=self._merging_options.metric_constructed_only,
                            suffix="_anglefiltered",
                        )
                    )

                # FINAL RECONSTRUCTION export: one drag-and-drop folder — the best scene (post-retri
                # when available), angle-filtered, tracks colorized from the source images, with the
                # prior-backed annex cameras riding along as posed-only frustums. Pure output; runs
                # last so it can never affect the solve or the other exports. Non-fatal on failure.
                if (
                    self._use_verified_pipeline
                    and self._export_final_reconstruction
                    and root_merged_result is not None
                    and root_merged_result.scene is not None
                ):
                    try:
                        best_scene = retri_scene if retri_scene is not None else root_merged_result.scene
                        if self._min_triangulation_angle_deg > 0:
                            best_scene = cluster_merging.filter_tracks_by_triangulation_angle(
                                best_scene, self._min_triangulation_angle_deg
                            )
                        # Fresh copy: annex cameras and colors must not leak into the scenes the
                        # metrics/exports above were computed from. Tracks are DEEP-copied —
                        # add_track stores references, and colorize mutates r/g/b in place.
                        from gtsam import SfmTrack as _SfmTrack

                        out_scene = GtsfmData(number_images=best_scene.number_images())
                        out_scene._image_info = best_scene._clone_image_info()
                        for _i in best_scene.get_valid_camera_indices():
                            out_scene.add_camera(_i, best_scene.get_camera(_i))
                        for _t in best_scene.tracks():
                            _tc = _SfmTrack(_t.point3())
                            for _k in range(_t.numberMeasurements()):
                                _mi, _uv = _t.measurement(_k)
                                _tc.addMeasurement(_mi, _uv)
                            out_scene.add_track(_tc)
                        n_annex_added = 0
                        for _i, _cam in annex.items():
                            if _cam is not None and out_scene.get_camera(_i) is None:
                                out_scene.add_camera(_i, _cam)
                                n_annex_added += 1
                        n_colored = _colorize_scene_tracks_in_place(out_scene, self.loader)
                        final_dir = base_output_paths.results / "final_reconstruction"
                        final_dir.mkdir(parents=True, exist_ok=True)
                        out_scene.export_as_colmap_text(final_dir)
                        logger.info(
                            "📦 Final reconstruction: %d cams (%d annex posed-only), %d tracks "
                            "(%d colorized, angle>=%.1f deg) → %s",
                            len(out_scene.get_valid_camera_indices()),
                            n_annex_added,
                            out_scene.number_tracks(),
                            n_colored,
                            self._min_triangulation_angle_deg,
                            final_dir,
                        )
                    except Exception as exc:
                        logger.warning("📦 Final reconstruction export failed (non-fatal): %s", exc)

        if merged_scene is not None and merged_scene.merge_success:
            logger.info(
                "Merged scene contains %d images and %d tracks.",
                merged_scene.num_images,
                merged_scene.num_tracks,
            )
        else:
            logger.warning("Merging failed, no final merged scene found.")

        # Log total time taken and save metrics report
        end_time = time.time()
        duration_sec = end_time - start_time
        logger.info(
            "🔥 GTSFM took %.1f %s to compute sparse multi-view result.",
            duration_sec / 60 if duration_sec >= 120 else duration_sec,
            "minutes" if duration_sec >= 120 else "seconds",
        )
        total_summary_metrics = GtsfmMetricsGroup(
            "total_summary_metrics", [GtsfmMetric("total_runtime_sec", duration_sec)]
        )
        base_metrics_groups.append(total_summary_metrics)

        save_metrics_reports(base_metrics_groups, str(base_output_paths.metrics))

    def _run_retriever(
        self, client: Client, output_paths: OutputPaths
    ) -> tuple[GtsfmMetricsGroup, VisibilityGraph, Optional[object]]:
        # Precomputed pair list (e.g. a 1DSfM EGs.txt): skip retrieval entirely and hand the
        # frontend exactly the benchmark's productive pairs — full-quality SIFT/matching/two-view
        # runs on only these. Any text file whose lines start with two integer image ids works.
        if self._precomputed_pairs_path:
            pairs = set()
            n_img = len(self.loader)
            with open(self._precomputed_pairs_path) as f:
                for line in f:
                    p = line.split()
                    if len(p) >= 2:
                        try:
                            a, b = int(p[0]), int(p[1])
                        except ValueError:
                            continue
                        if a != b and 0 <= a < n_img and 0 <= b < n_img:
                            pairs.add((min(a, b), max(a, b)))
            visibility_graph = sorted(pairs)
            logger.info(
                "📦 Precomputed pair list: %d pairs from %s (retrieval skipped).",
                len(visibility_graph), self._precomputed_pairs_path,
            )
            metrics = GtsfmMetricsGroup(
                "retriever_metrics", [GtsfmMetric("num_retrieved_pairs", len(visibility_graph))]
            )
            return metrics, visibility_graph, None

        # TODO(Frank): refactor to move more of this logic into ImagePairsGenerator
        retriever_start_time = time.time()
        batch_size = self.image_pairs_generator._batch_size

        transforms = self.image_pairs_generator.get_preprocessing_transforms()

        # Image_Batch_Futures is a list of Stacked Tensors with dimension (batch_size, Channels, H, W)
        image_batch_futures = self.loader.get_all_descriptor_image_batches_as_futures(client, batch_size, *transforms)

        image_fnames = self.loader.image_filenames()

        plots_output_dir = output_paths.plots
        with performance_report(filename="dask_reports/retriever.html"):
            visibility_graph = self.image_pairs_generator.run(
                client=client,
                image_batch_futures=image_batch_futures,
                image_fnames=image_fnames,
                plots_output_dir=plots_output_dir,
            )

        retriever = self.image_pairs_generator._retriever

        # Grab the similarity matrix BEFORE save_diagnostics clears it (sets to None).
        similarity_matrix = getattr(retriever, "_latest_similarity_matrix", None)
        if similarity_matrix is None:
            # Handle JointSimilaritySequentialRetriever wrapper.
            inner = getattr(retriever, "_similarity_retriever", None)
            if inner is not None:
                similarity_matrix = getattr(inner, "_latest_similarity_matrix", None)

        try:
            retriever.save_diagnostics(
                image_fnames=image_fnames,
                pairs=visibility_graph,
                plots_output_dir=plots_output_dir,
            )
        except Exception as exc:  # pragma: no cover - diagnostic path best-effort
            logger.warning("Failed to persist retriever diagnostics: %s", exc)

        retriever_metrics = self.image_pairs_generator._retriever.evaluate(len(self.loader), visibility_graph)
        retriever_duration_sec = time.time() - retriever_start_time
        retriever_metrics.add_metric(GtsfmMetric("retriever_duration_sec", retriever_duration_sec))
        logger.info("🚀 Image pair retrieval took %.2f min.", retriever_duration_sec / 60.0)

        return retriever_metrics, visibility_graph, similarity_matrix
