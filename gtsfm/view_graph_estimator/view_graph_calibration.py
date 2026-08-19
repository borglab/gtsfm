"""View graph calibration: estimate camera focal lengths from fundamental matrices.

Joint optimization of all focal lengths using Fetzer et al. (WACV 2020) residuals.
For each F-matrix edge, measures how close E = K2^T F K1 is to a valid essential matrix
(valid E has singular values σ1 = σ2, σ3 = 0).

Reference: Fetzer et al., "Stable Intrinsic Auto-Calibration from Fundamental Matrices
of Devices with Uncorrelated Camera Parameters", WACV 2020.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import gtsam
import numpy as np
from gtsam import Cal3Bundler, Cal3_S2, Cal3DS2
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

try:
    import poselib  # robust F + H estimation for the Fetzer gates

    _HAS_POSELIB = True
except ImportError:  # pragma: no cover
    _HAS_POSELIB = False

# Custom gtsam builds (borglab/gtsfm#1115) provide a closed-form SelfCalibrationFactor; standard
# wheels do not. The solver uses it when present and falls back to scipy otherwise — the ToL gold
# A/B showed the GATES carry the improvement (13.25% -> 5.32% median focal error on identical
# inputs), with the solver choice secondary.
# The closed-form focal factor is named SelfCalibrationFactor in current gtsam-develop and
# FetzerFactor in pre-rename builds; older builds also store the focal variables as Vector1
# instead of double. _solve_focals handles both names and both storage conventions.
_SELFCAL_FACTOR_CLS = getattr(gtsam, "SelfCalibrationFactor", None) or getattr(gtsam, "FetzerFactor", None)
_HAS_SELFCAL_FACTOR = _SELFCAL_FACTOR_CLS is not None
# COLMAP EstimateTwoViewGeometry model-selection threshold: PLANAR if H explains almost as many
# correspondences as F. Planar/panoramic pairs drive the closed-form Fetzer residual to its poles.
_MAX_H_INLIER_RATIO = 0.8

import gtsfm.common.types as gtsfm_types
from gtsfm.common.keypoints import Keypoints
from gtsfm.utils.logger import get_logger

logger = get_logger()


def estimate_fundamental_from_correspondences(
    coords_i1: np.ndarray,
    coords_i2: np.ndarray,
) -> Optional[np.ndarray]:
    """Estimate F-matrix from verified correspondences using 8-point algorithm.

    Args:
        coords_i1: (N, 2) pixel coordinates in image 1.
        coords_i2: (N, 2) pixel coordinates in image 2.

    Returns:
        3x3 fundamental matrix, or None if estimation fails.
    """
    if len(coords_i1) < 8:
        return None
    F, mask = cv2.findFundamentalMat(coords_i1, coords_i2, method=cv2.FM_8POINT)
    if F is None or F.shape != (3, 3):
        return None
    return F


def f_passes_focal_sanity(
    F: np.ndarray,
    pp1: np.ndarray,
    pp2: np.ndarray,
    f_init: float,
    lo: float = 0.4,
    hi: float = 2.5,
    max_resid: float = 0.01,
) -> bool:
    """True iff this F implies a well-defined focal: sweeping a shared focal over [lo,hi]*f_init,
    the essential-ness residual ((s2/s1-1)^2 + (s3/s1)^2 of K2^T F K1) must reach an INTERIOR
    minimum below max_resid. Near-planar/degenerate F's rail to the band edge or never converge —
    they poison the joint Fetzer solve (ToL: ungated -12.35% focal bias) while remaining fine for
    pose estimation, so this gate only guards calibration."""
    focals = np.linspace(lo * f_init, hi * f_init, 60)
    residuals = np.empty(len(focals))
    for i, f in enumerate(focals):
        K1 = np.array([[f, 0.0, pp1[0]], [0.0, f, pp1[1]], [0.0, 0.0, 1.0]])
        K2 = np.array([[f, 0.0, pp2[0]], [0.0, f, pp2[1]], [0.0, 0.0, 1.0]])
        s = np.linalg.svd(K2.T @ F @ K1, compute_uv=False)
        residuals[i] = (s[1] / s[0] - 1.0) ** 2 + (s[2] / s[0]) ** 2
    best = int(np.argmin(residuals))
    at_band_edge = best <= 1 or best >= len(focals) - 2
    return (not at_band_edge) and bool(residuals[best] <= max_resid)


def _estimate_fundamental_robust(
    coords_i1: np.ndarray, coords_i2: np.ndarray, threshold_px: float = 2.0
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    """Robust (LO-RANSAC) F via PoseLib + planar-config gate; falls back to the 8-point estimate
    when poselib is unavailable. Returns (F, drop_reason) with drop_reason in
    {None, 'planar', 'f_fail'}; F is None whenever drop_reason is not None."""
    if not _HAS_POSELIB:
        F = estimate_fundamental_from_correspondences(coords_i1, coords_i2)
        return (F, None) if F is not None else (None, "f_fail")
    try:
        F, f_info = poselib.estimate_fundamental(
            coords_i1, coords_i2, {"max_epipolar_error": threshold_px}, {}
        )
    except Exception:
        return None, "f_fail"
    F = np.asarray(F, dtype=np.float64)
    if F.shape != (3, 3) or not np.all(np.isfinite(F)) or np.allclose(F, 0.0):
        return None, "f_fail"
    n_F = int(f_info.get("num_inliers", 0))
    try:
        _, h_info = poselib.estimate_homography(
            coords_i1, coords_i2, {"max_reproj_error": threshold_px}, {}
        )
        n_H = int(h_info.get("num_inliers", 0))
    except Exception:
        n_H = 0
    if n_H > _MAX_H_INLIER_RATIO * max(n_F, 1):
        return None, "planar"
    return F, None


def _solve_focals(edges, cam_idx_to_var_idx, precomputed, initial_focals, jac_sparsity):
    """Joint focal solve: gtsam SelfCalibrationFactor graph (Cauchy 0.01, GLOMAP-matched) when the
    custom build provides it, else scipy least_squares with Cauchy loss. Returns (focals, info)."""
    if _HAS_SELFCAL_FACTOR:
        sorted_cams = sorted(cam_idx_to_var_idx, key=cam_idx_to_var_idx.get)

        def _gtsam_solve(as_vector: bool):
            graph = gtsam.NonlinearFactorGraph()
            edge_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Cauchy.Create(0.01),
                gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
            )
            for cam1, cam2, F, pp1, pp2 in edges:
                graph.add(
                    _SELFCAL_FACTOR_CLS(
                        gtsam.symbol("f", cam1), gtsam.symbol("f", cam2), F, pp1, pp2, edge_noise
                    )
                )
            values = gtsam.Values()
            for var_idx, cam in enumerate(sorted_cams):
                f0 = float(initial_focals[var_idx])
                values.insert(gtsam.symbol("f", cam), np.array([f0]) if as_vector else f0)
            params = gtsam.LevenbergMarquardtParams()
            params.setMaxIterations(200)
            result = gtsam.LevenbergMarquardtOptimizer(graph, values, params).optimize()
            if as_vector:
                focals = np.array([float(result.atVector(gtsam.symbol("f", c))[0]) for c in sorted_cams])
            else:
                focals = np.array([result.atDouble(gtsam.symbol("f", c)) for c in sorted_cams])
            return focals, graph.error(values), graph.error(result)

        # Current builds store the focal variable as double; pre-rename FetzerFactor builds expect
        # Vector1 (optimize() throws a GenericValue<double>-vs-Matrix RuntimeError). Try both.
        for as_vector in (False, True):
            try:
                focals, e0, e1 = _gtsam_solve(as_vector)
                return focals, (
                    f"gtsam {_SELFCAL_FACTOR_CLS.__name__} LM"
                    f"{' (Vector1 storage)' if as_vector else ''}, error {e0:.4f} -> {e1:.4f}"
                )
            except Exception as exc:
                last_exc = exc
        logger.warning("%s solve failed (%s); falling back to scipy.", _SELFCAL_FACTOR_CLS.__name__, last_exc)
    result = least_squares(
        _fetzer_residuals,
        x0=initial_focals,
        args=(edges, cam_idx_to_var_idx, precomputed),
        jac_sparsity=jac_sparsity,
        loss="cauchy",
        f_scale=0.1,
        bounds=(100.0, np.inf),
        max_nfev=200,
    )
    return result.x, f"scipy {result.message} in {result.nfev} evaluations, cost {result.cost:.4f}"


def _fetzer_residuals(
    focal_lengths: np.ndarray,
    edges: List[Tuple[int, int, np.ndarray, np.ndarray, np.ndarray]],
    cam_idx_to_var_idx: Dict[int, int],
    precomputed: Optional[Dict] = None,
) -> np.ndarray:
    """Compute Fetzer residuals for all edges (vectorized).

    For each edge, constructs E = K2^T F K1 from current focal lengths and measures
    deviation from a valid essential matrix via singular value ratios.

    Args:
        focal_lengths: Current focal length estimates, one per optimized camera.
        edges: List of (cam_idx1, cam_idx2, F, pp1, pp2) tuples.
        cam_idx_to_var_idx: Maps camera index to position in focal_lengths array.
        precomputed: Optional dict with precomputed index arrays for vectorization.

    Returns:
        Stacked residuals, 2 per edge.
    """
    n = len(edges)
    if precomputed is None:
        # Fallback to loop (used for final residual evaluation).
        residuals = np.zeros(2 * n)
        for i, (cam1, cam2, F, pp1, pp2) in enumerate(edges):
            f1 = focal_lengths[cam_idx_to_var_idx[cam1]]
            f2 = focal_lengths[cam_idx_to_var_idx[cam2]]
            K1 = np.array([[f1, 0, pp1[0]], [0, f1, pp1[1]], [0, 0, 1.0]])
            K2 = np.array([[f2, 0, pp2[0]], [0, f2, pp2[1]], [0, 0, 1.0]])
            E = K2.T @ F @ K1
            _, s, _ = np.linalg.svd(E)
            if s[0] > 1e-12:
                residuals[2 * i] = s[1] / s[0] - 1.0
                residuals[2 * i + 1] = s[2] / s[0]
        return residuals

    # Vectorized path using precomputed arrays.
    idx1 = precomputed["idx1"]
    idx2 = precomputed["idx2"]
    F_stack = precomputed["F_stack"]  # (n, 3, 3)
    pp1_stack = precomputed["pp1_stack"]  # (n, 2)
    pp2_stack = precomputed["pp2_stack"]  # (n, 2)

    f1 = focal_lengths[idx1]  # (n,)
    f2 = focal_lengths[idx2]  # (n,)

    # Build K1, K2 as batch: K = [[f, 0, px], [0, f, py], [0, 0, 1]]
    K1 = np.zeros((n, 3, 3))
    K1[:, 0, 0] = f1
    K1[:, 1, 1] = f1
    K1[:, 0, 2] = pp1_stack[:, 0]
    K1[:, 1, 2] = pp1_stack[:, 1]
    K1[:, 2, 2] = 1.0

    K2 = np.zeros((n, 3, 3))
    K2[:, 0, 0] = f2
    K2[:, 1, 1] = f2
    K2[:, 0, 2] = pp2_stack[:, 0]
    K2[:, 1, 2] = pp2_stack[:, 1]
    K2[:, 2, 2] = 1.0

    # E = K2^T @ F @ K1, batched.
    K2T = np.transpose(K2, (0, 2, 1))
    E = K2T @ F_stack @ K1  # (n, 3, 3)

    # Batch SVD.
    _, S, _ = np.linalg.svd(E)  # S shape (n, 3)

    residuals = np.zeros(2 * n)
    valid = S[:, 0] > 1e-12
    residuals[0::2] = np.where(valid, S[:, 1] / np.maximum(S[:, 0], 1e-12) - 1.0, 0.0)
    residuals[1::2] = np.where(valid, S[:, 2] / np.maximum(S[:, 0], 1e-12), 0.0)

    return residuals


def calibrate_view_graph(
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints: Dict[int, Keypoints],
    initial_intrinsics: Dict[int, gtsfm_types.CALIBRATION_TYPE],
    min_correspondences: int = 30,
    min_focal_ratio: float = 0.5,
    max_focal_ratio: float = 2.0,
    max_edge_error: float = 0.5,
    use_robust_gates: bool = True,
) -> Tuple[Dict[int, gtsfm_types.CALIBRATION_TYPE], set]:
    """Refine camera focal lengths via joint Fetzer optimization over all F-matrix edges.

    Also filters edges with high calibration error (GLOMAP FilterImagePairs).

    Args:
        v_corr_idxs_dict: Verified correspondence indices per image pair.
        keypoints: Keypoints keyed by image index.
        initial_intrinsics: Initial intrinsics keyed by image index.
        min_correspondences: Minimum correspondences to attempt F estimation.
        min_focal_ratio: Minimum allowed ratio of optimized/initial focal length.
        max_focal_ratio: Maximum allowed ratio of optimized/initial focal length.
        max_edge_error: Maximum Fetzer residual norm to keep an edge.

    Returns:
        Refined intrinsics dict (same keys as initial_intrinsics).
        Set of edge keys (i1, i2) to remove from the view graph.
    """
    # Step 1: Estimate F-matrices and collect optimization edges. With use_robust_gates (default),
    # each edge gets a robust PoseLib F plus two admission gates — planar-config (H/F inliers) and
    # focal-sanity (interior-minimum sweep). ToL gold A/B: ungated 8-point feeding of planar /
    # focal-degenerate edges (56% of that graph!) produced a -12.35% systematic focal bias; the
    # gated recipe cut median error 2.5x and removed the bias on identical inputs.
    edges = []  # (cam_idx1, cam_idx2, F, pp1, pp2)
    cameras_in_edges = set()
    num_planar = num_sanity = num_ffail = 0

    for (i1, i2), v_corr_idxs in v_corr_idxs_dict.items():
        if v_corr_idxs.shape[0] < min_correspondences:
            continue

        coords_i1 = keypoints[i1].coordinates[v_corr_idxs[:, 0]]
        coords_i2 = keypoints[i2].coordinates[v_corr_idxs[:, 1]]

        K1 = initial_intrinsics[i1].K()
        K2 = initial_intrinsics[i2].K()
        pp1 = np.array([K1[0, 2], K1[1, 2]])
        pp2 = np.array([K2[0, 2], K2[1, 2]])

        if use_robust_gates:
            F, drop_reason = _estimate_fundamental_robust(coords_i1, coords_i2)
            if drop_reason == "planar":
                num_planar += 1
                continue
            if F is None:
                num_ffail += 1
                continue
            f_init = 0.5 * (K1[0, 0] + K2[0, 0])
            if not f_passes_focal_sanity(F, pp1, pp2, f_init):
                num_sanity += 1
                continue
        else:
            F = estimate_fundamental_from_correspondences(coords_i1, coords_i2)
            if F is None:
                num_ffail += 1
                continue

        edges.append((i1, i2, F, pp1, pp2))
        cameras_in_edges.add(i1)
        cameras_in_edges.add(i2)

    if use_robust_gates:
        logger.info(
            "View graph calibration gates: %d edges kept | dropped %d planar, %d focal-sanity, "
            "%d F-fail (poselib=%s).",
            len(edges), num_planar, num_sanity, num_ffail, _HAS_POSELIB,
        )

    if not edges:
        logger.info("View graph calibration: no valid edges, skipping.")
        # Return the intrinsics UNCHANGED (as a dict). `list(initial_intrinsics)` would return the
        # camera-index KEYS, which downstream code then indexes as if it were the intrinsics dict
        # -> Cal3Bundler(pose, <int>) crash for any cluster with no F-edges.
        return dict(initial_intrinsics), set()

    # Step 2: Set up optimization variables.
    sorted_cameras = sorted(cameras_in_edges)
    cam_idx_to_var_idx = {cam: var for var, cam in enumerate(sorted_cameras)}
    initial_focals = np.array([initial_intrinsics[cam].K()[0, 0] for cam in sorted_cameras])

    logger.info(
        "View graph calibration (Fetzer): %d edges, %d cameras. " "Initial focals: min=%.1f, med=%.1f, max=%.1f",
        len(edges),
        len(sorted_cameras),
        initial_focals.min(),
        np.median(initial_focals),
        initial_focals.max(),
    )

    # Precompute arrays for vectorized residual evaluation.
    precomputed = {
        "idx1": np.array([cam_idx_to_var_idx[e[0]] for e in edges]),
        "idx2": np.array([cam_idx_to_var_idx[e[1]] for e in edges]),
        "F_stack": np.array([e[2] for e in edges]),  # (n, 3, 3)
        "pp1_stack": np.array([e[3] for e in edges]),  # (n, 2)
        "pp2_stack": np.array([e[4] for e in edges]),  # (n, 2)
    }

    # Jacobian sparsity: each edge's two residual rows depend ONLY on that edge's two camera focals
    # (idx1, idx2) out of all N cameras — a 99.9%-sparse Jacobian. Without this, least_squares builds a
    # DENSE numerical Jacobian, perturbing every one of the N focals per iteration (N x batched-SVD),
    # which hangs at scale (e.g. 2504 cameras x 117k edges). Passing jac_sparsity lets scipy estimate the
    # numerical Jacobian with graph-colored group perturbations + solve it sparsely -> seconds, not hours.
    n_edges = len(edges)
    n_cams = len(initial_focals)
    sparsity_rows = np.repeat(np.arange(2 * n_edges), 2)  # rows: [2k, 2k, 2k+1, 2k+1] per edge k
    sparsity_cols = np.empty(4 * n_edges, dtype=int)
    sparsity_cols[0::4] = precomputed["idx1"]  # residual 2k   depends on focal idx1
    sparsity_cols[1::4] = precomputed["idx2"]  # residual 2k   depends on focal idx2
    sparsity_cols[2::4] = precomputed["idx1"]  # residual 2k+1 depends on focal idx1
    sparsity_cols[3::4] = precomputed["idx2"]  # residual 2k+1 depends on focal idx2
    jac_sparsity = coo_matrix(
        (np.ones(4 * n_edges), (sparsity_rows, sparsity_cols)),
        shape=(2 * n_edges, n_cams),
    )

    # Step 3: Joint optimization (gtsam SelfCalibrationFactor when the build provides it, else
    # scipy Cauchy least squares — see _solve_focals).
    optimized_focals, solver_info = _solve_focals(
        edges, cam_idx_to_var_idx, precomputed, initial_focals, jac_sparsity
    )

    # Step 4: Validate and build refined intrinsics. applied_focals tracks the focal actually used
    # downstream per camera (optimized when accepted, initial otherwise) so Step 5 scores edges with
    # what the pipeline ships, not with discarded optimizer values.
    refined = dict(initial_intrinsics)
    applied_focals = initial_focals.copy()
    num_refined = 0
    num_rejected = 0

    for var_idx, cam_idx in enumerate(sorted_cameras):
        old_focal = initial_focals[var_idx]
        new_focal = optimized_focals[var_idx]
        ratio = new_focal / old_focal if old_focal > 0 else 0

        if min_focal_ratio <= ratio <= max_focal_ratio:
            applied_focals[var_idx] = new_focal
            old_K = initial_intrinsics[cam_idx]
            # We assume fx == fy here.
            if isinstance(old_K, Cal3Bundler):
                refined[cam_idx] = Cal3Bundler(
                    fx=float(new_focal),
                    k1=old_K.k1(),
                    k2=old_K.k2(),
                    u0=old_K.px(),
                    v0=old_K.py(),
                )
                num_refined += 1
            elif isinstance(old_K, Cal3_S2):
                refined[cam_idx] = Cal3_S2(
                    fx=float(new_focal),
                    fy=float(new_focal),
                    s=old_K.skew(),
                    u0=old_K.px(),
                    v0=old_K.py(),
                )
                num_refined += 1
            elif isinstance(old_K, Cal3DS2):
                refined[cam_idx] = Cal3DS2(
                    fx=float(new_focal),
                    fy=float(new_focal),
                    s=old_K.skew(),
                    u0=old_K.px(),
                    v0=old_K.py(),
                    k1=old_K.k1(),
                    k2=old_K.k2(),
                    p1=old_K.p1(),
                    p2=old_K.p2(),
                )
                num_refined += 1
        else:
            num_rejected += 1

    # Step 5: Filter edges with high calibration error (GLOMAP FilterImagePairs), scored at the
    # APPLIED focals so edges are judged by calibrations the pipeline actually uses.
    final_residuals = _fetzer_residuals(applied_focals, edges, cam_idx_to_var_idx)
    edges_to_remove = set()
    for i, (cam1, cam2, F, pp1, pp2) in enumerate(edges):
        edge_error = np.linalg.norm(final_residuals[2 * i : 2 * i + 2])
        if edge_error > max_edge_error:
            edges_to_remove.add((cam1, cam2))

    logger.info(
        "View graph calibration (Fetzer): refined %d, rejected %d / %d cameras. "
        "Optimized focals: min=%.1f, med=%.1f, max=%.1f. "
        "Filtered %d / %d edges by calibration error (threshold=%.2f). Solver: %s.",
        num_refined,
        num_rejected,
        len(sorted_cameras),
        optimized_focals.min(),
        np.median(optimized_focals),
        optimized_focals.max(),
        len(edges_to_remove),
        len(edges),
        max_edge_error,
        solver_info,
    )

    return refined, edges_to_remove


def compute_global_view_graph_intrinsics(
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints_list: List[Keypoints],
    one_view_data_dict: dict,
) -> Dict[int, gtsfm_types.CALIBRATION_TYPE]:
    """Estimate every camera's focal ONCE on the full verified view graph (global Fetzer).

    Unlike the per-cluster calibration -- which only sees a single cluster's edges and falls back to
    the VGGT-predicted focal for cameras lacking in-cluster F-edges -- this runs over ALL verified
    correspondences, giving the global estimation strength that pins focals close to GT. Initial focals
    come from the loader intrinsics (the EXIF-free 1.2*maxdim heuristic) converted to Cal3Bundler; VGGT
    focals are never used. Cameras with no qualifying F-edge keep the heuristic focal (still not VGGT).

    Returns Cal3Bundler intrinsics (matching the VGGT PinholeCameraCal3Bundler type used downstream),
    keyed by global camera index, covering every camera in one_view_data_dict.
    """
    initial_intrinsics: Dict[int, gtsfm_types.CALIBRATION_TYPE] = {}
    for idx, one_view_data in one_view_data_dict.items():
        K = one_view_data.intrinsics.K()
        initial_intrinsics[idx] = Cal3Bundler(float(K[0, 0]), 0.0, 0.0, float(K[0, 2]), float(K[1, 2]))

    keypoints = {idx: keypoints_list[idx] for idx in range(len(keypoints_list))}
    refined, _edges_to_remove = calibrate_view_graph(
        v_corr_idxs_dict=v_corr_idxs_dict,
        keypoints=keypoints,
        initial_intrinsics=initial_intrinsics,
    )
    # NOTE: cameras Fetzer cannot refine (no qualifying F-edge, or a focal-ratio/calibration-error
    # rejection) keep their 1.2*maxdim heuristic focal. A median-fill of those was TRIED and REVERTED:
    # the unrefined cameras skew high-focal (narrow-FOV/telephoto, few correspondences), and for those
    # the 1.2 heuristic is actually accurate (Brussels cams 12/39/68 true f/maxdim ~1.19-1.23) — filling
    # them to the dataset median (~1.025) hurt 3 cameras (~14% each) to help 1 (215). Keep the heuristic.
    return refined
