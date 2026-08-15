"""View graph calibration: estimate camera focal lengths from fundamental matrices.

Joint optimization of all focal lengths using Fetzer et al. (WACV 2020) residuals.
For each F-matrix edge, measures how close E = K2^T F K1 is to a valid essential matrix
(valid E has singular values σ1 = σ2, σ3 = 0).

Reference: Fetzer et al., "Stable Intrinsic Auto-Calibration from Fundamental Matrices
of Devices with Uncorrelated Camera Parameters", WACV 2020.
GLOMAP paper Section 3.5 — view graph calibration before global positioning.
"""

from typing import Dict, List, Optional, Tuple

import cv2
import gtsam
import numpy as np
from gtsam import Cal3Bundler, Cal3DS2, Cal3Fisheye, Cal3_S2

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints

logger = logger_utils.get_logger()

# Cauchy robust scale for the SelfCalibrationFactor graph. KEEP at 0.01: it is GLOMAP-faithful AND
# provides stability for the closed-form's unbounded [(f^2-K)/f^2] residual (loosening it
# lets ill-conditioned wide-FOV cams run to wrong focal extremes and regresses rotation).
CAUCHY_SCALE = 0.01
# GLOMAP feeds Fetzer only CALIBRATED(2)/UNCALIBRATED(3) pairs; PLANAR/PANORAMIC/
# DEGENERATE drive the closed-form residual denominators to their poles.
FETZER_ALLOWED_CONFIGS = frozenset({2, 3})


def estimate_fundamental_from_correspondences(
    coords_i1: np.ndarray,
    coords_i2: np.ndarray,
) -> Optional[np.ndarray]:
    """Estimate F-matrix from verified correspondences using the 8-point algorithm.

    Args:
        coords_i1: (N, 2) pixel coordinates in image 1.
        coords_i2: (N, 2) pixel coordinates in image 2.

    Returns:
        3x3 fundamental matrix, or None if estimation fails.
    """
    if len(coords_i1) < 8:
        return None
    F, _ = cv2.findFundamentalMat(coords_i1, coords_i2, method=cv2.FM_8POINT)
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
    """Return True if this fundamental matrix implies a well-defined focal length.

    A clean F has some focal f for which G(f) = K2(f)^T F K1(f) is essential-matrix-like: its
    singular values satisfy s1 == s2 and s3 == 0. A near-planar / degenerate F does not — its best
    focal either rails to the edge of the search range or never drives the residual low. Such F's
    blow up the closed-form Fetzer residual, so we keep them OUT of focal calibration (they remain
    fine for pose / rotation averaging — this gate only guards Fetzer). Bounded by construction,
    unlike Bougnoux focal-from-F, which returns NaN on real pairs.

    Sweeps a shared focal over [lo, hi] * f_init, scores each by the SVD-ratio residual
    (s2/s1 - 1)^2 + (s3/s1)^2, and accepts iff the best focal is an INTERIOR minimum (not at the
    band edge) with residual <= max_resid.
    """
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


def _refit_focal(old_K: gtsfm_types.CALIBRATION_TYPE, new_focal: float) -> Optional[gtsfm_types.CALIBRATION_TYPE]:
    """Return a copy of `old_K` with its focal length set to `new_focal`, keeping every other
    parameter (skew, principal point, distortion). Fetzer optimizes a single square-pixel focal,
    so fx = fy = new_focal. Returns None if the calibration model is unrecognized, so the caller
    can count it explicitly instead of silently leaving it unrefined.
    """
    f = float(new_focal)
    if isinstance(old_K, Cal3Bundler):
        return Cal3Bundler(fx=f, k1=old_K.k1(), k2=old_K.k2(), u0=old_K.px(), v0=old_K.py())
    if isinstance(old_K, Cal3_S2):
        return Cal3_S2(f, f, old_K.skew(), old_K.px(), old_K.py())
    if isinstance(old_K, Cal3DS2):
        return Cal3DS2(f, f, old_K.skew(), old_K.px(), old_K.py(), old_K.k1(), old_K.k2(), old_K.p1(), old_K.p2())
    if isinstance(old_K, Cal3Fisheye):
        return Cal3Fisheye(f, f, old_K.skew(), old_K.px(), old_K.py(), old_K.k1(), old_K.k2(), old_K.k3(), old_K.k4())
    return None


def calibrate_view_graph(
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints_list: List[Keypoints],
    initial_intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
    num_images: int,
    min_correspondences: int = 30,
    min_focal_ratio: float = 0.5,
    max_focal_ratio: float = 2.0,
    max_edge_error: float = 0.5,
    i2Fi1_dict: Optional[Dict[Tuple[int, int], np.ndarray]] = None,
    config_dict: Optional[Dict[Tuple[int, int], int]] = None,
) -> Tuple[List[gtsfm_types.CALIBRATION_TYPE], set]:
    """Refine camera focal lengths via joint Fetzer optimization over all F-matrix edges.

    Also filters edges with high calibration error (GLOMAP FilterImagePairs).

    Args:
        v_corr_idxs_dict: Verified correspondence indices per image pair.
        keypoints_list: Keypoints for all images.
        initial_intrinsics: Initial intrinsics (e.g., from EXIF or heuristic).
        num_images: Total number of images.
        min_correspondences: Minimum correspondences to attempt F estimation.
        min_focal_ratio: Minimum allowed ratio of optimized/initial focal length.
        max_focal_ratio: Maximum allowed ratio of optimized/initial focal length.
        max_edge_error: Maximum Fetzer residual norm to keep an edge.

    Returns:
        Refined intrinsics list (same length as initial_intrinsics).
        Set of edge keys (i1, i2) to remove from the view graph.
    """
    # Step 1: Collect F-matrices and optimization edges. Prefer the verifier's
    # robustly-estimated F (GLOMAP frontend flow — clean RANSAC F keeps the
    # closed-form residual's denominators away from their poles); fall back to
    # re-estimating from the verified correspondences only when unavailable.
    edges = []  # (cam_idx1, cam_idx2, F, pp1, pp2)
    cameras_in_edges = set()
    num_plumbed_F = 0
    num_dropped_by_config = 0
    num_dropped_by_focal_sanity = 0

    for (i1, i2), v_corr_idxs in v_corr_idxs_dict.items():
        if v_corr_idxs.shape[0] < min_correspondences:
            continue

        config = config_dict.get((i1, i2)) if config_dict is not None else None
        if config is not None and config not in FETZER_ALLOWED_CONFIGS:
            num_dropped_by_config += 1
            continue

        F = i2Fi1_dict.get((i1, i2)) if i2Fi1_dict is not None else None
        was_plumbed = F is not None
        if F is None:
            coords_i1 = keypoints_list[i1].coordinates[v_corr_idxs[:, 0]]
            coords_i2 = keypoints_list[i2].coordinates[v_corr_idxs[:, 1]]
            F = estimate_fundamental_from_correspondences(coords_i1, coords_i2)
        if F is None:
            continue

        K1 = initial_intrinsics[i1].K()
        K2 = initial_intrinsics[i2].K()
        pp1 = np.array([K1[0, 2], K1[1, 2]])
        pp2 = np.array([K2[0, 2], K2[1, 2]])

        # Robust F-list: drop near-planar/degenerate F's whose implied focal is garbage. They
        # pole/poison the focal calibration (the F is fine; this keeps it OUT of Fetzer only).
        f_init = 0.5 * (K1[0, 0] + K2[0, 0])
        if not f_passes_focal_sanity(F, pp1, pp2, f_init):
            num_dropped_by_focal_sanity += 1
            continue

        if was_plumbed:
            num_plumbed_F += 1
        edges.append((i1, i2, F, pp1, pp2))
        cameras_in_edges.add(i1)
        cameras_in_edges.add(i2)

    if not edges:
        logger.info("View graph calibration: no valid edges, skipping.")
        return list(initial_intrinsics), set()

    # Step 2: Set up optimization variables.
    sorted_cameras = sorted(cameras_in_edges)
    cam_idx_to_var_idx = {cam: var for var, cam in enumerate(sorted_cameras)}
    initial_focals = np.array([initial_intrinsics[cam].K()[0, 0] for cam in sorted_cameras])

    logger.info(
        "View graph calibration (Fetzer): %d edges (%d plumbed F, %d re-estimated, %d dropped by config, "
        "%d dropped by focal-sanity), %d cameras. Initial focals: min=%.1f, med=%.1f, max=%.1f",
        len(edges), num_plumbed_F, len(edges) - num_plumbed_F, num_dropped_by_config,
        num_dropped_by_focal_sanity, len(sorted_cameras),
        initial_focals.min(), np.median(initial_focals), initial_focals.max(),
    )

    # Step 3: Joint focal-length optimization — closed-form GTSAM SelfCalibrationFactor graph
    # solved with LM and a Cauchy(0.01) robust kernel (GLOMAP-matched).
    graph = gtsam.NonlinearFactorGraph()
    edge_noise = gtsam.noiseModel.Robust.Create(
        gtsam.noiseModel.mEstimator.Cauchy.Create(CAUCHY_SCALE),
        gtsam.noiseModel.Isotropic.Sigma(2, 1.0),
    )
    for cam1, cam2, F, pp1, pp2 in edges:
        graph.add(
            gtsam.SelfCalibrationFactor(
                gtsam.symbol("f", cam1), gtsam.symbol("f", cam2), F, pp1, pp2, edge_noise
            )
        )
    initial_values = gtsam.Values()
    for var_idx, cam in enumerate(sorted_cameras):
        initial_values.insert(gtsam.symbol("f", cam), float(initial_focals[var_idx]))

    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(200)
    opt = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, lm_params)
    gtsam_result = opt.optimize()

    optimized_focals = np.array(
        [gtsam_result.atDouble(gtsam.symbol("f", cam)) for cam in sorted_cameras]
    )
    solver_info = "graph error %.4f → %.4f" % (
        graph.error(initial_values), graph.error(gtsam_result)
    )

    # Step 4: Validate and build refined intrinsics.
    refined = list(initial_intrinsics)
    # Focal actually APPLIED downstream per camera: the optimized focal when accepted (in-band ratio
    # and a recognized calibration model), else the initial focal (rejected/unsupported cams, whose
    # optimized value is discarded). Step 5 scores edges with THIS, not the raw optimized_focals.
    applied_focals = initial_focals.copy()
    num_refined = 0
    num_rejected = 0
    num_unsupported = 0

    for var_idx, cam_idx in enumerate(sorted_cameras):
        old_focal = initial_focals[var_idx]
        new_focal = optimized_focals[var_idx]
        ratio = new_focal / old_focal if old_focal > 0 else 0

        if not (min_focal_ratio <= ratio <= max_focal_ratio):
            num_rejected += 1  # optimized focal too far from the initial guess — discard it.
            continue

        refit = _refit_focal(initial_intrinsics[cam_idx], new_focal)
        if refit is None:
            # In-band focal but an unhandled calibration model: keep the initial focal rather than
            # silently leaving applied_focals inconsistent with `refined`. Counted + logged below.
            num_unsupported += 1
            continue
        refined[cam_idx] = refit
        applied_focals[var_idx] = new_focal
        num_refined += 1

    # Step 5: Filter edges with high calibration error (GLOMAP FilterImagePairs).
    # Score at the APPLIED focals (optimized for accepted cams, initial for rejected) — NOT the
    # discarded optimized_focals — so edges aren't judged by a focal the pipeline never uses.
    final_residuals = _fetzer_residuals(applied_focals, edges, cam_idx_to_var_idx)
    edges_to_remove = set()
    for i, (cam1, cam2, F, pp1, pp2) in enumerate(edges):
        if np.linalg.norm(final_residuals[2 * i:2 * i + 2]) > max_edge_error:
            edges_to_remove.add((cam1, cam2))

    logger.info(
        "View graph calibration (Fetzer): refined %d, rejected %d, unsupported-model %d / %d cameras. "
        "Optimized focals: min=%.1f, med=%.1f, max=%.1f. "
        "Filtered %d / %d edges by calibration error (threshold=%.2f). "
        "Solver: %s.",
        num_refined, num_rejected, num_unsupported, len(sorted_cameras),
        optimized_focals.min(), np.median(optimized_focals), optimized_focals.max(),
        len(edges_to_remove), len(edges), max_edge_error,
        solver_info,
    )

    return refined, edges_to_remove
