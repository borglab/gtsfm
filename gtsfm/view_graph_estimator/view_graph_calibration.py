"""View graph calibration: estimate camera focal lengths from fundamental matrices.

Joint optimization of all focal lengths using Fetzer et al. (WACV 2020) residuals.
For each F-matrix edge, measures how close E = K2^T F K1 is to a valid essential matrix
(valid E has singular values σ1 = σ2, σ3 = 0).

Reference: Fetzer et al., "Stable Intrinsic Auto-Calibration from Fundamental Matrices
of Devices with Uncorrelated Camera Parameters", WACV 2020.
"""

import logging
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from gtsam import Cal3Bundler, Cal3_S2, Cal3DS2
from scipy.optimize import least_squares

import gtsfm.common.types as gtsfm_types
from gtsfm.common.keypoints import Keypoints

logger = logging.getLogger(__name__)


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
    # Step 1: Estimate F-matrices and collect optimization edges.
    edges = []  # (cam_idx1, cam_idx2, F, pp1, pp2)
    cameras_in_edges = set()

    for (i1, i2), v_corr_idxs in v_corr_idxs_dict.items():
        if v_corr_idxs.shape[0] < min_correspondences:
            continue

        coords_i1 = keypoints[i1].coordinates[v_corr_idxs[:, 0]]
        coords_i2 = keypoints[i2].coordinates[v_corr_idxs[:, 1]]

        F = estimate_fundamental_from_correspondences(coords_i1, coords_i2)
        if F is None:
            continue

        K1 = initial_intrinsics[i1].K()
        K2 = initial_intrinsics[i2].K()
        pp1 = np.array([K1[0, 2], K1[1, 2]])
        pp2 = np.array([K2[0, 2], K2[1, 2]])

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

    # Step 3: Joint optimization with Cauchy robust loss.
    result = least_squares(
        _fetzer_residuals,
        x0=initial_focals,
        args=(edges, cam_idx_to_var_idx, precomputed),
        loss="cauchy",
        f_scale=0.1,
        bounds=(100.0, np.inf),
        max_nfev=200,
    )

    optimized_focals = result.x

    # Step 4: Validate and build refined intrinsics.
    refined = dict(initial_intrinsics)
    num_refined = 0
    num_rejected = 0

    for var_idx, cam_idx in enumerate(sorted_cameras):
        old_focal = initial_focals[var_idx]
        new_focal = optimized_focals[var_idx]
        ratio = new_focal / old_focal if old_focal > 0 else 0

        if min_focal_ratio <= ratio <= max_focal_ratio:
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

    # Step 5: Filter edges with high calibration error.
    final_residuals = _fetzer_residuals(optimized_focals, edges, cam_idx_to_var_idx)
    edges_to_remove = set()
    for i, (cam1, cam2, F, pp1, pp2) in enumerate(edges):
        edge_error = np.linalg.norm(final_residuals[2 * i : 2 * i + 2])
        if edge_error > max_edge_error:
            edges_to_remove.add((cam1, cam2))

    logger.info(
        "View graph calibration (Fetzer): refined %d, rejected %d / %d cameras. "
        "Optimized focals: min=%.1f, med=%.1f, max=%.1f. "
        "Filtered %d / %d edges by calibration error (threshold=%.2f). "
        "Solver: %s in %d evaluations, cost %.4f → %.4f.",
        num_refined,
        num_rejected,
        len(sorted_cameras),
        optimized_focals.min(),
        np.median(optimized_focals),
        optimized_focals.max(),
        len(edges_to_remove),
        len(edges),
        max_edge_error,
        result.message,
        result.nfev,
        0.5 * np.sum(_fetzer_residuals(initial_focals, edges, cam_idx_to_var_idx) ** 2),
        result.cost,
    )

    return refined, edges_to_remove
