"""Tests for the robust (gated) Fetzer view-graph calibration."""

import numpy as np
from gtsam import Cal3Bundler

from gtsfm.common.keypoints import Keypoints
from gtsfm.view_graph_estimator import view_graph_calibration as vgc


def _skew(t):
    return np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])


def _synthetic_pair(f=600.0, n=200, seed=0, planar=False):
    """Correspondences from a known two-view geometry (f shared, pp at origin-offset 400,300)."""
    rng = np.random.RandomState(seed)
    K = np.array([[f, 0, 400.0], [0, f, 300.0], [0, 0, 1.0]])
    angle = 0.15
    R = np.array([
        [np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)],
    ])
    t = np.array([1.0, 0.15, 0.3])
    if planar:
        X = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-2, 2, n), np.full(n, 8.0)])
    else:
        X = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-2, 2, n), rng.uniform(5, 14, n)])
    x1 = (K @ X.T).T
    x1 = x1[:, :2] / x1[:, 2:3]
    X2 = (R @ X.T).T + t
    x2 = (K @ X2.T).T
    x2 = x2[:, :2] / x2[:, 2:3]
    E = _skew(t) @ R
    F = np.linalg.inv(K).T @ E @ np.linalg.inv(K)
    return x1, x2, F / np.linalg.norm(F)


def test_focal_sanity_accepts_clean_F_rejects_random() -> None:
    x1, x2, F = _synthetic_pair()
    pp = np.array([400.0, 300.0])
    assert vgc.f_passes_focal_sanity(F, pp, pp, f_init=650.0)  # near-truth init: interior min
    rng = np.random.RandomState(3)
    n_pass = sum(
        vgc.f_passes_focal_sanity(rng.rand(3, 3), pp, pp, f_init=650.0) for _ in range(20)
    )
    assert n_pass <= 2  # random matrices almost never imply a well-defined focal


def test_robust_F_planar_gate() -> None:
    x1, x2, _ = _synthetic_pair(planar=True)
    F, reason = vgc._estimate_fundamental_robust(x1, x2)
    if vgc._HAS_POSELIB:
        assert reason == "planar" and F is None
    else:
        assert F is not None or reason == "f_fail"


def test_calibrate_view_graph_recovers_focal() -> None:
    """3 cams, true f=600, EXIF-ish init f=700: the gated pipeline should pull focals toward truth."""
    pairs = [(0, 1), (1, 2), (0, 2)]
    keypoints = {}
    v_corr = {}
    store = {i: [] for i in range(3)}
    for k, (a, b) in enumerate(pairs):
        x1, x2, _ = _synthetic_pair(seed=k)
        ia = np.arange(len(store[a]), len(store[a]) + len(x1))
        ib = np.arange(len(store[b]), len(store[b]) + len(x2))
        store[a].extend(x1.tolist())
        store[b].extend(x2.tolist())
        v_corr[(a, b)] = np.column_stack([ia, ib]).astype(np.int64)
    for i in range(3):
        keypoints[i] = Keypoints(coordinates=np.array(store[i]))
    intr = {i: Cal3Bundler(700.0, 0.0, 0.0, 400.0, 300.0) for i in range(3)}
    refined, _ = vgc.calibrate_view_graph(v_corr, keypoints, intr, min_correspondences=30)
    focals = np.array([refined[i].fx() for i in range(3)])
    err0 = abs(700.0 - 600.0) / 600.0
    err1 = np.abs(focals - 600.0) / 600.0
    assert np.median(err1) < err0 * 0.35, f"focals {focals} did not converge toward 600"


def test_gates_off_matches_legacy_path() -> None:
    x1, x2, _ = _synthetic_pair(seed=9)
    keypoints = {0: Keypoints(coordinates=x1), 1: Keypoints(coordinates=x2)}
    v_corr = {(0, 1): np.column_stack([np.arange(len(x1))] * 2).astype(np.int64)}
    intr = {i: Cal3Bundler(650.0, 0.0, 0.0, 400.0, 300.0) for i in range(2)}
    refined, _ = vgc.calibrate_view_graph(
        v_corr, keypoints, intr, min_correspondences=30, use_robust_gates=False
    )
    assert 0 in refined and 1 in refined  # legacy path still runs end to end
