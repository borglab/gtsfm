"""Timing comparisons for CPU vs CUDA Sparse LM bundle adjustment.

These tests are intentionally marked ``slow`` so default CI unit tests skip them.
They are meant to be run on a CUDA-capable machine with benchmark datasets present::

    bash .github/scripts/download_single_benchmark.sh gerrard-hall-100 wget
    uv run pytest tests/bundle/test_cuda_ba_timing.py -m slow -s --no-cov

Authors: Kusum
"""

from __future__ import annotations

import os
import time
import unittest
from pathlib import Path
from typing import Optional, Tuple

import gtsam  # type: ignore
import pytest

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer, RobustBAMode
from gtsfm.common.gtsfm_data import GtsfmData

REPO_ROOT = Path(__file__).resolve().parents[2]
GERRARD_HALL_SPARSE = REPO_ROOT / "benchmarks" / "gerrard-hall" / "sparse"

# Isolate Sparse LM cost: reprojection factors only (no robust loss / extra priors).
# This keeps CPU vs CUDA comparison focused on the optimizer backend.
_TIMING_BA_KWARGS = dict(
    reproj_error_thresholds=[100.0],
    min_tracks_per_camera=0,
    robust_ba_mode=RobustBAMode.NONE,
    use_calibration_prior=False,
    use_karcher_mean_factor=False,
    use_pose_prior_all_cameras=False,
    use_pose_prior_first_camera=False,
    use_first_point_prior=False,
    max_iterations=50,
    optimizer_relative_cost_tol=1e-5,
)


def _gtsam_cuda_available() -> bool:
    return getattr(gtsam, "cuda", None) is not None


def _require_gerrard_hall() -> Path:
    if not GERRARD_HALL_SPARSE.exists():
        pytest.skip(
            "Gerrard Hall sparse model not found. Download with: "
            "bash .github/scripts/download_single_benchmark.sh gerrard-hall-100 wget"
        )
    return GERRARD_HALL_SPARSE


def _build_optimizer(*, use_cuda: bool) -> BundleAdjustmentOptimizer:
    return BundleAdjustmentOptimizer(
        use_cuda=use_cuda,
        cuda_linear_solver="PCG",
        # Fail loudly in timing runs so a silent CPU fallback cannot look like a CUDA win/loss.
        cuda_fallback_on_unsupported=False,
        cuda_collect_timing=True,
        **_TIMING_BA_KWARGS,
    )


def _time_simple_ba(
    data: GtsfmData, *, use_cuda: bool, warmups: int = 1, trials: int = 1
) -> Tuple[float, float, Optional[object]]:
    """Return (mean wall-clock seconds, final error, last CUDA diagnostics)."""
    durations = []
    final_error = float("nan")
    last_cuda_result = None

    for trial_idx in range(warmups + trials):
        ba = _build_optimizer(use_cuda=use_cuda)
        start = time.perf_counter()
        _, final_error = ba.run_simple_ba(data)
        wall = time.perf_counter() - start
        last_cuda_result = ba._last_cuda_result
        opt_duration = ba._last_optimization_duration_sec
        print(
            f"{'CUDA' if use_cuda else 'CPU'} "
            f"{'warmup' if trial_idx < warmups else 'trial'} "
            f"wall={wall:.3f}s opt={opt_duration:.3f}s error={final_error:.4f}"
        )
        if trial_idx >= warmups:
            # Prefer the optimizer-internal duration (exclude Values <-> GtsfmData conversion).
            durations.append(opt_duration if opt_duration is not None else wall)

    return sum(durations) / len(durations), final_error, last_cuda_result


@pytest.mark.slow
@pytest.mark.cuda
class TestCudaBundleAdjustmentTiming(unittest.TestCase):
    """Slow timing tests for CUDA Sparse LM on benchmark scenes."""

    def test_gerrard_hall_cpu_vs_cuda_timing(self):
        """Compare CPU LM vs CUDA Sparse LM wall time on Gerrard Hall (~100 images)."""
        if not _gtsam_cuda_available():
            self.skipTest("gtsam.cuda is unavailable (GTSAM was not built with CUDA support).")

        sparse_dir = _require_gerrard_hall()
        data = GtsfmData.read_colmap(str(sparse_dir))
        self.assertGreaterEqual(data.number_images(), 90)
        self.assertGreater(data.number_tracks(), 1000)

        print(
            f"\nGerrard Hall timing scene: "
            f"{data.number_images()} cameras, {data.number_tracks()} tracks"
        )

        cpu_sec, cpu_error, _ = _time_simple_ba(data, use_cuda=False, warmups=0, trials=1)
        cuda_sec, cuda_error, cuda_result = _time_simple_ba(data, use_cuda=True, warmups=1, trials=1)

        self.assertIsNotNone(cuda_result)
        backend = getattr(cuda_result, "backend", None)
        self.assertIsNotNone(backend)
        self.assertIn("Device", str(backend), msg=f"Expected CUDA Device backend, got {backend}")

        speedup = cpu_sec / cuda_sec if cuda_sec > 0 else float("inf")
        print("\n==== Gerrard Hall BA timing summary ====")
        print(f"CPU  optimize: {cpu_sec:.3f}s  final_error={cpu_error:.4f}")
        print(f"CUDA optimize: {cuda_sec:.3f}s  final_error={cuda_error:.4f}")
        print(f"Speedup (CPU/CUDA): {speedup:.2f}x")
        print("======================================\n")

        # Errors should be in the same ballpark; exact equality is not required across solvers.
        self.assertLess(abs(cpu_error - cuda_error) / max(abs(cpu_error), 1.0), 0.05)
        # Soft performance gate: CUDA should not be slower on this scene size.
        # Override with GTSFM_REQUIRE_CUDA_SPEEDUP=0 for machines where GPU contention is expected.
        require_speedup = os.environ.get("GTSFM_REQUIRE_CUDA_SPEEDUP", "1") != "0"
        if require_speedup:
            self.assertGreater(
                speedup,
                1.0,
                msg=f"Expected CUDA Sparse LM to beat CPU LM on Gerrard Hall, got {speedup:.2f}x",
            )


if __name__ == "__main__":
    unittest.main()
