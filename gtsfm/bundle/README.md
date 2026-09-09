# GTSFM - Bundle Adjustment

This is the bundle adjustment subsystem for GTSFM.

## Optimizers

### CPU Levenberg-Marquardt (Default)
By default, GTSfM optimizes bundle adjustment factor graphs using GTSAM's CPU `gtsam.LevenbergMarquardtOptimizer` (or `gtsam.GncLMOptimizer` when Graduated Non-Convexity is enabled).

### GPU Levenberg-Marquardt (GTSAM PR #2761)
GTSfM supports GPU-accelerated bundle adjustment via GTSAM's general CUDA Levenberg-Marquardt wrapper (`gtsam.cuda.SparseLevenbergMarquardtOptimizer`).

#### Requirements
- GTSAM built from source with `GTSAM_ENABLE_CUDA=ON` (and optionally `GTSAM_ENABLE_CUDSS=ON` for sparse direct Cholesky).
- When GTSAM is built without CUDA support, the `gtsam.cuda` namespace is absent.

#### Configuration Options
The following options can be set on `BundleAdjustmentOptions` or passed to `BundleAdjustmentOptimizer`:
- `use_cuda`: (bool, default `False`) Enable CUDA Sparse Levenberg-Marquardt optimization.
- `cuda_linear_solver`: (str, default `"PCG"`) Linear solver backend:
  - `"PCG"`: Matrix-free Preconditioned Conjugate Gradient on GPU. Requires only `GTSAM_ENABLE_CUDA=ON`.
  - `"CUDSS"`: cuDSS direct sparse Cholesky solver. Requires `GTSAM_ENABLE_CUDSS=ON` and NVIDIA cuDSS.
- `cuda_pcg_max_iterations`: (int, optional) Max inner iterations for the PCG solver.
- `cuda_pcg_relative_tolerance`: (float, optional) Relative tolerance for PCG convergence.
- `cuda_pcg_warm_start`: (bool, default `False`) Warm-start PCG from previous iteration.
- `cuda_pcg_convergence_check_interval`: (int, optional) Interval for checking PCG convergence.
- `cuda_fallback_on_unsupported`: (bool, default `True`) Gracefully fall back to CPU LM if CUDA runtime or factor structure is unsupported.
- `cuda_collect_timing`: (bool, default `False`) Collect CUDA kernel execution timings.

#### Hydra Configuration Example
```yaml
bundle_adjustment_module:
  _target_: gtsfm.bundle.bundle_adjustment.BundleAdjustmentOptimizer
  use_cuda: True
  cuda_linear_solver: "PCG"
  cuda_fallback_on_unsupported: True
  reproj_error_thresholds: [10, 5, 3]
  robust_ba_mode: "HUBER"
```
Or override on the CLI:
```bash
./run --config_name unified.yaml bundle_adjustment_module.use_cuda=True bundle_adjustment_module.cuda_linear_solver=PCG
```

Prefer grouping CUDA knobs via `BundleAdjustmentOptions` (and follow-up dataclass grouping for runner YAMLs) so main Hydra configs do not grow a long list of flat CUDA fields.

#### Timing (Gerrard Hall)
CPU vs CUDA Sparse LM timing can be measured on the Gerrard Hall COLMAP sparse model (~100 cameras, ~43k tracks):

```bash
bash .github/scripts/download_single_benchmark.sh gerrard-hall-100 wget
uv run pytest tests/bundle/test_cuda_ba_timing.py -m "slow and cuda" -s --no-cov
```

Representative local result on an RTX 3080 (PCG backend, reprojection-only graph, `max_iterations=50`):

| Backend | Optimize time | Notes |
|---|---|---|
| CPU LM | ~99 s | GTSAM `LevenbergMarquardtOptimizer` |
| CUDA Sparse LM | ~13 s | `gtsam.cuda.SparseLevenbergMarquardtOptimizer`, Device backend |
| Speedup | ~7.7× | Final objective within ~0.2% |

The pytest times the optimizer step (`_last_optimization_duration_sec`) after one CUDA warmup.

CI unit jobs skip this test (`-m "not slow"`). It also auto-skips when `gtsam.cuda` or the Gerrard Hall dataset is unavailable.
