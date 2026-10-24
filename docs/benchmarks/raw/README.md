# Dask execution profiles: evidence index

The main narrative is [the verification report](../dask_profiles_verification.md).
The [independent protocol and review](../dask_profiles_experiment.md) distinguishes
predeclared criteria, retained failures, and final verdicts.

| Evidence | Purpose |
| --- | --- |
| [Independent claims](independent-verifier/claims.json) | Machine-readable per-claim verdicts and limits |
| [Independent index](independent-verifier/manifest.json) | SHA256 values for 101 independently produced evidence files |
| [GPU experiment](gpu-experiment/README.md) | Apparatus, four measured trials, separate setup/warmup/cancellation/final-source attempts, reproduction and cleanup |
| [GPU index](gpu-experiment/artifact-manifest.json) | SHA256 and sizes for the frozen 562-artifact execution package |
| [Coordinator checks](coordinator-validation.json) | Separate comparison rerun, final tests/build/lint, AST equality and corrected setup errors |
| [Browser workflow](fullstack-ui-result.json) | Actual Chrome to FastAPI profile save/update/reload/error workflow |
| [Server restart](fullstack-restart-result.json) | Fresh backend process reads the browser-created persisted profile |
| [Browser screenshot](fullstack-ui.png) | Observed saved-profile interface from that workflow |

The raw package intentionally retains failed attempts and warning-bearing logs.
Data exports, logs and snapshots remain verbatim; duplicated reports inside the frozen
GPU package are preserved for its existing checksum index. The canonical independent
reports live under `independent-verifier/`.

Inputs and multi-gigabyte model weights are identified by immutable revisions and hashes
rather than copied here. The retained compute host and isolated source/assets are described
in the GPU report. These artifacts establish the explicitly scoped claims, not universal
dataset accuracy, internal bitwise equality, hard VRAM quotas, or a speedup.
