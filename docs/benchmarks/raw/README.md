# Dask configuration: evidence index

## Current: native Dask configuration

Read the [native verification report](../native_dask_verification.md) and the
[configuration guide](../../setup/dask-configuration.md). This replaces the
custom-profile implementation below with the existing Hydra/OmegaConf constructor
path and native Dask policies. The new experiment uses new outputs from the
refactored source; historical results are not its acceptance evidence.

| Evidence | Purpose |
| --- | --- |
| [Frozen CPU matrix](native-dask-independent/final-cpu-1/processes.json) | 14 real configuration, lifecycle and request-contract cases |
| [Independent source audit](native-dask-independent/final-source-audit.json) | Runtime hashes and removal of the bespoke profile system |
| [Strict GPU comparison](native-dask-independent/gpu-four-trial-comparison.json) | Complete exports and named pairs: exact bytes across all four current runs |
| [GPU controls review](native-dask-independent/gpu-controls-review.json) | Matched source, models, inputs, resolved algorithms, real workers and owned-job cleanup |
| [Current GPU package](native-dask-gpu/) | New manual/native matched HTTP-to-GPU trials, source and asset manifests |
| [Browser checks](native-dask/ui/result.json) | Real Chrome, intercepted requests; separate from actual GPU execution |
| [Dask skill installation](native-dask/skill-installation.json) | Pinned upstream community skill, one canonical shared owner |

The stable [one-page visual brief](../../../output/pdf/dask-profiles-verification.pdf)
shows the pipeline, replacement and experiment. [Delivery metadata](../native_dask_delivery.json)
records its source, checksums and scope. This is a maintainability result with exact
exported reconstruction on one four-image scene, without a speedup or accuracy claim.

## Historical: custom-profile prototype

The following evidence applies to the superseded implementation at `8dafa2de`.
Its source, methods and raw checksums are retained for traceability. Reproduce the
historical `verify_dask_profiles.py config` command at that revision; the unchanged
`compare` command remains the strict reconstruction comparator for current work.
The historical PDF is available from revision `012b79c8`; the stable PDF path now
contains the current brief when rendered.

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
