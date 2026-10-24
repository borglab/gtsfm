# GTSFM Studio Dask execution profiles verification protocol

Status: implementation protocol, 2026-10-24.

Hypothesis: a named host-local Dask execution profile can be validated once, persisted across Studio restarts, and resolved into the same runner arguments as the existing manual controls. The primary acceptance claim is repeatability and early configuration error detection; no speedup or hard GPU VRAM enforcement is assumed.

Baseline is the existing direct Studio form and `python -m gtsfm.runner` path at the pinned repository revision recorded in the project ledger. Treatment is the saved-profile API/UI path with the same loader, algorithm, dataset, seed/defaults, cache policy, and hardware. Each run records the resolved profile and command, requested versus observed worker count, threads, host-memory limits, GPU visibility, stage timings, process RSS, and GPU allocated/reserved memory where available.

Before bounded reconstruction runs, verify: profile create/update/select/reload in a fresh process; API-to-argv equivalence; rejection of invalid worker/thread/memory/GPU/scratch/mode values; one-worker RTX 3080 visibility; clean cancellation and Dask worker termination. For matched reconstruction repeats, compare selected image/pair identities, registered-camera set, tracks, aligned poses/intrinsics, reprojection metrics, and exports where available. Deterministic artifacts require exact equality; nondeterministic metrics use predeclared tolerances against a baseline repeat. Any untested mode remains conditional.
