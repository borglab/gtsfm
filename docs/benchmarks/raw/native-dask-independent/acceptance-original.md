# Native Dask configuration independent verification

Acceptance declared 2026-10-24 before new native-path runtime or reconstruction outputs.
Reviewer dispatch: `ctx_18159c53187d`; starting source `012b79c8`; prefeature reference `3306994e`.
This report replaces no historical evidence and makes no speedup claim.

## Predeclared acceptance

1. Production removes the bespoke profile store, schema, CRUD endpoints, memory parser,
   and frontend profile state. Native Dask YAML/environment configuration remains native;
   topology and memory-limit constructor arguments remain explicit existing runner controls.
   Existing Hydra/OmegaConf algorithm configuration, manual, Modal and legacy cluster config
   paths remain available. Stubbed SSH/GPU checks establish routing only, never deployment.
2. On installed Dask/distributed 2025.9.1, fresh bounded CPU subprocesses must observe real
   worker PIDs, worker/thread counts, per-worker parsed memory limits, native scratch and
   memory thresholds. Native environment values take precedence over native YAML. Explicit
   constructor/CLI settings take precedence where Dask supports those arguments. Unknown
   YAML keys must never be advertised as effective LocalCluster topology controls.
3. Existing Studio request -> runner argv/environment must retain ordinary manual values,
   device visibility and Hydra overrides. New native configuration controls, if present,
   must reach a fresh runner environment. No user home configuration or saved state is
   modified by these probes. Failure probes must retain the original failure and establish
   bounded worker termination; successful parsing alone is insufficient.
4. New GPU evidence must identify actual tested source files/hashes, complete argv and
   resolved algorithm configuration, model/input hashes, cache policy, seeds, GPU and
   workers. Compare new matched manual/native repeated runs with the existing strict
   `scripts/verify_dask_profiles.py compare` implementation. Preserve the original
   image/pair identities, camera metadata, track observations/colors, and finite geometry.
   Check complete final cameras/images/points exports and named retrieval pairs, not counts.
5. The frozen numerical criteria remain: a single orientation-preserving Sim(3) fit from
   at least three noncollinear camera centers; all camera centers <= `1e-5 * D`, all
   orientations <= `0.001` degree, all points <= `1e-4 * D`, all track reprojection errors
   <= `0.011` pixel; intrinsics `atol=rtol=1e-6`. `D` is baseline camera-center bounding-box
   diagonal. Nonfinite geometry fails. Ambiguous nonidentical observation mappings remain
   inconclusive; byte-identical complete exports resolve the exact-file case. Do not loosen
   thresholds or silently canonicalize away point identity/multiplicity.

Each claim receives **VERIFIED**, **NOT VERIFIED**, or **INCONCLUSIVE**. A completed valid
comparison missing its criterion is NOT VERIFIED; missing runtime, controls, or artifacts
is INCONCLUSIVE. Historical four-image RTX 3080 results do not verify the new code.

## Official architecture sources

- [Dask configuration](https://docs.dask.org/en/stable/configuration.html): native YAML,
  `DASK_CONFIG`, nested `DASK_...__...` environment keys, refresh and configuration lifetime.
- [Dask Python deployment API](https://docs.dask.org/en/stable/deploying-python.html):
  `LocalCluster` constructor arguments, worker processes and per-worker host memory.
- Installed `dask` and `distributed` both report `2025.9.1`; their actual implementation
  is the version-specific authority for these probes. Mutable stable docs are checked
  against installed source, not assumed to describe every older behavior.

The independent installed-source inspection confirms that `n_workers`,
`threads_per_worker`, `processes`, `memory_limit`, and `dashboard_address` are constructor
controls. Arbitrary YAML entries with those names do not configure LocalCluster. Native
memory spill/target/pause settings and `temporary-directory` are separate Dask config keys.

## Results

Pending final production contract and current runtime evidence. No acceptance verdict yet.
