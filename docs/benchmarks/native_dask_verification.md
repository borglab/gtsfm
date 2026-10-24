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

The current native-configuration integration is **VERIFIED within the tested scope**:
all 14 independent CPU cases passed, and all four new matched RTX 3080 runs produced
byte-identical complete final sparse exports and retrieval pairs (4 cameras, 1,509 tracks,
6 pairs). Source, input/model, resolved configuration, seed, worker and owned-job cleanup
controls passed independently. This is new-code evidence, not historical-run inference.
An unrelated MIRA lifecycle audit remains separately **INCONCLUSIVE** as detailed below;
no speedup or broader reconstruction claim is made.

| Claim | Verdict | Evidence and scope |
| --- | --- | --- |
| Native Dask YAML/environment policy precedence | **VERIFIED** | Fresh real workers see YAML target 0.61/connect 123s and environment target 0.67/connect 127s; explicit environment 30s is retained. |
| Existing project defaults remain when unconfigured | **VERIFIED** | Real production workers retain connect/tcp 300s and scheduler TTL None. |
| Hydra constructor file controls topology and memory | **VERIFIED** | Two distinct processes, two threads each, 512MiB per worker despite conflicting manual values; OmegaConf environment interpolation resolves upstream. |
| Native scratch, explicit constructor scratch, standard memory parsing | **VERIFIED** | Native environment beats YAML; explicit constructor local_directory wins; upstream parser handles decimal/binary units, auto/fraction and zero/unlimited. |
| Actual task execution and bounded cleanup | **VERIFIED** | Worker task returns 10; injected Client startup and pipeline-construction failures retain the sentinel and remove real workers before verifier teardown; no observed descendants remain after any case. |
| Studio request/argv/env and served bundle | **VERIFIED** | Actual FastAPI/JobManager, intercepted reconstruction dispatch; manual controls, cluster_config, algorithm Hydra override and CPU visibility reach the runner contract; profile routes/schema/persistence are absent. |
| Legacy SSH and GPU config compatibility | **VERIFIED**, routing only | Shipped string-host SSH and GPU YAML reach their existing constructor routes with manual options and I/O-worker selection; constructors are stubbed, so no deployment claim. |
| Manual/Modal browser controls | **VERIFIED**, UI only | Reviewed peer real-browser evidence with intercepted API, including stale profile storage; no live Modal job was launched. |
| Bespoke production profile system removed | **VERIFIED** | Source scan and actual ASGI checks; deleted store, profile-only worker probe/cache, CRUD, schemas and served-bundle contract. Historical verifier/raw remain unchanged. |
| New matched GPU reconstruction equivalence | **VERIFIED** | Four complete final exports and named pairs are byte-identical; baseline repeat and all four unchanged strict comparisons pass (4 cameras, 1,509 tracks, 6 pairs). |
| GPU experimental controls and owned-job cleanup | **VERIFIED** | Before/after source, assets and inputs match freeze; resolved algorithm configuration differs only in output_root; real worker controls and CUDA execution match; all observed PIDs absent, APIs closed, units unloaded and compute list empty. |
| Unrelated MIRA exact lifecycle identity across series | **INCONCLUSIVE** | Strict epoch create_time equality is **NOT VERIFIED** (135-second mismatch); post-run kernel identity is stable while btime moves, but no pre-run kernel start ticks were captured. |
| Native configuration documentation | **VERIFIED** | Restored guide resolves from README; separates constructor kwargs and policy, documents legacy scratch and import-time programmatic caveats, and limits remote claims. |

The current runner SHA256 is
`9a76568399ad840ee8ac51fa79eae297ba51c42c9aec763aa83e91b0e1515701`;
the frozen independent harness SHA256 is
`84e6324c5578225e0359ffb206b493cac34541d89ba729290fd000469e924435`.
[Final CPU results](raw/native-dask-independent/final-cpu-1/processes.json) record 14 zero
exits, no timeout, no source drift and no remaining observed process identities.
[Source audit](raw/native-dask-independent/final-source-audit.json) records exact hashes,
search scope and unchanged historical files. The missing guide identified in that capture
was subsequently restored and reviewed in [documentation-after.json](raw/native-dask-independent/documentation-after.json).

Constructor authority applies to constructor kwargs: the existing `--dask_tmpdir` still
sets Dask's programmatic temporary-directory policy if supplied, including with a constructor
file. An explicit file `local_directory` wins; otherwise workers can inherit that legacy
scratch policy. Runner import registers project defaults and reapplies collected native
YAML/environment values. Dask update_defaults can replace a pre-import programmatic value
equal to an old default; collected native values can also overwrite prior in-memory values.
Broad programmatic-before-runner-import precedence was not claimed or tested, and no
provenance-tracking layer was added.
The browser test and HTTP reconstruction tests are separate evidence layers, not a single
observed browser-to-GPU workflow.

Reproduce the bounded CPU matrix into a new artifact directory:

```sh
.venv/bin/python scripts/verify_native_dask.py --artifacts /tmp/native-dask-review-new
```

Each case has a 120-second deadline and records actual worker process identities.
No dependency installations, remote GPU launches, production edits or commits were performed
by this independent reviewer.

## Approved contract and upstream observations

Coordinator accepted the criteria in `msg_ad1ac7527d59` and supplied the final contract
in `msg_73899dfce3ed`: reuse `--cluster_config`, load with OmegaConf, and instantiate a
`_target_: distributed.LocalCluster` object with Hydra. The file supplies constructor
arguments without merging Studio's manual defaults. Existing `workers:` SSH/GPU files
remain supported. Dask policy files/environment remain a separate native mechanism.
This uses [Hydra instantiate](https://hydra.cc/docs/1.3/advanced/instantiate_objects/overview/)
and [OmegaConf YAML loading/interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html).

The initial upstream-only cases are retained in [upstream-1](raw/native-dask-independent/upstream-1/).
Both used actual Dask 2025.9.1 process workers: two distinct PIDs, two threads each,
536,870,912 bytes per worker, a submitted task returning 10, and no worker identities
remaining after close. YAML target `0.61` and scratch reached both workers; the env case
observed target `0.67` and the environment scratch instead. Deliberate inert YAML topology
keys did not affect explicit constructor values. Upstream memory parsing distinguishes
`512MiB` (536,870,912) from `512 MB` (512,000,000), treats zero/None as unlimited, and
accepts `auto` and fractional system-memory limits. These are upstream observations,
not production entrypoint or reconstruction acceptance.

Independent review identified that the existing runner's unconditional `dask.config.set`
forces TTL/communication timeouts over native YAML/env. Dask 2025.9.1 `update_defaults`
has another relevant behavior: a current value equal to the previous default is replaced
by the new default, even if explicitly supplied. Installed `refresh()` reloads YAML/env but prioritizes the oldest registered defaults;
it therefore cannot by itself preserve these later project fallback values. Production
probes include both unconfigured project defaults and nondefault/explicit upstream-default
communication timeouts, without introducing a custom precedence algorithm.

The [before-change runner probes](raw/native-dask-independent/runner-before/processes.json)
confirmed that timeout problem in actual production-created workers: YAML `123s` and env
`127s` both became `300s`. Both cases are retained as **NOT VERIFIED** for timeout
precedence, while their worker/thread/memory, target/scratch, task execution and process
cleanup checks passed. The initial cold runner import completed after an unusually long
startup; a subsequent fresh bounded import succeeded, so no import blocker is inferred.

The focused independent harness is [verify_native_dask.py](../../scripts/verify_native_dask.py).
It uses fresh importable-file subprocesses (120-second per-case deadline), records source
hashes before/after, observes worker PIDs and creation times, and closes workers before
reporting. Failure cases inject a recognizable exception with real workers already alive
and check whether production cleanup removed those identities **before** verifier safety
teardown. Its Studio case runs the actual FastAPI ASGI request and JobManager but intercepts
reconstruction dispatch; that case proves request/argv/environment routing only. Actual
reconstruction and HTTP-to-runner execution require the separately supplied new GPU trials.
The installed versions are Hydra 1.3.2, OmegaConf 2.3.0, FastAPI 0.141.1 and Dask/distributed
2025.9.1. The pre-existing independent comparison environment at
`/tmp/verify-this/dask-profiles-independent/venv/bin/python` exists; no packages were installed.

Before new GPU outputs, the coordinator approved identical authenticated loopback HTTP
transport for all four arms. The immutable [method addendum](raw/native-dask-independent/method-addendum.md)
records this refinement: A1/A2 use manual constructor values and B1/B2 use equivalent
Hydra LocalCluster YAML with conflicting manual form values intentionally ignored. No
identity, numerical, repeatability or other acceptance criterion changed.

Interim constructor observations are retained separately in
[constructor-interim-1](raw/native-dask-independent/constructor-interim-1/processes.json).
They establish real file-controlled two-worker/two-thread/512MiB execution despite
conflicting manual values, native memory-target/scratch precedence, constructor scratch
precedence, and standard zero/unlimited memory behavior. On runner source SHA256
`6cf8ee8514e8253bd89e2d151cf3965866d0ea6e091d125880ba401cae05df08`, injected Client startup
and pipeline-construction exceptions preserved the exact failure and removed the real
workers before verifier teardown; all recorded descendants were gone after case exit.
Timeout precedence still failed in that intermediate revision, and the legacy string-host
routing case reproduced `TypeError: string indices must be integers, not 'str'` after
stubbed SSH startup. These are retained failed/intermediate observations, not final-source
acceptance. The `hydra-env` case also recorded source drift and is **INCONCLUSIVE** as a
stable-source experiment, regardless of individual measured fields.

The initial Studio probe supplied a Hydra override absent from its chosen preset; Hydra
correctly rejected that verifier input. [studio-1](raw/native-dask-independent/studio-1/)
retains the error and classification. After inspecting the actual composed preset, the
corrected [studio-2](raw/native-dask-independent/studio-2/studio/result.json) passed all
request-routing and served-bundle checks. The valid test override was
`image_pairs_generator.retriever.min_score=0.4`; no production behavior was changed to
make the probe pass. The served JavaScript SHA256 was
`f2e16bb710148207f9de39805cfa95fad0dac051e2b09b20023456c5ff4d2f19`.

The intermediate dotted-default revision is retained in
[defaults-interim-1](raw/native-dask-independent/defaults-interim-1/runner-defaults/result.json):
real workers reported `30s`/`30s`/`5 minutes`, failing preservation of the original
`300s`/`300s`/`None` project defaults. Isolated native-API probes then distinguished three
Dask 2025.9.1 behaviors: `update_defaults` requires a nested dictionary; `refresh` prioritizes
older registered defaults; and `set` replaces nested dictionaries rather than recursively
merging them. The rejected `set(collect())` proposal is preserved in
[native-defaults-api-probe.json](raw/native-dask-independent/native-defaults-api-probe.json).
The successful isolated sequence is `update_defaults(nested)` followed by
`update(config.config, config.collect())`, with no `refresh`; it preserves project defaults
and reapplies native YAML/environment through Dask's own recursive merge. Its three fresh
no-override/30s/127s outcomes are in
[native-defaults-merge-probe.json](raw/native-dask-independent/native-defaults-merge-probe.json).
The final production-worker reruns above now verify integration of that sequence, including
explicit values equal to the upstream 30s default. The failed probes remain separate evidence
of intermediate revisions and rejected proposals; they are not final-source failures.


## Final architecture review

Against prefeature `3306994e`, `visualization/app.py`, `visualization/runtime.py`,
`gtsfm/utils/dask_stats.py` and frontend styles exactly match their original source.
The core net change is 112 changed lines in the runner and seven in frontend main.tsx:
existing cluster-file loading/dispatch, native Dask defaults, startup cleanup, string-host
compatibility, and UI help/accessibility/error-detail handling. The new packaged constructor
YAML and guide complete the configuration path. No new production parser, store, persistence
schema or worker-probe cache remains. Two private target helpers used only by tests are an
optional cleanup; they do not change the frozen runtime acceptance.

The [net-diff and skill review](raw/native-dask-independent/netdiff-skill-review.json)
also verifies all eight canonical installed Dask skill files against the supplied SHA256
and git-blob receipt. This establishes local file integrity, not independently fetched
upstream provenance or agent discovery execution. Its examples target Dask 2026.8.0;
actual project 2025.9.1 behavior and official sources governed this review, with no upgrade.

[Coordinator validation](raw/native-dask/coordinator-validation.json) reports 62 focused
pytest passes, formatting/lint checks and frontend check/build against the matching source
hashes. These are coordinator-observed results, separate from the independently executed
14-case real-worker matrix. [Peer browser evidence](raw/native-dask/ui/result.json) is
likewise explicitly limited to real UI interaction with intercepted API responses.


## New matched GPU results

The independent unchanged comparator produced
[gpu-four-trial-comparison.json](raw/native-dask-independent/gpu-four-trial-comparison.json):
**VERIFIED**, `exact_export_bytes=true`, `baseline_repeatable=true`. A1:A2, A1:B1, A1:B2
and B1:B2 all pass. Each reconstruction contains four registered cameras and 1,509 tracks;
all six retrieved image pairs match. Full `merged_retriangulated/cameras.txt`, `images.txt`,
`points3D.txt` and `similarity_named_pairs.txt` bytes match across all four trials, preserving
exact point IDs and observation multiplicity. The unchanged numerical checks also pass;
tiny reported Sim(3) roundoff is not a change in the byte-identical exports. The raw comparison
records the exact reproduction command and hashes of every compared file. The strict
comparator remains SHA256 `99d941d94b03087bd5db979dd6978af833f87fd2f2ff4da21e7ac8f6dd99612e`,
identical to its predeclared acceptance hash.

[Independent controls review](raw/native-dask-independent/gpu-controls-review.json)
checks each actual trial's source/assets/data before and after, request and command,
resolved Hydra YAML, instrumentation, process-local seeds and hash-seed roles, loaded model
module paths, real worker settings, HTTP receipts, telemetry and shutdown. The execution
order was A1/B1/A2/B2; comparison order is A1/A2/B1/B2. Every trial used the same first four
LundDoor images at resolution 518, `vggt_sift_frontend_megaloc`, Olsson loader, pinned warm
weights, fresh pipeline cache, Python/NumPy/Torch seeds 42, and no splat stage. Resolved
algorithm YAML differs only at output_root. Python hash seed is 42 in runner/resource-tracker
and Dask's 6640 in the worker, consistently in both arms.

All four real workers report one worker, two threads, 19,327,352,832 bytes (18GiB), CUDA
visibility `0`, daemon false and the shared scratch root. The native arms' conflicting
three-worker/one-thread/4GiB form values are ignored as intended. Instrumentation records
actual CUDA allocations and the expected VGGT/MegaLoc source modules. Both arms use the
same authenticated loopback HTTP transport: unauthenticated requests return 401 and actual
job POSTs return 202. All jobs complete with exit 0. Sampled GPU memory peaks range from
5,596 to 5,728MiB and temperature peaks from 36 to 37C; these are observations, not GPU
memory-limit guarantees. The cgroup CPU/RAM bounds are recorded in each telemetry stream.

All observed owned job PIDs are explicitly absent after exit, independent of epoch
creation-time comparisons. Each API thread stops, its port closes, its systemd unit is
unloaded, and the final GPU compute list and owned-unit list are empty. Source, installed
packages, pinned model assets and input hashes do not change. The source snapshot's Git
provenance remains `012b79c8`; coordinator subsequently committed matching runtime content
as `70a55d48`. Tested content is identified by retained file hashes, not an assertion that
the remote checkout had the later commit.

The independent [MIRA lifecycle review](raw/native-dask-independent/mira-lifecycle-review.json)
retains a separate post-series audit failure: the unrelated service has the same
PID/PPID/command, but its psutil epoch create_time differs by 135 seconds. Two subsequent
kernel snapshots show identical start ticks and boot ID while kernel btime and psutil
create_time both advance by three seconds, supporting clock-derived timestamp movement.
Systemd reports the service active with zero restarts. Because pre-run kernel identity was
not captured, whole-series exact MIRA lifecycle identity remains **INCONCLUSIVE**; strict
before/after epoch equality remains **NOT VERIFIED**. These facts do not erase the failed
audit or weaken the owned-job PID-absence checks.

This establishes exported sparse reconstruction equivalence for the controlled four-image
case and current configuration path. It does not establish internal tensor bitwise identity,
large-scene scaling, general accuracy equivalence, speedup, GPU-memory enforcement, live
Modal/SSH deployment, or a single observed browser-to-GPU workflow. Historical artifacts
and frozen acceptance/manifests were not changed.

The [independent evidence index](raw/native-dask-independent/README.md) distinguishes final
passes from retained intermediate failures; [manifest.json](raw/native-dask-independent/manifest.json)
hashes the independent artifacts, report and CPU harness without hashing itself.
