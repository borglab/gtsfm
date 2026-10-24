# Independent Dask profiles acceptance experiment

## Material Passport

- Origin skill: academic-research-suite / experiment-agent, plan and validate.
- Date: 2026-10-24.
- Verification status: independently VERIFIED for API/configuration, CPU reporting,
  actual persisted-profile CUDA execution, process/GPU cancellation cleanup, and exact
  exported sparse reconstruction equality on the declared scene. Speedup is INCONCLUSIVE.
- Version: dask-profiles-independent-v1.
- Orca Run: `run_936455410ccd`; verifier Dispatch: `ctx_c6e800ee03fa`;
  implementation Dispatch: `ctx_26544700eb86`; UI Dispatch: `ctx_55b24aa1acce`.
- GPU execution ownership update at 19:40 UTC: root message `msg_74b745aa975e` assigns
  dedicated execution Dispatch `ctx_4a5d065da4ec`; lead cedes compute in `msg_08d47bd77cb0`.
  The unchanged final protocol was sent to that peer as `msg_68802c076cff` before its series.
- Repository baseline: `3306994e2903a3fc65b01f952e1b5d7bc248cfe4`.
- Root scope decision `msg_c619c5019a6d`: this milestone accepts `execution_mode=local`
  only, on each Studio execution host. Remote-profile combinations must fail before
  dispatch; the existing manual remote/Modal path remains available. Running Studio
  locally on the existing remote3080 host is in scope. New propagation infrastructure is not.
- Scope: host-local execution configuration, bounded local CPU probes, and independent
  analysis of raw remote evidence supplied by the authorized execution worker. Initial
  preflights belonged to the implementation lead; subsequent trials belong exclusively to
  the root-authorized dedicated GPU peer. The verifier launches no remote jobs and edits no
  GPU harness/raw artifacts. No new paid resources or public services.

## Final independent verdict (2026-10-24)

| Claim | Verdict and scope |
| --- | --- |
| Profile CRUD survives fresh server processes | VERIFIED through actual HTTP and the production shared-store injection path |
| Manual/profile argv, environment and effective algorithm config | VERIFIED for CPU probes and actual CUDA trials; only declared output/evidence paths differ |
| Strict validation and incompatible run controls | VERIFIED on latest source: 23 invalid profile cases, corrupt store operations, unavailable GPU before launch, local-only boundary and conflicting controls including mismatched profile aliases |
| Concurrent persistence | VERIFIED in this bounded POSIX/local-filesystem run: 16/16 successful names retained within one server, and 16/16 across two servers, after restart |
| CPU Dask reporting | VERIFIED on latest source: 2 workers, 2 threads each, 512MiB each, correct scratch/PIDs; unset CUDA env is null and explicitly empty stays empty; weak-cache and process cleanup verified |
| Actual persisted profile to runner/CUDA worker | VERIFIED: one worker, two threads, 18GiB host limit, requested scratch, explicit CUDA device 0, actual MegaLoc and VGGT CUDA execution |
| Cancellation process/GPU cleanup | VERIFIED for actual JobManager cancellation during active work: runner exited -15, all 37 observed identities absent, empty GPU compute list; semaphore shutdown warning retained, all IPC cleanup not claimed |
| Matched MegaLoc/VGGT exported sparse reconstruction | VERIFIED: A1/B1/A2/B2 camera/image/point/named-pair files byte-identical, baseline repeatability passes; F1 and latest-source F2 separately match A1 |
| Generic observation-only matching with duplicate identities | INCONCLUSIVE for nonidentical ambiguous exports; 77 duplicate groups here. Exact-byte case preserves every original point ID and all 1509 tracks, with unchanged tolerances |
| Algorithmic speedup | INCONCLUSIVE: two repeats per arm, opposing paired directions, polling/copy/cache variability; no speedup claimed |
| Hard GPU VRAM cap / broader accuracy or scaling | Not provided or established by this milestone; host-memory limit is distinct from VRAM |

Durable independent evidence is indexed with SHA256 values in
[manifest.json](raw/independent-verifier/manifest.json). Key results are
[latest API config](raw/independent-verifier/config-run-9-current/configuration.json),
[four-arm comparison](raw/independent-verifier/gpu-four-trial-comparison.json),
[final-source confirmation](raw/independent-verifier/gpu-final-source-confirmation.json),
and [control/provenance audit](raw/independent-verifier/gpu-controls-audit.json).
The latest actual CPU reports cover
[explicitly empty](raw/independent-verifier/cpu-current-empty/cpu-worker-report.json) and
[unset](raw/independent-verifier/cpu-current-unset/cpu-worker-report.json) CUDA environments.
All earlier failed reports are retained separately. Root message `msg_7e03c67b9344`
authorized copying these independent artifacts beyond the original two-file boundary;
GPU harness/raw ownership stayed with the execution peer. Results below are judged by
observed executions and raw artifacts, not successful source edits alone.

The concurrency claim is bounded to the tested POSIX host and local filesystem. Current
source fails closed when `fcntl`/advisory locking is unavailable; Windows-native profile
storage and unsupported filesystems are outside this verified scope. The actual targets
are macOS and Linux/WSL. No untested platform guarantee is made.

## Falsifiable claims

1. A valid named execution profile survives a **fresh server process** using the same
   storage directory, and loading it resolves to the same effective runner arguments,
   environment, and algorithm configuration as its equivalent explicit manual controls.
2. Malformed or incompatible settings fail before launching a reconstruction. Positive
   integers must not silently clamp; unknown fields must not silently disappear.
3. A real Dask cluster obeys the requested number of worker processes, threads per
   worker, host-memory limit, scratch directory, and GPU visibility. A successful
   configuration parse is insufficient. GPU visibility does not establish a hard VRAM cap.
4. On the fixed tested scene, switching equivalent manual controls to a persisted profile
   preserves image/pair/track identities and exported geometry within the tolerances below.
   Counts, mocks, or the implementer's assessment do not establish this claim.
5. Any timing benefit is a separate descriptive observation, not a prerequisite or an
   assumed consequence of reusable profiles. No general speedup claim is predeclared.

Each claim receives `VERIFIED`, `NOT VERIFIED`, or `INCONCLUSIVE`. Missing baseline,
incomplete artifacts, failed execution, or invalid controls make the relevant claim
inconclusive. A completed valid comparison that misses its threshold is not verified.

## Configuration acceptance protocol

Use a temporary isolated results/profile directory. Start a loopback-only Studio process,
create a profile through its public API, read it back, update it, stop that process, then
start a distinct process with the same directory and retrieve the updated profile. Record
both server PIDs, payloads, status codes, persisted bytes, and returned profile content.
Delete the disposable profile after testing. Never touch the user's saved profiles.

For manual/profile equivalence, hold dataset, loader, model, overrides, output directory,
seed, GPU, and cache policy fixed. Compare the effective argument vector and environment
exactly; if the profile uses a separate CLI flag, expand it through the supported runner
resolution surface before comparing. Preserve the original argument vectors too. Compare
resolved Hydra configuration after excluding only documented output/provenance paths.
No broad normalization may erase an execution or algorithm difference.

Exercise the public invalid-config surface with zero, negative, fractional, boolean, and
malformed numeric-string worker/thread values; zero/negative/nonsense memory limits; unknown mode;
duplicate/negative/nonexistent GPU indices; CPU/GPU-incompatible settings; nonexistent or
unwritable scratch directory; unsupported fields; missing profile ID; conflicting legacy
cluster config and profile. Record every request and response. Host-dependent rejection
cases are applicable only on the host where they can be tested. Require a clear 4xx error
or CLI error and no child reconstruction process.
Integer strings used by HTML form fields (for example `"2"`) may normalize to integers
under the agreed schema; this is distinct from accepting `"1.5"`, `1.5`, or `true`.

Run the supported real probe on CPU locally and one GPU worker remotely. Inspect scheduler
state **and worker-executed observations**, including PID, nthreads, memory limit bytes,
local directory, hostname, `CUDA_VISIBLE_DEVICES`, torch CUDA availability/device count/name,
and a tiny CUDA allocation where requested. Check requested versus observed values. Verify
all worker PIDs exit after the probe. For job cancellation, start a bounded real job, cancel
through the API, and record process-group/worker exit; a `cancelled` response alone does not
prove cleanup. Ensure the dashboard does not expose a public interface.

## Reconstruction experiment declared before outputs

Order: **A1 manual, B1 persisted profile, A2 manual, B2 persisted profile**, sequentially on
the existing RTX 3080. Use the same fixed available scene/subset (prefer the existing
12-image Lund Door), MegaLoc/VGGT pipeline, source snapshot, hardware, software, seed,
resolution, preprocessing, partitioning, worker/thread counts, memory limits, and GPU IDs.
Any difference must be recorded and justified before a numerical comparison is accepted.

The historical Git baseline establishes the pre-feature manual interface. The matched
runtime experiment isolates manual versus profile resolution on the same implementation
snapshot; it does not by itself establish equivalence to an older source revision.

Warm model downloads are allowed only with the same model files for every trial. Record
their hashes/revisions and exact cache locations. Bypass or separately isolate descriptor,
matcher, and reconstruction-result caches for every measured trial, recording how this is
done. Reusing frontend result caches can support a narrower resume/configuration claim,
but cannot verify fresh MegaLoc/VGGT inference equivalence or a full-pipeline speedup.

Retain each trial separately, including failures and warmups. Never replace a failed trial
silently. No automatic retries or extra GPU work by the verifier. Keep bounded timeouts
and the existing resource limits; collect complete logs, stage and total wall time, sampled
process-tree RSS, GPU used/allocated/reserved memory where available, temperatures, and
final process/GPU state. System GPU memory is not PyTorch peak allocated memory.

### Required manifests and artifacts per trial

- Trial ID, arm, start/end time, exit code, timeout/cancellation status, exact argv and env.
- Input relative names, sizes, SHA256 values, selected-image list and ordering.
- Source-file SHA256 manifest and Git revision/dirty diff when available. The remote source
  copy has no `.git`; never label its revision verified from a local Git hash alone.
- Package versions, Python executable/version, driver/CUDA versions, GPU identity and RAM,
  CPU/RAM/thread settings, model-weight hashes/revisions, seed and deterministic settings.
- Raw requested/resolved profile, actual worker observations, resolved Hydra configuration,
  cache policy/directories, full logs, all metrics, telemetry and cleanup observations.
- Final `cameras.txt`, `images.txt`, `points3D.txt` from the same pipeline output stage and
  `plots/similarity_named_pairs.txt`. Preserve retrieval pair names, not just counts.

### Fixed comparison rules

The baseline repeat A1/A2 must first satisfy these same thresholds. If it does not,
reconstruction equivalence is **INCONCLUSIVE**, and tolerances are not enlarged after
seeing results. Compare B1 and B2 individually to A1 and to each other. Exact equality of
deterministic manifests, selected image order, retrieval pair identities, registered image
names, camera model/dimensions, and track observation identities is required. Point IDs and
camera IDs may be renumbered: canonicalize via image filenames and each track's sorted
`(filename, x, y)` observations at the export's 0.001-pixel precision. Track-count equality
without observation equality does not pass. Unobserved dense points require a separate
identity rule and are outside this sparse-track comparison.
Matched sparse-track RGB colors must also agree exactly.

Fit one orientation-preserving similarity transform from treatment camera centers to
baseline centers. Require at least three distinct noncollinear centers and nonzero baseline
camera bounding-box diagonal D. Use the same transform for cameras, rotations, and all
matched points; never align each track separately.

| Quantity | Maximum permitted difference |
| --- | --- |
| Intrinsic parameters | `atol=1e-6`, `rtol=1e-6`, per parameter |
| Aligned camera center | `1e-5 * D`, every camera |
| Aligned camera orientation | `0.001` degree, every camera |
| Aligned matched sparse point | `1e-4 * D`, every point |
| Exported mean-track reprojection error | `0.011` pixel, every track (2-decimal export) |

Exported observations and error fields are rounded, so this verifies exported sparse
reconstruction equivalence at the stated precision, not bitwise internal-state equality.
Keep exact file hashes as an additional stronger result if they match. Record the maximum
and distribution summaries, identity differences, fit scale/rotation/translation, and any
nonfinite or malformed values; nonfinite values never pass.

## Timing analysis

Report all four wall/stage times and memory peaks. With only two repetitions per arm,
report individual values, medians/ranges, and paired ratios descriptively; make no
significance or broad performance claim. Different cache states, hardware, work completed,
outputs, or concurrency invalidate a causal speed comparison. Durability and early error
detection are the intended primary improvement.

## Evidence and review ledger

- Initial source inspection: baseline `visualization.runtime.build_runner_args` clamps
  worker/thread values using `max(1, int(...))`; the invalid matrix must distinguish early
  rejection from this previous behavior. This is source evidence, not a runtime test.
- Baseline `GtsfmRunner._create_dask_client` uses `LocalCluster` for manual local controls;
  legacy CUDA cluster YAML selects `LocalCUDACluster`. No multi-host or pooled-VRAM claim.
- Baseline exports COLMAP text with observations rounded to 0.001 pixel and reprojection
  errors rounded to 0.01 pixel (`gtsfm/common/gtsfm_data.py`); those precisions bound the
  comparison. Similarity retrieval writes named-pair diagnostics.
- Memory was consulted only for prior architecture/SSH cautions; source behavior above was
  freshly inspected. No previous timing or hardware measurement is reused as current proof.
- Contract agreement, verifier execution, raw reconstruction comparison, defect review,
  and final per-claim verdicts: pending.
- Disposable initial-module probes confirmed boolean/fraction coercion and ignored unknown
  fields/future schema versions; malformed stored entries were skipped by listing. Raw
  evidence: `/tmp/verify-this/dask-profiles-independent/initial-source-probes.json`.
  These were reported to the implementer while code was changing and require a final rerun.
- The pinned baseline's actual argument builder accepted `num_workers=0` and
  `threads_per_worker=-2`, producing `1` for each. Raw evidence:
  `/tmp/verify-this/dask-profiles-independent/baseline-invalid-clamping.json`.
- The isolated CPU verification environment imports the real Studio API. Its packages are
  lightweight cached dependencies, not the full GPU reconstruction environment; this is
  startup evidence only.

## Reproduction commands and interim observations

The lead agreed to the CRUD API and injected real temporary `ExecutionProfileStore`
contract through Orca message `msg_b1ca74631a5b` before verifier creation. The verifier
creates actual loopback HTTP servers and binds both API and JobManager to that same
temporary store. This checks the supported injection path; it does not establish a
separate default-home storage binding or a launched reconstruction.

```bash
python scripts/verify_dask_profiles.py config --artifacts /tmp/dask-profile-acceptance-fresh
python scripts/verify_dask_profiles.py compare \
  --models /evidence/A1/model /evidence/A2/model /evidence/B1/model /evidence/B2/model \
  --pairs /evidence/A1/similarity_named_pairs.txt /evidence/A2/similarity_named_pairs.txt \
          /evidence/B1/similarity_named_pairs.txt /evidence/B2/similarity_named_pairs.txt \
  --output /evidence/independent-comparison.json
```

`config` needs Studio's lightweight dependencies; `compare` needs NumPy. The artifact
directory must be fresh to avoid overwriting saved state. Missing files, malformed exports,
ambiguous track identities, nonfinite geometry, and degenerate camera alignments fail visibly.
The four model directories must be the same final output stage; the script does not guess
among intermediate models. Commands above illustrate the interface, not completed GPU work.

Interim actual CPU evidence (`config-run-2/configuration.json` under
`/tmp/verify-this/dask-profiles-independent/`, 2026-10-24):

| Claim | Interim verdict | Evidence |
| --- | --- | --- |
| API create/update/read/delete across fresh server processes | VERIFIED, isolated injection scope | Three distinct server PIDs, returned updated profile equality, missing after delete/restart, server exit codes retained |
| Exact manual/profile argv and environment | VERIFIED, one CPU profile | Same dataset/output paths, MegaLoc/VGGT preset, two workers, three threads, 512MiB host memory, scratch path, CPU visibility |
| Invalid schema rejected | NOT VERIFIED in initial module | Booleans/fractions, unknown field/future schema, malformed/negative/duplicate GPU IDs accepted (nine cases) |
| Corruption rejected without mutation | NOT VERIFIED in initial module | Broken JSON refused; future version and null profile entry accepted or masked |
| Actual Dask worker contract / GPU / cancellation | INCONCLUSIVE | Production probe and raw remote evidence not yet received |
| Real reconstruction equivalence / speedup | INCONCLUSIVE | No matched real output trial set received |

The implementation was changing; these are actionable interim findings, not final defects
if subsequently fixed. Final verification must rerun against stable source hashes. The
first config report precedes the later immutable request-payload capture fix; the second
retains exact create/update payloads. Synthetic comparator self-checks under
`comparator-selfcheck/selfcheck.json` pass a known global Sim3 plus renumbered IDs and reject
a shifted point or observation. They establish comparator behavior only, not reconstruction
accuracy, equivalence, GPU execution, or performance.

Additional integration finding (pending fix/review): current
`visualization/runtime.py:1354` copies the originating profile lookup IDs into a remote
request and `visualization/runtime.py:1383` changes its target to local. The remote host
then looks up that host-local name in its own store. A no-network two-store public-function
repro produces `Execution profile not found` with an empty destination store, and
`requires a remote execution target` even with a copied same-name remote-mode profile.
Evidence: `/tmp/verify-this/dask-profiles-independent/remote-host-local-name-repro.json`.
The implementer and root coordinator received the finding through Orca. This is a
source-backed resolution repro, not an externally deployed remote-API test.
The root subsequently bounded the milestone to local-only profiles, so this finding is
resolved for milestone acceptance by clear early rejection of remote profile combinations,
without introducing propagation. The verifier checks this rejection and verifies an
ordinary manual remote request still reaches URL validation. That negative validation
check does not claim successful external remote execution.

### Subsequent independent checks (still awaiting final rerun)

Stable-source `config-run-3/configuration.json` verifies fresh-process persistence, exact
argv/environment, all 20 invalid profile cases, corruption refusal through list/save, and
the local-only remote boundary. Root-requested actual concurrent HTTP PUTs exposed a store
race: all **16** distinct-name writes returned 200, but **14** accepted names were absent
both immediately and after restart. This is a failed persistence requirement, not timing
noise. `os.replace` prevents partial JSON but does not serialize read/modify/write.
The lead received exact missing names and a reproducible config command. Later verifier
coverage also checks get/delete corruption, boolean run controls, and incompatible
cluster/input-worker overrides without launching reconstruction (missing dataset is a
fallback guard).

Actual bounded CPU telemetry probe:
`/tmp/verify-this/dask-profiles-independent/cpu_probe.py`, raw report
`cpu-run-1/cpu-worker-report.json` in that same artifact root. One actual `LocalCluster`
uses two worker processes, two threads each, 512MiB host-memory limit each, loopback host,
CUDA visibility empty, dashboard disabled, and an isolated temporary scratch directory.
It compares production `dask_stats.collect` against independent `Client.run` observations
using the supported `dask_worker` injection, and creates/closes three clients.

- VERIFIED in this bounded CPU/reporting scope: two workers, four total threads,
  536870912-byte limit per worker, both worker PIDs, empty CUDA visibility, and both worker
  PIDs gone after cluster closure. No runner-to-cluster propagation claim follows from
  constructing this explicit probe cluster.
- NOT VERIFIED in the initial telemetry change: production `local_directory` is null for
  both workers, while worker-injected observations return the real scratch paths.
  `_worker_probe` uses `get_worker()` outside task context under `Client.run`; it needs
  the supported injected worker object. Three closed clients also leave three entries
  in the global integer-client-ID cache, so that cache has no observed cleanup.
- Source review finds `cuda_memory_allocated` and `cuda_memory_reserved` in that static
  per-address cache. They cannot represent current memory throughout a run if never
  refreshed. Actual dynamic GPU allocation was not tested by the verifier; it must not
  be reported as a measured GPU failure. Remove those dynamic fields from static identity
  or sample them freshly with a timestamp.
- The CPU environment has lightweight Studio/Dask dependencies and no Torch; its reported
  CUDA unavailable result is not a test of the remote GPU environment. A direct local full
  runner import is unavailable because this minimal environment lacks Plotly and the full
  reconstruction dependency stack. This is an environment limitation, not an algorithm failure.

All defects above were sent to both lead and root through Orca. GPU/reconstruction and
actual Studio job cancellation remain pending raw implementation-host evidence.

### Current bounded results after fixes

Stable `config-run-5/configuration.json` independently passes single-process concurrent
persistence (16/16 successful saves survive restart), strict schema rejection, corruption
through all four store operations, conflicting run controls including booleans, local-only
remote boundary, fresh-process persistence, and exact argv/environment. No changed
algorithm configuration or wall-time claim is inferred from those checks.

Stable actual CPU `cpu-run-2/cpu-worker-report.json` verifies the telemetry fixes: both
reported scratch paths exactly match injected worker observations; 2 workers, 2 threads
and 536870912-byte limit each; PIDs 42036 and 42037 both gone after close. Three cached
clients become zero cache entries after client references are dropped and collected.
Dynamic CUDA allocated/reserved values were removed from static identity in the source;
the CPU probe does not measure GPU allocation or a hard VRAM cap.

Expanded, stable `config-run-6/configuration.json` also verifies rejection of an existing
regular file as scratch and rejection of unavailable `cuda:99999` before child launch.
The two remaining invalid-matrix failures are an unwritable existing scratch directory
and a new nested path below an unwritable parent; both actual API requests still succeed.
The verifier uses its own temporary directory with mode 0555 and restores permissions.
This is part of the original scratch acceptance criterion. Nonexistent directories may
legitimately be created under an agreed writable-parent policy; the initial protocol's
existence wording is clarified to distinguish a new creatable path from an unusable path.

Root-requested two-process audit, stable source:
`cross-process-run-1/cross-process.json` under the same artifact root. Two actual Studio
server processes share one temporary store; 16 distinct-name concurrent PUTs all return
200, but 8 accepted names are absent immediately and after a third fresh server process
reads the store. All three server exits and source hashes are retained. Cross-process
lossless persistence is **NOT VERIFIED**, even though the single-process threadpool test
passes. The root and lead received the exact repro; acceptance requires an interprocess
transaction lock or an explicit enforced single-writer scope with narrowed claims.
The same `config` command now reproduces both single-server and two-server batches, using
distinct profile names and the same temporary store, and reports each separately. Its
invalid matrix includes 23 cases, including file/unwritable-directory/unwritable-parent
scratch. All API/server URLs, PIDs, exit codes, immutable payloads and source hashes are
retained. Previous reports are not overwritten.

`config-run-4` is a separately retained INCONCLUSIVE import attempt during an incomplete
shared-source edit; it is neither a measured algorithm failure nor a final source defect.

Root-owned real-browser evidence was read independently from
`/tmp/gtsfm-profiles-fullstack/ui-result.json`: actual HTTP responses show canonical profile
create at 2 threads, update to 3, fresh GET retaining 3, and a 400 response for malformed
memory. The root records successful Chrome draft/reload/error display checks with no
mocked API and stable source hashes during that run. The verifier did not rerun those
browser interactions and does not count this as GPU, reconstruction or cancellation proof.

## Independent review of implementation-host preflight raw files

Directly read `docs/benchmarks/raw/rtx3080_nvidia_smi.txt`,
`rtx3080_torch_probe.txt`, and `rtx3080_dask_probe.txt` after they appeared in the shared
repository. The single-line hardware query reports RTX 3080, 10240 MiB total, 317 MiB used
and zero utilization. Torch output reports version 2.7.0+cu128, CUDA available, one RTX 3080,
successful CUDA tensor sum 65536, and 262144 allocated / 2097152 reserved bytes.

The standalone guarded Dask JSON reports requested one worker / one thread / 1GB, and
observed one process worker PID 303756, one thread, 1000000000-byte memory limit, actual
scratch path, CUDA available/device count 1 and RTX 3080 identity. These observations are
consistent with a real accessible GPU worker. They do not show that a saved Studio profile
produced that cluster, nor that MegaLoc/VGGT inference completed.

The worker's `cuda_visible_devices` is an empty string while CUDA is available; the probe
implementation uses an empty default for an absent environment variable, so this raw field
cannot distinguish unset (host default visibility) from explicitly empty (hide devices).
GPU metadata establishes accessibility, not selected-device isolation. Worker-local tensor
execution or actual model inference remains a separate requirement.

The raw Dask file contains no post-close PID check, process exit receipt, or post-run GPU
process snapshot. The implementation document's clean-shutdown sentence is therefore an
author assertion at this point, not independently verified cleanup. The known failed stdin
spawn harness attempt and pinned VGGT source import are described by the implementation
ledger; no reconstruction or geometry result is inferred from either. Root and lead received
these evidence gaps through Orca before final review. Full matched raw trials were pending
at that preflight checkpoint; completed results follow.

## Completed matched experiment and final-source confirmation

The independent verifier read the execution peer's complete raw artifacts and frozen
harness, not just its outcome summary. The harness SHA256 is
`313bcda0fcb3dcfd65839f7e5fe2beb6ec1ec406608d78844b0d909590d8fd1d`.
Its instrumentation seeds and records Hydra/CUDA/module provenance; it does not replace
model inference. The [series declaration](raw/gpu-experiment/series-declaration.json)
was frozen before A1 outputs. Its 852-file source manifest hashes to the declared
`362ebc3e92ce628f8ca2c6e953857c2782ff9e1c432955b4fba4e606d83b609d`.
The post-series check reports no source or harness drift. Final checks also retain unchanged
input names/sizes/SHA256, both model-weight SHA256 values, and all 157 DINOv2 source hashes.
The remote source copy has no Git metadata; these manifests identify the actual run source.

All four runs use the first four real Lund Door JPEGs, EXIF calibration without `data.mat`
or ground-truth poses, the actual MegaLoc/SIFT/vanilla VGGT/GTSAM preset at 518 resolution,
crop input and default BF16 on the RTX 3080. Pipeline result caches start empty every trial;
model weights are fixed and warm. Driver Python/NumPy/Torch seeds are 42; actual worker
Python hash seeds are Dask's 6640, consistently by role across arms. One worker, two threads,
18GiB **host** memory, explicit CUDA device 0 and loopback dashboard are observed in every
trial. Actual cgroup limits are 24GiB and four CPU cores. All four successful trial processes,
workers and resource trackers are absent afterward; GPU compute-app lists are empty.

Independently compared contracts and resolved Hydra configurations differ only in the
declared trial output root; captured execution-related environments differ only in
evidence/live-output paths.
Actual worker logs show MegaLoc and VGGT modules and CUDA allocations. The configured
profile is read from persisted bytes and goes through real `JobManager.start` in B1/B2.
This closes the earlier runner-propagation and actual GPU-execution gaps. It is a backend
JobManager execution/cancellation check; the GPU trials do not independently drive an HTTP
job/cancel endpoint or browser run submission.

### Export equality and conservative ambiguity handling

All four final model files and named-pair files are byte-identical across **A1, B1, A2, B2**:

| Artifact | SHA256 |
| --- | --- |
| `cameras.txt` | `dcc89f348d073e7c54e835a4f99dbcc448be1f80e086ccc6dec8b5e5faf46282` |
| `images.txt` | `6720ea33d1c563c815f91cd3083645ed30260955060d912e0b17fed0bdf78543` |
| `points3D.txt` | `def842b8ce19248eed97e5095fd9229ec1115a7ecd1753459e1987319e90dea0` |
| `similarity_named_pairs.txt` | `72000710af4383c65770877a829aff2247fd4bde8f1a5270586d2e60c3be5802` |

There are four nondegenerate registered cameras and 1509 sparse tracks. The first generic
observation-only matcher correctly stopped: 77 groups share rounded observation identities
(78 additional tracks), with no repeated observation within a track. That initial
INCONCLUSIVE result is retained in
[gpu-canonical-ambiguity.json](raw/independent-verifier/gpu-canonical-ambiguity.json).

The verifier was then completed to implement the **predeclared stronger exact-byte rule**:
only when every complete model and pair file is identical does it retain original point
IDs, preserving every track and observation multiplicity. For nonidentical exports, the
original strict observation matcher still rejects ambiguity. No threshold or tolerance
was relaxed and no tracks were dropped. Nonfinite values, bad links, within-track duplicate
observations and degenerate camera geometry still fail. Copied-fixture regression checks
reject changed/nonfinite data and retain the original renumbered-Sim3 behavior; those checks
are tool tests, not extra reconstruction trials.

A1/A2 baseline repeatability and A1/B1, A1/B2, B1/B2 comparisons pass. Maximum residuals
from fitting an identity-equivalent Sim3 are camera-center `2.54e-17 * D`, point `3.57e-15 * D`,
rotation `2.42e-6` degree and reprojection delta zero. These tiny residuals are numerical
alignment noise; complete serialized values are identical. This verifies exported sparse
reconstruction equality at the output precision, not all internal tensors or absolute
ground-truth accuracy, and it does not establish other-scene or large-scale behavior.

F1 and F2 are kept **outside** the frozen four-arm series. F1 tests the first two metadata/
validation changes; F2 tests the latest telemetry fallback plus profile alias validation.
Their source declarations differ from the frozen manifest only in `dask_stats.py` and
`execution_profiles.py`. Latest F2 hashes are `524f5100...` and `b7623a40...`, respectively,
and match the latest independent CPU/API probes. Both confirmations exit zero, match A1's
complete export hashes and effective algorithm config, and leave no observed PIDs or GPU
compute apps. F1 is retained rather than being overwritten when the final telemetry edit
arrived. Neither confirmation is pooled into performance repetitions.

### Timing, memory and cancellation scope

| Trial | Arm | Reported harness seconds | Pipeline metric seconds | Sampled GPU used peak, MiB |
| --- | --- | ---: | ---: | ---: |
| A1 | manual | 47.725 | 31.3387 | 6708 |
| B1 | saved profile | 44.915 | 28.9453 | 5586 |
| A2 | manual | 43.332 | 28.7758 | 5596 |
| B2 | saved profile | 43.653 | 25.0509 | 6796 |

Reported harness time is wall-clock elapsed minus a fixed seven-second passive post-exit
settle; it includes launch, polling and copying overhead and is not pure compute time.
Manual median/range is 45.528 / 43.332–47.725 seconds; profile median/range is
44.284 / 43.653–44.915. Paired profile/manual ratios are 0.941 and 1.007, with opposite
directions. With two repetitions per arm, fixed order and cache/poll/copy variability,
**speedup is INCONCLUSIVE**. No causal performance or memory improvement is claimed.

Every measured run's recorded Torch peak allocated value is 5044550144 bytes. Process-tree
RSS sums include shared pages; cgroup peaks include charged descendants and page cache;
sampled GPU used memory is a different counter from Torch allocated/reserved memory. The
12-image component preflight's reserved counter (14856224768 bytes) must not be described
as physically resident VRAM on the 10240MiB device. The four-image scene was selected for
safe integrated trials before the measured declaration. The empty first three-image
preflight attempt and successful attempt2, other preflights, warmup, cancellation and final
source confirmations are preserved separately, not silently substituted into the four arms.

Actual `cancel1` starts from persisted profile bytes, reaches active MegaLoc CUDA work,
then calls `JobManager.cancel`. The reviewed harness performs passive settling and checks,
with no successful-path cleanup sweep. Runner return code is -15; all 37 observed process
identities are absent and the GPU compute list is empty, confirmed again by final per-PID
checks. Shutdown reports six semaphore objects for resource-tracker cleanup, so this is
**process/GPU cleanup**, not a claim of warning-free shutdown or independently inspected IPC
cleanup. Final GPU state is 201MiB, 31C, zero utilization, no compute apps; no experimental
systemd units remain. The verifier itself launched no remote GPU work.

### Statistical interpretation audit

All 11 methodology categories were considered. Simpson's paradox, collider adjustment,
base-rate neglect and reverse causality have no estimator here. Ecological and Berkson
inferences are avoided by restricting conclusions to this capacity-selected scene and host.
Regression-to-mean/order effects prevent a speedup inference; survivorship is addressed by
retaining failures/warmups separately; look-elsewhere and forking-path concerns are bounded
by unchanged predeclared tolerances and disclosed exact-byte handling; correlation is not
promoted to a causal timing claim. No significance test, population confidence interval,
general accuracy claim or broad speedup claim is made.
