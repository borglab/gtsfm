# Mac-hosted Studio → SSH computation verification

## Acceptance and observed results

Studio stays on the Mac. A named execution configuration populates editable
controls and launches remote computation without per-run remote-terminal steps.
The destination checkout/environment are installed once.

| Requirement | Observed evidence |
| --- | --- |
| Local named configurations | Actual browser save, edit, reload and page reload; persistence tests cover separate processes and concurrent delete/save. |
| Controls determine execution | Real worker observations verify 1 worker, 2 threads, 18 GiB memory, CUDA device 0 and the selected scratch directory. |
| Automatic startup and input staging | Mac HTTP → OpenSSH → Windows → WSL2 → RTX 3080, using four locally uploaded Lund Door images. The acceptance runs use real HTTP and reconstruction. |
| Logs and local results | Actual browser logs, downloaded complete exports, 4 reconstructed cameras and 1,509 tracks; all 12 photo requests across three result scenes succeed after computation. |
| Process ownership | Running-job cancellation was tested before the master merge. A final-source SIGTERM test during VGGT computation verifies the Mac server, owned SSH connections, remote service and GPU process all exit. |
| Native Dask cluster | Real SSHCluster scheduler and two workers over loopback SSH, including runner/Hydra and device-environment propagation. Physical multi-host SfM was not run. |

After integrating master `adc793547b63374a89cd6215f7edeb901d4569e9`,
**152 focused Python tests**, **19 viewer tests**, **9 browser tests**, the
frontend type check/build, and the exact repository CI lint check passed.
The browser regression suite uses provider fixtures; it is separate from the
real RTX acceptance run. Review fixes cover credential isolation, saved controls,
stale asynchronous updates, worker device binding, shutdown, history/retry,
photo identity/cache handling and cross-process store locking.

## Controlled reconstruction comparison

Four sequential trials alternated unmodified master and the implementation:
**E1 baseline → F1 treatment → E2 baseline → F2 treatment**. F1 used the real Mac
browser to save/reload settings, upload inputs, run, inspect logs and view results.
The other trials used the actual job APIs. No reconstruction outputs were mocked.

Both arms used the same four image bytes, pinned model assets, RTX 3080 (10 GiB),
Python environment, seeds (Python/NumPy/Torch 42), Olsson loader, resolution 518,
`vggt_sift_frontend_megaloc`, exact `sift_brute_force` matching and no splatting.
Each source's pipeline cache was emptied before each trial; model weights stayed
warm. The effective result configurations matched after normalizing input/output
paths. Worker observations, rather than request fields alone, checked settings.

Master ignored the correspondence-preset flag for the cached VGGT optimizer.
The baseline therefore used the equivalent explicit Hydra matcher override;
the treatment selected the preset normally. The implementation now applies
frontend presets before constructing cached or uncached optimizers, so displayed
selections reach the effective pipeline. Defaults are unchanged.

All **10 complete camera/image/point/pair export hashes matched across all four
trials**. A subsequent final-source smoke run F3 matched the same ten hashes.
Every trial produced 4 cameras and 1,509 final tracks. Representative final hashes:

| Export | SHA256 |
| --- | --- |
| `merged_retriangulated/cameras.txt` | `7f566f2cfe6e422c17a7df4ed3294916c864bdec590b482471b02c5b15279989` |
| `merged_retriangulated/images.txt` | `3d842782b00a5f4d96767e0f1305efa6e13fc6aee0426b75ee9841f1ec1292ea` |
| `merged_retriangulated/points3D.txt` | `d46c90b30018798994c6e0b1e9866015d0d14f730ff990cb63477d0f345b0bba` |
| `similarity_named_pairs.txt` | `72000710af4383c65770877a829aff2247fd4bde8f1a5270586d2e60c3be5802` |

Final audit: all **341 production-source files** in the frozen manifest match
between Mac and remote snapshot; all **174 pinned asset hashes** match their
reference. All five worker guards pass. The shared remote environment was left
unchanged; both arms used the same isolated overlay for current-master
`poselib==2.0.5` and `tornado==6.5.9`, with Dask/distributed 2025.9.1 and
Torch 2.7.0+cu128.

## Resource controls and limits

Each trial ran inside a systemd user cgroup: 24 GiB RAM, no swap, 4 CPU cores,
900-second wall cap. A 1 Hz supervisor stopped on temperature ≥80°C, sampled GPU
memory ≥9,700 MiB, available host memory <4 GiB, or elapsed time >840 seconds.
All five trials exited successfully with no cutoff.

| Trial | Sampled peak GPU MiB | Peak temperature °C | Cgroup peak GiB |
| --- | ---: | ---: | ---: |
| E1 | 6,178 | 34 | 20.51 |
| F1 | 5,554 | 39 | 12.62 |
| E2 | 9,406 | 40 | 12.61 |
| F2 | 6,970 | 38 | 12.60 |
| F3 | 6,486 | 37 | 20.54 |

PyTorch peaks were 4.70 GiB allocated / 8.48 GiB reserved in every trial.
These differ from sampled total-device VRAM. Cgroup peaks include file cache;
summed process RSS also double-counts shared pages. Sampled peaks may miss spikes.

This establishes no observed reconstruction-output regression on this four-image,
exact-matcher fixture. It does not establish universal accuracy, large-scene
scalability, Slurm/PBS support or physical multi-host performance. Browser-arm
elapsed time includes interaction, so no speedup or timing-equivalence claim is
made. General default-FLANN output determinism is not established.

## Retained failures and evidence

Initial default-FLANN trials varied even between two runs of unmodified master
(1,508 versus 1,506 tracks). An attempted preset comparison exposed the ignored
VGGT preset flag. Those runs are retained separately and are not counted as exact
acceptance trials. After fixing preset application, the fresh E1/F1/E2/F2 protocol
above used equivalent, verified brute-force settings without loosening comparison.

An earlier pre-merge readiness failure occurred before job submission; its cause
is unproven because stderr was unavailable then. Startup stderr is now retained,
and readiness may retry up to three times before submission. A possibly accepted
job POST is never automatically replayed. The F3 smoke harness omitted finalizing
its report's success flag; an independent `final-audit.json` checks all ten actual
hashes, effective configurations, worker guards, supervisors and cleanup without
altering that original report.

Raw reports, screenshots, source manifests, instrumentation, telemetry and failed
attempts stay outside the PR in `/tmp/gtsfm-studio-ssh-merged-evidence` on the Mac
and the isolated `gtsfm-studio-ssh-merged-20261029` remote directory. Earlier
pre-merge evidence remains in `/tmp/gtsfm-studio-ssh-evidence`; its report is
preserved in commit `27215a40`. Final remote-service and GPU-process checks are
empty. No experiment dumps are added to the implementation diff.

## Reproduce

Follow [Studio execution configurations](../setup/studio-execution.md), install
the compatible environment/model assets once, and save an SSH configuration.
Run `scripts/verify_studio_ssh.py` against a disposable local Studio workspace
and the same dataset. It defaults to `sift_brute_force`, compares complete exports
between repeats, and supports `--reference` for a prior harness report. Freeze
inputs, source, models, seeds, effective settings, cache policy, resource limits
and telemetry for a controlled baseline/treatment study. The CLI complements
the actual browser acceptance test; it does not replace browser coverage.
