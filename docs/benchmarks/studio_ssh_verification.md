# Mac-hosted Studio → SSH computation verification

## Acceptance and observed results

Studio remains on the Mac. A named execution configuration is saved there,
reloaded into editable controls, and used to start computation without per-run
remote-terminal steps. The remote checkout/environment are installed once.

| Requirement | Evidence |
| --- | --- |
| Named local persistence, reload, edits | Actual browser save/reload and server restart; atomic store tests include independent processes and delete/save races. |
| Displayed controls are submitted settings | Browser control assertions plus real Dask-worker observations: 1 worker, 2 threads, 18 GiB host-memory limit, CUDA device 0, selected scratch directory. |
| Remote startup and input staging | Actual Mac HTTP → OpenSSH → Windows/WSL2 → RTX 3080 runs using four locally uploaded Lund Door images. No HTTP or pipeline output mocks in these runs. |
| Progress, logs, results, photos | Actual browser job/log display; complete exports downloaded to the Mac; viewer displays 4 cameras and 1,509 tracks; photographs remain available after the connection stops. |
| Cancellation and process ownership | Actual running-job cancellation and SIGTERM of the Mac Studio server; owned remote service, feature SSH processes and GPU compute processes verified absent afterward. |
| Native SSH worker clusters | Real native SSHCluster scheduler and two separate workers over loopback SSH; runner/Hydra integration and worker environment propagation exercised. Physical multi-host SfM was not run. |

Focused checks: **101 Python tests**, **8 viewer tests**, frontend type check/build,
and repository-style Python lint passed. Two independent agent reviews found
issues in credential isolation, async UI updates, device binding, shutdown,
photo identity/cache behavior and store locking; these were fixed and retested.
The UI isolation probe covers seven branches using intercepted provider responses;
it is separate from the real RTX acceptance runs.

## Reconstruction equivalence protocol

The final repeat pair used the same source snapshot, four image files, model
assets, hardware and settings. The worker observer checked the effective
configuration rather than only the request. Both trials used random/NumPy/Torch
seed 42, a single GPU worker with 2 threads and 18 GiB, image resolution 518,
and `vggt_sift_frontend_megaloc` without splatting. The test launcher applied
24 GiB cgroup RAM, no swap, 4 CPU cores and a 900-second wall limit. Dask/distributed
was 2025.9.1, Torch 2.7.0+cu128, on an RTX 3080 with 10 GiB VRAM.

All **10** camera/image/point/pair export hashes matched between final repeats
`b2eeb12b5505` and `6c341f2ab4e2`. The final reconstruction and named pairs also
matched the archived controlled baseline at immutable revision
[`a6c6e014`](https://github.com/borglab/gtsfm/blob/a6c6e0145d18c3001dedc65294985027fa408a5e/docs/benchmarks/raw/independent-verifier/gpu-export-hashes.json):

| Export | SHA256 |
| --- | --- |
| cameras.txt | `dcc89f348d073e7c54e835a4f99dbcc448be1f80e086ccc6dec8b5e5faf46282` |
| images.txt | `6720ea33d1c563c815f91cd3083645ed30260955060d912e0b17fed0bdf78543` |
| points3D.txt | `def842b8ce19248eed97e5095fd9229ec1115a7ecd1753459e1987319e90dea0` |
| named pairs | `72000710af4383c65770877a829aff2247fd4bde8f1a5270586d2e60c3be5802` |

Local input hashes were unchanged before/after the final pair. All **174 pinned
asset hashes** matched the earlier baseline manifest; production source hashes
matched on the Mac and remote snapshot. Both worker-control guards passed.
Observed PyTorch allocator peaks were about 4.69 GiB allocated / 8.47 GiB reserved;
these are not a whole-process RSS or total-device VRAM measurement.

These are functional and exact-output checks on a four-image fixture. Pipeline
caches were not reset for the final pair, and the baseline was not rerun as a
contemporaneous timing control. No speedup, timing equivalence, general accuracy,
large-scene scalability, Slurm/PBS, or physical multi-host performance claim is made.
Exact exported bytes establish no observed reconstruction-output regression on
this fixture, not a universal no-regression guarantee.

## Failures retained separately

An earlier final-harness attempt (`95dcaf77680e`) failed during service readiness,
before remote submission or reconstruction. A fresh direct readiness check
succeeded. The original stderr was unavailable, so the transport cause is
unproven. The implementation now retains startup stderr and makes up to three
bounded readiness attempts **before submission**; a possibly accepted job POST
is never automatically replayed. This failed probe is excluded from successful
reconstruction trials. Earlier development snapshots also revealed the missing
photo-name mapping and Hydra-container conversion; neither was counted as final
acceptance evidence.

Raw reports, screenshots, instrumentation, failure receipts and the full export
comparison are retained outside the implementation diff in the task's local
`/tmp/gtsfm-studio-ssh-evidence` evidence directory. Remote trials used an isolated
`gtsfm-studio-ssh-20261029` snapshot and left no experiment service or GPU job running.
The earlier baseline's immutable archive remains linked above. No experiment
output dumps are added to this PR.

## Reproduce

Follow [Studio execution configurations](../setup/studio-execution.md), install
the same model/environment once on the destination, and configure a bounded test
interpreter when using a shared GPU. Run `scripts/verify_studio_ssh.py` against a
disposable local Studio workspace, a locally saved SSH configuration, and the
same four-image dataset. It checks all complete export hashes between repeats;
`--reference` compares a prior harness report. Capture seeds, input/model/source
hashes, effective worker observations, resource limits and cache state alongside
any scientific comparison. The CLI check complements the recorded browser test;
it does not replace browser interaction coverage.

Tested production source hashes:

| File | SHA256 |
| --- | --- |
| `gtsfm/runner.py` | `1315320c2c6568e422d493001af3357e13db71b6ed1a7d00d768b14c04c21fc5` |
| `visualization/app.py` | `7d4df1070ee2a97a80958c95b5d5fd8db84011e239dff0f71920be2cb73e7f4b` |
| `visualization/runtime.py` | `abb4ad1e3dc84f3e299b7b94fc567ab52f1ee5d366114bb68dbd99dd17da13b7` |
| `visualization/datasets.py` | `b61fc93f35203d184ac27ee721980e6e240bc14d902693c100551a83bf95a66a` |
| `visualization/photos.py` | `f6f74553ec70ed8e7d1cf1621cc1b384ba0f369ce3684f26436abf53c93a4b6f` |
| `visualization/execution.py` | `c0f3698c09bc2883aac32753b8e417d182c7599c1ad2690f41715afb239aaae5` |
| `visualization/ssh_workspace.py` | `207e6525fd05c255569255668879d93f1ece307a357ee441ca59223bf9a035d3` |
| `visualization/frontend/src/main.tsx` | `103d53289ff7ea142d64f0191a20b3d3aa7e0ae764fa2f27c6f7e23db86acf6a` |
| `visualization/static/studio.js` | `912885fffeb08d571808104b88dddd28bbc3c5006101ffce01d798d4f3268ec4` |
