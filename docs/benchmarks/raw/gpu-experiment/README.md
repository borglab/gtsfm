# Real RTX 3080 execution-profile experiment

The frozen A1/B1/A2/B2 series completed with actual MegaLoc retrieval, SIFT/TwoWayMatcher frontend, vanilla VGGT geometry and GTSAM refinement. Each final `merged_retriangulated` COLMAP model has four registered cameras and 1,509 sparse tracks. The three exported model files and named retrieval-pair file have identical SHA256 values across all four trials. The independent verifier reported the four-trial export comparison VERIFIED, including finite geometry, linked observations and nondegenerate alignment checks; counts alone were not the acceptance test. Its initial canonical-track comparison found 77 duplicate rounded observation-key groups, retained that failed attempt, and used the predeclared stronger exact-byte path for these identical files without loosening tolerances.

| Trial | Actual launch | Harness elapsed, seconds | Logged sparse pipeline, seconds | Sampled GPU peak, MiB |
| --- | --- | ---: | ---: | ---: |
| A1 | Manual runner | 47.725 | 31.3 | 6708 |
| B1 | Persisted profile → JobManager.start → runner | 44.915 | 28.9 | 5586 |
| A2 | Manual runner | 43.332 | 28.8 | 5596 |
| B2 | Persisted profile → JobManager.start → runner | 43.653 | 25.1 | 6796 |

These are descriptive observations, with two repeats per arm and no speedup claim. Harness elapsed uses wall-clock timestamps, includes launch/poll/artifact-copy overhead, and subtracts a fixed seven-second passive post-exit settling interval. `ended - started` includes that interval. Stage timings come from the unmodified pipeline logs. See `series-clarifications.json` for timing, seed, RSS and cleanup boundaries.

## Frozen controls and reproducibility

- `series-declaration.json` was written and sent before A1 outputs. It fixes order A1/B1/A2/B2, images, hashes, preset, resource settings, seeds, cache policy, stage, and unchanged independent tolerances.
- The dataset contains the first four real Lund Door JPEGs, `DSC_0001.JPG` through `DSC_0004.JPG`, copied without `data.mat`; Olsson therefore uses EXIF calibration and no ground-truth extrinsics. This is a bounded four-image experiment, not a full-scene scalability result.
- Every trial used one Dask process worker, two threads, 18 GiB worker host-memory limit, explicit `CUDA_VISIBLE_DEVICES=0`, isolated absolute scratch, resolution 518 and crop mode. Actual worker observations, CUDA allocations, model module paths, process arguments and environments are retained per trial.
- Startup `random`, NumPy and Torch seeds are 42; the model's geometry seed is 42. Parent/resource-tracker `PYTHONHASHSEED` is 42 and Dask's worker pre-spawn hash seed is 6640, consistently across all four trials. Do not describe every process hash seed as 42.
- Every trial starts with an empty isolated `source/cache`; generated caches are retained under its evidence directory. Model weights and DINOv2 source remain warm and identical. `TORCHINDUCTOR_COMPILE_THREADS=1` applies equally to all measured arms; the separate warmup had the default compiler pool.
- The Linux source copy has no root `.git`. `source-file-manifest.json` verifies all 852 transferred tracked/new feature files against actual bytes, with no changes after the series. `initial/environment-manifest.json` additionally records initialized optional source, DINOv2 hashes, model-weight hashes and installed versions. `source-git-diff.patch`, `source-git-state.txt` and `frozen-feature-source/` preserve the local baseline/feature provenance. Local Git HEAD was freshly checked as `3306994e2903a3fc65b01f952e1b5d7bc248cfe4`; that hash alone is not remote-source verification.
- VGGT is official revision `44b3afbd1869d8bde4894dd8ea1e293112dd5eba`, weights SHA256 `d15bf50a8615c8225ed48b51ea5cac673d82442ec0309036df555a053253afe0`. LightGlue source is initialized at `746fac2c042e05d1865315b1413419f1c1e7ba55`. FastVGGT is absent and actual VGGT module paths are recorded. MegaLoc weights SHA256 is `c6d01ea3ff7858175f66e2c25305585f9229aab809a3f248e9e42eb258383242`; logs confirm successful pretrained-weight loading rather than random fallback.

The exact guarded harness is `scripts/experiment_dask_profiles_gpu.py`, frozen SHA256 `313bcda0fcb3dcfd65839f7e5fe2beb6ec1ec406608d78844b0d909590d8fd1d`. Its embedded `INSTRUMENTATION` string is the complete isolated `sitecustomize.py`; it records and seeds but does not replace inference. Run it as an actual file, never Python stdin for multiprocessing.

Remote frozen root: `/home/jasdeep/gtsfm-profile-gpu-20261024`. Interpreter: `/home/jasdeep/gtsfm_eval/.venv/bin/python`, reused without changes. SSH route: `ssh -o BatchMode=yes -o ConnectTimeout=8 jasdeep-pc 'wsl.exe -d ShineCompute -- bash -s'`, feeding a shell script on stdin. Every actual trial command is in the declaration and each `summary.json`. The service envelope was:

```sh
systemd-run --user --unit=gtsfm-profile-A1 --wait --collect \
  -p MemoryMax=24G -p MemorySwapMax=0 -p CPUQuota=400% -p RuntimeMaxSec=960 \
  --setenv=PATH=/home/jasdeep/gtsfm_eval/.venv/bin:/usr/local/bin:/usr/bin:/bin:/usr/lib/wsl/lib \
  /home/jasdeep/gtsfm_eval/.venv/bin/python \
  /home/jasdeep/gtsfm-profile-gpu-20261024/experiment_dask_profiles_gpu.py \
  --base /home/jasdeep/gtsfm-profile-gpu-20261024 --trial A1 --arm manual \
  --dataset /home/jasdeep/gtsfm-profile-gpu-20261024/dataset-first4 --wall-seconds 900
```

Use a fresh isolated base and unique trial/unit names to reproduce; the harness refuses an existing trial evidence directory. It stops at GPU temperature ≥80°C, sampled physical GPU usage ≥9700 MiB, available system RAM <4 GiB, or its wall timeout. Actual cgroup samples confirm `memory.max=25769803776` and `cpu.max=400000 100000`; GPU visibility is not a hard VRAM quota. Sum-RSS includes shared pages; cgroup memory is retained separately. The existing MIRA service and shared interpreter/source/dependencies were preserved.

## Artifacts and retained attempts

For each `series/{A1,B1,A2,B2}`:

- Final model: `output/results/merged_retriangulated/{cameras,images,points3D}.txt`.
- Named pairs: `output/results/plots/similarity_named_pairs.txt`.
- Resolved Hydra: `output/results/config.yaml` and independent `hydra-*.yaml` captures.
- Contract/actual launch: `contract.json`, `process-*.json`, `summary.json`; B trials also retain `start.json`, `final-job.json`, and `output/execution_profile.json`.
- Complete runner log, metrics and outputs, one-second `telemetry.jsonl`, per-process `python-*.jsonl` CUDA/model observations, generated pipeline caches, explicit PID existence checks and final process/GPU snapshots.

`measured-series-summary.json` extracts times, peaks, worker details, seeds and artifact hashes without discarding raw values. Named output paths are explicit; no intermediate-stage guessing or geometry normalization was used to obtain file equality.

`initial/cancel1` is an actual persisted-profile JobManager job cancelled during observed active Dask reconstruction work. JobManager performed cancellation; the harness only waited passively and checked processes. The runner returned -15, all 37 observed runner/worker/compiler/resource-tracker processes were absent afterward, and the GPU compute-process query was empty. A resource-tracker warning reports six semaphore objects at shutdown, so this establishes process/GPU cleanup, not warning-free or complete IPC cleanup.

`initial/warmup4` is the separate full integrated capacity/setup warmup, not one of the measured repeats. It completed and exported the same final model. `initial/preflight-vggt3` is the retained first harness failure before inference (service PATH and image-loader helper signature); the corrected `preflight-vggt3-attempt2` and four-image real VGGT preflight succeeded. The twelve-image VGGT preflight also **succeeded**, before an attempted safety stop could act: 28.23 seconds, sampled physical GPU usage 9988 MiB, PyTorch peak allocated 8,290,189,824 bytes and peak reserved 14,856,224,768 bytes. Reserved allocation on WSL is not physical GPU usage. No twelve-image CUDA OOM or cutoff failure is claimed; the smaller integrated subset was chosen for operating headroom.

`final-source/` retains separate F1 and latest-source F2 treatments. The four frozen trials retain the earlier accepted feature source. F1 tested profile module `b7623a...` and telemetry `1e6cfa...`; F2 tests the final telemetry fallback `524f5100c19c5c26f0b389481179f8fbdd1ec1cd4ccb7a6ce00e245c3d3d8046` with the same profile module. Only the two declared validation/telemetry files differ from the frozen series; no reconstruction math changed. Full manifests and pre-run declarations distinguish these follow-ups from the four-trial experiment.

No commits were made by the executor and no shared production files were edited by it. Lead integration, independent acceptance and publication are coordinator-owned.

Final-source F1 and F2 both completed and exported the same three COLMAP file hashes and named-pair hash as A1. `final-source/followup-summary.json` retains their individual elapsed values and cleanup observations. `host/runner-contract-audit.json` verifies actual runner argv against recorded manifests, actual environments after excluding only the two evidence/live paths, and resolved Hydra configs after excluding only `output_root`.

Final state: `host/final-drift-and-pid-checks.json` verifies all input names/sizes/hashes, pinned weight and DINOv2 source hashes, the frozen 852-file source, latest F2 source and every observed trial PID. No observed trial PID remains; the pre-existing MIRA PID 34238 remains alive. `host/final-gpu.txt` reports 201 MiB, 31 C and 0% utilization, the compute-process query is empty, and no experiment unit remains loaded. No GPU job is left running.
