# Frozen native-Dask GPU execution evidence

Four fresh RTX 3080 reconstructions completed in the declared order A1 manual, B1 native YAML,
A2 manual, B2 native YAML. Every run used a real authenticated loopback `POST /api/jobs`, the
returned job ID and actual managed runner, pinned first-four Lund Door images, MegaLoc/SIFT,
vanilla VGGT and GTSAM. No reconstruction algorithm or source was edited during the series.

The unchanged strict comparator reports **VERIFIED**, including the manual baseline repeat.
Complete final `cameras.txt`, `images.txt`, `points3D.txt` and named retrieval pairs are
byte-identical across all four runs: four cameras and 1,509 tracks. This establishes these
exported results on this four-image fixture; it is not an internal bitwise-equality or speedup claim.
See [strict-comparison.json](strict-comparison.json); the independent reviewer received all inputs.

| Trial | Arm | Runner observation (s) | Harness (s) | Peak sampled physical GPU (MiB) |
| --- | --- | ---: | ---: | ---: |
| A1 | manual | 38.914 | 53.522 | 5596 |
| B1 | native | 37.813 | 52.289 | 5606 |
| A2 | manual | 37.826 | 52.215 | 5640 |
| B2 | native | 40.059 | 54.415 | 5728 |

Runner observation is a monotonic interval from discovering the managed runner until its exit
poll, with one-second sampling. Harness time includes verification, startup and cleanup;
per-stage pipeline logs and metrics are retained. These timings are descriptive only.
The `systemd-run` completion text printed an implausible 256 KiB memory peak: use live cgroup
`memory.peak`, process RSS and GPU observations in each `telemetry.jsonl` instead.

## Matched controls and actual boundary

- Manual requests: one worker, two threads, `18GiB`, CUDA device 0, absolute scratch.
- Native requests deliberately sent three workers, one thread and `4GiB`, plus the existing
  `cluster_config` path. That file instantiated `distributed.LocalCluster` with one worker,
  two threads, `18GiB`, the same scratch, and loopback dashboard on an ephemeral port.
- All actual workers reported one worker, two threads, **19,327,352,832 bytes**, CUDA visibility
  `0`, intended scratch and daemon false. The guard observed these before inference and
  would abort on a mismatch; positive/negative CPU-only guard probes are in `initial/`.
- Both arms used the same native `DASK_CONFIG` policy and environment, random/NumPy/Torch seed
  42, CPU library thread limits 2, Inductor compile threads 1, fresh `source/cache`, and warm
  pinned model weights. Resolved Hydra YAML, module paths and per-process Torch memory are retained.
- Outer caps for every run: `MemoryMax=24G`, `MemorySwapMax=0`, `CPUQuota=400%`,
  `RuntimeMaxSec=960`. Harness cutoff: 900 seconds, GPU temperature 80 C, physical GPU
  memory 9,700 MiB or system available RAM below 4 GiB. No cutoff fired.
- Each `start.json` records HTTP 202 and actual job ID; `api-server*.json` records loopback
  address, unauthenticated HTTP 401 and verified server closure. Disposable tokens are not saved.

## Frozen provenance

Grant `msg_1a2a8118126a` preceded inference. The isolated source has no Git checkout; the
pre-commit provenance is `012b79c84dbc248213fb4d12e8ea3832bf266dd6`, and root later committed
unchanged production bytes in `70a55d48`. Content hashes, not the later commit label, identify
what ran. The 390-file overlay archive and manifest, 1,025 actual source-file hashes, 174
asset/control hashes, four input hashes and exact grant receipt are in `initial/`.
Every trial's before/after source/asset/input manifests matched. The executed harness SHA256
is `871deb3c2068b541f08b499c4a51658714b8b12d3c81fb087f019ff1d71378a2`.
Vanilla VGGT source/weight pins and DINO/MegaLoc/LightGlue references are retained; FastVGGT
and the deleted bespoke profile module were absent from the isolated source.

## Cleanup and preserved audit limitation

All **12 observed owned PIDs** and their process groups are absent, all four API server threads
and ports are closed, no `native-dask-*` units remain, and the GPU compute list is empty
(idle 201 MiB, 32 C at final check). Model/input hashes and the package-version inventory match the initial snapshot.
See [cleanup/final-cleanup.json](cleanup/final-cleanup.json).

The strict MIRA `psutil.create_time()` comparison failed and is preserved in
[cleanup/audit-attempt1.json](cleanup/audit-attempt1.json). PID 34238, parent 11126 and command
remained the same. Two post-run snapshots show kernel start ticks fixed at 34,251,871 while
Linux `btime` and derived creation time both shift three seconds. Systemd records the original
October 18 start, active service and `NRestarts=0`; the service was never touched. **Exact
full-series kernel identity remains INCONCLUSIVE because pre-run kernel start ticks were
not captured.** This caveat is not silently converted into a passing timestamp assertion.

## Artifact map and exact commands

- `A1/`, `B1/`, `A2/`, `B2/`: full logs, command/request contracts, worker/resource observations,
  resolved Hydra, pipeline caches and complete outputs, including
  `output/results/merged_retriangulated/` and `output/results/plots/similarity_named_pairs.txt`.
- [execution-summary.json](execution-summary.json): compact observed execution/resource evidence.
- [reproduction.json](reproduction.json) and [run-trials.sh](run-trials.sh): actual SSH/stdin,
  bounded systemd commands, comparator invocation and environment/source paths.
- [artifact-manifest.json](artifact-manifest.json): every artifact's byte count and SHA256,
  excluding the manifest itself. Existing evidence directories refuse overwrite.

The historical experiment and assets were preserved. No dependency, user configuration,
MIRA service, production/frontend source or independent-verifier file was changed by this
GPU dispatch; only the owned harness rename/adaptation and this raw evidence tree changed.
