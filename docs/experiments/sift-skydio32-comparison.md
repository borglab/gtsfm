# SIFT comparison on Skydio-32

This follow-up to the [Door comparison](sift-door-comparison.md) uses the local
32-image Skydio crane benchmark. The production pipeline remains at master
`8f21351c`; only the benchmark harness and experiment-only extractor adapters
are added on `codex/sift-rootsift-comparison`. No PyVLFeat backend is run.

All four methods use the `unified` configuration, shared intrinsics, 760-pixel
short-side images (760 × 1014 after resizing), five NetVLAD retrieval neighbors,
a sequential lookahead of ten, mutual brute-force matching with ratio 0.8,
OpenCV RANSAC verification, two-view BA, and the same GTSFM backend. Retrieval
selects 303 pairs from 32 images. Reference intrinsics and poses are loaded from
`colmap_crane_mast_32imgs`; these are COLMAP reconstruction references, not
independently measured ground truth. COLMAP SIFT is used only for comparison.

The requested feature limit remains 5,000. COLMAP can exceed that limit; actual
counts are recorded separately. The two RootSIFT variants are the existing
standalone Ayush descriptor and a control applying the same normalization to
OpenCV's original descriptors. Both use the float32 compatibility conversion
described in the Door report.

## Reproduction

From this worktree, using the same GTSFM environment as the Door experiment:

```bash
PYTHONPATH="$PWD" python scripts/benchmark_sift_variants.py \
  --loader colmap \
  --dataset-dir /home/dellaert/git/gtsfm/benchmarks/skydio-32/colmap_crane_mast_32imgs \
  --images-dir /home/dellaert/git/gtsfm/benchmarks/skydio-32/images \
  --max-resolution 760 \
  --output-root /tmp/skydio32-sift-new \
  --methods opencv colmap rootsift_legacy rootsift
```

Full logs, resolved configurations, raw metrics, and reconstructions are saved
under `/tmp/gtsfm-sift-skydio32-comparison/`. Runs are sequential, with one Dask
worker and one worker thread. NetVLAD retrieval may reuse cached descriptors.
Runtime is not a controlled comparison because other workloads share the host.

The standard bundle-adjustment AUC is the metric stage used by the historical
ETH3D PR table. Current master also runs an additional post-merge
retriangulation/BA stage; its scores are reported separately. This is one trial
per method on one larger local scene, not an ETH3D reproduction.

## Fixed reference-image evaluation

The saved OpenCV BA metrics cover only 31 reference cameras (465 pairs), even
though the input dataset contains 32 images. One camera excluded earlier from
reconstruction is omitted from that metric's denominator. To compare all
extractors fairly, the primary results are recomputed from the saved camera
poses against all 32 reference images: 496 pairs, with every pair involving a
missing estimated camera counted as a failure. Correspondence is established
by filename, not image ID. Relative pose errors need no global alignment.

```bash
PYTHONPATH="$PWD" python scripts/evaluate_sift_comparison.py \
  --reference-model /home/dellaert/git/gtsfm/benchmarks/skydio-32/colmap_crane_mast_32imgs \
  --output-root /tmp/gtsfm-sift-skydio32-comparison
```

The saved pipeline metrics are retained alongside this reevaluation. For
OpenCV, the difference in BA AUC is the denominator correction, 465/496. No
production evaluation or reconstruction code is changed in this experiment.

## Results

Primary scores use all 496 reference-image pairs. AUC values are percentages.

Standard BA output:

| Extractor | Cameras | Tracks | AUC@1° | AUC@2.5° | AUC@5° | AUC@10° | AUC@20° |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenCV SIFT | 29/32 | 1,167 | 47.52 | 64.33 | 71.92 | 76.51 | 79.16 |
| COLMAP SIFT | 7/32 | 2 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| Ayush standalone RootSIFT | Failed | — | — | — | — | — | — |
| OpenCV + RootSIFT normalization | 29/32 | 1,365 | 44.84 | 61.78 | 70.03 | 75.31 | 78.52 |

Additional post-merge retriangulation/BA:

| Extractor | Cameras | Tracks | AUC@1° | AUC@2.5° | AUC@5° | AUC@10° | AUC@20° |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenCV SIFT | 29/32 | 1,377 | 46.86 | 64.00 | 71.93 | 76.83 | 79.34 |
| COLMAP SIFT | 7/32 | 3 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| OpenCV + RootSIFT normalization | 29/32 | 1,442 | 37.33 | 56.05 | 65.62 | 72.34 | 76.66 |

The normalized OpenCV variant retains more tracks (1,365 vs 1,167) but has
lower pose AUC in this trial. Both it and plain OpenCV miss the same three
cameras: `S1014644.JPG`, `S1014656.JPG`, and `S1014724.JPG`. The modest benefit
observed on Door does not generalize to this Skydio trial. This is not enough
evidence for a general ranking of SIFT implementations.

COLMAP's process returned success, but reconstruction quality failed: its first
BA stage reduced 1,815 tracks to 2 amid widespread cheirality exceptions. The
standalone RootSIFT path similarly reduced 896 tracks to 1, then exited with
`RuntimeError: input stream error` while deserializing the merged result.
Neither failure should be treated as a clean descriptor-quality comparison.

Both failures were repeated using cached features, matches, and two-view
estimates in `/tmp/gtsfm-sift-skydio32-repeat`. The saved two-view reports were
identical between the original and repeated runs. COLMAP again retained
7 cameras and 2 tracks; standalone RootSIFT again collapsed to 1 track and
failed with the same serialization error. The failures have not been fixed
by changing the production backend in this experiment.

Feature counts: OpenCV and both RootSIFT variants use 1,713–3,847 features per
image (mean 2,960.94); COLMAP uses 2,893–4,667 (mean 3,796.81). The same requested
5,000 limit is therefore inactive on this dataset.

Validation: all four resolved configurations match except extractor class and
output path. Fifteen tests passed, including fixed-denominator coverage,
similarity invariance, and continuing the benchmark after a failed variant.
The exact CI flake8 check passed. The harness now records failures, continues
to later variants, and returns a nonzero overall status if any process fails.

See [machine-readable results](sift-skydio32-comparison.json) for commands,
package versions, dataset checksums, raw-versus-corrected AUC, and execution
statuses. Raw per-pair fixed-reference errors are saved in
`/tmp/gtsfm-sift-skydio32-comparison/full-reference-metrics.json`.
