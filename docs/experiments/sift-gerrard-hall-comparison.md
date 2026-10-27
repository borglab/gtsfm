# SIFT comparison on Gerrard Hall

This follows the [Skydio-32 comparison](sift-skydio32-comparison.md), using all
100 locally available Gerrard Hall images. The reference calibration and poses
come from the local COLMAP reconstruction, not independent ground truth. This
can favor agreement with a COLMAP-derived reference. COLMAP is used only as a
comparison extractor; no VLFeat backend is run.

Each method uses 760-pixel short-side images (760 × 1140), shared intrinsics,
the `unified` configuration, five NetVLAD neighbors plus a sequential lookahead
of ten (974 pairs), mutual brute-force matching with ratio 0.8, OpenCV RANSAC,
two-view BA, and the same GTSFM backend. The requested feature limit is 5,000.
OpenCV produced 3,621–5,000 features per image (mean 4,894); COLMAP produced
6,073–9,855 (mean 7,530). Therefore the COLMAP comparison does not have an equal
feature budget. Each reconstruction uses one worker and one worker thread;
overlapping runs and warmed caches make runtime comparisons inappropriate.

## Rotation solver

The default Shonan solver completed for COLMAP and standalone RootSIFT. For
OpenCV and normalized RootSIFT it spent more than 18 and 16 minutes, respectively,
in rotation averaging while memory and swap usage grew. Both were deliberately
stopped to protect the shared machine; they did not fail an algorithmic test.

To obtain a consistent comparison, all four methods were rerun using the same
experiment-only rank-three Shonan objective, chordal initialization and weights,
but with global-optimality certification disabled. This returns a local result;
it does not provide the original solver's global-optimality guarantee. Cached
features, matches and two-view estimates were reused. The saved two-view reports
are identical between original and fixed-rank runs for every method. On the two
methods where the default solver completed, first-BA scores agree to four decimal
places with the fixed-rank runs. The production solver and defaults are unchanged.

## Results

Scores are recomputed against all 100 reference images and all 4,950 possible
image pairs, with missing cameras counted as failures. AUC values are percentages
using the maximum of relative rotation and translation-direction errors.

First bundle adjustment, using the same fixed-rank solver for every row:

| Extractor | Cameras | Tracks | AUC@1° | AUC@2.5° | AUC@5° | AUC@10° | AUC@20° |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenCV SIFT | 100/100 | 14,022 | 86.81 | 94.67 | 97.31 | 98.65 | 99.32 |
| COLMAP SIFT | 100/100 | 26,059 | 93.88 | 97.52 | 98.76 | 99.38 | 99.69 |
| Ayush standalone RootSIFT | 98/100 | 9,414 | 77.48 | 88.54 | 92.25 | 94.12 | 95.07 |
| OpenCV + RootSIFT normalization | 100/100 | 15,013 | 89.66 | 95.82 | 97.91 | 98.95 | 99.48 |

The standalone variant misses `IMG_2386.JPG` and `IMG_2387.JPG`; as described in
the [Door report](sift-door-comparison.md), it recomputes descriptors after the
keypoint conversion discards orientation and octave information. The normalized
OpenCV control preserves those frames and gives the controlled test of applying
Ayush's normalization to OpenCV's descriptors. Its 5° AUC is 0.60 percentage
points higher than plain OpenCV, and it produces 991 more tracks. This is one
scene and one run, not a general ranking.

Additional post-merge retriangulation/BA, which is a separate pipeline stage:

| Extractor | Cameras | Tracks | AUC@1° | AUC@2.5° | AUC@5° | AUC@10° | AUC@20° |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenCV SIFT | 100/100 | 14,195 | 30.06 | 64.13 | 81.78 | 90.81 | 95.36 |
| COLMAP SIFT | 100/100 | 26,327 | 38.30 | 73.66 | 86.67 | 93.27 | 96.61 |
| Ayush standalone RootSIFT | 98/100 | 9,558 | 20.13 | 55.13 | 75.01 | 85.39 | 90.66 |
| OpenCV + RootSIFT normalization | 100/100 | 15,203 | 36.17 | 71.49 | 85.53 | 92.70 | 96.31 |

Post-merge scores fall noticeably for every variant. The cause was not
investigated here, so the first-BA scores should not be presented as the final
pipeline output. Among the 974 selected pairs, the numbers with verified
rotation and translation errors both below 5° are 638 for OpenCV, 713 for
COLMAP, 542 for standalone RootSIFT and 656 for normalized RootSIFT. These are
frontend pair results, not global-reconstruction pose scores.

## Reproduction

With a GTSFM environment and the NetVLAD checkpoint available, run from this
worktree (the `--uncertified-rotation` flag changes only this experiment):

```bash
PYTHONPATH="$PWD" python scripts/benchmark_sift_variants.py \
  --loader colmap \
  --dataset-dir /home/dellaert/git/gtsfm/benchmarks/gerrard-hall/sparse \
  --images-dir /home/dellaert/git/gtsfm/benchmarks/gerrard-hall/images \
  --max-resolution 760 --uncertified-rotation \
  --output-root /tmp/gerrard-sift-new \
  --methods opencv colmap rootsift_legacy rootsift
PYTHONPATH="$PWD" python scripts/evaluate_sift_comparison.py \
  --reference-model /home/dellaert/git/gtsfm/benchmarks/gerrard-hall/sparse \
  --output-root /tmp/gerrard-sift-new
```

The [machine-readable results](sift-gerrard-hall-comparison.json) contain
environment versions, input checksums, exact commands and process statuses.
Full logs, metrics, configurations and reconstructions are under
`/tmp/gtsfm-sift-gerrard-fixed-rank`. The original attempts are under
`/tmp/gtsfm-sift-gerrard-hall-comparison`, `/tmp/gtsfm-sift-gerrard-colmap`,
`/tmp/gtsfm-sift-gerrard-legacy`, and `/tmp/gtsfm-sift-gerrard-normalized`.
