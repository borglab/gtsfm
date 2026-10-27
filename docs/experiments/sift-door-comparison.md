# SIFT comparison on Lund Door

This experiment compares feature extractors using GTSFM master `8f21351c`, on the
12-image `tests/data/set1_lund_door` benchmark. It does not run VLFeat. The revived
VLFeat PR remains preserved separately as `codex/revive-vlfeat-sift`, commit
`b35a262b`, with no uncommitted changes.

The original ETH3D table is a historical reference, not a result reproduced by
this Door experiment. Its COLMAP and OpenCV means can be recovered exactly by
averaging the 13 scene rows in the [PR's attached TSV](https://github.com/borglab/gtsfm/pull/842#issuecomment-2960751350).
The discussion does not include a complete environment or the `sift_colmap`
configuration used for those runs.

## Configuration

All variants use `unified`, a 1296-pixel short-side resolution limit, shared
intrinsics, NetVLAD retrieval with five neighbors and sequential lookahead ten,
mutual brute-force matching with ratio threshold 0.8, the same OpenCV RANSAC
verifier and two-view BA, and the same GTSFM reconstruction backend. There are
65 retrieved pairs from 12 images. Ground truth is the Olsson `data.mat` file.
MVS and Gaussian splatting are disabled. Each run uses one worker and one worker
thread, with BLAS/OMP thread limits of one.

The requested feature limit is 5,000 for both backends. OpenCV returns exactly
5,000 per image; the existing COLMAP wrapper returns 8,173–8,550 descriptors.
Its default normalization is `L1_ROOT`. This is a comparison of the existing
extractors, not an equal-descriptor-budget experiment.

## RootSIFT variants

- `rootsift_legacy` runs the existing `RootSIFTDescriptor.describe()` on the same
  selected OpenCV keypoints. Conversion through GTSFM `Keypoints` discards the
  original orientation and octave metadata before descriptors are recomputed.
- `rootsift` applies the same L1 normalization and square root directly to the
  baseline OpenCV descriptors, keeping the original keypoint frames. This is
  the controlled normalization comparison.

Both adapters cast the final descriptors to float32. Under NumPy 2.2.6, the
legacy `np.finfo(float).eps` expression promotes the result to float64, which
OpenCV's brute-force matcher does not accept. No production extractor is
modified by this experiment.

## Reproduction

Use an installed GTSFM environment and ensure the existing NetVLAD checkpoint
is available under `thirdparty/hloc/weights/VGG16-NetVLAD-Pitts30K.mat` (otherwise
the model loader downloads it). From the repository root:

```bash
PYTHONPATH="$PWD" python scripts/benchmark_sift_variants.py \
  --output-root /tmp/sift-door-new --methods opencv colmap
# Once both baselines complete:
PYTHONPATH="$PWD" python scripts/benchmark_sift_variants.py \
  --output-root /tmp/sift-door-new --methods rootsift_legacy rootsift
```

The script refuses to overwrite an existing method output directory. It records
the exact command, package versions, process status, and execution time. The
runner saves resolved configurations and raw metrics. GTSFM caches are local to
the checkout; use a fresh worktree for cold feature-extraction runs. Timing is
not a controlled comparison: retrieval can be cached and other workloads were
active on the machine.

## Interpretation

Report `bundle_adjustment_metrics` pose AUC to match the metric column used in
the original PR's table. Current master also performs an additional post-merge
retriangulation/BA stage; its separate scores are retained in the results and
must not be confused with the original BA metric. A single small scene cannot
establish the relative performance on the ETH3D training set.

## Results

All four runs reconstructed 12/12 cameras. AUC values below are percentages.

Standard bundle adjustment (the metric stage used in the PR table):

| Extractor | AUC@1° | AUC@2.5° | AUC@5° | AUC@10° | AUC@20° | Tracks |
|---|---:|---:|---:|---:|---:|---:|
| OpenCV SIFT | 84.68 | 93.87 | 96.94 | 98.47 | 99.23 | 6,981 |
| COLMAP SIFT (default RootSIFT) | 85.60 | 94.24 | 97.12 | 98.56 | 99.28 | 11,281 |
| Ayush standalone RootSIFT | 84.57 | 93.83 | 96.91 | 98.46 | 99.23 | 4,380 |
| OpenCV + RootSIFT normalization | 84.91 | 93.96 | 96.98 | 98.49 | 99.25 | 7,078 |

Additional post-merge retriangulation/BA on current master:

| Extractor | AUC@1° | AUC@2.5° | AUC@5° | AUC@10° | AUC@20° |
|---|---:|---:|---:|---:|---:|
| OpenCV SIFT | 46.65 | 73.23 | 86.04 | 93.02 | 96.51 |
| COLMAP SIFT (default RootSIFT) | 49.25 | 75.42 | 87.55 | 93.78 | 96.89 |
| Ayush standalone RootSIFT | 48.84 | 74.66 | 86.84 | 93.42 | 96.71 |
| OpenCV + RootSIFT normalization | 47.11 | 73.42 | 86.13 | 93.06 | 96.53 |

On Door, the controlled RootSIFT normalization changes AUC@1° from 84.68% to
84.91% and tracks from 6,981 to 7,078. The standalone implementation has
similar pose accuracy but substantially fewer tracks. These are single-run
observations; the small pose differences do not establish a robust ranking.
The later retriangulation/BA stage lowers pose AUC for every variant; its
behavior was recorded, not changed as part of this feature comparison.

Validation: the 12 relevant tests passed, including equivalence to Ayush's
normalization and actual OpenCV matching of both adapters. The exact repository
CI flake8 command passed. All four resolved configs were compared and differ
only in extractor class and output location.

Package versions: NumPy 2.2.6, OpenCV 5.0.0.93, PyCOLMAP 3.11.1, GTSAM 4.3.0.
See [machine-readable results](sift-door-comparison.json) for all versions,
dataset checksums, commands, and values. Full logs, configurations, metrics,
and reconstructions are in `/tmp/gtsfm-sift-door-comparison/`.
