# Replace standalone RootSIFT with joint OpenCV extraction

`RootSIFTDetectorDescriptor` applies L1 normalization and an elementwise square
root to the descriptors returned by `SIFTDetectorDescriptor.detect_and_describe()`.
OpenCV computes these descriptors using its original orientation, octave, and
scale information. The old `RootSIFTDescriptor.describe(image, keypoints)` path
recomputed SIFT descriptors after conversion through GTSFM `Keypoints`, which
stores coordinates, scales, and responses but drops orientation and octave.
The normalization formula itself was already correct; the replacement avoids
that lossy round trip and a second descriptor computation.

The output stays float32, including with NumPy 2, for OpenCV matcher compatibility.
The new class also has its own detector-descriptor cache namespace. Plain SIFT
remains available and the default extractor is unchanged.

## API migration

Replace the removed standalone `gtsfm.frontend.descriptor.rootsift.RootSIFTDescriptor`
with the combined extractor:

```python
from gtsfm.frontend.detector_descriptor.rootsift import RootSIFTDetectorDescriptor

keypoints, descriptors = RootSIFTDetectorDescriptor(max_keypoints=5000).detect_and_describe(image)
```

For an existing SIFT runner configuration, override:

```text
cluster_optimizer.correspondence_generator.detector_descriptor.detector_descriptor_obj._target_=gtsfm.frontend.detector_descriptor.rootsift.RootSIFTDetectorDescriptor
```

This is an API change: the replacement detects its own keypoints and does not
provide `describe(image, arbitrary_keypoints)`. No production callers of the old
class were found in the repository; its unit and reproducibility tests are
replaced with tests for joint extraction. Upright SIFT is outside this change.

## Recorded comparison

These are existing runs, not benchmarks rerun for this change. The benchmark
harness and original reports are preserved on `codex/sift-rootsift-comparison`
at commit `f558a601`; production code in those runs was based on master
`8f21351c`. The three variants used identical OpenCV detections and feature
limits. The legacy adapter cast its output to float32 so that NumPy 2 promotion
did not prevent matching. The combined adapter normalized the original OpenCV
descriptors; a regression test checks the replacement against that formula.

All rows use the first bundle-adjustment output. AUC values are percentages.

| Dataset | Extractor | Cameras | Tracks | Pose AUC@1° | Pose AUC@5° |
|---|---|---:|---:|---:|---:|
| Lund Door | OpenCV SIFT | 12/12 | 6,981 | 84.68 | 96.94 |
| Lund Door | Standalone RootSIFT (old) | 12/12 | 4,380 | 84.57 | 96.91 |
| Lund Door | OpenCV + RootSIFT (replacement) | 12/12 | 7,078 | 84.91 | 96.98 |
| Skydio-32 | OpenCV SIFT | 29/32 | 1,167 | 47.52 | 71.92 |
| Skydio-32 | Standalone RootSIFT (old) | Failed | — | — | — |
| Skydio-32 | OpenCV + RootSIFT (replacement) | 29/32 | 1,365 | 44.84 | 70.03 |
| Gerrard Hall | OpenCV SIFT | 100/100 | 14,022 | 86.81 | 97.31 |
| Gerrard Hall | Standalone RootSIFT (old) | 98/100 | 9,414 | 77.48 | 92.25 |
| Gerrard Hall | OpenCV + RootSIFT (replacement) | 100/100 | 15,013 | 89.66 | 97.91 |

The replacement retains substantially more tracks than standalone RootSIFT on
Door and Gerrard Hall, and improves camera coverage and pose accuracy on
Gerrard Hall. Compared with plain OpenCV, normalization increases track counts
on all three scenes but lowers pose AUC on Skydio. These results support replacing
the lossy standalone path, not claiming that RootSIFT always outperforms SIFT.

### Conditions and limitations

- Runs used GTSAM 4.3.0, NumPy 2.2.6, and OpenCV 5.0.0.93; the same `unified`
  backend, shared intrinsics, 5,000-feature limit, five NetVLAD neighbors plus
  sequential lookahead ten, mutual brute-force matching with ratio 0.8, OpenCV
  RANSAC, and two-view BA. Door used a 1296-pixel short-side limit and 65 retrieved
  pairs; Skydio used 760 pixels and 303 pairs; Gerrard Hall used 760 pixels and
  974 pairs.
- Door uses the Olsson reference. Skydio and Gerrard Hall use COLMAP-derived
  reference poses, not independently measured ground truth. No COLMAP feature
  extractor is used by the replacement.
- Skydio and Gerrard Hall scores use all reference-image pairs (496 and 4,950),
  counting missing cameras as failures. Pose error is the maximum of relative
  rotation and translation-direction error. Skydio's saved OpenCV pipeline score
  omitted a reference camera; the table uses the corrected denominator.
- Gerrard Hall used the same experiment-only rank-three Shonan solve with
  certification disabled for all three methods. Default OpenCV and normalized
  runs were stopped because of time and memory growth. Production solver settings
  are unchanged by this PR.
- Standalone RootSIFT on Skydio collapsed from 896 tracks to one during BA and
  then failed during result deserialization. This repeated with cached frontend
  outputs; it is not a clean measure of descriptor quality or proof of the
  failure's cause.
- These are single-scene trials, not an ETH3D reproduction or a VLFeat comparison.
  There is no controlled runtime comparison.

The later post-merge retriangulation/BA stage is separate from the first-BA
scores above. Its pose AUC@5° was:

| Dataset | OpenCV SIFT | Standalone RootSIFT | OpenCV + RootSIFT |
|---|---:|---:|---:|
| Lund Door | 86.04 | 86.84 | 86.13 |
| Skydio-32 | 71.93 | Failed | 65.62 |
| Gerrard Hall | 81.78 | 75.01 | 85.53 |

The post-merge degradation has not been diagnosed. First-BA scores should not
be described as the final pipeline accuracy.
