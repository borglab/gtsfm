# Zhu et al. motion-averaging baseline

This baseline reimplements the local/global rotation and translation steps from
[Zhu et al., CVPR 2018, Sections 4.3–4.6](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhu_Very_Large-Scale_Global_CVPR_2018_paper.pdf).
It accepts a verified COLMAP database, so experiments can share image pairs,
keypoints, matches, and camera metadata. Pair filtering applies to a copy.

1. Recursively partition the inlier-weighted graph with normalized cuts and
   add neighboring separator cameras to each local incremental reconstruction.
2. Form relative rotations and scale-bearing displacements from local
   camera poses and selected landmarks. Initialize partition transforms from
   common estimated cameras.
3. Alternate interior and separator solves for rotations, then translations.
4. Carry the local intrinsics into triangulation and final sparse BA. Evaluate
   using GTSfM's pose metrics. Reference poses enter only the evaluator.

The implementation uses SciPy sparse trust-region solves and initializes
similarity transforms from common cameras. The paper uses Levenberg–Marquardt
and an epipolar-geometry-based initialization. The implementation runs on one
host and uses PyCOLMAP to refine the final sparse model. The paper uses
distributed final BA.

## Reproduce

Run from the repo root in the pinned GTSfM environment. Each run writes a
verified database snapshot, local reconstructions, final COLMAP geometry, and
`metrics.json` to a new output directory. The reported `frontend_sha256`
identifies the filtered `frontend.db` snapshot used by the reconstruction.
Reuse that snapshot to repeat a run with the same verified inputs.

```sh
python -m scripts.run_distributed_motion_averaging \
  --database tests/data/set1_lund_door/colmap.db \
  --images tests/data/set1_lund_door/images \
  --olsson-reference tests/data/set1_lund_door \
  --max-cameras 6 --min-pair-inliers 3800 --output .context/door-zhu2018
```

The Door threshold makes this small fixture exercise both interior and
separator cameras. This graph has 25 verified pairs, three primary
partitions, and eight separator cameras.

The Reichstag fixture uses a fresh database with CPU SIFT and exhaustive pair
verification. Start with a new database path.

```sh
mkdir -p .context
python - <<'PY'
import pycolmap

database = ".context/reichstag-sift.db"
images = "tests/data/imb_reichstag/images"
pycolmap.set_random_seed(0)
pycolmap.extract_features(
    database,
    images,
    sift_options=pycolmap.SiftExtractionOptions(num_threads=2, max_num_features=4096),
    device=pycolmap.Device.cpu,
)
pycolmap.match_exhaustive(
    database,
    sift_options=pycolmap.SiftMatchingOptions(num_threads=2),
    device=pycolmap.Device.cpu,
)
PY
python -m scripts.run_distributed_motion_averaging \
  --database .context/reichstag-sift.db --images tests/data/imb_reichstag/images \
  --yfcc-imb-reference tests/data/imb_reichstag \
  --max-cameras 6 --output .context/reichstag-zhu2018
```

## Observed results

| Fixture | Final registered / input | Final points | Pose AUC at 5° after averaging | Pose AUC at 5° after final BA |
| --- | ---: | ---: | ---: | ---: |
| Lund Door | 12 / 12 | 11,118 | 0.9048 | 0.9068 |
| Reichstag, first frontend | 10 / 10 | 2,193 | 0.5291 | 0.4952 |
| Reichstag, fresh frontend | 10 / 10 | 2,195 | 0.4910 | 0.5252 |

Every Reichstag camera touches the partition cut, so this fixture exercises the
global solve and final reconstruction. Rebuilding the frontend produced the
same 45 image pairs but different image IDs and inlier sets. Final BA lowered
pose AUC in the first build and raised it in the second. Reusing the first
verified database reproduced its partition membership, point count, and final
pose AUC. Keep the verified database fixed for method comparisons. A scaling
comparison also needs matched image sets and separate frontend, local,
averaging, and refinement timings.
