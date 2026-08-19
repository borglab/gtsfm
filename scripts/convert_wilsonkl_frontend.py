"""Convert a wilsonkl/SfM_Init (1DSfM) dataset release into a precomputed-frontend file.

The 1DSfM benchmark ships the frontend every published method consumed: EGs.txt (verified
view-graph edges), coords.txt (per-image SIFT keypoints, original pixel frame), tracks.txt
(multi-view tracks as (image, key) pairs). This converts them into the npz consumed by
SceneOptimizer(precomputed_frontend_path=...), replacing retrieval + matching + two-view:

  keypoints are rescaled from the original frame to the loader frame (same
  get_downsampling_factor_per_axis the loader applies, using the original dims recorded in
  coords.txt headers), and per-edge verified correspondences are derived from the tracks
  restricted to EGs edges.

Usage:
  python scripts/convert_wilsonkl_frontend.py --gt_dir <dataset dir> --output <out.npz> \
      [--max_resolution 760]
"""
import argparse
import re
import time

import numpy as np

from gtsfm.utils.images import get_downsampling_factor_per_axis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max_resolution", type=int, default=760,
                    help="loader max_resolution (short side), must match the run config")
    args = ap.parse_args()
    t0 = time.time()

    # ---- coords.txt: per-image keypoints in ORIGINAL pixel frame + original dims ----
    kp = {}  # img -> np.ndarray (N,2) in loader frame
    dims = {}  # img -> (loader_w, loader_h) expected at run time (EXIF-rotation guard)
    header_re = re.compile(r"#index = (\d+),.*?keys = (\d+), px = ([\d.]+), py = ([\d.]+)")
    cur_img, cur_rows, scale_uv = None, None, (1.0, 1.0)

    def flush():
        if cur_img is not None and cur_rows:
            kp[cur_img] = np.array(cur_rows, dtype=np.float32)

    with open(f"{args.gt_dir}/coords.txt") as f:
        for line in f:
            if line.startswith("#index"):
                flush()
                m = header_re.search(line)
                cur_img = int(m.group(1))
                px, py = float(m.group(3)), float(m.group(4))
                w0, h0 = 2 * px, 2 * py
                su, sv, new_h, new_w = get_downsampling_factor_per_axis(int(h0), int(w0), args.max_resolution)
                scale_uv = (su, sv)
                dims[cur_img] = (new_w, new_h)
                cur_rows = []
            else:
                p = line.split()
                if len(p) >= 3:
                    cur_rows.append((float(p[1]) * scale_uv[0], float(p[2]) * scale_uv[1]))
    flush()
    n_images = max(kp) + 1
    print(f"[+{time.time()-t0:.0f}s] coords: {len(kp)} images with keypoints (n_images={n_images})")

    # ---- EGs.txt: the verified edge set ----
    edges = set()
    with open(f"{args.gt_dir}/EGs.txt") as f:
        for line in f:
            p = line.split()
            if len(p) >= 2:
                a, b = int(p[0]), int(p[1])
                edges.add((min(a, b), max(a, b)))
    print(f"[+{time.time()-t0:.0f}s] EGs: {len(edges)} edges")

    # ---- tracks.txt -> per-edge correspondences (restricted to EGs edges) ----
    corr = {e: [] for e in edges}
    n_tracks = 0
    with open(f"{args.gt_dir}/tracks.txt") as f:
        first = f.readline()  # track count header
        for line in f:
            p = line.split()
            if not p:
                continue
            L = int(p[0])
            ms = [(int(p[1 + 2 * k]), int(p[2 + 2 * k])) for k in range(L)]
            n_tracks += 1
            for x in range(L):
                for y in range(x + 1, L):
                    (ia, ka), (ib, kb) = ms[x], ms[y]
                    if ia == ib:
                        continue
                    if ia > ib:
                        (ia, ka), (ib, kb) = (ib, kb), (ia, ka)
                    lst = corr.get((ia, ib))
                    if lst is not None:
                        lst.append((ka, kb))
    corr = {e: np.array(c, dtype=np.int32) for e, c in corr.items() if len(c) >= 12}
    n_corr = sum(len(c) for c in corr.values())
    print(f"[+{time.time()-t0:.0f}s] tracks: {n_tracks} -> {len(corr)} edges with >=12 "
          f"correspondences ({n_corr} total)")

    # ---- flat-encode and save ----
    kp_offsets = np.zeros(n_images + 1, dtype=np.int64)
    for i in range(n_images):
        kp_offsets[i + 1] = kp_offsets[i] + (len(kp[i]) if i in kp else 0)
    kp_flat = np.concatenate([kp[i] for i in range(n_images) if i in kp]) if kp else np.zeros((0, 2), np.float32)
    E = sorted(corr)
    edge_arr = np.array(E, dtype=np.int32)
    corr_offsets = np.zeros(len(E) + 1, dtype=np.int64)
    for k, e in enumerate(E):
        corr_offsets[k + 1] = corr_offsets[k] + len(corr[e])
    corr_flat = np.concatenate([corr[e] for e in E]) if E else np.zeros((0, 2), np.int32)
    img_dims = np.full((n_images, 2), -1, dtype=np.int32)
    for i, (w, h) in dims.items():
        img_dims[i] = (w, h)
    np.savez_compressed(
        args.output, n_images=n_images, kp_offsets=kp_offsets, kp_flat=kp_flat,
        edges=edge_arr, corr_offsets=corr_offsets, corr_flat=corr_flat,
        max_resolution=args.max_resolution, img_dims=img_dims,
    )
    print(f"[+{time.time()-t0:.0f}s] wrote {args.output}")


if __name__ == "__main__":
    main()
