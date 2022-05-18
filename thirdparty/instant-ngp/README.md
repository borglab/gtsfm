### Instructions to use Instant-NGP from GTSFM results

```bash
# 1. Convert GTSFM results (COLMAP-style) to NeRF-style transforms.json file
python thirdparty/instant-ngp/colmap2nerf.py --images /path/to/images --text results/ba_output --out results/nerf                           
# 2. Calculate the overlap area of frustums and transform the overlapped area to the unit cube
python gtsfm/utils/overlap_frustums.py -t results/nerf/transforms.json -o results/nerf
```

Then, collect the `results/nerf/transforms-out.json` together with the image dataset to run NeRF task using Instant-NGP.