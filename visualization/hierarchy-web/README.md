# Hierarchical Clustering Visualization

Interactive Three.js web visualization of the GTSfM hierarchical clustering pipeline for 3D reconstruction. Shows how individual VGGT cluster reconstructions are progressively merged into a complete building model, with per-point alpha blending during merge transitions.

## Quick Start

### 1. Place GTSfM results

Copy the GTSfM `results/` output folder into this directory:

```
visualization/hierarchy-web/
├── data/
│   └── gerrard-hall-vggt/
│       └── results/
│           ├── vggt/                  # Root VGGT cluster
│           │   ├── points3D.txt
│           │   ├── images.txt
│           │   └── cameras.txt
│           ├── C_1/vggt/              # Leaf cluster
│           ├── C_2/vggt/              # Leaf cluster
│           ├── ...
│           ├── C_4/merged/            # Intermediate merge
│           └── merged/               # Final merged reconstruction
│               ├── points3D.txt
│               ├── images.txt
│               └── cameras.txt
├── hierarchy-vggt.html
├── js/
└── README.md
```

Each cluster folder must contain `points3D.txt` (COLMAP format) and `images.txt` (camera poses). The tree structure is defined in `js/data-loader-vggt.js` — update `getStructure()` if the hierarchy changes.

### 2. Start a local server

```bash
cd visualization/hierarchy-web
python3 -m http.server 8080
```

### 3. Open in browser

Navigate to [http://localhost:8080/hierarchy-vggt.html](http://localhost:8080/hierarchy-vggt.html)

## Features

- **Automatic orientation**: Parses COLMAP camera poses from `images.txt` to compute the correct scene orientation mathematically (no hardcoded rotations). Works for any dataset.
- **Squareness-optimized layout**: Recursively subdivides the screen into tiles, trying all possible row/column groupings to maximize tile squareness and minimize wasted space.
- **Per-point alpha blending**: During merge transitions, matched points smoothly interpolate from child to merged positions. Unmatched points fly to/from their nearest counterparts.
- **Recording**: Click the Record button to capture the visualization as a `.webm` video.

## Controls

| Button | Action |
|--------|--------|
| **Play/Pause** | Auto-advance through all timeline events |
| **Prev/Next** | Step through events one at a time |
| **Reset** | Return to the first event |
| **Record** | Start/stop recording as `.webm` video |

Mouse: drag to orbit, scroll to zoom, right-click drag to pan.

## Architecture

| File | Purpose |
|------|---------|
| `js/data-loader-vggt.js` | Loads point clouds and camera extrinsics, computes scene orientation from camera poses |
| `js/layout-engine-squareness.js` | Recursive rectangle layout with exhaustive partition search for optimal squareness |
| `js/animation-engine-squareness.js` | Timeline system with per-point merge interpolation |
| `js/main-hierarchy-vggt.js` | App entry point, Three.js scene, UI controls, and recording |
| `js/camera-engine.js` | Camera flythrough support |
| `js/interaction-engine.js` | Mouse interaction (hover/click on clusters) |

## Adapting to New Datasets

1. Place the new GTSfM results in `data/<dataset-name>/results/`
2. Update `getStructure()` in `js/data-loader-vggt.js` to reflect the new cluster tree
3. Update the data path in `loadPointCloud()` and `loadCameraExtrinsics()`
4. The orientation and layout will adapt automatically
