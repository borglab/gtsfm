# Hierarchical Clustering Visualization

Interactive Three.js web visualization of the GTSfM hierarchical clustering pipeline for 3D reconstruction. The visualization shows the complete end-to-end process of how individual VGGT cluster reconstructions are progressively merged into a complete building model.

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

Each cluster folder must contain `points3D.txt` (COLMAP format with 3D coordinates and RGB color) and `images.txt` (camera poses with quaternions and translations). The tree structure is defined in `js/data-loader-vggt.js` — update `getStructure()` if the hierarchy changes.

### 2. Start a local server

```bash
cd visualization/hierarchy-web
python3 -m http.server 8080
```

### 3. Open in browser

Navigate to [http://localhost:8080/hierarchy-vggt.html](http://localhost:8080/hierarchy-vggt.html)

## Controls

| Button | Action |
|--------|--------|
| **Play/Pause** | Auto-advance through all timeline events |
| **Prev/Next** | Step through events one at a time |
| **Reset** | Return to the first event |
| **Record** | Start/stop recording the visualization as a `.webm` video |

Mouse: drag to orbit, scroll to zoom, right-click drag to pan. Click anywhere on the timeline progress bar to jump to a specific event.

## Recording

Click the **Record** button to start capturing the visualization. The button turns red and pulses while recording. Click **Stop** to end the recording — the video file automatically downloads as a `.webm` file (timestamped filename) to your browser's default download location.

---

## How the Visualization Works

The visualization displays the hierarchical clustering process used by the GTSfM pipeline. The pipeline reconstructs a 3D scene by first running VGGT (Visual Geometry Grounded Transformer) on small clusters of images, then progressively merging neighboring clusters until a single complete reconstruction is produced. This visualization animates that process step by step.

### What You See

1. **Leaf clusters appear** — Individual VGGT reconstructions (point clouds) are placed on screen in allocated tiles
2. **Clusters merge** — When sibling clusters merge, individual 3D points smoothly fly from their child-cluster positions to their merged positions
3. **Final building** — The last merge event produces the complete 3D reconstruction of the building

The visualization is driven by a **timeline of events**. Leaf placement events happen first (deepest clusters first, then shallower ones), followed by merge events (deepest merges first, up to the root).

---

## Algorithms

### 1. Scene Orientation (Camera-Based)

**Problem**: The SfM reconstruction places the building in an arbitrary coordinate system determined by the pipeline's internal optimization. There is no guarantee that "up" in the reconstruction corresponds to "up" on screen. Without correction, the building appears sideways or at an arbitrary angle.

**Solution**: We compute the correct orientation mathematically from the actual camera poses stored in `images.txt` (COLMAP format).

**How it works**:

1. **Parse camera extrinsics**: Each camera in `images.txt` has a quaternion `(qw, qx, qy, qz)` representing the rotation from world to camera frame, and a translation `(tx, ty, tz)` in the camera frame.

2. **Compute camera world positions**: For each camera, the world-space position is `C = -R^T * t`, where `R` is the 3x3 rotation matrix built from the quaternion and `R^T` is its transpose (camera-to-world transform).

3. **Find the "up" direction**: Each camera has an "up" direction in world space computed as `R^T * [0, -1, 0]` (negated because COLMAP's Y-axis points down). Averaging all camera up-vectors across the dataset gives the true vertical axis in the reconstruction's coordinate system. In practice, photographers hold cameras roughly level, so these vectors are highly consistent (measured consistency of 0.9757 out of 1.0 for the Gerrard Hall dataset).

4. **Find the "front" direction**: The vector from the camera centroid (average position of all cameras) to the building centroid (average position of all 3D points) gives the dominant viewing direction. This is orthogonalized against the up-vector to ensure perpendicularity.

5. **Build rotation matrix**: From the up and front vectors, an orthonormal basis is constructed using cross products:
   - `scene_Y` = normalized up direction
   - `scene_Z` = front direction, orthogonalized against scene_Y
   - `scene_X` = cross(scene_Y, scene_Z)

   These three vectors form the rows of a 3x3 rotation matrix that transforms the reconstruction's world frame into Three.js screen space (Y-up, Z-toward-viewer).

**Why this approach**: This is fully data-driven — no hardcoded rotation angles. When a new dataset is provided (different building, different camera arrangement), the orientation is automatically computed from that dataset's camera poses.

**Implementation**: `loadCameraExtrinsics()` and `computeSceneOrientation()` in `js/data-loader-vggt.js`.

---

### 2. Cluster Placement Algorithm (Squareness-Optimized Layout)

**Problem**: The hierarchical tree of clusters must be arranged on screen so that (a) each cluster fills its allocated space well, (b) sibling clusters are adjacent (for natural merge animations), and (c) the screen is filled without excessive white space.

**Solution**: A recursive rectangle subdivision algorithm that exhaustively searches all possible row/column groupings at each tree level and picks the one that maximizes the worst-case tile squareness.

**How it works**:

1. **Compute the root rectangle**: A 16:9 rectangle is sized based on the number of leaf clusters and the largest cluster's bounding sphere radius:
   ```
   tileSize = maxRadius * 2.2
   totalArea = tileSize^2 * leafCount * 1.1
   ROOT_H = sqrt(totalArea / (16/9))
   ROOT_W = ROOT_H * 16/9
   ```

2. **Weight each child by leaf count**: At each internal node, children are assigned weights equal to their number of descendant leaf clusters. For example, at the root with children `[vggt(1), C_1(1), C_2(1), C_3(1), C_4/merged(3)]`, the weights are `[1, 1, 1, 1, 3]`.

3. **Generate all possible partitions**: For `n` children, there are `2^(n-1)` ways to partition them into contiguous groups (each of the `n-1` gaps between children is either a row/column break or not). For 5 children, that's 16 possible groupings.

4. **Evaluate each partition in both orientations**: Each partition is tested as both "rows" (groups stacked vertically) and "columns" (groups side by side horizontally) — giving `2^(n-1) * 2` total candidates.

   For each candidate:
   - Each group (row/column) receives height/width proportional to its total weight
   - Within each group, children receive width/height proportional to their individual weights
   - The **squareness** of each child's tile is computed as `min(width, height) / max(width, height)` (1.0 = perfect square, 0.0 = infinitely elongated)
   - The **worst-case squareness** across all tiles is recorded

5. **Pick the best**: The partition with the highest worst-case squareness wins. This ensures no tile is excessively elongated.

6. **Recurse**: Each child's allocated rectangle is then recursively subdivided using the same algorithm for its own children.

7. **Fit clusters into tiles**: Each leaf cluster's 3D point cloud is scaled to fit inside its tile:
   ```
   fitDim = min(tile_width, tile_height) * 0.85
   fitScale = fitDim / (2 * cluster_radius)
   ```
   The 0.85 factor provides a 15% margin so clusters don't touch tile edges. The division by `2 * radius` (diameter) ensures the full bounding sphere fits within the tile.

**Example**: For the Gerrard Hall dataset with weights `[1,1,1,1,3]`, the winning layout is two rows — `[vggt, C_1, C_2]` on top and `[C_3, C_4/merged]` on bottom — with a worst-case tile squareness of ~0.43, compared to ~0.25 for the naive all-columns approach.

**Implementation**: `assignLeafTiles()` in `js/layout-engine-squareness.js`.

---

### 3. Merge Animation (Per-Point Alpha Blending)

**Problem**: When two or more child clusters merge into a parent cluster, the visualization needs to show this transition smoothly. The child point clouds and merged point cloud have different numbers of points and different point positions — some points exist only in children, some only in the merged result, and some correspond between both.

**Solution**: Per-point interpolation where each point individually transitions from its child position to its merged position, with three categories of points handled differently.

**How it works**:

1. **Point matching**: Before any animation, a nearest-neighbor matching step establishes correspondence between child points and merged points.
   - For each child point, the nearest merged point is found (brute-force nearest neighbor in 3D)
   - Candidates are sorted by distance. A threshold is set at `4x the 90th percentile distance` — pairs below this threshold are considered matches
   - A greedy 1-to-1 assignment ensures each merged point is matched to at most one child point

2. **Three categories of points during a merge**:
   - **Matched points** (~60-80% of points): These exist in both child and merged results. During the animation, each point smoothly interpolates from its child-cluster position to its merged position using cubic easing (`easeInOutCubic`). Three temporary `THREE.Points` objects are created — one for matched, one for child-only, one for merged-only.
   - **Child-only points** (exist only in children, not in merged): These fly toward their nearest merged point and fade out slightly (`opacity 1 → 0.7`) during the transition.
   - **Merged-only points** (exist only in merged result, not in children): These fly from their nearest child point to their final merged position, fading in during the transition.

3. **Animation timeline**: Each merge transition lasts 2.5 seconds. The easing function `easeInOutCubic` produces smooth acceleration and deceleration. At the end of the transition, the temporary point clouds are disposed and the merged cluster's original point cloud is made visible.

4. **Auto-play guard**: The auto-play system waits for the current merge animation to fully complete before advancing to the next event. This prevents race conditions where overlapping animations could produce incorrect visual states.

**Implementation**: `playMergeTransition()` and `update()` in `js/animation-engine-squareness.js`, point matching in `computePointMatching()` in `js/data-loader-vggt.js`.

---

### 4. The Final Visualization

The complete visualization sequence for the Gerrard Hall dataset has **9 events**:

- **Events 1-7** (leaf placements): The 7 leaf VGGT clusters appear one at a time in their allocated tiles, with a fade-in animation. Deeper clusters in the tree appear first.
- **Event 8** (intermediate merge): The 3 clusters in the C_4 subtree (`C_4/vggt`, `C_4_1/vggt`, `C_4_2/vggt`) merge into `C_4/merged`. Individual points fly from their child positions to their merged positions.
- **Event 9** (final merge): All remaining clusters (`vggt`, `C_1/vggt`, `C_2/vggt`, `C_3/vggt`, `C_4/merged`) merge into the root `merged` cluster, producing the complete 3D reconstruction of Gerrard Hall. The final point cloud is displayed centered on screen.

The end result shows the fully reconstructed building with RGB colors from the original photographs, oriented upright and facing the viewer.

---

## File Architecture

| File | Purpose |
|------|---------|
| `hierarchy-vggt.html` | Main page with Three.js imports, UI controls, and CSS styling |
| `js/data-loader-vggt.js` | Loads `points3D.txt` and `images.txt`, defines tree structure, computes scene orientation and point matching |
| `js/layout-engine-squareness.js` | Recursive rectangle layout with exhaustive partition search for optimal tile squareness |
| `js/animation-engine-squareness.js` | Timeline event system with per-point merge interpolation and fade animations |
| `js/main-hierarchy-vggt.js` | App entry point: Three.js scene, camera, renderer, orbit controls, UI wiring, recording |
| `js/camera-engine.js` | Camera flythrough support for automated camera movements |
| `js/interaction-engine.js` | Mouse hover/click interaction with point cloud clusters |

## Adapting to New Datasets

1. Place the new GTSfM results in `data/<dataset-name>/results/`
2. Update `getStructure()` in `js/data-loader-vggt.js` to reflect the new cluster tree hierarchy
3. Update the data path in `loadPointCloud()` and `loadCameraExtrinsics()`
4. The orientation and layout algorithms will adapt automatically to the new data — no hardcoded values need changing
