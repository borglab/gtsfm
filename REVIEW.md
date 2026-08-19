# PR #1117 — VGGT Verified Pipeline

**16 commits · +1913 / −80 · 23 files · targets `mapping-vggt`**

This branch turns the per-cluster VGGT reconstruction into a **verified-view-graph pipeline** and stacks
four accuracy/robustness features on top of it, ending with a **VGGT-Omega** geometry arm. It's large
because it was developed as one experimental thread; this doc decomposes it into **6 self-contained
feature units** that map 1:1 to the clean branches we'll land into `master`.

## Headline result — IMC Phototourism *Grand Place Brussels* (234 images)

| Arm (config) | AUC@3° (full) | AUC@3° (constructed) | Cameras |
|---|---|---|---|
| Prior VGGT baseline (non-verified) | ~0.675 | — | ~219 |
| **VGGT — verified** (`vggt_sift_…_verified`) | **0.6968** | 0.7213 | **230** |
| **VGGT-Omega — verified** (`vggt_omega_…_verified`) | **0.7191** | **0.7380** | **231** |

Controlled A/B: the omega vs VGGT rows differ **only** in the per-cluster geometry model — same frontend,
partition, Fetzer, and retri.

## How to review this efficiently

1. **Start with the configs** — they are the table of contents. Diff
   `gtsfm/configs/vggt_sift_frontend_megaloc_phototourism_verified.yaml` (every new flag is commented in place).
2. **Then `scene_optimizer.py`** (+205) — the orchestration: global two-view verification → verified
   partition → global Fetzer → post-merge retriangulation.
3. **Then the per-feature files** in the order below.
4. **Mechanical vs load-bearing:** `cluster_vggt_omega_with_frontend.py` (+404) and
   `vggt_omega_geometry_transformer.py` (+347) are **vendored from PR #1116** (Harneet) — review at the
   interface level, not line-by-line. The load-bearing *new* logic is in `scene_optimizer.py`,
   `cluster_merging.py`, `cluster_vggt_with_frontend.py`, and `view_graph_calibration.py`.

## New config flags (the feature switches — all default to the prior behavior)

| Flag | Where | Default | Effect |
|---|---|---|---|
| `use_verified_pipeline` | top-level | `false` | global two-view verification + verified-graph METIS partition + post-merge retriangulation |
| `use_view_graph_calibration` | optimizer | `false` | per-edge scipy Fetzer focal refinement |
| `use_global_view_graph_calibration` | optimizer | `false` | single **global** Fetzer over the verified graph (takes precedence) |
| `calibration_prior_focal_sigma` | ba_options | n/a | anchors focals in cluster (5px) & merge (10px) BAs |
| `reuse_global_correspondences` | optimizer | `false` | build per-cluster tracks from the global verified frontend (skip per-cluster frontend) — pure speedup |
| `recover_trackless_cameras_in_retriangulation` | merging | `false` | inject good-pose/no-track cams into the post-merge retri for a geometric second chance |

**Every flag off ⇒ byte-identical to the prior baseline.** This is the key reviewer reassurance: the diff
is additive and gated.

---

## Feature units → suggested clean branches (with landing order)

**Dependency order: ① must land first** (everything reads its `ClusterContext` fields + `use_verified_pipeline`).
**②③④⑤ are independent** of each other on top of ①. **⑥ is independent** but its config overlays ①–⑤.

### ① Verified view-graph pipeline *(foundation)*
- **Commits:** `aeba561a`, `86a5a3ca`, `1c5ef76f`, `1dc132ba`
- **What:** Runs one global two-view verification pass over the full MegaLoc retrieval graph, partitions
  METIS on the **verified** subgraph (edges where `TwoViewResult.valid()`), and adds a **post-merge
  retriangulation+BA** stage (`results/merged_retriangulated/`). Several follow-ups fix worker-OOM/dask-nesting
  by running the global frontend **inline** instead of via nested `worker_client()` submission.
- **Key files:** `scene_optimizer.py` (orchestration), `cluster_mvo.py`, `two_view_estimator.py`
  (`create_two_view_results_inline`), `frontend/correspondence_generator/*` (`generate_correspondences_inline`),
  `cluster_optimizer_base.py` (new `ClusterContext` fields).
- **Review focus:** the inline-vs-dask execution model (the OOM fixes) and the verified-graph construction.

### ② Global Fetzer focal calibration + focal-flow anchoring
- **Commits:** `e08be7c8`, `20adfc40`, `7b641621`
- **What:** Estimates focals with scipy Fetzer over the verified F-matrices (`compute_global_view_graph_intrinsics`),
  then **anchors** those focals through the cluster BA (σ=5px) and merge BA (σ=10px) so BAs optimize
  *poses, not focals* (kills the focal/depth ambiguity that was diverging separator cameras across clusters).
- **Key files:** `graph_optimizer/view_graph_estimator/view_graph_calibration.py` (+44), `scene_optimizer.py`,
  the `ba_options.calibration_prior_*` plumbing.
- **Note:** Fetzer initializes from the loader heuristic (1.2·maxdim) and **never reads VGGT/omega focals** —
  important for the omega arm (its anisotropic fx/fy is irrelevant; a single-focal `Cal3Bundler` is produced).

### ③ Peak frontend (config + PoseLib verifier)
- **Commits:** `170dc610`, `77796d20`
- **What:** Adopts the "gp-glomap-parity" frontend — `PoseLibVerifier` (5-point + LO-RANSAC), 8192 ColmapSIFT
  keypoints, 30/0.15 inlier gate — and retunes the Brussels baseline + verified configs.
- **Key files:** `configs/vggt_sift_frontend_megaloc_phototourism.yaml` (+ verified), `two_view_estimator.py`,
  **`pyproject.toml`** (+`poselib>=2.0`).
- **Review focus:** the new `poselib` dependency + verifier swap. Mostly config; lowest-risk unit.

### ④ Global correspondence reuse *(speedup)*
- **Commits:** `1bc73192`, `ede5a9ba`, `8d6948ba`
- **What:** Each cluster builds its 2D tracks from the **already-computed** global verified correspondences
  instead of re-running its own frontend (which only cache-read the same data, serially + redundantly across
  overlapping clusters). Pure speedup — per-edge `v_corr` is identical. *(One commit, `ede5a9ba`, reverts an
  experimental per-cluster triangulated-structure path that inflated clusters to 11k–15k tracks → OOM; it keeps
  the focal-flow fix. See "reverted experiments" below.)*
- **Key files:** `cluster_vggt_with_frontend.py` (the `reuse_global_correspondences` branch),
  `ClusterContext.global_v_corr_idxs_dict` / `global_keypoints`.
- **Cache token:** adds `/gcorr` to the optimizer `__repr__` (cluster-cache key).

### ⑤ Trackless-camera recovery
- **Commits:** `bbd79ff7` (build-side "all-measurements"/`/allkpts`), `1af1e43e` (retriangulation recovery)
- **What:** Two orthogonal levers to rescue cameras that get good poses but no VGGT-depth track. (a) The
  depth-lift build no longer lets zero-confidence VGGT depth *veto* a verified SIFT measurement (`/allkpts`).
  (b) Cameras dropped at the root merge's track filter are **captured and injected** into the post-merge
  retriangulation for a geometric (depth-independent) second chance; those that still can't triangulate are
  cleanly dropped. **Provably can't perturb the constructed cameras** (recovered cams have <15 tracks ⇒
  excluded from the retri BA factor graph).
- **Key files:** `cluster_merging.py` (+38, capture + `trackless_cameras` on the merge result),
  `scene_optimizer.py` (`_run_post_merge_retriangulation` inject + first-class diagnostic logging),
  `cluster_vggt_with_frontend.py` (build).
- **Result note:** on Brussels this recovers cam 49; 104/207 remain (they're a pose-constraint problem — see
  follow-ups).

### ⑥ VGGT-Omega geometry integration
- **Commits:** `dc9824af` (integration), `7a7c51ca` (preprocessing-dispatch fix)
- **What:** Vendors VGGT-Omega (from PR #1116) and runs it **through our verified optimizer**
  (`ClusterVGGTWithFrontend` via its injected `geometry_transformer`), *not* the bundled
  `ClusterVGGTOmegaWithFrontend` (a master-based copy lacking Fetzer/reuse/recovery). Two small enabling changes:
  (a) guard `transformer.config` access (omega has none) at `__init__` + `__repr__`; (b) **generalize image
  preprocessing to follow the transformer** — new `GeometryTransformer.load_image_batch` (base = VGGT loader,
  omega overrides → its own 512/16-aligned loader). This is the cleanest standalone refactor in the PR and is a
  **no-op for all VGGT configs**.
- **Key files:** `frontend/geometry_transformer.py` (+20, the abstraction),
  `frontend/vggt_omega_geometry_transformer.py` (+347, vendored),
  `cluster_optimizer/cluster_vggt_omega_with_frontend.py` (+404, vendored), `cluster_vggt.py` (loader dispatch),
  `cluster_optimizer/__init__.py` (register), `configs/vggt_omega_*` (×2), `.gitmodules` +
  `thirdparty/vggt-omega` submodule, `scripts/download_model_weights.sh`, `utils/torch.py`.
- **⚠️ Reviewer/ops notes:** weights are **gated, `cc-by-nc-4.0`, non-redistributable**, **CUDA-only**.
  Submodule + `--hf_token` download required; omega module imports are lazy so non-omega paths/tests are
  unaffected. **The geometry-transformer abstraction (a) is worth landing as its own tiny branch first** —
  omega then becomes a pure add-on.

---

## Reverted experiments (intentionally *not* in the final pipeline)
- **Per-cluster triangulated structure** (`use_triangulated_structure: true`) — inflated clusters to 11k–15k
  tracks → 85–200s BAs → worker OOM. Reverted to VGGT depth-lift (`ede5a9ba`); the flag remains but defaults
  `false`.

## Known limitations / planned follow-ups (out of scope for this PR)
1. **Brussels 104/207 still unrecovered** — they never acquire tracks at *any* stage, so they're carried
   through the Sim3 merge unconstrained (one is ~90°-flipped). This is a **pose-constraint problem →
   PnP/resection**, not a geometry-model one (omega didn't and couldn't fix it).
2. **One Sim3-flipped camera** (max rot err 90.05° vs 0.13° median) dominates the mean rotation error —
   separate AUC lever.
3. **Omega licensing/ops** — NC weights must stay out of any commercial/redistributed artifact.

## How to run / verify
```bash
git submodule update --init --recursive                 # omega arm only
bash scripts/download_model_weights.sh --hf_token <tok>  # omega arm only (CUDA)
uv run ./run --config_name vggt_omega_sift_frontend_megaloc_phototourism_verified \
  loader._target_=gtsfm.loader.Colmap loader.dataset_dir=$DATA \
  +loader.colmap_files_subdir=sfm_updated +loader.use_gt_intrinsics=false
```
Drop `_omega` from the config name for the VGGT arm. Metrics land in `results/merged_retriangulated/…json`
(`pose_auc_@3.0_deg`, `number_cameras_merged`).
