#!/usr/bin/env bash
# Build a COLMAP frontend database.db (keypoints + geometrically-verified two_view_geometries) for a
# dataset, to feed the GTSFM colmapdb configs (vggt[_omega]_sift_frontend_megaloc_phototourism_colmapdb).
# This replaces GTSFM's Dask SIFT frontend, which OOMs on large (1000+-image) scenes — COLMAP's native
# CUDA SIFT + vocab-tree matching is far faster and memory-efficient. Run on a GPU node.
#
# Usage:
#   bash scripts/build_colmap_db.sh <DATASET_DIR> [IMAGE_SUBDIR=images] [MAX_FEATURES=8192]
# Example:
#   bash scripts/build_colmap_db.sh /storage/scratch1/5/<user>/.../benchmarks/st_peters_square
#
# Produces <DATASET_DIR>/database.db (+ a vocab tree alongside it). Then run, e.g.:
#   ./run --config_name vggt_omega_sift_frontend_megaloc_phototourism_colmapdb \
#         loader._target_=gtsfm.loader.Colmap loader.dataset_dir=<DATASET_DIR> \
#         +loader.colmap_files_subdir=sparse +loader.use_gt_intrinsics=false
set -euo pipefail

DATASET_DIR="${1:?usage: build_colmap_db.sh <DATASET_DIR> [IMAGE_SUBDIR] [MAX_FEATURES]}"
IMAGE_SUBDIR="${2:-images}"
MAX_FEATURES="${3:-8192}"

# Resolve any ~/scratch symlink to the real /storage path so the COLMAP apptainer container can see it.
DATASET_DIR="$(cd "$DATASET_DIR" && pwd -P)"
IMG="$DATASET_DIR/$IMAGE_SUBDIR"
DB="$DATASET_DIR/database.db"
VOCAB="$DATASET_DIR/vocab_tree_flickr100K_words256K.bin"
# Bind the data dir into the (containerized) COLMAP so it can read images + write the db.
export APPTAINER_BIND="${APPTAINER_BIND:+$APPTAINER_BIND,}$DATASET_DIR"

[ -d "$IMG" ] || { echo "ERROR: image dir not found: $IMG"; exit 1; }

# On PACE, `colmap` is provided as a module-defined apptainer wrapper. Make it available even when this
# script runs as a (non-interactive) subprocess: init Lmod, then load the module.
if ! command -v colmap >/dev/null 2>&1; then
  if ! command -v module >/dev/null 2>&1; then
    for f in /etc/profile.d/z00_lmod.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
      [ -f "$f" ] && source "$f" && break
    done
  fi
  module load colmap 2>/dev/null || true
fi
command -v colmap >/dev/null 2>&1 || {
  echo "ERROR: 'colmap' is not available. Run 'module load colmap' first (and 'export -f colmap' if invoking"
  echo "       this as 'bash …'), or 'source scripts/build_colmap_db.sh …' so the colmap function is in scope."
  exit 1
}

echo "📂 dataset : $DATASET_DIR"
echo "🖼️  images  : $IMG  ($(ls "$IMG" | wc -l) files)"
echo "🗄️  database: $DB"

# 1) Feature extraction — try GPU SIFT, fall back to CPU if it can't get an OpenGL context (headless node).
echo "🔍 feature_extractor (GPU SIFT, max_num_features=$MAX_FEATURES)…"
if ! colmap feature_extractor --database_path "$DB" --image_path "$IMG" \
      --SiftExtraction.use_gpu 1 --SiftExtraction.max_num_features "$MAX_FEATURES"; then
  echo "⚠️  GPU SIFT failed (likely no GL context on this headless node) — retrying on CPU…"
  rm -f "$DB"
  colmap feature_extractor --database_path "$DB" --image_path "$IMG" \
      --SiftExtraction.use_gpu 0 --SiftExtraction.max_num_features "$MAX_FEATURES"
fi

# 2) Matching — vocab-tree (sublinear). Exhaustive is O(N^2) and far too slow for 1000+-image scenes.
if [ ! -f "$VOCAB" ]; then
  echo "⬇️  fetching vocab tree…"
  wget -nc -q https://demuc.de/colmap/vocab_tree_flickr100K_words256K.bin -O "$VOCAB"
fi
echo "🔗 vocab_tree_matcher (GPU)…"
colmap vocab_tree_matcher --database_path "$DB" \
    --VocabTreeMatching.vocab_tree_path "$VOCAB" --SiftMatching.use_gpu 1

# 3) Report — want non-zero "verified pairs" (that's two_view_geometries, what GTSFM reads).
echo "📊 database_stats:"
colmap database_stats --database_path "$DB"
echo "✅ done → $DB"
