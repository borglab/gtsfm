#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <dataset_name> [tracker]"
  echo "Example: $0 gerrard-hall"
  echo "Example: $0 gerrard-hall vggsfm"
  exit 1
fi

DATASET_NAME="$1"
TRACKER="${2:-vggt}"

if [[ "${TRACKER}" != "vggt" && "${TRACKER}" != "vggsfm" ]]; then
  echo "Error: tracker must be one of: vggt, vggsfm"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_DIR="${REPO_ROOT}/benchmarks/${DATASET_NAME}"
RESULTS_ROOT="${REPO_ROOT}/pipeline/results/${DATASET_NAME}"
CLUSTER_TREE_PATH="${RESULTS_ROOT}/1-partition/results/cluster_tree.pkl"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Error: dataset directory not found: ${DATASET_DIR}"
  exit 1
fi

BASELINE_DIR="$(find "${DATASET_DIR}" -mindepth 1 -maxdepth 4 -type d \( -name "sparse" -o -name "colmap" \) -print -quit)"
if [[ -z "${BASELINE_DIR}" ]]; then
  echo "Error: baseline directory not found. Expected one of:"
  echo "  any 'sparse' or 'colmap' directory under ${DATASET_DIR}"
  exit 1
fi

# Ensure conda activation works in non-interactive shells.
if [[ "${CONDA_DEFAULT_ENV:-}" != "gtsfm-v2" ]]; then
  if command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)" || true
  fi

  if [[ -z "${CONDA_EXE:-}" ]]; then
    for candidate in \
      "${HOME}/miniconda3/etc/profile.d/conda.sh" \
      "${HOME}/anaconda3/etc/profile.d/conda.sh" \
      "/opt/conda/etc/profile.d/conda.sh"; do
      if [[ -f "${candidate}" ]]; then
        # shellcheck disable=SC1090
        source "${candidate}"
        break
      fi
    done
  fi

  if ! command -v conda >/dev/null 2>&1; then
    echo "Error: conda command is not available in this shell."
    echo "Please run with gtsfm-v2 already active, or ensure conda is installed and initialized."
    exit 1
  fi

  conda activate gtsfm-v2
fi

# partition
python "${REPO_ROOT}/pipeline/1-partition/partition_metis_megaloc.py" \
  --dataset_dir "${DATASET_DIR}" \
  --output_root "${RESULTS_ROOT}/1-partition"

# vggt reconstruction + cluster ba
python "${REPO_ROOT}/pipeline/2-reconstruction/vggt/run_on_cluster.py" \
  --cluster_tree_path "${CLUSTER_TREE_PATH}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_root "${RESULTS_ROOT}/2-reconstruction/vggt_cluster_run" \
  --ba_tracker "${TRACKER}"

python "${REPO_ROOT}/pipeline/utils/check_tracks.py" \
  --recon_root "${RESULTS_ROOT}/2-reconstruction/vggt_cluster_run/results" \
  --images_root "${DATASET_DIR}" \
  --model_name vggt

python "${REPO_ROOT}/pipeline/2-reconstruction/vggt/run_on_cluster.py" \
  --cluster_tree_path "${CLUSTER_TREE_PATH}" \
  --dataset_dir "${DATASET_DIR}" \
  --output_root "${RESULTS_ROOT}/2-reconstruction/vggt_cluster_run" \
  --ba_tracker "${TRACKER}" \
  --use_ba \
  --ba_output_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run"

python "${REPO_ROOT}/pipeline/utils/check_tracks.py" \
  --recon_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run/results" \
  --images_root "${DATASET_DIR}" \
  --model_name vggt

python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
  --baseline "${BASELINE_DIR}" \
  --root "${RESULTS_ROOT}/2-reconstruction/vggt_cluster_run" \
  --recon_name vggt \
  --csv_output "${RESULTS_ROOT}/2-reconstruction/vggt_cluster_run/vggt_eval/cluster_pose_metrics.csv"
python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
  --baseline "${BASELINE_DIR}" \
  --root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run" \
  --recon_name vggt \
  --csv_output "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run/vggt_ba_eval/cluster_pose_metrics.csv"

# alignment
eval_reconstruction() {
  local current_model_dir="$1"
  local output_dir="$2"
  local recon_name
  recon_name="$(basename "${current_model_dir}")"
  python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs.py" \
    --baseline "${BASELINE_DIR}" \
    --current "${current_model_dir}" \
    --output "${output_dir}"
  python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
    --baseline "${BASELINE_DIR}" \
    --root "${output_dir}" \
    --recon_name "${recon_name}" \
    --csv_output "${output_dir}/vggt_eval/cluster_pose_metrics.csv"
}

## case 1
python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
  --cluster_tree_path "${CLUSTER_TREE_PATH}" \
  --input_root "${RESULTS_ROOT}/2-reconstruction/vggt_cluster_run" \
  --output_root "${RESULTS_ROOT}/4-alignment"
eval_reconstruction \
  "${RESULTS_ROOT}/4-alignment/results/merged_pre_ba" \
  "${RESULTS_ROOT}/4-alignment/results"

## case 2
python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
  --cluster_tree_path "${CLUSTER_TREE_PATH}" \
  --input_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run" \
  --input_model_name vggt \
  --output_root "${RESULTS_ROOT}/4-alignment-clusterba"
eval_reconstruction \
  "${RESULTS_ROOT}/4-alignment-clusterba/results/merged_pre_ba" \
  "${RESULTS_ROOT}/4-alignment-clusterba/results"

## case 3
python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
  --cluster_tree_path "${CLUSTER_TREE_PATH}" \
  --input_root "${RESULTS_ROOT}/2-reconstruction/vggt_cluster_run" \
  --output_root "${RESULTS_ROOT}/5-global_ba" \
  --run_colmap_ba \
  --convert_ba_to_txt
eval_reconstruction \
  "${RESULTS_ROOT}/5-global_ba/results/merged_colmap_ba_txt" \
  "${RESULTS_ROOT}/5-global_ba/results"

## case 4
python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
  --cluster_tree_path "${CLUSTER_TREE_PATH}" \
  --input_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run" \
  --output_root "${RESULTS_ROOT}/5-global_ba-cluster_ba" \
  --input_model_name vggt \
  --run_colmap_ba \
  --convert_ba_to_txt
eval_reconstruction \
  "${RESULTS_ROOT}/5-global_ba-cluster_ba/results/merged_colmap_ba_txt" \
  "${RESULTS_ROOT}/5-global_ba-cluster_ba/results"
