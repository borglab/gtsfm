#!/usr/bin/env bash

set -euo pipefail
export HF_HOME=/nethome/xzhang979/nvme/cache

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: $0 <dataset_name> [tracker] [--reconstruction_method {vggt_cluster|pi3}]"
  echo "Example: $0 gerrard-hall"
  echo "Example: $0 gerrard-hall vggsfm"
  echo "Example: $0 gerrard-hall --reconstruction_method pi3"
  exit 0
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset_name> [tracker] [--reconstruction_method {vggt_cluster|pi3}]"
  echo "Example: $0 gerrard-hall"
  echo "Example: $0 gerrard-hall vggsfm"
  echo "Example: $0 gerrard-hall --reconstruction_method pi3"
  exit 1
fi

DATASET_NAME="$1"
shift

TRACKER="vggt"
RECONSTRUCTION_METHOD="vggt_cluster"

if [[ $# -gt 0 ]]; then
  case "$1" in
    vggt|vggsfm|colmap)
      TRACKER="$1"
      shift
      ;;
  esac
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reconstruction_method)
      if [[ $# -lt 2 ]]; then
        echo "Error: --reconstruction_method requires a value"
        exit 1
      fi
      RECONSTRUCTION_METHOD="$2"
      if [[ "${RECONSTRUCTION_METHOD}" != "vggt_cluster" && "${RECONSTRUCTION_METHOD}" != "pi3" ]]; then
        echo "Error: --reconstruction_method must be one of: vggt_cluster, pi3"
        exit 1
      fi
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 <dataset_name> [tracker] [--reconstruction_method {vggt_cluster|pi3}]"
      echo "Example: $0 gerrard-hall"
      echo "Example: $0 gerrard-hall vggsfm"
      echo "Example: $0 gerrard-hall --reconstruction_method pi3"
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'"
      exit 1
      ;;
  esac
done

if [[ "${TRACKER}" != "vggt" && "${TRACKER}" != "vggsfm" && "${TRACKER}" != "colmap" ]]; then
  echo "Error: tracker must be one of: vggt, vggsfm, colmap"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DATASET_DIR="${REPO_ROOT}/benchmarks/${DATASET_NAME}"
if [[ "${RECONSTRUCTION_METHOD}" == "pi3" ]]; then
  RESULTS_ROOT="${REPO_ROOT}/pipeline/results/${DATASET_NAME}_pi3"
else
  RESULTS_ROOT="${REPO_ROOT}/pipeline/results/${DATASET_NAME}"
fi
CLUSTER_TREE_PATH="${RESULTS_ROOT}/1-partition/results/cluster_tree.pkl"

RECON_RUN_NAME="vggt_cluster_run"
RECON_MODEL_NAME="${TRACKER}"
if [[ "${RECONSTRUCTION_METHOD}" == "pi3" ]]; then
  RECON_RUN_NAME="pi3_run"
  RECON_MODEL_NAME="pi3"
fi
RECON_OUTPUT_ROOT="${RESULTS_ROOT}/2-reconstruction/${RECON_RUN_NAME}"

if [[ ! -d "${DATASET_DIR}" ]]; then
  echo "Error: dataset directory not found: ${DATASET_DIR}"
  exit 1
fi

BASELINE_DIR="$(find "${DATASET_DIR}" -mindepth 1 -maxdepth 4 -type d \( -name "sparse" -o -name "colmap" -o -name "sfm" \) -print -quit)"
if [[ -z "${BASELINE_DIR}" ]]; then
  echo "Error: baseline directory not found. Expected one of:"
  echo "  any 'sparse', 'colmap', or 'sfm' directory under ${DATASET_DIR}"
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

# reconstruction
if [[ "${RECONSTRUCTION_METHOD}" == "pi3" ]]; then
  python "${REPO_ROOT}/pipeline/2-reconstruction/Pi3/run_on_cluster.py" \
    --cluster_tree_path "${CLUSTER_TREE_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_root "${RECON_OUTPUT_ROOT}" \
    --model_name "${RECON_MODEL_NAME}"
else
  python "${REPO_ROOT}/pipeline/2-reconstruction/vggt/run_on_cluster.py" \
    --cluster_tree_path "${CLUSTER_TREE_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_root "${RECON_OUTPUT_ROOT}" \
    --ba_tracker "${TRACKER}"
fi

python "${REPO_ROOT}/pipeline/utils/check_tracks.py" \
  --recon_root "${RECON_OUTPUT_ROOT}/results" \
  --images_root "${DATASET_DIR}" \
  --model_name "${RECON_MODEL_NAME}"

if [[ "${RECONSTRUCTION_METHOD}" != "pi3" ]]; then
  python "${REPO_ROOT}/pipeline/2-reconstruction/vggt/run_on_cluster.py" \
    --cluster_tree_path "${CLUSTER_TREE_PATH}" \
    --dataset_dir "${DATASET_DIR}" \
    --output_root "${RECON_OUTPUT_ROOT}" \
    --ba_tracker "${TRACKER}" \
    --use_ba \
    --ba_output_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run"

  python "${REPO_ROOT}/pipeline/utils/check_tracks.py" \
    --recon_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run/results" \
    --images_root "${DATASET_DIR}" \
    --model_name "${TRACKER}"
fi

python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
  --baseline "${BASELINE_DIR}" \
  --root "${RECON_OUTPUT_ROOT}" \
  --recon_name "${RECON_MODEL_NAME}" \
  --csv_output "${RECON_OUTPUT_ROOT}/${RECON_MODEL_NAME}_eval/cluster_pose_metrics.csv"

if [[ "${RECONSTRUCTION_METHOD}" != "pi3" ]]; then
  python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
    --baseline "${BASELINE_DIR}" \
    --root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run" \
    --recon_name "${TRACKER}" \
    --csv_output "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run/${TRACKER}_ba_eval/cluster_pose_metrics.csv"
fi

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
  --input_root "${RECON_OUTPUT_ROOT}" \
  --input_model_name "${RECON_MODEL_NAME}" \
  --output_root "${RESULTS_ROOT}/4-alignment"
eval_reconstruction \
  "${RESULTS_ROOT}/4-alignment/results/merged_pre_ba" \
  "${RESULTS_ROOT}/4-alignment/results"

if [[ "${RECONSTRUCTION_METHOD}" != "pi3" ]]; then
  ## case 2
  python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
    --cluster_tree_path "${CLUSTER_TREE_PATH}" \
    --input_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run" \
    --input_model_name "${TRACKER}" \
    --output_root "${RESULTS_ROOT}/4-alignment-clusterba"
  eval_reconstruction \
    "${RESULTS_ROOT}/4-alignment-clusterba/results/merged_pre_ba" \
    "${RESULTS_ROOT}/4-alignment-clusterba/results"
fi

## case 3
python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
  --cluster_tree_path "${CLUSTER_TREE_PATH}" \
  --input_root "${RECON_OUTPUT_ROOT}" \
  --input_model_name "${RECON_MODEL_NAME}" \
  --output_root "${RESULTS_ROOT}/5-global_ba" \
  --run_colmap_ba \
  --convert_ba_to_txt
eval_reconstruction \
  "${RESULTS_ROOT}/5-global_ba/results/merged_colmap_ba_txt" \
  "${RESULTS_ROOT}/5-global_ba/results"

if [[ "${RECONSTRUCTION_METHOD}" != "pi3" ]]; then
  ## case 4
  python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
    --cluster_tree_path "${CLUSTER_TREE_PATH}" \
    --input_root "${RESULTS_ROOT}/3-cluster_ba/vggt_cluster_run" \
    --output_root "${RESULTS_ROOT}/5-global_ba-cluster_ba" \
    --input_model_name "${TRACKER}" \
    --run_colmap_ba \
    --convert_ba_to_txt
  eval_reconstruction \
    "${RESULTS_ROOT}/5-global_ba-cluster_ba/results/merged_colmap_ba_txt" \
    "${RESULTS_ROOT}/5-global_ba-cluster_ba/results"
fi
