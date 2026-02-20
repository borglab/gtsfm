#!/usr/bin/env bash

set -euo pipefail

TRACKER="vggt"
SINGLE_CLUSTER=0

usage() {
  echo "Usage: $0 [tracker] [--single_cluster]"
  echo "Example: $0"
  echo "Example: $0 vggsfm"
  echo "Example: $0 vggt --single_cluster"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    vggt|vggsfm)
      TRACKER="$1"
      shift
      ;;
    --single_cluster)
      SINGLE_CLUSTER=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument '$1'"
      usage
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_ROOT="${REPO_ROOT}/benchmarks/eth3dmvs"
PARTITION_MODE_TAG="metis"
if [[ "${SINGLE_CLUSTER}" -eq 1 ]]; then
  PARTITION_MODE_TAG="single_cluster"
fi
RUN_TAG="${TRACKER}_${PARTITION_MODE_TAG}"
RESULTS_ROOT_BASE="${REPO_ROOT}/pipeline/results/eth3dmvs_${RUN_TAG}"

if [[ ! -d "${DATASET_ROOT}" ]]; then
  echo "Error: dataset directory not found: ${DATASET_ROOT}"
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

prepare_eval_baseline() {
  local source_baseline_dir="$1"
  local prepared_baseline_dir="$2"

  mkdir -p "${prepared_baseline_dir}"
  cp "${source_baseline_dir}/cameras.txt" "${prepared_baseline_dir}/cameras.txt"
  cp "${source_baseline_dir}/points3D.txt" "${prepared_baseline_dir}/points3D.txt"

  # Keep COLMAP image records, but normalize NAME to basename to match reconstruction outputs.
  awk '
    BEGIN {non_comment_idx = 0}
    {
      if ($0 ~ /^#/ || $0 ~ /^[[:space:]]*$/) {
        print $0
        next
      }

      if (non_comment_idx % 2 == 0) {
        name = $10
        sub(/^.*\//, "", name)
        printf "%s %s %s %s %s %s %s %s %s %s\n", $1, $2, $3, $4, $5, $6, $7, $8, $9, name
      } else {
        print $0
      }

      non_comment_idx++
    }
  ' "${source_baseline_dir}/images.txt" > "${prepared_baseline_dir}/images.txt"
}

run_scene() {
  local split="$1"
  local scene_name="$2"
  local dataset_dir="${DATASET_ROOT}/${split}/${scene_name}"
  local results_root="${RESULTS_ROOT_BASE}/${split}/${scene_name}"
  local cluster_tree_path="${results_root}/1-partition/results/cluster_tree.pkl"
  local scene_images_dir
  local baseline_dir
  local eval_baseline_dir

  echo "========================================"
  echo "Running ETH3D scene: ${split}/${scene_name}"
  echo "Dataset: ${dataset_dir}"
  echo "Results: ${results_root}"

  if [[ -d "${dataset_dir}/dslr_calibration_undistorted" ]]; then
    baseline_dir="${dataset_dir}/dslr_calibration_undistorted"
  else
    baseline_dir="$(find "${dataset_dir}" -mindepth 1 -maxdepth 4 -type d \( -name "sparse" -o -name "colmap" \) -print -quit)"
  fi

  if [[ -z "${baseline_dir}" ]]; then
    echo "Error: baseline directory not found for scene ${split}/${scene_name}."
    echo "Expected: ${dataset_dir}/dslr_calibration_undistorted"
    exit 1
  fi

  for required_file in cameras.txt images.txt points3D.txt; do
    if [[ ! -f "${baseline_dir}/${required_file}" ]]; then
      echo "Error: baseline file missing: ${baseline_dir}/${required_file}"
      exit 1
    fi
  done

  if [[ -d "${dataset_dir}/images/dslr_images_undistorted" ]]; then
    scene_images_dir="${dataset_dir}/images/dslr_images_undistorted"
  elif [[ -d "${dataset_dir}/images" ]]; then
    scene_images_dir="${dataset_dir}/images"
  else
    echo "Error: image directory not found for scene ${split}/${scene_name}."
    echo "Expected one of:"
    echo "  ${dataset_dir}/images/dslr_images_undistorted"
    echo "  ${dataset_dir}/images"
    exit 1
  fi

  eval_baseline_dir="${results_root}/_eval_baseline"
  prepare_eval_baseline "${baseline_dir}" "${eval_baseline_dir}"

  echo "Images: ${scene_images_dir}"
  echo "Baseline (source): ${baseline_dir}"
  echo "Baseline (eval): ${eval_baseline_dir}"

  # partition
  partition_cmd=(
    python "${REPO_ROOT}/pipeline/1-partition/partition_metis_megaloc.py"
    --dataset_dir "${dataset_dir}"
    --images_dir "${scene_images_dir}"
    --output_root "${results_root}/1-partition"
  )
  if [[ "${SINGLE_CLUSTER}" -eq 1 ]]; then
    partition_cmd+=(--single_cluster)
  fi
  "${partition_cmd[@]}"

  # vggt reconstruction + cluster ba
  python "${REPO_ROOT}/pipeline/2-reconstruction/vggt/run_on_cluster.py" \
    --cluster_tree_path "${cluster_tree_path}" \
    --dataset_dir "${dataset_dir}" \
    --images_root "${scene_images_dir}" \
    --output_root "${results_root}/2-reconstruction/vggt_cluster_run" \
    --ba_tracker "${TRACKER}"

  python "${REPO_ROOT}/pipeline/utils/check_tracks.py" \
    --recon_root "${results_root}/2-reconstruction/vggt_cluster_run/results" \
    --images_root "${dataset_dir}" \
    --model_name vggt

  python "${REPO_ROOT}/pipeline/2-reconstruction/vggt/run_on_cluster.py" \
    --cluster_tree_path "${cluster_tree_path}" \
    --dataset_dir "${dataset_dir}" \
    --images_root "${scene_images_dir}" \
    --output_root "${results_root}/2-reconstruction/vggt_cluster_run" \
    --ba_tracker "${TRACKER}" \
    --use_ba \
    --ba_output_root "${results_root}/3-cluster_ba/vggt_cluster_run"

  python "${REPO_ROOT}/pipeline/utils/check_tracks.py" \
    --recon_root "${results_root}/3-cluster_ba/vggt_cluster_run/results" \
    --images_root "${dataset_dir}" \
    --model_name vggt

  python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
    --baseline "${eval_baseline_dir}" \
    --root "${results_root}/2-reconstruction/vggt_cluster_run" \
    --recon_name vggt \
    --csv_output "${results_root}/2-reconstruction/vggt_cluster_run/vggt_eval/cluster_pose_metrics.csv"

  python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
    --baseline "${eval_baseline_dir}" \
    --root "${results_root}/3-cluster_ba/vggt_cluster_run" \
    --recon_name vggt \
    --csv_output "${results_root}/3-cluster_ba/vggt_cluster_run/vggt_ba_eval/cluster_pose_metrics.csv"

  eval_reconstruction() {
    local current_model_dir="$1"
    local output_dir="$2"
    local fallback_model_dir="${3:-}"
    local recon_name

    if [[ ! -f "${current_model_dir}/images.txt" && ! -f "${current_model_dir}/images.bin" ]]; then
      if [[ -n "${fallback_model_dir}" && ( -f "${fallback_model_dir}/images.txt" || -f "${fallback_model_dir}/images.bin" ) ]]; then
        echo "Model not found at ${current_model_dir}; falling back to ${fallback_model_dir}"
        current_model_dir="${fallback_model_dir}"
      else
        echo "Error: evaluation model missing at ${current_model_dir}"
        if [[ -n "${fallback_model_dir}" ]]; then
          echo "Fallback model also missing at ${fallback_model_dir}"
        fi
        exit 1
      fi
    fi

    recon_name="$(basename "${current_model_dir}")"

    python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs.py" \
      --baseline "${eval_baseline_dir}" \
      --current "${current_model_dir}" \
      --output "${output_dir}"

    python "${REPO_ROOT}/gtsfm/evaluation/compare_colmap_outputs_by_cluster.py" \
      --baseline "${eval_baseline_dir}" \
      --root "${output_dir}" \
      --recon_name "${recon_name}" \
      --csv_output "${output_dir}/vggt_eval/cluster_pose_metrics.csv"
  }

  # case 1
  python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
    --cluster_tree_path "${cluster_tree_path}" \
    --input_root "${results_root}/2-reconstruction/vggt_cluster_run" \
    --output_root "${results_root}/4-alignment"

  eval_reconstruction \
    "${results_root}/4-alignment/results/merged_pre_ba" \
    "${results_root}/4-alignment/results" \
    "${results_root}/4-alignment/results/vggt_original"

  # case 2
  python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
    --cluster_tree_path "${cluster_tree_path}" \
    --input_root "${results_root}/3-cluster_ba/vggt_cluster_run" \
    --input_model_name vggt \
    --output_root "${results_root}/4-alignment-clusterba"

  eval_reconstruction \
    "${results_root}/4-alignment-clusterba/results/merged_pre_ba" \
    "${results_root}/4-alignment-clusterba/results" \
    "${results_root}/4-alignment-clusterba/results/vggt_original"

  # case 3
  python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
    --cluster_tree_path "${cluster_tree_path}" \
    --input_root "${results_root}/2-reconstruction/vggt_cluster_run" \
    --output_root "${results_root}/5-global_ba" \
    --run_colmap_ba \
    --convert_ba_to_txt

  eval_reconstruction \
    "${results_root}/5-global_ba/results/merged_colmap_ba_txt" \
    "${results_root}/5-global_ba/results" \
    "${results_root}/5-global_ba/results/vggt_original"

  # case 4
  python "${REPO_ROOT}/pipeline/4-alignment/alignment.py" \
    --cluster_tree_path "${cluster_tree_path}" \
    --input_root "${results_root}/3-cluster_ba/vggt_cluster_run" \
    --output_root "${results_root}/5-global_ba-cluster_ba" \
    --input_model_name vggt \
    --run_colmap_ba \
    --convert_ba_to_txt

  eval_reconstruction \
    "${results_root}/5-global_ba-cluster_ba/results/merged_colmap_ba_txt" \
    "${results_root}/5-global_ba-cluster_ba/results" \
    "${results_root}/5-global_ba-cluster_ba/results/vggt_original"
}

mapfile -t SCENES < <(find "${DATASET_ROOT}" -mindepth 2 -maxdepth 2 -type d | sort)

if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "Error: no scene directories found under ${DATASET_ROOT}"
  exit 1
fi

echo "Found ${#SCENES[@]} ETH3D scenes."
echo "Run tag: ${RUN_TAG}"
echo "Results root base: ${RESULTS_ROOT_BASE}"
FAILED_SCENES=()
SUCCEEDED_COUNT=0
for scene_path in "${SCENES[@]}"; do
  split="$(basename "$(dirname "${scene_path}")")"
  scene_name="$(basename "${scene_path}")"

  # Execute each scene in an isolated shell so a failure does not stop the full dataset run.
  set +e
  (
    set -euo pipefail
    run_scene "${split}" "${scene_name}"
  )
  scene_rc=$?
  set -e

  if [[ ${scene_rc} -ne 0 ]]; then
    FAILED_SCENES+=("${split}/${scene_name}")
    echo "Scene failed (${scene_rc}): ${split}/${scene_name}"
  else
    SUCCEEDED_COUNT=$((SUCCEEDED_COUNT + 1))
    echo "Scene completed: ${split}/${scene_name}"
  fi
done

echo "Completed ETH3D scenes: ${SUCCEEDED_COUNT}/${#SCENES[@]} succeeded."
if [[ ${#FAILED_SCENES[@]} -gt 0 ]]; then
  echo "Failed scenes (${#FAILED_SCENES[@]}):"
  printf '  - %s\n' "${FAILED_SCENES[@]}"
fi
