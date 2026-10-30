#!/bin/bash
#SBATCH --job-name=gtsfm_british_museum_rmetis
#SBATCH --partition=hays-lab
#SBATCH --account=hays-lab
#SBATCH --qos=short
#SBATCH --output=logs/gtsfm_british_museum_rmetis_%j.out
#SBATCH --error=logs/gtsfm_british_museum_rmetis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=a40:2
#SBATCH --mem-per-gpu=48G
#SBATCH --exclude="baymax,puma"
#SBATCH --time=04:00:00

set -euo pipefail
export PYTHONUNBUFFERED=TRUE

SCENE_NAME="british_museum"

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: sbatch $0 <input_root_directory> <output_root_directory> [scene_name]"
  echo "Example: sbatch $0 data outputs/recursive_metis_british_museum"
  echo "Runs GTSFM on single scene: $SCENE_NAME"
  exit 1
fi

if [[ $# -eq 3 ]]; then
  SCENE_NAME="$3"
fi

INPUT_ROOT_DIR="$1"
OUTPUT_ROOT_DIR="$2"
CONFIG_NAME="${CONFIG_NAME:-vggt_sift_frontend_megaloc_phototourism}"
LOADER_NAME="colmap_no_intrinsics"
COLMAP_FILES_SUBDIR="/coc/flash5/akrishnan86/pt_ground_truth/${SCENE_NAME}/dense/sparse"
GRAPH_PARTITIONER="${GRAPH_PARTITIONER:-recursive_metis}"
CLUSTER_CFG="cluster_gpu_slurm_${SLURM_JOB_ID}.yaml"

dataset_dir="${INPUT_ROOT_DIR%/}/${SCENE_NAME}"
if [[ ! -d "$dataset_dir" ]]; then
  echo "Error: scene directory does not exist: $dataset_dir"
  exit 1
fi

source ~/.bashrc
cd /coc/flash5/akrishnan86/gtsfm
mkdir -p logs
conda activate gtsfm-v1

python -c "import dask_cuda; print('dask-cuda OK')"

NUM_GPUS=$(python - <<'PY'
import os
cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
print(len([x for x in cvd.split(",") if x.strip()])) if cvd else print(1)
PY
)

{
  echo "workers:"
  for ((i=1; i<=NUM_GPUS; i++)); do
    echo "  - host: localhost"
    echo "    gpus: 1"
    echo "    system_memory: \"40GB\""
  done
} > "gtsfm/configs/${CLUSTER_CFG}"

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Generated gtsfm/configs/${CLUSTER_CFG} with ${NUM_GPUS} workers"

output_root="${OUTPUT_ROOT_DIR%/}"

echo "Running GTSFM on: $dataset_dir"
echo "  config: $CONFIG_NAME, loader: $LOADER_NAME, partitioner: $GRAPH_PARTITIONER, output: $output_root"
if [[ -n "${MIN_SCORE:-}" ]]; then
  echo "  retriever min_score: $MIN_SCORE"
fi
export GTSFM_CACHE_SUBDIR="pt_${SCENE_NAME}_retri_fetzer"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS - 1)))
export PYTORCH_NVML_BASED_CUDA_CHECK=1

./run \
  --config_name "$CONFIG_NAME" \
  --loader "$LOADER_NAME" \
  --dataset_dir "$dataset_dir" \
  --num_workers "$NUM_GPUS" --cluster_config "$CLUSTER_CFG" \
  --graph_partitioner "$GRAPH_PARTITIONER" \
  --output_root "$output_root" \
  --worker_memory_limit "48GB" \
  +loader.colmap_files_subdir="$COLMAP_FILES_SUBDIR" \
  '~graph_partitioner.min_cameras_to_partition' \
  '~graph_partitioner.split_oversized_nodes' \
  ${MIN_SCORE:+image_pairs_generator.retriever.min_score=$MIN_SCORE}
