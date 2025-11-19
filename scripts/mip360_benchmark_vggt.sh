# Script to launch jobs over various ETH3D datasets & front-ends.
# See https://www.eth3d.net/data/schoeps2017cvpr.pdf for more details.

USER_ROOT=$1
CLUSTER_CONFIG=$2

now=$(date +"%Y%m%d_%H%M%S")

DATA_ROOT=/home/tdriver6/Documents/360_v2

# Includes all "high-resolution multi-view" datasets from 'training' split (i.e. w/ public GT data)
# See https://www.eth3d.net/datasets for more information.
datasets=(
    #bicycle
    #bonsai
    #counter
    #garden
    #kitchen
    #room
    #stump
    flowers
    treehill
)

for dataset in ${datasets[@]}; do
    echo "----------------------------------------"
    echo "Dataset: ${dataset}"
    echo "----------------------------------------"

    images_dir="${DATA_ROOT}/${dataset}/images"
    colmap_files_dirpath="${DATA_ROOT}/${dataset}/sparse/0"

    OUTPUT_ROOT=${USER_ROOT}/${now}/${dataset}_results
    mkdir -p $OUTPUT_ROOT

    ./run --loader colmap --dataset_dir $colmap_files_dirpath --images_dir $images_dir --config_name vggt_mip360 --graph_partitioner metis --worker_memory_limit=24GB \
    2>&1 | tee $OUTPUT_ROOT/out.log

    mv results $OUTPUT_ROOT
done
