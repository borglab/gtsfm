# Script to launch jobs over various ETH3D datasets & front-ends.
# See https://www.eth3d.net/data/schoeps2017cvpr.pdf for more details.

USER_ROOT=$1
CLUSTER_CONFIG=$2

now=$(date +"%Y%m%d_%H%M%S")

#ETH3D_ROOT=/usr/local/gtsfm-data/eth3d_datasets/multi_view_training_dslr_undistorted
#ETH3D_ROOT=/home/tdriver6/Downloads/eth3d_datasets
ETH3D_ROOT=/home/tdriver6/Documents/eth3d

# Includes all "high-resolution multi-view" datasets from 'training' split (i.e. w/ public GT data)
# See https://www.eth3d.net/datasets for more information.
datasets=(
    courtyard
    delivery_area
    electro
    facade
    kicker
    meadow
    office
    pipes
    playground
    relief_2
    relief
    terrace
    terrains
)

for dataset in ${datasets[@]}; do
    echo "Dataset: ${dataset}"

    images_dir="${ETH3D_ROOT}/${dataset}/images"
    colmap_files_dirpath="${ETH3D_ROOT}/${dataset}/dslr_calibration_undistorted"

    OUTPUT_ROOT=${USER_ROOT}/${now}/${now}__${dataset}__results__vggt__megaloc0.2__max_frame_lookahead2
    mkdir -p $OUTPUT_ROOT

    ./run --loader colmap --dataset_dir $colmap_files_dirpath --images_dir $images_dir --config_name vggt --graph_partitioner metis --worker_memory_limit=24GB \
    2>&1 | tee $OUTPUT_ROOT/out.log

    mv results $OUTPUT_ROOT
done
