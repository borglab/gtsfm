# Script to launch jobs over various ETH3D datasets & front-ends.
# See https://www.eth3d.net/data/schoeps2017cvpr.pdf for more details.

USER_ROOT=$1
CLUSTER_CONFIG=$2

now=$(date +"%Y%m%d_%H%M%S")

ETH3D_ROOT=/usr/local/gtsfm-data/eth3d_datasets/multi_view_training_dslr_undistorted
#ETH3D_ROOT=/home/tdriver6/Downloads/eth3d_datasets

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

max_frame_lookahead_sizes=(
    #0
    #5
    10
    #15
)

num_matched_sizes=(
    5
    #10
    #15
    #20
    #25
)

correspondence_generator_config_names=(
    sift
    disk
    lightglue
    superglue
    loftr
)

if [[ $CLUSTER_CONFIG ]]
then
    CLUSTER_ARGS="--cluster_config $CLUSTER_CONFIG"
else
    CLUSTER_ARGS=""
fi


for num_matched in ${num_matched_sizes[@]}; do
    for max_frame_lookahead in ${max_frame_lookahead_sizes[@]}; do
        for dataset in ${datasets[@]}; do
            if [[ $num_matched == 0 && $max_frame_lookahead == 0 ]]
            then
                # Matches must come from at least some retriever.
                continue
            fi

            for correspondence_generator_config_name in ${correspondence_generator_config_names[@]}; do

                if [[ $correspondence_generator_config_name == *"sift"* ]]
                then
                    num_workers=1
                elif [[ $correspondence_generator_config_name == *"lightglue"* ]]
                then
                    num_workers=1
                elif [[ $correspondence_generator_config_name == *"superglue"* ]]
                then
                    num_workers=1
                elif [[ $correspondence_generator_config_name == *"loftr"* ]]
                then
                    num_workers=1
                elif [[ $correspondence_generator_config_name == *"disk"* ]]
                then
                    num_workers=1
                fi

                echo "Dataset: ${dataset}"
                echo "Num matched: ${num_matched}"
                echo "Max frame lookahead: ${max_frame_lookahead}"
                echo "Correspondence Generator: ${correspondence_generator_config_name}"
                echo "Num workers: ${num_workers}"

                images_dir="${ETH3D_ROOT}/${dataset}/images"
                colmap_files_dirpath="${ETH3D_ROOT}/${dataset}/dslr_calibration_undistorted"
                # images_dir="${ETH3D_ROOT}/${dataset}_dslr_undistorted/${dataset}/images"
                # colmap_files_dirpath="${ETH3D_ROOT}/${dataset}_dslr_undistorted/${dataset}/dslr_calibration_undistorted"

                OUTPUT_ROOT=${USER_ROOT}/${now}/${now}__${dataset}__results__num_matched${num_matched}__maxframelookahead${max_frame_lookahead}__760p__unified_${correspondence_generator_config_name}
                mkdir -p $OUTPUT_ROOT

                ./run \
                --loader colmap_loader \
                --run_mvs false \
                --config_name unified \
                --correspondence_generator_config_name $correspondence_generator_config_name \
                --share_intrinsics \
                --dataset_dir $colmap_files_dirpath \
                --images_dir $images_dir \
                --use_gt_intrinsics \
                --use_gt_extrinsics \
                --num_workers 1 \
                --num_matched $num_matched \
                --max_frame_lookahead $max_frame_lookahead \
                --worker_memory_limit "32GB" \
                --output_root $OUTPUT_ROOT \
                --max_resolution 760 \
                $CLUSTER_ARGS \
                2>&1 | tee $OUTPUT_ROOT/out.log
            done
        done
    done
done


