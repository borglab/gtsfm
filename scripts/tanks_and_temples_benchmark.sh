


# Script to launch jobs over various datasets & front-ends.

USER_ROOT=$1
CLUSTER_CONFIG=$2

now=$(date +"%Y%m%d_%H%M%S")


datasets=(
    # Tanks and Temples Dataset.
    # barn-tanks-and-temples-410
    # truck-251
    # meetingroom-371
    courthouse-1106
    ignatius-263
    church-507
)

max_frame_lookahead_sizes=(
    # 0
    # 5
    10
    #15
)

num_matched_sizes=(
    # 0
    5
    # 10
    # 15
    # 20
    # 25
)

correspondence_generator_config_names=(
    sift
    lightglue
    superglue
    loftr
    disk
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


            INTRINSICS_ARGS="--share_intrinsics"
  

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
                elif [[ $correspondence_generator_config_name == *"disk"* ]]
                then
                    num_workers=1
                elif [[ $correspondence_generator_config_name == *"loftr"* ]]
                then
                    num_workers=1
                fi

                echo "Dataset: ${dataset}"
                echo "Num matched: ${num_matched}"
                echo "Max frame lookahead: ${max_frame_lookahead}"
                echo "Correspondence Generator: ${correspondence_generator_config_name}"
                echo "Num workers: ${num_workers}"
                echo "Intrinsics: ${INTRINSICS_ARGS}"

                if [[ $dataset == *"truck-251"* ]]
                then
                    dataset_root=/usr/local/gtsfm-data/TanksAndTemples/Truck
                elif [[ $dataset == *"meetingroom-371"* ]]
                then
                    dataset_root=/usr/local/gtsfm-data/TanksAndTemples/Meetingroom
                elif [[ $dataset == *"courthouse-1106"* ]]
                then
                    dataset_root=/usr/local/gtsfm-data/Tanks_and_Temples_Courthouse_1106
                elif [[ $dataset == *"ignatius-263"* ]]
                then
                    dataset_root=/usr/local/gtsfm-data/TanksAndTemples/Ignatius
                elif [[ $dataset == *"church-507"* ]]
                then
                    dataset_root=/usr/local/gtsfm-data/TanksAndTemples/Church
                fi

                OUTPUT_ROOT=${USER_ROOT}/${now}/${now}__${dataset}__results__num_matched${num_matched}__maxframelookahead${max_frame_lookahead}__760p__unified_${correspondence_generator_config_name}
                mkdir -p $OUTPUT_ROOT

                python gtsfm/runner/run_scene_optimizer_1dsfm.py \
                --mvs_off \
                --config unified \
                --correspondence_generator_config_name $correspondence_generator_config_name \
                --dataset_root $dataset_root \
                --num_workers $num_workers \
                --num_matched $num_matched \
                --max_frame_lookahead $max_frame_lookahead \
                --worker_memory_limit "32GB" \
                --output_root $OUTPUT_ROOT \
                --max_resolution 760 \
                $INTRINSICS_ARGS $CLUSTER_ARGS \
                2>&1 | tee $OUTPUT_ROOT/out.log

            done
        done
    done
done

