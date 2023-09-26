# Script to launch jobs over various datasets & front-ends.

USER_ROOT=$1
CLUSTER_CONFIG=$2

now=$(date +"%Y%m%d_%H%M%S")


datasets=(
	gendarmenmarkt-1463
	palace-fine-arts-281
	skydio-crane-mast-501
	2011205_rc3
	south-building-128
	gerrard-hall-100
	)

max_frame_lookahead_sizes=(
	0
	5
	10
	15
	)

num_matched_sizes=(
	0
	5
	10
	15
	20
	25
	)

correspondence_generator_config_names=(
	sift
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
			if [[ $dataset == *"gendarmenmarkt-1463"* && $max_frame_lookahead != 0 ]]
			then
				# Gendarmenmarkt images have no natural order.
				continue
			fi

			if [[ $num_matched == 0 && $max_frame_lookahead == 0 ]]
			then
				# Matches must come from at least some retriever.
				continue
			fi

			for correspondence_generator_config_name in ${correspondence_generator_config_names[@]}; do
				
				if [[ $correspondence_generator_config_name == *"sift"* ]]
				then
					num_workers=10
				elif [[ $correspondence_generator_config_name == *"lightglue"* ]]
				then
					num_workers=1
				elif [[ $correspondence_generator_config_name == *"superglue"* ]]
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

				if [[ $dataset == *"palace-fine-arts-281"* ]]
				then
					loader=olsson
					dataset_root=/usr/local/gtsfm-data/palace-fine-arts-281
				elif [[ $dataset == *"skydio-crane-mast-501"* ]]
				then
					loader=colmap
					images_dir=/usr/local/gtsfm-data/skydio-crane-mast-501/skydio-crane-mast-501-images
					colmap_files_dirpath=/usr/local/gtsfm-data/skydio-crane-mast-501/skydio-501-colmap-pseudo-gt
				elif [[ $dataset == *"2011205_rc3"* ]]
				then
					loader=astrovision
					data_dir=/usr/local/gtsfm-data/2011205_rc3
				elif [[ $dataset == *"south-building-128"* ]]
				then
					loader=colmap
					images_dir=/usr/local/gtsfm-data/south-building-128/images
					colmap_files_dirpath=/usr/local/gtsfm-data/south-building-128/colmap-2023-07-28-txt
				elif [[ $dataset == *"gerrard-hall-100"* ]]
				then
					loader=colmap
					images_dir=/usr/local/gtsfm-data/gerrard-hall-100/images
					colmap_files_dirpath=/usr/local/gtsfm-data/gerrard-hall-100/colmap-3.7-sparse-txt-2023-07-27
				elif [[ $dataset == *"gendarmenmarkt-1463"* ]]
				then
					loader=colmap
					images_dir=/usr/local/gtsfm-data/Gendarmenmarkt/images
					colmap_files_dirpath=/usr/local/gtsfm-data/Gendarmenmarkt/gendarmenmark/im_size_full/0
				fi

				OUTPUT_ROOT=${USER_ROOT}/${now}/${now}__${dataset}__results__num_matched${num_matched}__maxframelookahead${max_frame_lookahead}__760p__unified_${correspondence_generator_config_name}
				mkdir -p $OUTPUT_ROOT

				if [[ $loader == *"olsson"* ]]
				then
					python gtsfm/runner/run_scene_optimizer_olssonloader.py \
					--mvs_off \
					--config unified \
					--correspondence_generator_config_name $correspondence_generator_config_name \
					--share_intrinsics \
					--dataset_root $dataset_root \
					--num_workers $num_workers \
					--num_matched $num_matched \
					--max_frame_lookahead $max_frame_lookahead \
					--worker_memory_limit "32GB" \
					--output_root $OUTPUT_ROOT \
					--max_resolution 760 \
					$CLUSTER_ARGS \
					2>&1 | tee $OUTPUT_ROOT/out.log
				elif [[ $loader == *"colmap"* ]]
				then
					python gtsfm/runner/run_scene_optimizer_colmaploader.py \
					--mvs_off \
					--config unified \
					--correspondence_generator_config_name $correspondence_generator_config_name \
					--share_intrinsics \
					--images_dir $images_dir \
					--colmap_files_dirpath $colmap_files_dirpath \
					--num_workers $num_workers \
					--num_matched $num_matched \
					--max_frame_lookahead $max_frame_lookahead \
					--worker_memory_limit "32GB" \
					--output_root $OUTPUT_ROOT \
					--max_resolution 760 \
					$CLUSTER_ARGS \
					2>&1 | tee $OUTPUT_ROOT/out.log
				elif [[ $loader == *"astrovision"* ]]
				then
					python gtsfm/runner/run_scene_optimizer_astrovision.py \
					--mvs_off \
					--config unified \
					--correspondence_generator_config_name $correspondence_generator_config_name \
					--share_intrinsics \
					--data_dir $data_dir \
					--num_workers $num_workers \
					--num_matched $num_matched \
					--max_frame_lookahead $max_frame_lookahead \
					--worker_memory_limit "32GB" \
					--output_root $OUTPUT_ROOT \
					--max_resolution 760 \
					$CLUSTER_ARGS \
					2>&1 | tee $OUTPUT_ROOT/out.log
				fi
			done
		done
	done
done


