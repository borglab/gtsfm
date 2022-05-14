mkdir data/hilti_exp06
mkdir data/hilti_exp06/lidar
mkdir data/hilti_exp06/calibration
cp ../fastlio_result/exp06/fastlio_odom.g2o data/hilti_exp06/lidar/fastlio2.g2o
cp ../fastlio_result/exp06/exp06_output_constraints.txt data/hilti_exp06/lidar/constraints.txt
cp data/hilti/calibration/* data/hilti_exp06/calibration/
tar -C data/hilti_exp06/ -xvf ../fastlio_result/exp06/exp06_image_pcd.tar image/
mv data/hilti_exp06/image data/hilti_exp06/images


python gtsfm/runner/run_scene_optimizer_hilti.py --dataset_dirpath data/hilti_exp06 --config deep_front_end_hilti --matching_regime rig_hilti --threads_per_worker 1 --num_workers 3 --subsample --max_length 100