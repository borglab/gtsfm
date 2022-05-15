mkdir data/hilti_exp01
mkdir data/hilti_exp01/lidar
mkdir data/hilti_exp01/calibration
cp ../fastlio_result/exp01/fastlio_odom.g2o data/hilti_exp01/lidar/fastlio2.g2o
cp ../pose_slam_final_results/exp01/exp01_output_constraints.txt data/hilti_exp01/lidar/constraints.txt
cp data/hilti/calibration/* data/hilti_exp01/calibration/
tar -C data/hilti_exp01/ -xvf ../fastlio_result/exp01/exp01_image_pcd.tar image/
mv data/hilti_exp01/image data/hilti_exp01/images


python gtsfm/runner/run_frontend_hilti_nodask.py --dataset_dirpath data/hilti_exp01