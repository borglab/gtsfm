from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, LocalCluster, performance_report
import time
import os
import shutil
from datetime import datetime
import psycopg2

import gtsam
from gtsam import Pose3, PinholeCameraCal3Bundler, Unit3, Rot3

from gtsfm.loader.yfcc_imb_loader import YfccImbLoader
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
from gtsfm.utils import viz
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.two_view_estimator import TwoViewEstimator, run_two_view_estimator_as_futures
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.data_association.point3d_initializer import (
    Point3dInitializer,
    TriangulationExitCode,
    TriangulationOptions,
    TriangulationSamplingMode,
)

def main():
    """Main function containing all computational logic"""
    
    # Create result folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"two_view_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    # Create image subfolder
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to folder: {results_dir}")
    
    # Set database connection parameters
    db_params = {
        'host': 'localhost',
        'port': '5432',
        'database': 'postgres',
        'user': 'postgres',
        'password': '0504'  
    }
    
    # Create database tables (if not exist)
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Create two view estimator result 
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS two_view_results (
            id SERIAL PRIMARY KEY,
            i1 INTEGER NOT NULL,
            i2 INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            verified_corr_count INTEGER,
            inlier_ratio FLOAT,
            rotation_matrix TEXT,
            translation_direction TEXT,
            success BOOLEAN NOT NULL,
            computation_time FLOAT,
            worker_name TEXT
        );
        """)
        
        # Create two view estimator report
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS two_view_reports (
            id SERIAL PRIMARY KEY,
            i1 INTEGER NOT NULL,
            i2 INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            pre_ba_inlier_ratio FLOAT,
            post_ba_inlier_ratio FLOAT,
            post_isp_inlier_ratio FLOAT,
            report_data TEXT
        );
        """)
        
        conn.commit()
        cursor.close()
        conn.close()
        print("Database tables created/verified")
    except Exception as e:
        print(f"Failed to create database tables: {e}")
    
    # 1. Prepare input data - executed locally
    print("Preparing input data...")
    cwd = Path.cwd()
    folder_path = cwd.parent / "tests" / "data" / "imb_reichstag"

    indices = [0, 1, 2, 3]  
    num_images = len(indices)

    # Load images and intrinsics - executed locally
    loader = YfccImbLoader(str(folder_path))
    images = [loader.get_image(i) for i in indices]
    camera_intrinsics = [loader.get_camera_intrinsics_full_res(i) for i in indices]

    # Feature detection and description - executed locally
    detector_descriptor = SIFTDetectorDescriptor()
    print("Detecting keypoints....")
    features = [detector_descriptor.detect_and_describe(image) for image in images]
    keypoints_list = []
    descriptors_list = []

    for i, (kp, desc) in enumerate(features):
        # Keep raw coordinate arrays, do not convert to Keypoints object
        keypoints_list.append(kp)  
        descriptors_list.append(desc)
        print(f"Image {indices[i]}: {desc.shape[0]} Keypoints")

    # Feature matching - executed locally
    matcher = TwoWayMatcher(ratio_test_threshold=0.8)
    print("Matching keypoints...")

    # Create all possible image pair combinations
    image_pairs = [(i, j) for i in range(num_images) for j in range(i+1, num_images)]
    putative_corr_idxs_dict = {}

    for i1, i2 in image_pairs:
        image_shape_i1 = images[i1].value_array.shape
        image_shape_i2 = images[i2].value_array.shape
        
        match_indices = matcher.match(
            keypoints_list[i1],  
            keypoints_list[i2],  
            descriptors_list[i1], 
            descriptors_list[i2], 
            image_shape_i1,
            image_shape_i2
        )
        
        if match_indices.shape[0] > 0:
            putative_corr_idxs_dict[(i1, i2)] = match_indices
            print(f"Image pair ({indices[i1]}, {indices[i2]}): {match_indices.shape[0]} matches")

    # Create verifier and inlier processor - executed locally
    verifier = Ransac(use_intrinsics_in_verification=False, estimation_threshold_px=2)
    inlier_support_processor = InlierSupportProcessor(
        min_num_inliers_est_model=20,
        min_inlier_ratio_est_model=0.1
    )

    # Create two-view estimator with DB connection
    triangulation_options = TriangulationOptions(
        mode=TriangulationSamplingMode.NO_RANSAC,
        min_triangulation_angle=1.0,
        reproj_error_threshold=4.0
    )

    two_view_estimator = TwoViewEstimator(
        verifier=verifier,
        inlier_support_processor=inlier_support_processor,
        bundle_adjust_2view=True,
        eval_threshold_px=4,
        triangulation_options=triangulation_options,
        postgres_params=db_params  
    )

    # Set empty pose priors and ground truth cameras - executed locally
    relative_pose_priors = {}
    gt_cameras = [None] * num_images
    gt_scene_mesh = None

    # 2. Connect to remote Dask cluster - assuming scheduler is running
    print("Connecting to remote Dask cluster...")

    scheduler_address = "tcp://localhost:8788"  
    client = Client(scheduler_address)
    print(f"Connected to Dask cluster: {client.dashboard_link}")
    
    print(f"Cluster workers: {len(client.scheduler_info()['workers'])}")
    for worker_id, worker_info in client.scheduler_info()['workers'].items():
        host = worker_info.get('host', 'unknown')
        port = worker_info.get('port', 'unknown')
        worker_address = f"{host}:{port}" if 'port' in worker_info else host
        print(f"  - Worker: {worker_id}, 地址: {worker_address}")

    try:
        # Create result dictionary - executed locally
        computation_results = {
            "start_time": time.time(),
            "image_pairs": image_pairs,
            "pair_compute_times": {},
            "two_view_outputs": {}
        }

        # 3. Send data to cluster - prepare for remote execution
        # 4. Run estimation remotely - save performance report
        print("Running distributed two-view estimation...")
        with performance_report(filename=str(results_dir / "dask_performance_report.html")):
            start_time = time.time()
            
            two_view_output_dict = run_two_view_estimator_as_futures(
                client=client,
                two_view_estimator=two_view_estimator,
                keypoints_list=keypoints_list,
                putative_corr_idxs_dict=putative_corr_idxs_dict,
                camera_intrinsics=camera_intrinsics,
                relative_pose_priors=relative_pose_priors,
                gt_cameras=gt_cameras,
                gt_scene_mesh=gt_scene_mesh
            )

            total_time = time.time() - start_time
            print(f"Distributed computation complete. Time: {total_time:.2f} 秒")

        # 5. Query database for results
        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
            
            print("\Querying two-view results from database:")
            cursor.execute("""
            SELECT i1, i2, verified_corr_count, inlier_ratio, success, computation_time, worker_name
            FROM two_view_results
            WHERE timestamp >= %s
            ORDER BY i1, i2
            """, (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),))
            
            results = cursor.fetchall()
            
            print("Image Pair | Verified Matches | Inlier Ratio | Success | Computation Time (s) | Worker")
            print("-" * 80)

            for row in results:
                i1, i2, corr_count, inlier_ratio, success, comp_time, worker = row
                print(f"({i1}, {i2}) | {corr_count:8d} | {inlier_ratio:.4f} | {success} | {comp_time:.4f} | {worker}")
            
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Database query failed: {e}")

        # Process results - executed locally
        computation_results["total_time"] = total_time
        computation_results["two_view_outputs"] = {}

        # Create summary file for processed results
        summary_file = results_dir / "results_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Two-View Estimation Summary - {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
            f.write(f"Number of images: {num_images}\n")
            f.write(f"Number of image pairs: {len(image_pairs)}\n\n")
            f.write("Results per image pair:\n")
            f.write("-"*50 + "\n")

        print("\nAnalyzing results:")
        for (i1, i2), (i2Ri1, i2Ui1, v_corr_idxs, pre_ba_report, post_ba_report, post_isp_report) in two_view_output_dict.items():
            # Store result
            computation_results["two_view_outputs"][(i1, i2)] = {
                "i2Ri1": i2Ri1,
                "i2Ui1": i2Ui1,
                "verified_corr_count": len(v_corr_idxs) if v_corr_idxs is not None else 0,
                "pre_ba_inlier_ratio": pre_ba_report.inlier_ratio_est_model if pre_ba_report else None,
                "post_ba_inlier_ratio": post_ba_report.inlier_ratio_est_model if post_ba_report else None,
                "post_isp_inlier_ratio": post_isp_report.inlier_ratio_est_model if post_isp_report else None
            }
            
            # Append results to summary file
            with open(summary_file, "a") as f:
                f.write(f"\nImage Pair ({indices[i1]}, {indices[i2]}):\n")
                
            if i2Ri1 is not None and i2Ui1 is not None:
                # If relative pose estimation succeeded
                ypr_deg = np.degrees(i2Ri1.xyz()) if i2Ri1 else None
                
                print(f"Image Pair ({indices[i1]}, {indices[i2]}):")
                print(f"  - Number of verified correspondences: {len(v_corr_idxs)}")
                print(f"  - Relative rotation (yaw, pitch, roll): {ypr_deg}")
                print(f"  - Relative translation direction: {i2Ui1.point3().T}")
                print(f"  - Inlier ratio: {post_isp_report.inlier_ratio_est_model:.4f}")
                
                # Add detailed info to summary file
                with open(summary_file, "a") as f:
                    f.write(f"  - Number of verified correspondences: {len(v_corr_idxs)}\n")
                    f.write(f"  - Relative rotation (yaw, pitch, roll): {ypr_deg}\n")
                    f.write(f"  - Relative translation direction: {i2Ui1.point3().T}\n")
                    f.write(f"  - Inlier ratio: {post_isp_report.inlier_ratio_est_model:.4f}\n")
                
                # Visualize partial matches and save – no display
                if len(v_corr_idxs) > 0:
                    max_viz_corrs = min(100, len(v_corr_idxs))
                    

                    try:
                        # Method 1: Directly use raw coordinate arrays
                        correspondence_image = viz.plot_twoview_correspondences(
                            images[i1], images[i2], 
                            keypoints_list[i1], keypoints_list[i2], 
                            v_corr_idxs[:max_viz_corrs], 
                            max_corrs=max_viz_corrs
                        )
                    except TypeError as e:
                        print(f"Visualization error: {e}")
                        print("Trying alternative method...")
                        
                        # Method 2: Use named arguments to explicitly create Keypoints objects
                        viz_keypoints_i1 = Keypoints(coordinates=keypoints_list[i1])
                        viz_keypoints_i2 = Keypoints(coordinates=keypoints_list[i2])
                        
                        correspondence_image = viz.plot_twoview_correspondences(
                            images[i1], images[i2], 
                            viz_keypoints_i1, viz_keypoints_i2, 
                            v_corr_idxs[:max_viz_corrs], 
                            max_corrs=max_viz_corrs
                        )
                    
                    # Save image without displaying it
                    plt.figure(figsize=(12, 10))
                    plt.imshow(correspondence_image.value_array)
                    plt.title(f"图像对 ({indices[i1]}, {indices[i2]}) 验证的对应点")
                    plt.savefig(images_dir / f"correspondences_{indices[i1]}_{indices[i2]}.png")
                    plt.close()  
            else:
                print(f"Image Pair ({indices[i1]}, {indices[i2]}): Relative pose estimation failed")
                # Add failure info to summary file
                with open(summary_file, "a") as f:
                    f.write("  - Relative pose estimation failed\n")


        import pickle
        
        # Remove unserializable objects
        serializable_results = {
            "start_time": computation_results["start_time"],
            "total_time": computation_results["total_time"],
            "image_pairs": computation_results["image_pairs"],
            "pair_results": {}
        }

        for (i1, i2), results in computation_results["two_view_outputs"].items():
            serializable_results["pair_results"][(i1, i2)] = {
                "verified_corr_count": results["verified_corr_count"],
                "pre_ba_inlier_ratio": results["pre_ba_inlier_ratio"],
                "post_ba_inlier_ratio": results["post_ba_inlier_ratio"],
                "post_isp_inlier_ratio": results["post_isp_inlier_ratio"],
                "i2Ri1_ypr_deg": np.degrees(results["i2Ri1"].xyz()).tolist() if results["i2Ri1"] else None,
                "i2Ui1_point3": results["i2Ui1"].point3().tolist() if results["i2Ui1"] else None
            }

        # Save results to pickle file
        with open(results_dir / "computation_results.pkl", "wb") as f:
            pickle.dump(serializable_results, f)

        # Try to fetch and save Dask task stream (if available)
        try:
            task_graph = client.get_task_stream()
            with open(results_dir / "task_stream.json", "w") as f:
                import json
                json.dump(task_graph, f)
        except:
            print("Unable to fetch Dask task stream")

        print(f"All results have been saved to folder: {results_dir}")
    
    finally:
        client.close()
        print("Dask client closed")

if __name__ == '__main__':
    main()