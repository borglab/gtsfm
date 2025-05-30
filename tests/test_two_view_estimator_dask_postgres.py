"""
Distributed Two-View Estimator Test with PostgreSQL Integration

This module implements a comprehensive test for the two-view estimator using:
- Distributed Dask cluster with remote workers
- PostgreSQL database integration for result storage
- YAML-based configuration management
- SSH tunnel establishment for secure communication

Features:
- Automatic SSH tunnel setup for remote workers
- Database initialization and result storage
- Performance monitoring and visualization
- Configurable cluster settings via YAML

Author: Zongyue Liu
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from dask.distributed import Client, performance_report
import time
import subprocess
import signal
import atexit
import socket
import os
from datetime import datetime
import psycopg2
import yaml

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

# Move all process management to global scope
processes = []

def cleanup():
    """Terminate all started processes"""
    for p in processes:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except:
                p.kill()
    print("All cluster processes cleaned up.")

# Register cleanup globally
atexit.register(cleanup)

def check_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Kill process using the specified port"""
    try:
        result = subprocess.run(['lsof', '-i', f':{port}', '-t'], capture_output=True, text=True)
        pids = result.stdout.strip().split('\n')
        for pid in pids:
            if pid:
                print(f"Killing process {pid} using port {port}")
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)
        return True
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False

def load_config(config_file='gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

def initialize_database(db_params):
    """Initialize database tables for two-view estimation"""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        
        # Create two-view results table
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
        
        # Create two-view reports table
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
        print("Database tables created/verified successfully")
        return True
    except Exception as e:
        print(f"Failed to initialize database: {e}")
        return False

def setup_cluster_infrastructure(config):
    """Set up SSH tunnels and start Dask scheduler"""
    # Extract configuration
    username = config['username']
    scheduler_port = config['scheduler']['port']
    dashboard_port = config['scheduler']['dashboard']
    workers = config['workers']
    workers_addresses = config['workers_addresses']
    db_port = str(config['database']['port'])
    
    # Free up ports if they are in use
    ports_to_check = [scheduler_port, dashboard_port] + list(workers_addresses.values())
    
    for port in ports_to_check:
        if check_port_in_use(port):
            print(f"Port {port} is in use. Attempting to free it...")
            if not kill_process_on_port(port):
                print(f"Failed to free port {port}. Please manually close the application using it.")
                exit(1)
            time.sleep(3)
    
    # Establish SSH tunnels for each worker
    for server in workers:
        server_address = f"{username}@{server}"
        worker_port = workers_addresses[server]
        
        print(f"Establishing SSH tunnel to {server}...")
        ssh_tunnel_cmd = [
            "ssh", "-N", "-f", 
            "-R", f"{scheduler_port}:localhost:{scheduler_port}", 
            "-R", f"{db_port}:localhost:{db_port}", 
            "-L", f"{worker_port}:localhost:{worker_port}", 
            server_address
        ]
        
        ssh_tunnel_proc = subprocess.Popen(ssh_tunnel_cmd)
        processes.append(ssh_tunnel_proc)
        print(f"SSH tunnel established. Process ID: {ssh_tunnel_proc.pid}")
        time.sleep(5)
    
    # Start Dask scheduler
    print("Starting Dask scheduler...")
    dask_scheduler_cmd = [
        "conda", "run", "-n", "gtsfm-v1", 
        "dask-scheduler", "--port", str(scheduler_port), 
        "--dashboard-address", f":{dashboard_port}"
    ]
    
    dask_scheduler_proc = subprocess.Popen(dask_scheduler_cmd)
    processes.append(dask_scheduler_proc)
    print(f"Dask scheduler started. Process ID: {dask_scheduler_proc.pid}")
    time.sleep(10)
    
    # Start Dask workers on remote servers
    for server in workers:
        server_address = f"{username}@{server}"
        worker_port = workers_addresses[server]
        
        print(f"Starting Dask worker on remote server {server}...")
        remote_cmd = (
            f"ssh -t {server_address} 'bash -c \""
            f"export PATH=/home/{username}/miniconda3/bin:$PATH && "
            f"source /home/{username}/miniconda3/etc/profile.d/conda.sh && "
            "conda activate gtsfm-v1 && "
            f"dask-worker tcp://localhost:{scheduler_port} "
            f"--listen-address tcp://0.0.0.0:{worker_port} "
            f"--contact-address tcp://localhost:{worker_port}"
            "\"'"
        )
        
        dask_worker_proc = subprocess.Popen(remote_cmd, shell=True)
        processes.append(dask_worker_proc)
        print(f"Remote Dask worker started. Process ID: {dask_worker_proc.pid}")
        time.sleep(5)
    
    return scheduler_port, processes

def main():
    """Main function containing all computational logic"""
    
    # === LOCAL VERSION CHECKING ===
    import os
    import subprocess
    import socket
    
    hostname = socket.gethostname()
    
    try:
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()[:8]
        git_branch = subprocess.check_output(['git', 'branch', '--show-current']).decode().strip()
        print(f"LOCAL VERSION CHECK: Host={hostname}, Branch={git_branch}, Commit={git_hash}")
    except:
        print(f"LOCAL VERSION CHECK: Host={hostname}, Git info unavailable")
    
    # Check local Keypoints file
    try:
        keypoints_file = 'gtsfm/common/keypoints.py'
        mtime = os.path.getmtime(keypoints_file)
        mod_time = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
        print(f"LOCAL VERSION CHECK: Keypoints.py modified at {mod_time}")
    except:
        print(f"LOCAL VERSION CHECK: Could not check keypoints.py modification time")
    
    # === END LOCAL VERSION CHECKING ===
    
    # Load configuration
    config = load_config()
    
    # Extract database parameters
    db_params = {
        'host': config['database']['host'],
        'port': config['database']['port'],
        'database': config['database']['database'],
        'user': config['database']['user'],
        'password': config['database']['password']
    }
    
    # Create result folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"two_view_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    
    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to folder: {results_dir}")
    
    # Initialize database
    print("Initializing PostgreSQL database...")
    if not initialize_database(db_params):
        print("Failed to initialize database. Exiting.")
        exit(1)
    
    # Set up cluster infrastructure
    print("Setting up distributed cluster...")
    scheduler_port, processes = setup_cluster_infrastructure(config)
    
    # Prepare input data locally
    print("Preparing input data...")
    cwd = Path.cwd()
    folder_path = cwd / "tests" / "data" / "imb_reichstag"

    indices = [0, 1, 2, 3]  
    num_images = len(indices)

    # Load images and camera intrinsics
    loader = YfccImbLoader(str(folder_path))
    images = [loader.get_image(i) for i in indices]
    camera_intrinsics = [loader.get_camera_intrinsics_full_res(i) for i in indices]

    # Feature detection and description
    detector_descriptor = SIFTDetectorDescriptor()
    print("Detecting keypoints...")
    features = [detector_descriptor.detect_and_describe(image) for image in images]
    keypoints_list = []
    descriptors_list = []

    for i, (kp, desc) in enumerate(features):
        # Convert to Keypoints objects with image IDs
        keypoints_obj = Keypoints(coordinates=kp)
        keypoints_obj.image_id = indices[i]  # Set image ID for database storage
        keypoints_list.append(keypoints_obj)
        descriptors_list.append(desc)
        print(f"Image {indices[i]}: {desc.shape[0]} keypoints")

    # Feature matching
    matcher = TwoWayMatcher(ratio_test_threshold=0.8)
    print("Matching keypoints...")

    image_pairs = [(i, j) for i in range(num_images) for j in range(i+1, num_images)]
    putative_corr_idxs_dict = {}

    for i1, i2 in image_pairs:
        image_shape_i1 = images[i1].value_array.shape
        image_shape_i2 = images[i2].value_array.shape
        
        match_indices = matcher.match(
            keypoints_list[i1].coordinates,
            keypoints_list[i2].coordinates,
            descriptors_list[i1], 
            descriptors_list[i2], 
            image_shape_i1,
            image_shape_i2
        )
        
        if match_indices.shape[0] > 0:
            putative_corr_idxs_dict[(i1, i2)] = match_indices
            print(f"Image pair ({indices[i1]}, {indices[i2]}): {match_indices.shape[0]} matches")

    # Create verifier and inlier processor
    verifier = Ransac(use_intrinsics_in_verification=False, estimation_threshold_px=2)
    inlier_support_processor = InlierSupportProcessor(
        min_num_inliers_est_model=20,
        min_inlier_ratio_est_model=0.1
    )

    # Create two-view estimator with database integration
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

    # Set empty pose priors and ground truth cameras
    relative_pose_priors = {}
    gt_cameras = [None] * num_images
    gt_scene_mesh = None

    # Connect to Dask cluster
    print("Connecting to Dask cluster...")
    client = Client(f"tcp://localhost:{scheduler_port}")
    print(f"Connected to Dask cluster: {client.dashboard_link}")
    
    # Store database parameters in client metadata for workers to access
    client.set_metadata('db_params', db_params)
    
    print(f"Cluster workers: {len(client.scheduler_info()['workers'])}")
    for worker_id, worker_info in client.scheduler_info()['workers'].items():
        host = worker_info.get('host', 'unknown')
        port = worker_info.get('port', 'unknown')
        worker_address = f"{host}:{port}" if 'port' in worker_info else host
        print(f"  - Worker: {worker_id}, Address: {worker_address}")

    try:
        # Run distributed two-view estimation
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
            print(f"Distributed computation completed in {total_time:.2f} seconds")

        # Query database for results
        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()
            
            print("\nQuerying two-view results from database:")
            cursor.execute("""
            SELECT i1, i2, verified_corr_count, inlier_ratio, success, computation_time, worker_name
            FROM two_view_results
            WHERE timestamp >= %s
            ORDER BY i1, i2
            """, (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),))
            
            results = cursor.fetchall()
            
            print("Image Pair | Verified Matches | Inlier Ratio | Success | Computation Time (s) | Worker")
            print("-" * 90)
            for row in results:
                i1, i2, corr_count, inlier_ratio, success, comp_time, worker = row
                print(f"({i1}, {i2}) | {corr_count:8d} | {inlier_ratio:.4f} | {success} | {comp_time:.4f} | {worker}")
            
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Database query failed: {e}")

        # Process and save results
        print("\nProcessing results...")
        
        summary_file = results_dir / "results_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Two-View Estimation Summary - {timestamp}\n")
            f.write("="*50 + "\n\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
            f.write(f"Number of images: {num_images}\n")
            f.write(f"Number of image pairs: {len(image_pairs)}\n\n")
            f.write("Results per image pair:\n")
            f.write("-"*50 + "\n")

        for (i1, i2), (i2Ri1, i2Ui1, v_corr_idxs, pre_ba_report, post_ba_report, post_isp_report) in two_view_output_dict.items():
            
            with open(summary_file, "a") as f:
                f.write(f"\nImage Pair ({indices[i1]}, {indices[i2]}):\n")
                
            if i2Ri1 is not None and i2Ui1 is not None:
                ypr_deg = np.degrees(i2Ri1.xyz()) if i2Ri1 else None
                
                print(f"Image Pair ({indices[i1]}, {indices[i2]}):")
                print(f"  - Verified correspondences: {len(v_corr_idxs)}")
                print(f"  - Relative rotation (yaw, pitch, roll): {ypr_deg}")
                print(f"  - Relative translation direction: {i2Ui1.point3().T}")
                print(f"  - Inlier ratio: {post_isp_report.inlier_ratio_est_model:.4f}")
                
                with open(summary_file, "a") as f:
                    f.write(f"  - Verified correspondences: {len(v_corr_idxs)}\n")
                    f.write(f"  - Relative rotation (yaw, pitch, roll): {ypr_deg}\n")
                    f.write(f"  - Relative translation direction: {i2Ui1.point3().T}\n")
                    f.write(f"  - Inlier ratio: {post_isp_report.inlier_ratio_est_model:.4f}\n")
                
                # Visualize correspondences
                if len(v_corr_idxs) > 0:
                    max_viz_corrs = min(100, len(v_corr_idxs))
                    
                    try:
                        correspondence_image = viz.plot_twoview_correspondences(
                            images[i1], images[i2], 
                            keypoints_list[i1], keypoints_list[i2], 
                            v_corr_idxs[:max_viz_corrs], 
                            max_corrs=max_viz_corrs
                        )
                        
                        plt.figure(figsize=(12, 10))
                        plt.imshow(correspondence_image.value_array)
                        plt.title(f"Image Pair ({indices[i1]}, {indices[i2]}) Verified Correspondences")
                        plt.savefig(images_dir / f"correspondences_{indices[i1]}_{indices[i2]}.png")
                        plt.close()
                    except Exception as e:
                        print(f"Visualization error for pair ({indices[i1]}, {indices[i2]}): {e}")
            else:
                print(f"Image Pair ({indices[i1]}, {indices[i2]}): Relative pose estimation failed")
                with open(summary_file, "a") as f:
                    f.write("  - Relative pose estimation failed\n")

        print(f"\nAll results saved to: {results_dir}")
        
        # Keep processes alive
        print("Processes will continue running. Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Termination signal received, cleaning up...")
    finally:
        # Clean up
        client.close()
        print("Dask client closed")

if __name__ == '__main__':
    main()