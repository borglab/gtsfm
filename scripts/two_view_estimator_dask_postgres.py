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

import atexit
import os
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import psycopg2
import yaml
from dask.distributed import Client, performance_report
from dotenv import load_dotenv

# Load environment variables from .env file at the start
load_dotenv()

import gtsfm.utils.viz as viz
from gtsfm.data_association.point3d_initializer import TriangulationOptions, TriangulationSamplingMode
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.loader.yfcc_imb_loader import YfccImbLoader
from gtsfm.two_view_estimator import TwoViewEstimator, run_two_view_estimator_as_futures
from gtsfm.utils.ssh_tunneling import SSHTunnelManager

processes = []


def cleanup() -> None:
    """Terminate all started processes"""
    for p in processes:
        if p.poll() is None:
            try:
                p.terminate()
                p.wait(timeout=5)
            except Exception:
                p.kill()
    print("All cluster processes cleaned up.")


atexit.register(cleanup)


def check_port_in_use(port) -> bool:
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def kill_process_on_port(port) -> bool:
    """Kill process using the specified port"""
    try:
        result = subprocess.run(["lsof", "-i", f":{port}", "-t"], capture_output=True, text=True)
        pids = result.stdout.strip().split("\n")
        for pid in pids:
            if pid:
                print(f"Killing process {pid} using port {port}")
                os.kill(int(pid), signal.SIGTERM)
                time.sleep(1)
        return True
    except Exception as e:
        print(f"Error killing process on port {port}: {e}")
        return False


def load_config(config_file="gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml"):
    """Load configuration from YAML file and merge with environment variables"""
    try:
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

        # Override with environment variables if they exist
        if "SSH_USERNAME" in os.environ:
            config["username"] = os.environ["SSH_USERNAME"]

        if "database" in config:
            if "POSTGRES_HOST" in os.environ:
                config["database"]["host"] = os.environ["POSTGRES_HOST"]
            if "POSTGRES_PORT" in os.environ:
                config["database"]["port"] = int(os.environ["POSTGRES_PORT"])
            if "POSTGRES_DATABASE" in os.environ:
                config["database"]["database"] = os.environ["POSTGRES_DATABASE"]
            if "POSTGRES_USER" in os.environ:
                config["database"]["user"] = os.environ["POSTGRES_USER"]
            if "POSTGRES_PASSWORD" in os.environ:
                config["database"]["password"] = os.environ["POSTGRES_PASSWORD"]

        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)


def setup_cluster_infrastructure(config_file: str) -> tuple[int, list[subprocess.Popen]]:
    """Set up SSH tunnels and start Dask scheduler"""
    tunnel_manager = SSHTunnelManager(config_file)
    scheduler_port = tunnel_manager.setup_complete_infrastructure()
    return scheduler_port, tunnel_manager.processes


def test_database_connection(db_params):
    """Test database connection from current process"""
    try:
        conn = psycopg2.connect(**db_params)
        cursor = conn.cursor()
        cursor.execute("SELECT version()")
        version = cursor.fetchone()
        print(f"Database connection successful: {version[0]}")
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        print(f"Database connection failed: {e}")
        return False


def main():
    """Main function containing all computational logic"""

    # Load configuration - default can be here at the application level
    config_file = "gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml"
    config = load_config(config_file)

    # Extract database parameters
    db_params = {
        "host": config["database"]["host"],
        "port": config["database"]["port"],
        "database": config["database"]["database"],
        "user": config["database"]["user"],
        "password": config["database"]["password"],
    }

    # Create result folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"two_view_results_{timestamp}")
    results_dir.mkdir(exist_ok=True)

    images_dir = results_dir / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Saving results to folder: {results_dir}")

    # Set up cluster infrastructure
    print("Setting up distributed cluster...")

    scheduler_port, processes = setup_cluster_infrastructure(config_file)

    # Prepare input data locally
    print("Preparing input data...")

    cwd = Path.cwd()
    folder_path = cwd / "tests" / "data" / "imb_reichstag"

    indices = [0, 1, 2, 3, 4]
    num_images = len(indices)

    # Load images and camera intrinsics
    loader = YfccImbLoader(str(folder_path))
    images = [loader.get_image(i) for i in indices]
    camera_intrinsics = [loader.get_camera_intrinsics_full_res(i) for i in indices]

    # Feature detection and description
    detector_descriptor = SIFTDetectorDescriptor()
    print("[LOCAL MACHINE] Detecting keypoints...")

    features = [detector_descriptor.detect_and_describe(image) for image in images]
    keypoints_list = []
    descriptors_list = []

    for i, (kp, desc) in enumerate(features):
        kp.image_id = indices[i]  # Set image ID for database storage
        keypoints_list.append(kp)  # Directly append the Keypoints object
        descriptors_list.append(desc)
        print(f"Image {indices[i]}: {desc.shape[0]} keypoints")

    # Feature matching
    matcher = TwoWayMatcher(ratio_test_threshold=0.8)
    print("[LOCAL MACHINE] Matching keypoints...")

    image_pairs = [(i, j) for i in range(num_images) for j in range(i + 1, num_images)]
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
            image_shape_i2,
        )

        if match_indices.shape[0] > 0:
            putative_corr_idxs_dict[(i1, i2)] = match_indices
            print(f"Image pair ({indices[i1]}, {indices[i2]}): {match_indices.shape[0]} matches")

    # Create verifier and inlier processor
    verifier = Ransac(use_intrinsics_in_verification=False, estimation_threshold_px=2)
    inlier_support_processor = InlierSupportProcessor(min_num_inliers_est_model=20, min_inlier_ratio_est_model=0.1)

    # Create two-view estimator with database integration
    triangulation_options = TriangulationOptions(
        mode=TriangulationSamplingMode.NO_RANSAC, min_triangulation_angle=1.0, reproj_error_threshold=4.0
    )

    two_view_estimator = TwoViewEstimator(
        verifier=verifier,
        inlier_support_processor=inlier_support_processor,
        bundle_adjust_2view=True,
        eval_threshold_px=4,
        triangulation_options=triangulation_options,
        postgres_params=db_params,
    )

    # Set empty pose priors and ground truth cameras
    relative_pose_priors = {}
    gt_cameras = [None] * num_images
    gt_scene_mesh = None

    # Connect to Dask cluster
    print("Connecting to Dask cluster...")
    client = Client(f"tcp://localhost:{scheduler_port}")
    print(f"Connected to Dask cluster: {client.dashboard_link}")

    # Wait for all workers to connect
    expected_workers = len(config["workers"])
    print(f"Waiting for {expected_workers} workers to connect...")

    timeout = 60  # seconds
    start_wait = time.time()
    while len(client.scheduler_info()["workers"]) < expected_workers:
        if time.time() - start_wait > timeout:
            print(f"Timeout: Only {len(client.scheduler_info()['workers'])} workers connected")
            break
        print(f"Currently {len(client.scheduler_info()['workers'])}/{expected_workers} workers connected. Waiting...")
        time.sleep(3)

    print(f"Cluster workers: {len(client.scheduler_info()['workers'])}")
    for worker_id, worker_info in client.scheduler_info()["workers"].items():
        host = worker_info.get("host", "unknown")
        port = worker_info.get("port", "unknown")
        worker_address = f"{host}:{port}" if "port" in worker_info else host
        print(f"  - Worker: {worker_id}, Address: {worker_address}")
    try:
        # Test database connection
        if not test_database_connection(db_params):
            print("Database connection test failed. Exiting.")
            return

        # Run distributed two-view estimation
        print("[MAIN] Running distributed two-view estimation...")

        with performance_report(filename=str(results_dir / "dask_performance_report.html")):
            start_time = time.time()

            print("[MAIN] Calling run_two_view_estimator_as_futures...")

            two_view_output_dict = run_two_view_estimator_as_futures(
                client=client,
                two_view_estimator=two_view_estimator,
                keypoints_list=keypoints_list,
                putative_corr_idxs_dict=putative_corr_idxs_dict,
                camera_intrinsics=camera_intrinsics,
                relative_pose_priors=relative_pose_priors,
                gt_cameras=gt_cameras,
                gt_scene_mesh=gt_scene_mesh,
            )

            print(f"[MAIN] run_two_view_estimator_as_futures returned {len(two_view_output_dict)} results")

            total_time = time.time() - start_time
            print(f"[MAIN] Distributed computation completed in {total_time:.2f} seconds")

        # Query database for results
        print("[MAIN] Querying database for results...")

        try:
            conn = psycopg2.connect(**db_params)
            cursor = conn.cursor()

            print("\n[MAIN] Two-view results from database:")

            cursor.execute(
                """
            SELECT i1, i2, timestamp, verified_corr_count, inlier_ratio, success, computation_time, worker_name
            FROM two_view_results
            WHERE timestamp >= %s
            ORDER BY i1, i2
            """,
                (datetime.now().replace(hour=0, minute=0, second=0, microsecond=0),),
            )

            results = cursor.fetchall()
            print(
                "Image Pair | Timestamp                  | Verified Matches | Inlier Ratio | Success | "
                "Computation Time (s) | Worker"
            )
            print("-" * 120)
            for row in results:
                i1, i2, timestamp, corr_count, inlier_ratio, success, comp_time, worker = row
                print(
                    f"({i1:2d}, {i2:2d}) | {timestamp} | {corr_count:8d} | "
                    f"{inlier_ratio:.4f} | {success} | {comp_time:.4f} | {worker}"
                )

            cursor.close()
            conn.close()
        except Exception as e:
            print(f"[MAIN] Database query failed: {e}")

        # Process and save results
        print("\n[MAIN] Processing results...")

        summary_file = results_dir / "results_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Two-View Estimation Summary - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total time: {total_time:.2f} seconds\n")
            f.write(f"Number of images: {num_images}\n")
            f.write(f"Number of image pairs: {len(image_pairs)}\n\n")
            f.write("Results per image pair:\n")
            f.write("-" * 50 + "\n")

        for (i1, i2), two_view_output in two_view_output_dict.items():
            # Extract fields from the dataclass
            i2Ri1 = two_view_output.i2Ri1
            i2Ui1 = two_view_output.i2Ui1
            v_corr_idxs = two_view_output.v_corr_idxs
            pre_ba_report = two_view_output.pre_ba_report
            post_ba_report = two_view_output.post_ba_report
            post_isp_report = two_view_output.post_isp_report

            with open(summary_file, "a") as f:
                f.write(f"\nImage Pair ({indices[i1]}, {indices[i2]}):\n")

            if i2Ri1 is not None and i2Ui1 is not None:
                ypr_deg = np.degrees(i2Ri1.xyz()) if i2Ri1 else None

                print(f"[MAIN] Image Pair ({indices[i1]}, {indices[i2]}):")
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
                            images[i1],
                            images[i2],
                            keypoints_list[i1],
                            keypoints_list[i2],
                            v_corr_idxs[:max_viz_corrs],
                            max_corrs=max_viz_corrs,
                        )

                        plt.figure(figsize=(12, 10))
                        plt.imshow(correspondence_image.value_array)
                        plt.title(f"Image Pair ({indices[i1]}, {indices[i2]}) Verified Correspondences")
                        plt.savefig(images_dir / f"correspondences_{indices[i1]}_{indices[i2]}.png")
                        plt.close()
                    except Exception as e:
                        print(f"[MAIN] Visualization error for pair ({indices[i1]}, {indices[i2]}): {e}")

            else:
                print(f"[MAIN] Image Pair ({indices[i1]}, {indices[i2]}): Relative pose estimation failed")

                with open(summary_file, "a") as f:
                    f.write("  - Relative pose estimation failed\n")

        print(f"\n[MAIN] All results saved to: {results_dir}")

        # Keep processes alive
        print("[MAIN] Processes will continue running. Press Ctrl+C to stop...")

        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[MAIN] Termination signal received, cleaning up...")

    except Exception as e:
        print(f"[MAIN] ❌ Error occurred: {e}")

    finally:
        # Clean up
        print("[MAIN] Closing Dask client...")

        client.close()
        print("[MAIN] Dask client closed")


if __name__ == "__main__":
    main()
