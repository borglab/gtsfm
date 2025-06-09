"""
Realistic Keypoints Serialization Bug Test - WITH TYPE VALIDATION

This test uses the REAL two-view estimator workflow and adds detailed type validation
to prove that coordinates becomes a Keypoints object during serialization.

Key addition: Type checking to validate our hypothesis about the bug.
"""

from pathlib import Path
import numpy as np
import time
import atexit
import subprocess
import signal
import socket
import os
from datetime import datetime
from dask.distributed import Client

from gtsam import Pose3, PinholeCameraCal3Bundler, Unit3, Rot3

from gtsfm.loader.yfcc_imb_loader import YfccImbLoader
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
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
from gtsfm.utils.ssh_tunneling import SSHTunnelManager

# Global process management
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

def debug_keypoints_type(keypoints_obj, label="keypoints"):
    """
    Debug function to thoroughly examine keypoints object type corruption.
    This function will be called when errors occur to validate our hypothesis.
    """
    print(f"\n=== TYPE VALIDATION FOR {label.upper()} ===")
    print(f"{label} object type: {type(keypoints_obj)}")
    print(f"{label} object repr: {repr(keypoints_obj)}")
    
    if hasattr(keypoints_obj, 'coordinates'):
        coords_type = type(keypoints_obj.coordinates)
        print(f"{label}.coordinates type: {coords_type}")
        print(f"{label}.coordinates repr: {repr(keypoints_obj.coordinates)}")
        
        # This is the KEY CHECK for our hypothesis
        if coords_type == Keypoints:
            print(f"🎯 HYPOTHESIS CONFIRMED! {label}.coordinates became a Keypoints object!")
            print(f"   Expected: numpy.ndarray")
            print(f"   Actual: {coords_type}")
            
            # Check if this nested Keypoints has coordinates too (recursive corruption)
            if hasattr(keypoints_obj.coordinates, 'coordinates'):
                nested_coords_type = type(keypoints_obj.coordinates.coordinates)
                print(f"   {label}.coordinates.coordinates type: {nested_coords_type}")
                print(f"   This indicates recursive Keypoints nesting!")
                
            return True  # Bug confirmed
        elif coords_type == np.ndarray:
            print(f"✅ {label}.coordinates is correctly np.ndarray")
            coords_shape = keypoints_obj.coordinates.shape
            coords_dtype = keypoints_obj.coordinates.dtype
            print(f"   Shape: {coords_shape}, Dtype: {coords_dtype}")
            return False  # No bug detected
        else:
            print(f"⚠️  {label}.coordinates has unexpected type: {coords_type}")
            return False
    else:
        print(f"❌ {label} object has no coordinates attribute!")
        return False

def debug_type_corruption_on_worker(keypoints_i1, keypoints_i2, test_name="worker_test"):
    """
    Function that runs on remote workers to detect type corruption.
    This will be called when the main test fails to examine the state.
    """
    import socket
    import numpy as np
    from gtsfm.common.keypoints import Keypoints
    
    worker_name = socket.gethostname()
    print(f"\n[{worker_name}] === TYPE CORRUPTION DEBUG ON WORKER ===")
    print(f"[{worker_name}] Test: {test_name}")
    
    # Check types on the worker side
    bug_detected = False
    
    print(f"[{worker_name}] Checking keypoints_i1...")
    if debug_keypoints_type(keypoints_i1, "keypoints_i1"):
        bug_detected = True
    
    print(f"[{worker_name}] Checking keypoints_i2...")
    if debug_keypoints_type(keypoints_i2, "keypoints_i2"):
        bug_detected = True
    
    # Try to reproduce the exact error that occurs in ransac.py:104
    if not bug_detected:
        print(f"[{worker_name}] No type corruption detected. Testing chain call...")
        try:
            # This is the exact line that fails in ransac.py
            test_indices = np.array([0, 1])
            coords_i1 = keypoints_i1.extract_indices(test_indices).coordinates
            coords_i2 = keypoints_i2.extract_indices(test_indices).coordinates
            print(f"[{worker_name}] ✅ Chain call succeeded")
            print(f"[{worker_name}] Result types: {type(coords_i1)}, {type(coords_i2)}")
        except Exception as e:
            print(f"[{worker_name}] ❌ Chain call failed: {e}")
            if "'Keypoints' object is not subscriptable" in str(e):
                print(f"[{worker_name}] 🎯 This is the target error!")
                bug_detected = True
    
    return {
        "worker": worker_name,
        "bug_detected": bug_detected,
        "keypoints_i1_coords_type": str(type(keypoints_i1.coordinates)) if hasattr(keypoints_i1, 'coordinates') else "NO_COORDS",
        "keypoints_i2_coords_type": str(type(keypoints_i2.coordinates)) if hasattr(keypoints_i2, 'coordinates') else "NO_COORDS"
    }

def load_config(config_file='gtsfm/configs/local_scheduler_postgres_remote_cluster.yaml'):
    """Load configuration from YAML file"""
    import yaml
    try:
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

def setup_cluster_infrastructure(config):
    """Set up SSH tunnels and start Dask scheduler"""
    tunnel_manager = SSHTunnelManager()
    tunnel_manager.config = config
    scheduler_port = tunnel_manager.setup_complete_infrastructure()
    return scheduler_port, tunnel_manager.processes

def main():
    """Main function - using REAL two-view estimator with TYPE VALIDATION"""
    print("="*80)
    print("REALISTIC KEYPOINTS SERIALIZATION BUG TEST - WITH TYPE VALIDATION")
    print("="*80)
    print("Using REAL two-view estimator workflow to trigger the bug")
    print("Target error: 'Keypoints' object is not subscriptable")
    print("Hypothesis: self.coordinates becomes Keypoints instead of np.ndarray")
    print("This test will VALIDATE the type corruption hypothesis")
    print("="*80)
    
    # Load configuration
    config = load_config()
    
    # Create result folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = Path(f"realistic_bug_test_with_validation_{timestamp}")
    results_dir.mkdir(exist_ok=True)
    print(f"Saving results to folder: {results_dir}")
    
    try:
        # Set up cluster infrastructure
        print("\n1. Setting up distributed cluster...")
        scheduler_port, cluster_processes = setup_cluster_infrastructure(config)
        
        global processes
        processes = cluster_processes
        
        # Prepare REAL input data - using actual test images
        print("2. Preparing REAL input data...")
        cwd = Path.cwd()
        folder_path = cwd / "tests" / "data" / "imb_reichstag"

        # Use just 2 images to simplify but keep it real
        indices = [0, 1]  
        num_images = len(indices)

        # Load REAL images and camera intrinsics
        loader = YfccImbLoader(str(folder_path))
        images = [loader.get_image(i) for i in indices]
        camera_intrinsics = [loader.get_camera_intrinsics_full_res(i) for i in indices]

        # Feature detection and description - REAL SIFT features
        detector_descriptor = SIFTDetectorDescriptor()
        print("3. Detecting keypoints with REAL SIFT...")
        features = [detector_descriptor.detect_and_describe(image) for image in images]
        keypoints_list = []
        descriptors_list = []

        for i, (kp, desc) in enumerate(features):
            # Convert to Keypoints objects with image IDs - EXACTLY like real system
            keypoints_obj = Keypoints(coordinates=kp)
            keypoints_obj.image_id = indices[i]
            keypoints_list.append(keypoints_obj)
            descriptors_list.append(desc)
            print(f"   Image {indices[i]}: {desc.shape[0]} keypoints")
            
            # Validate types locally before sending to workers
            print(f"   Local validation for image {indices[i]}:")
            debug_keypoints_type(keypoints_obj, f"keypoints_image_{indices[i]}")

        # Feature matching - REAL matching
        matcher = TwoWayMatcher(ratio_test_threshold=0.8)
        print("4. Matching keypoints with REAL matcher...")

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
                print(f"   Image pair ({indices[i1]}, {indices[i2]}): {match_indices.shape[0]} matches")

        # Create REAL verifier and inlier processor
        verifier = Ransac(use_intrinsics_in_verification=False, estimation_threshold_px=2)
        inlier_support_processor = InlierSupportProcessor(
            min_num_inliers_est_model=20,
            min_inlier_ratio_est_model=0.1
        )

        # Create REAL two-view estimator WITHOUT database integration
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
            # postgres_params=None  # Remove DB complexity
        )

        # Set empty pose priors and ground truth cameras
        relative_pose_priors = {}
        gt_cameras = [None] * num_images
        gt_scene_mesh = None

        # Connect to Dask cluster
        print("5. Connecting to Dask cluster...")
        client = Client(f"tcp://localhost:{scheduler_port}")
        print(f"   Connected to Dask cluster: {client.dashboard_link}")
        
        print(f"   Cluster workers: {len(client.scheduler_info()['workers'])}")
        for worker_id, worker_info in client.scheduler_info()['workers'].items():
            host = worker_info.get('host', 'unknown')
            port = worker_info.get('port', 'unknown')
            worker_address = f"{host}:{port}" if 'port' in worker_info else host
            print(f"     - Worker: {worker_id}, Address: {worker_address}")

        try:
            # Run REAL distributed two-view estimation
            print("6. Running REAL distributed two-view estimation...")
            print("   🎯 This should trigger the serialization bug!")
            
            start_time = time.time()
            
            # This is the EXACT same call that triggers the bug in the real system!
            two_view_output_dict = run_two_view_estimator_as_futures(
                client=client,
                two_view_estimator=two_view_estimator,
                keypoints_list=keypoints_list,  # 🚨 This gets serialized to workers
                putative_corr_idxs_dict=putative_corr_idxs_dict,
                camera_intrinsics=camera_intrinsics,
                relative_pose_priors=relative_pose_priors,
                gt_cameras=gt_cameras,
                gt_scene_mesh=gt_scene_mesh
            )

            total_time = time.time() - start_time
            print(f"   ✅ Distributed computation completed in {total_time:.2f} seconds")
            print("   ❌ No serialization bug was triggered")
            bug_found = False

        except Exception as e:
            error_msg = str(e)
            print(f"   ❌ Error occurred: {error_msg}")
            
            # Check if this is the target bug
            is_target_bug = "'Keypoints' object is not subscriptable" in error_msg
            if is_target_bug:
                print("   🎯 SUCCESS! Found the target serialization bug!")
                print("   This confirms the coordinates corruption during distributed processing")
                
                # NOW DO THE TYPE VALIDATION!
                print("\n" + "="*60)
                print("PERFORMING TYPE VALIDATION ON WORKERS")
                print("="*60)
                
                # Run type debugging on workers to validate our hypothesis
                print("   Submitting type validation tasks to workers...")
                debug_futures = client.map(
                    debug_type_corruption_on_worker, 
                    [keypoints_list[0]] * 3,  # Send same keypoints to multiple workers
                    [keypoints_list[1]] * 3,
                    [f"validation_test_{i}" for i in range(3)]
                )
                
                print("   Gathering type validation results...")
                debug_results = client.gather(debug_futures)
                
                print("\n   TYPE VALIDATION RESULTS:")
                print("   " + "-"*50)
                type_corruption_confirmed = False
                
                for result in debug_results:
                    worker = result['worker']
                    bug_detected = result['bug_detected']
                    coords_i1_type = result['keypoints_i1_coords_type']
                    coords_i2_type = result['keypoints_i2_coords_type']
                    
                    print(f"   Worker {worker}:")
                    print(f"     Bug detected: {bug_detected}")
                    print(f"     keypoints_i1.coordinates type: {coords_i1_type}")
                    print(f"     keypoints_i2.coordinates type: {coords_i2_type}")
                    
                    if "'Keypoints'" in coords_i1_type or "'Keypoints'" in coords_i2_type:
                        print(f"     🎯 TYPE CORRUPTION CONFIRMED on {worker}!")
                        type_corruption_confirmed = True
                    
                if type_corruption_confirmed:
                    print("\n   🎯 HYPOTHESIS VALIDATED!")
                    print("   ✅ coordinates DID become Keypoints objects instead of np.ndarray")
                    print("   ✅ This proves the serialization corruption mechanism")
                else:
                    print("\n   ⚠️  Hypothesis not confirmed by type validation")
                    print("   The error may be caused by a different mechanism")
                
                # Save the validation results
                validation_file = results_dir / "type_validation_results.txt"
                with open(validation_file, "w") as f:
                    f.write(f"Type Validation Results - {timestamp}\n")
                    f.write("="*50 + "\n\n")
                    f.write(f"Target Error: {error_msg}\n")
                    f.write(f"Type Corruption Confirmed: {type_corruption_confirmed}\n\n")
                    f.write("Worker Results:\n")
                    for result in debug_results:
                        f.write(f"  {result}\n")
                
                import traceback
                print("\n   Full traceback:")
                traceback.print_exc()
                
                bug_found = True
            else:
                print(f"   Different error occurred: {error_msg}")
                bug_found = False
                import traceback
                traceback.print_exc()

        # Process and save results
        print("\n7. Processing results...")
        
        summary_file = results_dir / "results_summary.txt"
        with open(summary_file, "w") as f:
            f.write(f"Realistic Keypoints Serialization Bug Test with Type Validation - {timestamp}\n")
            f.write("="*70 + "\n\n")
            f.write("This test validates the hypothesis that serialization corruption\n")
            f.write("causes coordinates to become Keypoints objects instead of np.ndarray.\n\n")
            f.write(f"Bug Found: {'YES' if 'bug_found' in locals() and bug_found else 'NO'}\n")
            f.write(f"Type Corruption Confirmed: {'YES' if 'type_corruption_confirmed' in locals() and type_corruption_confirmed else 'NO'}\n")
            f.write(f"Test completed in: {total_time if 'total_time' in locals() else 'N/A'} seconds\n")

        # Summary
        print("\n" + "="*80)
        print("REALISTIC BUG TEST WITH TYPE VALIDATION SUMMARY")
        print("="*80)
        
        if 'bug_found' in locals() and bug_found:
            print("🎯 SUCCESS: The serialization bug was reproduced!")
            print("   ✅ This confirms the bug occurs in real distributed workflows")
            if 'type_corruption_confirmed' in locals() and type_corruption_confirmed:
                print("   ✅ TYPE CORRUPTION HYPOTHESIS CONFIRMED!")
                print("   ✅ coordinates became Keypoints objects during serialization")
                print("   ✅ This proves the exact mechanism of the bug")
            else:
                print("   ⚠️  Type corruption hypothesis not confirmed")
                print("   ⚠️  The bug may have a different mechanism")
        else:
            print("✅ No serialization bug found")
            print("   This suggests:")
            print("   1. The current implementation has defensive coding that prevents the bug")
            print("   2. The bug may have been fixed in recent commits")
            print("   3. The bug may require even more specific conditions")

        print(f"\nTest completed at {datetime.now()}")
        print(f"Results saved to: {results_dir}")
        
        # Keep processes alive
        print("\n🔥 Keeping processes alive...")
        print("Press Ctrl+C to stop...")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⚠️ Termination signal received")
        
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test setup failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        if 'client' in locals():
            try:
                client.close()
                print("✅ Dask client closed")
            except:
                pass

if __name__ == "__main__":
    main()
