"""Helper functions for running scene optimizer on subgraphs.

Authors: Zongyue Liu
"""

import time
from typing import Dict, List, Optional, Tuple, Any, Union

import dask
import numpy as np
from dask.distributed import Client
from gtsam import Pose3, Rot3, Unit3, Cal3Bundler, Cal3_S2, PinholeCameraCal3Bundler, PinholeCameraCal3_S2

import gtsfm.common.types as gtsfm_types
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.graph_partitioner.single_partition import SinglePartition
from gtsfm.scene_optimizer import SceneOptimizer
import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

# Define calibration and camera types explicitly
CalibType = Union[Cal3Bundler, Cal3_S2, None]
CameraType = Union[PinholeCameraCal3Bundler, PinholeCameraCal3_S2, None]


class SubgraphInput:
    """Container for inputs to scene optimizer for a single subgraph."""
    
    def __init__(
        self,
        subgraph_id: int,
        original_indices: List[int],
        keypoints_list: List[Keypoints],
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        images: List[Any],
        camera_intrinsics: List[Optional[CalibType]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        gt_poses: List[Optional[Pose3]],
        gt_cameras: List[Optional[CameraType]],
    ):
        """Initialize a subgraph input container.
        
        Args:
            subgraph_id: ID of the subgraph.
            original_indices: Mapping from subgraph indices to original indices.
            keypoints_list: List of keypoints for images in the subgraph.
            i2Ri1_dict: Dictionary of relative rotations for edges in the subgraph.
            i2Ui1_dict: Dictionary of relative translations for edges in the subgraph.
            v_corr_idxs_dict: Dictionary of verified correspondences for edges in the subgraph.
            two_view_reports: Dictionary of two-view reports for edges in the subgraph.
            images: List of images (as futures or delayed objects) for the subgraph.
            camera_intrinsics: List of camera intrinsics for images in the subgraph.
            absolute_pose_priors: List of absolute pose priors for images in the subgraph.
            relative_pose_priors: Dictionary of relative pose priors for edges in the subgraph.
            gt_poses: List of ground truth poses for images in the subgraph.
            gt_cameras: List of ground truth cameras for images in the subgraph.
        """
        self.subgraph_id = subgraph_id
        self.original_indices = original_indices
        self.keypoints_list = keypoints_list
        self.i2Ri1_dict = i2Ri1_dict
        self.i2Ui1_dict = i2Ui1_dict
        self.v_corr_idxs_dict = v_corr_idxs_dict
        self.two_view_reports = two_view_reports
        self.images = images
        self.camera_intrinsics = camera_intrinsics
        self.absolute_pose_priors = absolute_pose_priors
        self.relative_pose_priors = relative_pose_priors
        self.gt_poses = gt_poses
        self.gt_cameras = gt_cameras
        self.num_images = len(original_indices)


def prepare_subgraph_inputs(
    graph_partitioner: SinglePartition,
    similarity_matrix: np.ndarray,
    keypoints_list: List[Keypoints],
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
    images: List[Any],
    camera_intrinsics: List[Optional[CalibType]],
    absolute_pose_priors: List[Optional[PosePrior]],
    relative_pose_priors: Dict[Tuple[int, int], PosePrior],
    gt_poses: List[Optional[Pose3]],
    gt_cameras: List[Optional[CameraType]],
) -> List[SubgraphInput]:
    """Prepare inputs for scene optimizer for each subgraph.
    
    Args:
        graph_partitioner: Graph partitioner instance.
        similarity_matrix: Similarity matrix between images.
        keypoints_list: List of keypoints for all images.
        i2Ri1_dict: Dictionary of relative rotations.
        i2Ui1_dict: Dictionary of relative translations.
        v_corr_idxs_dict: Dictionary of verified correspondences.
        two_view_reports: Dictionary of two-view reports.
        images: List of images (as futures or delayed objects).
        camera_intrinsics: List of camera intrinsics.
        absolute_pose_priors: List of absolute pose priors.
        relative_pose_priors: Dictionary of relative pose priors.
        gt_poses: List of ground truth poses.
        gt_cameras: List of ground truth cameras.
        
    Returns:
        List of SubgraphInput objects, one for each subgraph.
    """
    # Partition the graph into subgraphs
    subgraphs = graph_partitioner.partition(similarity_matrix)
    
    logger.info(f"Graph partitioned into {len(subgraphs)} subgraphs")
    
    # Prepare inputs for each subgraph
    subgraph_inputs = []
    for i, subgraph_edges in enumerate(subgraphs):
        # Get unique nodes (image indices) in this subgraph
        subgraph_nodes = sorted(set(node for edge in subgraph_edges for node in edge))
        logger.info(f"Subgraph {i+1}/{len(subgraphs)} has {len(subgraph_nodes)} nodes and {len(subgraph_edges)} edges")
        
        # Create mapping from original indices to subgraph indices
        orig_to_subgraph = {orig: i for i, orig in enumerate(subgraph_nodes)}
        
        # Filter keypoints
        subgraph_keypoints = [keypoints_list[i] for i in subgraph_nodes]
        
        # Filter dictionaries based on edges
        subgraph_i2Ri1 = {}
        subgraph_i2Ui1 = {}
        subgraph_v_corr_idxs = {}
        subgraph_two_view_reports = {}
        subgraph_relative_pose_priors = {}
        
        for i1, i2 in subgraph_edges:
            if (i1, i2) in i2Ri1_dict:
                new_i1, new_i2 = orig_to_subgraph[i1], orig_to_subgraph[i2]
                subgraph_i2Ri1[(new_i1, new_i2)] = i2Ri1_dict[(i1, i2)]
                subgraph_i2Ui1[(new_i1, new_i2)] = i2Ui1_dict[(i1, i2)]
                subgraph_v_corr_idxs[(new_i1, new_i2)] = v_corr_idxs_dict[(i1, i2)]
                subgraph_two_view_reports[(new_i1, new_i2)] = two_view_reports[(i1, i2)]
                
                if (i1, i2) in relative_pose_priors:
                    subgraph_relative_pose_priors[(new_i1, new_i2)] = relative_pose_priors[(i1, i2)]
        
        # Filter lists based on node indices
        subgraph_images = [images[i] for i in subgraph_nodes]
        subgraph_intrinsics = [camera_intrinsics[i] for i in subgraph_nodes]
        subgraph_absolute_pose_priors = [absolute_pose_priors[i] for i in subgraph_nodes]
        subgraph_gt_poses = [gt_poses[i] for i in subgraph_nodes]
        subgraph_gt_cameras = [gt_cameras[i] for i in subgraph_nodes]
        
        # Create subgraph input
        subgraph_input = SubgraphInput(
            subgraph_id=i,
            original_indices=subgraph_nodes,
            keypoints_list=subgraph_keypoints,
            i2Ri1_dict=subgraph_i2Ri1,
            i2Ui1_dict=subgraph_i2Ui1,
            v_corr_idxs_dict=subgraph_v_corr_idxs,
            two_view_reports=subgraph_two_view_reports,
            images=subgraph_images,
            camera_intrinsics=subgraph_intrinsics,
            absolute_pose_priors=subgraph_absolute_pose_priors,
            relative_pose_priors=subgraph_relative_pose_priors,
            gt_poses=subgraph_gt_poses,
            gt_cameras=subgraph_gt_cameras,
        )
        
        subgraph_inputs.append(subgraph_input)
    
    return subgraph_inputs


def process_subgraph(
    scene_optimizer: SceneOptimizer,
    subgraph_input: SubgraphInput,
    gt_scene_mesh=None,
) -> Tuple[GtsfmData, List[int]]:
    """Process a single subgraph using scene optimizer.
    
    Args:
        scene_optimizer: Scene optimizer instance.
        subgraph_input: Input data for the subgraph.
        gt_scene_mesh: Ground truth scene mesh (optional).
        
    Returns:
        Tuple containing:
        - GtsfmData result for the subgraph
        - List of original indices for the subgraph
    """
    logger.info(f"Processing subgraph {subgraph_input.subgraph_id+1} with {subgraph_input.num_images} images")
    
    # Create computation graph for this subgraph
    subgraph_start_time = time.time()
    
    subgraph_result, _, _ = scene_optimizer.create_computation_graph(
        keypoints_list=subgraph_input.keypoints_list,
        i2Ri1_dict=subgraph_input.i2Ri1_dict,
        i2Ui1_dict=subgraph_input.i2Ui1_dict,
        v_corr_idxs_dict=subgraph_input.v_corr_idxs_dict,
        two_view_reports=subgraph_input.two_view_reports,
        num_images=subgraph_input.num_images,
        images=subgraph_input.images,
        camera_intrinsics=subgraph_input.camera_intrinsics,
        absolute_pose_priors=subgraph_input.absolute_pose_priors,
        relative_pose_priors=subgraph_input.relative_pose_priors,
        cameras_gt=subgraph_input.gt_cameras,
        gt_wTi_list=subgraph_input.gt_poses,
        gt_scene_mesh=gt_scene_mesh,
    )
    
    # Compute result
    result = dask.compute(subgraph_result)[0]
    subgraph_duration = time.time() - subgraph_start_time
    logger.info(f"Subgraph {subgraph_input.subgraph_id+1} processing completed in {subgraph_duration:.2f} seconds")
    
    return result, subgraph_input.original_indices


def run_scene_optimizer_as_futures(
    client: Client,
    scene_optimizer: SceneOptimizer,
    subgraph_inputs: List[SubgraphInput],
    gt_scene_mesh=None,
) -> List[dask.delayed]:
    """Run scene optimizer for each subgraph as futures.
    
    Args:
        client: Dask client for parallel computation.
        scene_optimizer: Scene optimizer instance.
        subgraph_inputs: List of inputs for each subgraph.
        gt_scene_mesh: Ground truth scene mesh (optional).
        
    Returns:
        List of futures for each subgraph result.
    """
    futures = []
    
    for subgraph_input in subgraph_inputs:
        # Create a future for processing this subgraph
        future = client.submit(
            process_subgraph,
            scene_optimizer=scene_optimizer,
            subgraph_input=subgraph_input,
            gt_scene_mesh=gt_scene_mesh,
        )
        futures.append(future)
    
    return futures


def run_scene_optimizer_for_subgraphs(
    client: Client,
    scene_optimizer: SceneOptimizer,
    graph_partitioner: SinglePartition,
    similarity_matrix: np.ndarray,
    keypoints_list: List[Keypoints],
    i2Ri1_dict: Dict[Tuple[int, int], Rot3],
    i2Ui1_dict: Dict[Tuple[int, int], Unit3],
    v_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
    num_images: int,
    images: List[Any],
    camera_intrinsics: List[Optional[CalibType]],
    absolute_pose_priors: List[Optional[PosePrior]],
    relative_pose_priors: Dict[Tuple[int, int], PosePrior],
    cameras_gt: List[Optional[CameraType]],
    gt_wTi_list: List[Optional[Pose3]],
    gt_scene_mesh=None,
) -> List[Tuple[GtsfmData, List[int]]]:
    """Run scene optimizer separately for each subgraph.
    
    Args:
        client: Dask client for parallel computation.
        scene_optimizer: Scene optimizer instance.
        graph_partitioner: Graph partitioner instance.
        similarity_matrix: Similarity matrix between images.
        keypoints_list: List of keypoints for all images.
        i2Ri1_dict: Dictionary of relative rotations.
        i2Ui1_dict: Dictionary of relative translations.
        v_corr_idxs_dict: Dictionary of verified correspondences.
        two_view_reports: Dictionary of two-view reports.
        num_images: Total number of images.
        images: List of images (as futures or delayed objects).
        camera_intrinsics: List of camera intrinsics.
        absolute_pose_priors: List of absolute pose priors.
        relative_pose_priors: Dictionary of relative pose priors.
        cameras_gt: List of ground truth cameras.
        gt_wTi_list: List of ground truth poses.
        gt_scene_mesh: Ground truth scene mesh (optional).
        
    Returns:
        List of tuples containing:
        - GtsfmData result for each subgraph
        - List of original indices for each subgraph
    """
    # Prepare inputs for each subgraph
    subgraph_inputs = prepare_subgraph_inputs(
        graph_partitioner=graph_partitioner,
        similarity_matrix=similarity_matrix,
        keypoints_list=keypoints_list,
        i2Ri1_dict=i2Ri1_dict,
        i2Ui1_dict=i2Ui1_dict,
        v_corr_idxs_dict=v_corr_idxs_dict,
        two_view_reports=two_view_reports,
        images=images,
        camera_intrinsics=camera_intrinsics,
        absolute_pose_priors=absolute_pose_priors,
        relative_pose_priors=relative_pose_priors,
        gt_poses=gt_wTi_list,
        gt_cameras=cameras_gt,
    )
    
    # Run scene optimizer for each subgraph
    futures = run_scene_optimizer_as_futures(
        client=client,
        scene_optimizer=scene_optimizer,
        subgraph_inputs=subgraph_inputs,
        gt_scene_mesh=gt_scene_mesh,
    )
    
    # Gather results
    results = client.gather(futures)
    
    return results


def merge_subgraph_results(
    subgraph_results: List[Tuple[GtsfmData, List[int]]],
    num_images: int
) -> GtsfmData:
    """Merge results from multiple subgraphs into a single GtsfmData object.
    
    Args:
        subgraph_results: List of tuples (GtsfmData, original_indices) for each subgraph.
        num_images: Total number of images in the original scene.
        
    Returns:
        A merged GtsfmData object.
    """
    merged_data = GtsfmData(number_images=num_images)
    
    for subgraph_data, original_indices in subgraph_results:
        # Mapping from subgraph indices to original indices
        subgraph_to_orig = {i: orig for i, orig in enumerate(original_indices)}
        
        # Add cameras with original indices
        for subgraph_idx in subgraph_data.get_valid_camera_indices():
            orig_idx = subgraph_to_orig[subgraph_idx]
            merged_data.add_camera(orig_idx, subgraph_data.get_camera(subgraph_idx))
        
        # Add tracks with updated measurement indices
        for track in subgraph_data.get_tracks():
            # Create a new track with updated indices
            new_track = copy_track_with_remapped_indices(track, subgraph_to_orig)
            merged_data.add_track(new_track)
    
    return merged_data


def copy_track_with_remapped_indices(track, index_mapping):
    """Create a copy of an SfmTrack with remapped measurement indices.
    
    Args:
        track: Original SfmTrack.
        index_mapping: Mapping from subgraph indices to original indices.
        
    Returns:
        A new SfmTrack with remapped indices.
    """
    from gtsam import SfmTrack
    
    new_track = SfmTrack()
    new_track.setPoint3(track.point3())
    
    for i in range(track.number_measurements()):
        measurement = track.measurement(i)
        cam_idx = track.camera_idx(i)
        orig_cam_idx = index_mapping[cam_idx]
        new_track.add_measurement(measurement, orig_cam_idx)
    
    return new_track