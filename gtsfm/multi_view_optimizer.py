"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
import os
from pathlib import Path
from dask.delayed import Delayed
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point3,
    Pose3,
    Rot3,
    Unit3,
)

from gtsfm.common.image import Image
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.io as io
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics
import gtsfm.utils.reprojection as reproj_utils
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.data_association.data_assoc import DataAssociation

METRICS_PATH = Path(__file__).resolve().parent.parent.parent / "result_metrics"

# Paths to Save Output in React Folders.
REACT_METRICS_PATH = Path(__file__).resolve().parent.parent / "rtf_vis_tool" / "src" / "result_metrics"

logger = logger_utils.get_logger()


class MultiViewOptimizer:
    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
        bundle_adjustment_module: BundleAdjustmentOptimizer
    ) -> None:
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = data_association_module
        self.ba_optimizer = bundle_adjustment_module

    def create_computation_graph(
        self,
        images_graph: List[Delayed],
        num_images: int,
        keypoints_graph: List[Delayed],
        i2Ri1_graph: Dict[Tuple[int, int], Delayed],
        i2Ui1_graph: Dict[Tuple[int, int], Delayed],
        v_corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        intrinsics_graph: List[Delayed],
        gt_poses_graph: List[Delayed] = None,
    ) -> Tuple[Delayed, Delayed, Optional[Delayed], Optional[Delayed]]:
        """Creates a computation graph for multi-view optimization.

        Args:
            num_images: number of images in the scene.
            keypoints_graph: keypoints for images, each wrapped up as Delayed.
            i2Ri1_graph: relative rotations for image pairs, each value wrapped up as Delayed.
            i2Ui1_graph: relative unit-translations for image pairs, each value wrapped up as Delayed.
            v_corr_idxs_graph: indices of verified correspondences for image pairs, wrapped up as Delayed.
            intrinsics_graph: intrinsics for images, wrapped up as Delayed.

        Returns:
            The input to bundle adjustment, wrapped up as Delayed.
            The final output, wrapped up as Delayed.
            Dictionary containing metrics, wrapped up as Delayed
        """
        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(prune_to_largest_connected_component)(i2Ri1_graph, i2Ui1_graph)

        pruned_i2Ri1_graph = pruned_graph[0]
        pruned_i2Ui1_graph = pruned_graph[1]

        wRi_graph = self.rot_avg_module.create_computation_graph(num_images, pruned_i2Ri1_graph)
        wti_graph = self.trans_avg_module.create_computation_graph(num_images, pruned_i2Ui1_graph, wRi_graph)
        init_cameras_graph = dask.delayed(init_cameras)(wRi_graph, wti_graph, intrinsics_graph)

        ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
            num_images, init_cameras_graph, v_corr_idxs_graph, keypoints_graph, images_graph
        )

        auxiliary_graph_list = [
            dask.delayed(io.save_json_file)(
                os.path.join("result_metrics", "data_association_metrics.json"), data_assoc_metrics_graph
            ),

            # duplicate dask variable to save data_association_metrics within React directory
            dask.delayed(io.save_json_file)(
                os.path.join(REACT_METRICS_PATH, "data_association_metrics.json"), data_assoc_metrics_graph
            )
        ]

        # dummy graph to force an immediate dump of data association metrics
        ba_input_graph = dask.delayed(lambda x, y: (x, y))(ba_input_graph, auxiliary_graph_list)[0]

        ba_result_graph = self.ba_optimizer.create_computation_graph(ba_input_graph)

        save_track_patches_viz = True
        if save_track_patches_viz:

            # save vstacked-patched for each 3d track
            track3d_vis_graph = dask.delayed(save_ba_output_track_visualizations)(ba_result_graph, images_graph)

            # as visualization tasks are not to be provided to the user, we create a
            # dummy computation of concatenating viz tasks with the output graph,
            # forcing computation of viz tasks
            ba_result_aux_graph = dask.delayed(lambda x, y: (x, y))(ba_result_graph, track3d_vis_graph)

            # return the entry with just the sfm result
            ba_result_graph = ba_result_aux_graph[0]

        # filter by reprojection error threshold
        ba_result_graph = dask.delayed(filter_ba_result)(ba_result_graph, self.ba_optimizer.output_reproj_error_thresh)


        if gt_poses_graph is None:
            return ba_input_graph, ba_result_graph, None, None

        metrics_graph = dask.delayed(metrics.compute_averaging_metrics)(
            i2Ui1_graph, wRi_graph, wti_graph, gt_poses_graph
        )
        saved_metrics_graph = dask.delayed(io.save_json_file)(
            "result_metrics/averaging_metrics.json", metrics_graph
        )

        # duplicate dask variable to save optimizer_metrics within React directory
        react_saved_metrics_graph = dask.delayed(io.save_json_file)(
            os.path.join(REACT_METRICS_PATH, "averaging_metrics.json"), metrics_graph
        )

        return ba_input_graph, ba_result_graph, saved_metrics_graph, react_saved_metrics_graph


def save_ba_output_track_visualizations(optimized_data: GtsfmData, images: List[Image]) -> None:
    """Bin reprojection errors per track, and for each bin, save vstacked-patches for each 3d track

    Args:
        optimized_data: optimized camera poses and 3d point tracks.
        images: a list of all images in scene (optional and only for track patch visualization).
    """
    for j in range(optimized_data.number_tracks()):
        track_3d = optimized_data.get_track(j)
        track_errors, _ = reproj_utils.compute_track_reprojection_errors(optimized_data._cameras, track_3d)
        avg_track_reproj_error = np.mean(track_errors)
        if np.isnan(avg_track_reproj_error):
            # ignore NaN tracks
            continue
        io.save_track3d_visualizations(j, [track_3d], images, save_dir=os.path.join("plots", "tracks_3d", f"{int(np.round(avg_track_reproj_error))}"))


def filter_ba_result(optimized_data: GtsfmData, output_reproj_error_thresh: float) -> GtsfmData:
    """Return a new GtsfmData object, preserving only tracks with low average reprojection error.

    Args:
        optimized_data: optimized camera poses and 3d point tracks.

    Returns:
        filtered_result: optimized camera poses and filtered 3d point tracks.
    """
    # filter the largest errors
    filtered_result = optimized_data.filter_landmarks(output_reproj_error_thresh)

    metrics_dict = {}
    metrics_dict["after_filtering"] = filtered_result.aggregate_metrics()

    logger.info("[Result] Number of tracks after filtering: %d", metrics_dict["after_filtering"]["number_tracks"])
    logger.info("[Result] Mean track length %.3f", metrics_dict["after_filtering"]["3d_track_lengths"]["mean"])
    logger.info("[Result] Median track length %.3f", metrics_dict["after_filtering"]["3d_track_lengths"]["median"])
    filtered_result.log_scene_reprojection_error_stats()

    io.save_json_file(os.path.join(METRICS_PATH, "bundle_adjustment_filtering_metrics.json"), metrics_dict)

    return filtered_result



def prune_to_largest_connected_component(
    rotations: Dict[Tuple[int, int], Optional[Rot3]], unit_translations: Dict[Tuple[int, int], Optional[Unit3]],
) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
    """Process the graph of image indices with Rot3s/Unit3s defining edges, and select the largest connected component.

    Args:
        rotations: dictionary of relative rotations for pairs.
        unit_translations: dictionary of relative unit-translations for pairs.

    Returns:
        Subset of rotations which are in the largest connected components.
        Subset of unit_translations which are in the largest connected components.
    """
    print("Multiview input edges: ", rotations.keys())
    input_edges = [k for (k, v) in rotations.items() if v is not None]
    nodes_in_pruned_graph = graph_utils.get_nodes_in_largest_connected_component(input_edges)

    # select the edges with nodes in the pruned graph
    selected_edges = []
    for i1, i2 in rotations.keys():
        if i1 in nodes_in_pruned_graph and i2 in nodes_in_pruned_graph:
            selected_edges.append((i1, i2))

    print("Multiview pruned edges: ", selected_edges)
    # return the subset of original input
    return (
        {k: rotations[k] for k in selected_edges},
        {k: unit_translations[k] for k in selected_edges},
    )


def init_cameras(
    wRi_list: List[Optional[Rot3]], wti_list: List[Optional[Point3]], intrinsics_list: List[Cal3Bundler],
) -> Dict[int, PinholeCameraCal3Bundler]:
    """Generate camera from valid rotations and unit-translations.

    Args:
        wRi_list: rotations for cameras.
        wti_list: translations for cameras.
        intrinsics_list: intrinsics for cameras.
    Returns:
        Valid cameras.
    """
    cameras = {}

    for idx, (wRi, wti) in enumerate(zip(wRi_list, wti_list)):
        if wRi is not None and wti is not None:
            cameras[idx] = PinholeCameraCal3Bundler(Pose3(wRi, wti), intrinsics_list[idx])

    return cameras
