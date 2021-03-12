"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid, John Lambert
"""
from typing import Any, Dict, List, Optional, Tuple

import dask
import networkx as nx
import os
from dask.delayed import Delayed
from gtsam import (
    Cal3Bundler,
    PinholeCameraCal3Bundler,
    Point3,
    Pose3,
    Rot3,
    Unit3,
)

import gtsfm.utils.io as io
import gtsfm.utils.metrics as metrics
from gtsfm.averaging.rotation.rotation_averaging_base import RotationAveragingBase
from gtsfm.averaging.translation.translation_averaging_base import TranslationAveragingBase
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.data_association.data_assoc import DataAssociation


class MultiViewOptimizer:
    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        data_association_module: DataAssociation,
    ) -> None:
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = data_association_module
        self.ba_optimizer = BundleAdjustmentOptimizer()

    def create_computation_graph(
        self,
        num_images: int,
        keypoints_graph: List[Delayed],
        i2Ri1_graph: Dict[Tuple[int, int], Delayed],
        i2Ui1_graph: Dict[Tuple[int, int], Delayed],
        v_corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        intrinsics_graph: List[Delayed],
        gt_poses_graph: List[Delayed] = None,
    ) -> Tuple[Delayed, Delayed]:
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
        """
        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(select_largest_connected_component)(i2Ri1_graph, i2Ui1_graph)

        pruned_i2Ri1_graph = pruned_graph[0]
        pruned_i2Ui1_graph = pruned_graph[1]

        wRi_graph = self.rot_avg_module.create_computation_graph(num_images, pruned_i2Ri1_graph)

        wti_graph = self.trans_avg_module.create_computation_graph(num_images, pruned_i2Ui1_graph, wRi_graph)

        init_cameras_graph = dask.delayed(init_cameras)(wRi_graph, wti_graph, intrinsics_graph)

        ba_input_graph, data_assoc_metrics_graph = self.data_association_module.create_computation_graph(
            num_images, init_cameras_graph, v_corr_idxs_graph, keypoints_graph
        )

        auxiliary_graph_list = [
            dask.delayed(io.save_json_file)(
                os.path.join("result_metrics", "data_association_metrics.json"), data_assoc_metrics_graph
            )
        ]

        # dummy graph to force an immediate dump of data association metrics
        ba_input_graph = dask.delayed(lambda x, y: (x, y))(ba_input_graph, auxiliary_graph_list)[0]

        ba_result_graph = self.ba_optimizer.create_computation_graph(ba_input_graph)

        if gt_poses_graph is None:
            return ba_input_graph, ba_result_graph, None

        metrics_graph = dask.delayed(metrics.compute_averaging_metrics)(
            i2Ui1_graph, wRi_graph, wti_graph, gt_poses_graph
        )
        saved_metrics_graph = dask.delayed(io.save_json_file)(
            "result_metrics/multiview_optimizer_metrics.json", metrics_graph
        )
        return ba_input_graph, ba_result_graph, saved_metrics_graph


def select_largest_connected_component(
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
    input_edges = [k for (k, v) in rotations.items() if v is not None]

    # create a graph from all edges which have an essential matrix
    result_graph = nx.Graph()
    result_graph.add_edges_from(input_edges)

    # get the largest connected components
    largest_cc = max(nx.connected_components(result_graph), key=len)
    result_subgraph = result_graph.subgraph(largest_cc).copy()

    # get the remaining edges and construct the dictionary back
    pruned_edges = list(result_subgraph.edges())

    # as the edges are non-directional, they might have flipped and should be corrected
    selected_edges = []
    for i1, i2 in pruned_edges:
        if (i1, i2) in rotations:
            selected_edges.append((i1, i2))
        else:
            selected_edges.append((i2, i1))

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
