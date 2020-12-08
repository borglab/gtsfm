"""The main class which integrates all the modules.

Authors: Ayush Baid
"""
from typing import Dict, List, Optional, Tuple

import dask
import networkx as nx
import numpy as np
from dask.delayed import Delayed
from gtsam import Rot3, Unit3

from common.keypoints import Keypoints
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase
from frontend.matcher.matcher_base import MatcherBase
from frontend.verifier.verifier_base import VerifierBase
from averaging.rotation.rotation_averaging_base import RotationAveragingBase
from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase
from loader.loader_base import LoaderBase


class FeatureExtractor:
    """Wrapper for running detection and description on each image."""

    def __init__(self, detector_descriptor: DetectorDescriptorBase):
        self.detector_descriptor = detector_descriptor

    def create_computation_graph(self,
                                 image_graph: List[Delayed]) -> List[Delayed]:
        return self.detector_descriptor.create_computation_graph(image_graph)


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in
    the dataset."""

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase):
        self.matcher = matcher
        self.verifier = verifier

    def create_computation_graph(self,
                                 image_pair_indices: List[Tuple[int, int]],
                                 detection_graph: List[Delayed],
                                 description_graph: List[Delayed],
                                 camera_intrinsics_graph: List[Delayed],
                                 exact_intrinsics: bool = True
                                 ) -> Tuple[Dict[Tuple[int, int], Delayed],
                                            Dict[Tuple[int, int], Delayed],
                                            Dict[Tuple[int, int], Delayed]]:

        # graph for matching to obtain putative correspondences
        matcher_graph = self.matcher.create_computation_graph(
            image_pair_indices, description_graph)

        # verification on putative correspondences to obtain relative pose
        # and verified correspondences
        i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph = \
            self.verifier.create_computation_graph(
                detection_graph, matcher_graph,
                camera_intrinsics_graph, exact_intrinsics
            )

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph


class MultiViewOptimizer:
    def __init__(self,
                 rot_avg_module: RotationAveragingBase,
                 trans_avg_module: TranslationAveragingBase):
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module

    def create_computation_graph(
            self,
            num_images: int,
            keypoints_graph: List[Delayed],
            i2Ri1_graph: Dict[Tuple[int, int], Delayed],
            i2Ui1_graph: Dict[Tuple[int, int], Delayed],
            v_corr_idxs_graph: Dict[Tuple[int, int], Delayed]) -> Delayed:
        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(self.select_largest_connected_component)(
            i2Ri1_graph, i2Ui1_graph)

        pruned_i2Ri1_graph = pruned_graph[0]
        pruned_i2Ui1_graph = pruned_graph[1]

        wRi_graph = self.rot_avg_module.create_computation_graph(
            num_images, pruned_i2Ri1_graph)

        wTi_graph = self.trans_avg_module.create_computation_graph(
            num_images, pruned_i2Ui1_graph, wRi_graph
        )

        sfmresult = dask.delayed(Dict())

        return sfmresult

    @classmethod
    def select_largest_connected_component(
            cls,
            rotations: Dict[Tuple[int, int], Optional[Rot3]],
            unit_translations: Dict[Tuple[int, int], Optional[Unit3]]
    ) -> Tuple[
            Dict[Tuple[int, int], Rot3],
            Dict[Tuple[int, int], Unit3]]:
        """Process the graph of image indices with Rot3s/Unit3s defining edges, and select the largest connected component."""

        input_edges = [
            k for (k, v) in rotations.items() if v is not None]

        # create a graph from all edges which have an essential matrix
        result_graph = nx.Graph()
        result_graph.add_edges_from(input_edges)

        # get the largest connected components
        largest_cc = max(nx.connected_components(result_graph), key=len)
        result_subgraph = result_graph.subgraph(largest_cc).copy()

        # get the remaining edges and construct the dictionary back
        pruned_edges = list(result_subgraph.edges())

        # return the subset of original input
        return {k: rotations[k] for k in pruned_edges}, \
            {k: unit_translations[k] for k in pruned_edges}


class SceneOptimizer:
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(self,
                 detector_descriptor: DetectorDescriptorBase,
                 matcher: MatcherBase,
                 verifier: VerifierBase,
                 rot_avg_module: RotationAveragingBase,
                 trans_avg_module: TranslationAveragingBase
                 ) -> None:

        self.feature_extractor = FeatureExtractor(
            detector_descriptor
        )

        self.two_view_estimater = TwoViewEstimator(
            matcher, verifier
        )

        self.multiview_optimizer = MultiViewOptimizer(
            rot_avg_module, trans_avg_module
        )

    def create_computation_graph(self,
                                 num_images: int,
                                 image_pair_indices: List[Tuple[int, int]],
                                 image_graph: List[Delayed],
                                 camera_intrinsics_graph: List[Delayed],
                                 exact_intrinsics: bool = True
                                 ) -> Tuple[List[Delayed],
                                            Delayed,
                                            Delayed,
                                            Dict[Tuple[int, int], Delayed]]:
        # detection and description graph
        detection_graph, description_graph = \
            self.feature_extractor.create_computation_graph(image_graph)

        # estimate two-view geometry and get indices of verified correspondences.
        i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph = \
            self.two_view_estimater.create_computation_graph(
                image_pair_indices,
                detection_graph,
                description_graph,
                camera_intrinsics_graph,
                exact_intrinsics
            )

        sfmresult_graph = self.multiview_optimizer.create_computation_graph(
            num_images,
            detection_graph,
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph
        )

        return sfmresult_graph
