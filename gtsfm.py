"""The main class which integrates all the modules.

Authors: Ayush Baid
"""
import abc
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


class GTSFM(metaclass=abc.ABCMeta):
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(self,
                 detector_descriptor: DetectorDescriptorBase,
                 matcher: MatcherBase,
                 verifier: VerifierBase,
                 rotation_averaging_module: RotationAveragingBase,
                 translation_averaging_module: TranslationAveragingBase
                 ) -> None:
        self.detector_descriptor = detector_descriptor
        self.matcher = matcher
        self.verifier = verifier
        self.rotation_averaging_module = rotation_averaging_module
        self.translation_averaging_module = translation_averaging_module

    def run(self,
            loader: LoaderBase,
            exact_intrinsics: bool = True) -> Tuple[
            List[Keypoints],
            List[Optional[Rot3]],
            List[Optional[Unit3]],
            Dict[Tuple[int, int], np.ndarray]]:
        """Run structure-from-motion on supplied loader.

        Args:
            loader: loader with input dataset.
            exact_intrinsics (optional): Flag denoting intrinsics can be
                                         treated as exact in the verification
                                         stage. Defaults to True.

        Returns:
            List of keypoints detected in each image.
            List of global rotations (after rotation averaging).
            List of global translations (after translation averaging).
            Indices of verified correspondences between pairs of images.
        """
        # run detection and description for all images in the loader
        keypoints_list = []
        descriptors_list = []

        for i in range(len(loader)):
            keypoints, descriptors = \
                self.detector_descriptor.detect_and_describe(
                    loader.get_image(i))

            keypoints_list.append(keypoints)
            descriptors_list.append(descriptors)

        # perform matching and verification on valid image pairs in the loader
        if exact_intrinsics:
            verification_fn = self.verifier.verify_with_exact_intrinsics
        else:
            verification_fn = self.verifier.verify_with_approximate_intrinsics

        i2Ri1s = dict()
        i2Ui1s = dict()
        v_corr_idxs_dict = dict()
        for (i1, i2) in loader.get_valid_pairs():
            corr_idxs = self.matcher.match(
                descriptors_list[i1], descriptors_list[i2])

            i2Ri1, i2Ui1, v_corr_idxs = verification_fn(
                keypoints_list[i1],
                keypoints_list[i2],
                corr_idxs,
                loader.get_camera_intrinsics(i1),
                loader.get_camera_intrinsics(i2),
            )

            v_corr_idxs_dict[(i1, i2)] = v_corr_idxs
            if i2Ri1 is not None:
                # TODO: confirm with john if this is the right way to representation rotations and unit translations (i.e. (i1, i2) or (i2, i1))
                i2Ri1s[(i1, i2)] = i2Ri1
                i2Ui1s[(i1, i2)] = i2Ui1

        i2Ri1s, i2Ui1s = self.select_largest_connected_component(i2Ri1s, i2Ui1s)

        wRi_list = self.rotation_averaging_module.run(len(loader), i2Ri1s)

        wti_list = self.translation_averaging_module.run(
            len(loader), i2Ui1s, wRi_list)

        return keypoints_list, wRi_list, wti_list, v_corr_idxs_dict

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
            self.detector_descriptor.create_computation_graph(image_graph)

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

        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(self.select_largest_connected_component)(
            i2Ri1_graph, i2Ui1_graph)

        pruned_i2Ri1_graph = pruned_graph[0]
        pruned_i2Ui1_graph = pruned_graph[1]

        wRi_graph = self.rotation_averaging_module.create_computation_graph(
            num_images, pruned_i2Ri1_graph)

        wTi_graph = self.translation_averaging_module.create_computation_graph(
            num_images, pruned_i2Ui1_graph, wRi_graph
        )

        return detection_graph, wRi_graph, wTi_graph, v_corr_idxs_graph
