"""The main class which integrates all the modules.

Authors: Ayush Baid
"""
from typing import Dict, List, Optional, Tuple

import dask
import networkx as nx
from dask.delayed import Delayed
from gtsam import PinholeCameraCal3Bundler, Rot3, Unit3, Cal3Bundler, Pose3

from averaging.rotation.rotation_averaging_base import RotationAveragingBase
from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase
from bundle.bundle_adjustment import BundleAdjustmentOptimizer
from data_association.dummy_da import DummyDataAssociation
from frontend.detector_descriptor.detector_descriptor_base import \
    DetectorDescriptorBase
from frontend.matcher.matcher_base import MatcherBase
from frontend.verifier.verifier_base import VerifierBase


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
                                 image_pair_indices: Tuple[int, int],
                                 detection_graph: List[Delayed],
                                 descriptor_graph: List[Delayed],
                                 camera_intrinsics_graph: List[Delayed],
                                 exact_intrinsics: bool = True
                                 ) -> Tuple[Delayed, Delayed, Delayed]:
        """ Create delayed tasks for matching and verification. """

        # graph for matching to obtain putative correspondences
        matcher_graph = self.matcher.create_computation_graph(image_pair_indices, descriptor_graph)

        # verification on putative correspondences to obtain relative pose
        # and verified correspondences
        i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph = \
            self.verifier.create_computation_graph(image_pair_indices,
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
        self.data_association_module = DummyDataAssociation(0.5, 2)
        self.ba_optimizer = BundleAdjustmentOptimizer()

    def create_computation_graph(
            self,
            num_images: int,
            keypoints_graph: List[Delayed],
            i2Ri1_graph: Dict[Tuple[int, int], Delayed],
            i2Ui1_graph: Dict[Tuple[int, int], Delayed],
            v_corr_idxs_graph: Dict[Tuple[int, int], Delayed],
            intrinsics_graph: List[Delayed]) -> Delayed:
        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(self.select_largest_connected_component)(
            i2Ri1_graph, i2Ui1_graph)

        pruned_i2Ri1_graph = pruned_graph[0]
        pruned_i2Ui1_graph = pruned_graph[1]

        wRi_graph = self.rot_avg_module.create_computation_graph(
            num_images, pruned_i2Ri1_graph)

        wti_graph = self.trans_avg_module.create_computation_graph(
            num_images, pruned_i2Ui1_graph, wRi_graph)

        init_cameras_graph = dask.delayed(self.init_cameras)(
            wRi_graph, wti_graph, intrinsics_graph)

        ba_input_graph = self.data_association_module.create_computation_graph(
            init_cameras_graph, v_corr_idxs_graph, keypoints_graph)

        ba_result_graph = self.ba_optimizer.create_computation_graph(
            ba_input_graph)

        return ba_result_graph

    @classmethod
    def select_largest_connected_component(
            cls,
            rotations: Dict[Tuple[int, int], Optional[Rot3]],
            unit_translations: Dict[Tuple[int, int], Optional[Unit3]]
    ) -> Tuple[
            Dict[Tuple[int, int], Rot3],
            Dict[Tuple[int, int], Unit3]]:
        """Process the graph of image indices with Rot3s/Unit3s defining edges,
        and select the largest connected component."""

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

        # as the edges are non-directional, they might have flipped and should
        # be corrected
        selected_edges = []
        for i1, i2 in pruned_edges:
            if (i1, i2) in rotations:
                selected_edges.append((i1, i2))
            else:
                selected_edges.append((i2, i1))

        # return the subset of original input
        return {k: rotations[k] for k in selected_edges}, \
            {k: unit_translations[k] for k in selected_edges}

    @classmethod
    def init_cameras(
            cls,
            wRi_list: List[Optional[Rot3]],
            wti_list: List[Optional[Unit3]],
            intrinsics_list: List[Cal3Bundler]
    ) -> Dict[int, PinholeCameraCal3Bundler]:
        """Generate camera from valid rotations and unit-translations."""

        cameras = {}

        for idx, (wRi, wti) in enumerate(zip(wRi_list, wti_list)):
            if wRi is not None and wti is not None:
                cameras[idx] = PinholeCameraCal3Bundler(
                    Pose3(wRi, wti), intrinsics_list[idx])

        return cameras


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
            detector_descriptor)

        self.two_view_estimator = TwoViewEstimator(
            matcher, verifier)

        self.multiview_optimizer = MultiViewOptimizer(
            rot_avg_module, trans_avg_module)

    def create_computation_graph(self,
                                 num_images: int,
                                 image_pair_indices: List[Tuple[int, int]],
                                 image_graph: List[Delayed],
                                 camera_intrinsics_graph: List[Delayed],
                                 use_intrinsics_in_verification: bool = True
                                 ) -> Tuple[List[Delayed],
                                            Delayed,
                                            Delayed,
                                            Dict[Tuple[int, int], Delayed]]:
        """ The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times"""
        # detection and description graph
        detection_graph = []
        descriptor_graph = []
        for delayed_image in image_graph:
            delayed_dets, delayed_descs = self.feature_extractor.create_computation_graph(delayed_image)
            detection_graph += [delayed_dets]
            descriptor_graph += [delayed_descs]

        # estimate two-view geometry and get indices of verified correspondences.
        i2Ri1_graph = []
        i2Ui1_graph = []
        v_corr_idxs_graph = []
        for (i1, i2) in image_pair_indices:
            i2Ri1, i2Ui1, v_corr_idxs = self.two_view_estimator.create_computation_graph(
                (i1,i2),
                detection_graph,
                descriptor_graph,
                camera_intrinsics_graph,
                use_intrinsics_in_verification
            )
            i2Ri1_graph[(i1,i2)] = i2Ri1
            i2Ui1_graph[(i1,i2)] = i2Ui1
            v_corr_idxs_graph[(i1,i2)] = v_corr_idxs

        sfmResult_graph = self.multiview_optimizer.create_computation_graph(
            num_images,
            detection_graph,
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph,
            camera_intrinsics_graph
        )

        return sfmResult_graph
