"""The main class which integrates all the modules.

Authors: Ayush Baid
"""
import os
from typing import Any, Dict, List, Optional, Tuple

import dask
import matplotlib.pyplot as plt
import networkx as nx
from dask.delayed import Delayed
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, Unit3

import utils.io as io_utils
import utils.viz as viz_utils
from averaging.rotation.rotation_averaging_base import RotationAveragingBase
from averaging.translation.translation_averaging_base import (
    TranslationAveragingBase,
)
from bundle.bundle_adjustment import BundleAdjustmentOptimizer
from data_association.data_assoc import DataAssociation
from frontend.detector_descriptor.detector_descriptor_base import (
    DetectorDescriptorBase,
)
from frontend.matcher.matcher_base import MatcherBase
from frontend.verifier.verifier_base import VerifierBase


class FeatureExtractor:
    """Wrapper for running detection and description on each image."""

    def __init__(self, detector_descriptor: DetectorDescriptorBase):
        self.detector_descriptor = detector_descriptor

    def create_computation_graph(
        self, image_graph: Delayed
    ) -> Tuple[Delayed, Delayed]:
        """ Given an image, create detection and descriptor generation tasks """
        return self.detector_descriptor.create_computation_graph(image_graph)


class TwoViewEstimator:
    """Wrapper for running two-view relative pose estimation on image pairs in
    the dataset."""

    def __init__(self, matcher: MatcherBase, verifier: VerifierBase):
        self.matcher = matcher
        self.verifier = verifier

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        descriptors_i1_graph: Delayed,
        descriptors_i2_graph: Delayed,
        camera_intrinsics_i1_graph: Delayed,
        camera_intrinsics_i2_graph: Delayed,
        exact_intrinsics: bool = True,
    ) -> Tuple[Delayed, Delayed, Delayed]:
        """Create delayed tasks for matching and verification."""

        # graph for matching to obtain putative correspondences
        corr_idxs_graph = self.matcher.create_computation_graph(
            descriptors_i1_graph, descriptors_i2_graph
        )

        # verification on putative correspondences to obtain relative pose
        # and verified correspondences
        (
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph,
        ) = self.verifier.create_computation_graph(
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
            camera_intrinsics_i1_graph,
            camera_intrinsics_i2_graph,
            exact_intrinsics,
        )

        return i2Ri1_graph, i2Ui1_graph, v_corr_idxs_graph


class MultiViewOptimizer:
    def __init__(
        self,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        config: Any,
    ):
        self.rot_avg_module = rot_avg_module
        self.trans_avg_module = trans_avg_module
        self.data_association_module = DataAssociation(
            config.reproj_error_thresh,
            config.min_track_len,
            config.triangulation_mode,
            config.num_ransac_hypotheses,
        )
        self.ba_optimizer = BundleAdjustmentOptimizer()

    def create_computation_graph(
        self,
        num_images: int,
        keypoints_graph: List[Delayed],
        i2Ri1_graph: Dict[Tuple[int, int], Delayed],
        i2Ui1_graph: Dict[Tuple[int, int], Delayed],
        v_corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        intrinsics_graph: List[Delayed],
    ) -> Delayed:
        # prune the graph to a single connected component.
        pruned_graph = dask.delayed(self.select_largest_connected_component)(
            i2Ri1_graph, i2Ui1_graph
        )

        pruned_i2Ri1_graph = pruned_graph[0]
        pruned_i2Ui1_graph = pruned_graph[1]

        wRi_graph = self.rot_avg_module.create_computation_graph(
            num_images, pruned_i2Ri1_graph
        )

        wti_graph = self.trans_avg_module.create_computation_graph(
            num_images, pruned_i2Ui1_graph, wRi_graph
        )

        init_cameras_graph = dask.delayed(self.init_cameras)(
            wRi_graph, wti_graph, intrinsics_graph
        )

        ba_input_graph = self.data_association_module.create_computation_graph(
            init_cameras_graph, v_corr_idxs_graph, keypoints_graph
        )

        ba_result_graph = self.ba_optimizer.create_computation_graph(
            ba_input_graph
        )

        return ba_result_graph

    @classmethod
    def select_largest_connected_component(
        cls,
        rotations: Dict[Tuple[int, int], Optional[Rot3]],
        unit_translations: Dict[Tuple[int, int], Optional[Unit3]],
    ) -> Tuple[Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3]]:
        """Process the graph of image indices with Rot3s/Unit3s defining edges,
        and select the largest connected component."""

        input_edges = [k for (k, v) in rotations.items() if v is not None]

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
        return (
            {k: rotations[k] for k in selected_edges},
            {k: unit_translations[k] for k in selected_edges},
        )

    @classmethod
    def init_cameras(
        cls,
        wRi_list: List[Optional[Rot3]],
        wti_list: List[Optional[Unit3]],
        intrinsics_list: List[Cal3Bundler],
    ) -> Dict[int, PinholeCameraCal3Bundler]:
        """Generate camera from valid rotations and unit-translations."""

        cameras = {}

        for idx, (wRi, wti) in enumerate(zip(wRi_list, wti_list)):
            if wRi is not None and wti is not None:
                cameras[idx] = PinholeCameraCal3Bundler(
                    Pose3(wRi, wti), intrinsics_list[idx]
                )

        return cameras


class SceneOptimizer:
    """Wrapper combining different modules to run the whole pipeline on a
    loader."""

    def __init__(
        self,
        detector_descriptor: DetectorDescriptorBase,
        matcher: MatcherBase,
        verifier: VerifierBase,
        rot_avg_module: RotationAveragingBase,
        trans_avg_module: TranslationAveragingBase,
        config: Any,
        debug_mode: bool = False,
    ) -> None:

        self.feature_extractor = FeatureExtractor(detector_descriptor)

        self.two_view_estimator = TwoViewEstimator(matcher, verifier)

        self.multiview_optimizer = MultiViewOptimizer(
            rot_avg_module, trans_avg_module, config
        )

        self._debug_mode = debug_mode

    def __visualize_twoview_correspondences(
        self,
        image_i1_graph: Delayed,
        image_i2_graph: Delayed,
        corr_idxs_graph: Delayed,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        file_name: str,
    ) -> None:
        plot = viz_utils.plot_twoview_correspondences(
            image_i1_graph,
            image_i2_graph,
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
        )

        io_utils.save_image(plot, file_name)

    def __visualize_sfm_result(
        self, sfm_data: Delayed, folder_name: str
    ) -> None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")

        viz_utils.plot_sfm_data_3d(sfm_data, ax)
        viz_utils.set_axes_equal(ax)

        # save the 3D plot in the original view
        plt.savefig(os.path.join(folder_name, "3d.png"))

        # save the BEV representation
        ax.view_init(azim=0, elev=100)
        plt.savefig(os.path.join(folder_name, "bev.png"))

    def create_computation_graph(
        self,
        num_images: int,
        image_pair_indices: List[Tuple[int, int]],
        image_graph: List[Delayed],
        camera_intrinsics_graph: List[Delayed],
        use_intrinsics_in_verification: bool = True,
    ) -> Tuple[Delayed, List[Delayed]]:
        """ The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times"""

        # optional graph elements for visualizations
        viz_graph_list = []

        # detection and description graph
        keypoints_graph_list = []
        descriptors_graph_list = []
        for delayed_image in image_graph:
            (
                delayed_dets,
                delayed_descs,
            ) = self.feature_extractor.create_computation_graph(delayed_image)
            keypoints_graph_list += [delayed_dets]
            descriptors_graph_list += [delayed_descs]

        # estimate two-view geometry and get indices of verified correspondences.
        i2Ri1_graph_dict = {}
        i2Ui1_graph_dict = {}
        v_corr_idxs_graph_dict = {}
        for (i1, i2) in image_pair_indices:
            (
                i2Ri1,
                i2Ui1,
                v_corr_idxs,
            ) = self.two_view_estimator.create_computation_graph(
                keypoints_graph_list[i1],
                keypoints_graph_list[i2],
                descriptors_graph_list[i1],
                descriptors_graph_list[i2],
                camera_intrinsics_graph[i1],
                camera_intrinsics_graph[i2],
                use_intrinsics_in_verification,
            )
            i2Ri1_graph_dict[(i1, i2)] = i2Ri1
            i2Ui1_graph_dict[(i1, i2)] = i2Ui1
            v_corr_idxs_graph_dict[(i1, i2)] = v_corr_idxs

            if self._debug_mode:
                viz_graph_list.append(
                    dask.delayed(self.__visualize_twoview_correspondences)(
                        image_graph[i1],
                        image_graph[i2],
                        v_corr_idxs,
                        keypoints_graph_list[i1],
                        keypoints_graph_list[i2],
                        "plots/correspondences/{}_{}.png".format(i1, i2),
                    )
                )

        sfmResult_graph = self.multiview_optimizer.create_computation_graph(
            num_images,
            keypoints_graph_list,
            i2Ri1_graph_dict,
            i2Ui1_graph_dict,
            v_corr_idxs_graph_dict,
            camera_intrinsics_graph,
        )

        if self._debug_mode:
            filtered_sfm_data = dask.delayed(sfmResult_graph.filter_landmarks)(
                2
            )

            viz_graph_list.append(
                dask.delayed(self.__visualize_sfm_result)(
                    filtered_sfm_data, "plots/results/"
                )
            )

        return sfmResult_graph, viz_graph_list
