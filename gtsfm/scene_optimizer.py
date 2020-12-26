"""The main class which integrates all the modules.

Authors: Ayush Baid, John Lambert
"""
import logging
import os
import sys
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import dask
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from dask.delayed import Delayed
from dask.distributed import Client, LocalCluster, performance_report
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, Unit3

import gtsfm.utils.io as io_utils
import gtsfm.utils.serialization  # import needed to register serialization fns
import gtsfm.utils.viz as viz_utils
from gtsfm.averaging.rotation.rotation_averaging_base import (
    RotationAveragingBase,
)
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.averaging.translation.averaging_1dsfm import (
    TranslationAveraging1DSFM,
)
from gtsfm.averaging.translation.translation_averaging_base import (
    TranslationAveragingBase,
)
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.data_association.data_assoc import (
    DataAssociation,
    TriangulationParam,
)
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import (
    DetectorDescriptorBase,
)
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
from gtsfm.frontend.verifier.degensac import Degensac
from gtsfm.frontend.verifier.verifier_base import VerifierBase
from gtsfm.loader.folder_loader import FolderLoader

# configure loggers to avoid DEBUG level stdout messages
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)


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
    ) -> Tuple[Delayed, Delayed]:
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

        return ba_input_graph, ba_result_graph

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
    ) -> None:

        self.feature_extractor = FeatureExtractor(detector_descriptor)

        self.two_view_estimator = TwoViewEstimator(matcher, verifier)

        self.multiview_optimizer = MultiViewOptimizer(
            rot_avg_module, trans_avg_module, config
        )

        self._save_viz = config.save_viz

    def __visualize_twoview_correspondences(
        self,
        image_i1_graph: Delayed,
        image_i2_graph: Delayed,
        corr_idxs_graph: Delayed,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        file_name: str,
    ) -> None:
        plot_img = viz_utils.plot_twoview_correspondences(
            image_i1_graph,
            image_i2_graph,
            keypoints_i1_graph,
            keypoints_i2_graph,
            corr_idxs_graph,
        )

        io_utils.save_image(plot_img, file_name)

    def __visualize_sfm_data(self, sfm_data: Delayed, folder_name: str) -> None:
        fig = plt.figure()
        ax = fig.gca(projection="3d")

        viz_utils.plot_sfm_data_3d(sfm_data, ax)
        viz_utils.set_axes_equal(ax)

        # save the 3D plot in the original view
        fig.savefig(os.path.join(folder_name, "3d.png"))

        # save the BEV representation
        default_camera_elevation = 100  # in metres above ground
        ax.view_init(azim=0, elev=default_camera_elevation)
        fig.savefig(os.path.join(folder_name, "bev.png"))

        plt.close(fig)

    def __visualize_camera_poses(
        self,
        pre_ba_sfm_data: Delayed,
        post_ba_sfm_data: Delayed,
        folder_name: str,
    ) -> None:
        # extract camera poses
        pre_ba_poses = []
        for i in range(pre_ba_sfm_data.number_cameras()):
            pre_ba_poses.append(pre_ba_sfm_data.camera(i).pose())

        post_ba_poses = []
        for i in range(post_ba_sfm_data.number_cameras()):
            post_ba_poses.append(post_ba_sfm_data.camera(i).pose())

        fig = plt.figure()
        ax = fig.gca(projection="3d")

        viz_utils.plot_poses_3d(pre_ba_poses, ax, center_marker_color="c")
        viz_utils.plot_poses_3d(post_ba_poses, ax, center_marker_color="k")

        # save the 3D plot in the original view
        fig.savefig(os.path.join(folder_name, "poses_3d.png"))

        # save the BEV representation
        default_camera_elevation = 100  # in metres above ground
        ax.view_init(azim=0, elev=default_camera_elevation)
        fig.savefig(os.path.join(folder_name, "poses_bev.png"))

        plt.close(fig)

    def create_computation_graph(
        self,
        num_images: int,
        image_pair_indices: List[Tuple[int, int]],
        image_graph: List[Delayed],
        camera_intrinsics_graph: List[Delayed],
        use_intrinsics_in_verification: bool = True,
    ) -> Delayed:
        """ The SceneOptimizer plate calls the FeatureExtractor and TwoViewEstimator plates several times"""

        # optional graph elements for visualizations, not returned to the user.
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

            if self._save_viz:
                os.makedirs("plots/correspondences", exist_ok=True)
                viz_graph_list.append(
                    dask.delayed(self.__visualize_twoview_correspondences)(
                        image_graph[i1],
                        image_graph[i2],
                        v_corr_idxs,
                        keypoints_graph_list[i1],
                        keypoints_graph_list[i2],
                        "plots/correspondences/{}_{}.jpg".format(i1, i2),
                    )
                )

        (
            ba_input_graph,
            ba_output_graph,
        ) = self.multiview_optimizer.create_computation_graph(
            num_images,
            keypoints_graph_list,
            i2Ri1_graph_dict,
            i2Ui1_graph_dict,
            v_corr_idxs_graph_dict,
            camera_intrinsics_graph,
        )

        if self._save_viz:
            filtered_sfm_data_graph = dask.delayed(
                ba_output_graph.filter_landmarks
            )(config.reproj_error_thresh)

            os.makedirs("plots/ba_input", exist_ok=True)
            os.makedirs("plots/results", exist_ok=True)

            viz_graph_list.append(
                dask.delayed(self.__visualize_sfm_data)(
                    ba_input_graph, "plots/ba_input/"
                )
            )

            viz_graph_list.append(
                dask.delayed(self.__visualize_sfm_data)(
                    filtered_sfm_data_graph, "plots/results/"
                )
            )

            viz_graph_list.append(
                dask.delayed(self.__visualize_camera_poses)(
                    ba_input_graph, filtered_sfm_data_graph, "plots/results"
                )
            )

        # as visualization tasks are not to be provided to the user, we create a
        # dummy computation of concatenating viz tasks with the output graph,
        # forcing computation of viz tasks
        output_graph = dask.delayed(lambda x, y: [x] + y)(
            ba_output_graph, viz_graph_list
        )

        # return the entry with just the sfm result
        return output_graph[0]


if __name__ == "__main__":
    loader = FolderLoader(
        os.path.join("tests", "data", "set1_lund_door"), image_extension="JPG"
    )

    config = SimpleNamespace(
        **{
            "reproj_error_thresh": 5,
            "min_track_len": 3,
            "triangulation_mode": TriangulationParam.NO_RANSAC,
            "num_ransac_hypotheses": 20,
            "save_viz": True,
        }
    )
    obj = SceneOptimizer(
        detector_descriptor=SIFTDetectorDescriptor(),
        matcher=TwoWayMatcher(),
        verifier=Degensac(),
        rot_avg_module=ShonanRotationAveraging(),
        trans_avg_module=TranslationAveraging1DSFM(),
        config=config,
    )

    sfm_result_graph = obj.create_computation_graph(
        len(loader),
        loader.get_valid_pairs(),
        loader.create_computation_graph_for_images(),
        loader.create_computation_graph_for_intrinsics(),
        use_intrinsics_in_verification=True,
    )

    # create dask client
    cluster = LocalCluster(n_workers=2, threads_per_worker=4)
    client = Client(cluster)

    with performance_report(filename="dask-report.html"):
        sfm_result = sfm_result_graph.compute()
