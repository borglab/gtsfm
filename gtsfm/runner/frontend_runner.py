"""Runner for the front-end only (when back-end results are not desired or required).

Author: John Lambert
"""
from typing import Tuple

import dask
from dask.delayed import Delayed

from gtsfm.feature_extractor import FeatureExtractor
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.two_view_estimator import TwoViewEstimator


def run_frontend(
    loader: LoaderBase, feature_extractor: FeatureExtractor, two_view_estimator: TwoViewEstimator
) -> Tuple[Delayed, Delayed]:
    """Creates the front-end computation graph, and then runs it.

    Note: Copied from SceneOptimizer class, without back-end code.

    Args:
        loader: image loader.
        feature_extractor: feature extractor module to use.
        two_view_estimator: two-view estimator module to use.

    Returns:
        keypoints_list: detected keypoints for each image.
        i2Ri1_dict: dictionary of relative rotations for each image pair.
        i2Ui1_dict: dictionary of relative unit translation directions for each image pair.
        v_corr_idxs_dict: verified correspondence indices for each image pair.
    """
    image_pair_indices = loader.get_valid_pairs()
    image_graph = loader.create_computation_graph_for_images()
    camera_intrinsics_graph = loader.create_computation_graph_for_intrinsics()
    image_shape_graph = loader.create_computation_graph_for_image_shapes()

    # detection and description graph
    keypoints_graph_list = []
    descriptors_graph_list = []
    for delayed_image in image_graph:
        delayed_dets, delayed_descs = feature_extractor.create_computation_graph(delayed_image)
        keypoints_graph_list += [delayed_dets]
        descriptors_graph_list += [delayed_descs]

    # estimate two-view geometry and get indices of verified correspondences.
    i2Ri1_graph_dict = {}
    i2Ui1_graph_dict = {}
    v_corr_idxs_graph_dict: Dict[Tuple[int, int], Delayed] = {}
    for (i1, i2) in image_pair_indices:
        (i2Ri1, i2Ui1, v_corr_idxs, two_view_report) = two_view_estimator.create_computation_graph(
            keypoints_graph_list[i1],
            keypoints_graph_list[i2],
            descriptors_graph_list[i1],
            descriptors_graph_list[i2],
            camera_intrinsics_graph[i1],
            camera_intrinsics_graph[i2],
            image_shape_graph[i1],
            image_shape_graph[i2],
        )
        i2Ri1_graph_dict[(i1, i2)] = i2Ri1
        i2Ui1_graph_dict[(i1, i2)] = i2Ui1
        v_corr_idxs_graph_dict[(i1, i2)] = v_corr_idxs

    with dask.config.set(scheduler="single-threaded"):
        keypoints_list, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict = dask.compute(
            keypoints_graph_list, i2Ri1_graph_dict, i2Ui1_graph_dict, v_corr_idxs_graph_dict
        )

    return keypoints_list, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict