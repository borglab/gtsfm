"""Runner for the front-end only (when back-end results are not desired or required).

Author: John Lambert
"""
from typing import Dict, List, Tuple, Union

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Rot3, Unit3

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.correspondence_generator.det_desc_correspondence_generator import DetDescCorrespondenceGenerator
from gtsfm.frontend.correspondence_generator.image_correspondence_generator import ImageCorrespondenceGenerator
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.two_view_estimator import TwoViewEstimator


def run_frontend(
    loader: LoaderBase,
    correspondence_generator: Union[DetDescCorrespondenceGenerator, ImageCorrespondenceGenerator],
    two_view_estimator: TwoViewEstimator,
) -> Tuple[
    List[Keypoints], Dict[Tuple[int, int], Rot3], Dict[Tuple[int, int], Unit3], Dict[Tuple[int, int], np.ndarray]
]:
    """Creates the front-end computation graph, and then runs it.

    Note: Copied from SceneOptimizer class, without back-end code.

    Args:
        loader: image loader.
        correspondence_generator: correspondence generator module to use.
        two_view_estimator: two-view estimator module to use.

    Returns:
        keypoints_list: detected keypoints for each image.
        i2Ri1_dict: dictionary of relative rotations for each image pair.
        i2Ui1_dict: dictionary of relative unit translation directions for each image pair.
        v_corr_idxs_dict: verified correspondence indices for each image pair.
    """
    image_pair_indices = loader.get_valid_pairs()
    camera_intrinsics = loader.get_all_intrinsics()
    image_shapes = loader.get_image_shapes()

    (delayed_keypoints, delayed_putative_corr_idxs_dict,) = correspondence_generator.apply(
        delayed_images=loader.create_computation_graph_for_images(),
        image_shapes=loader.get_image_shapes(),
        image_pair_indices=image_pair_indices,
    )

    with dask.config.set(scheduler="single-threaded"):
        keypoints_list, putative_corr_idxs_dict = dask.compute(delayed_keypoints, delayed_putative_corr_idxs_dict)

    # estimate two-view geometry and get indices of verified correspondences.
    i2Ri1_graph_dict = {}
    i2Ui1_graph_dict = {}
    v_corr_idxs_graph_dict: Dict[Tuple[int, int], Delayed] = {}

    for (i1, i2) in image_pair_indices:
        i2Ri1, i2Ui1, v_corr_idxs, _ = two_view_estimator.create_computation_graph(
            keypoints_i1_graph=keypoints_list[i1],
            keypoints_i2_graph=keypoints_list[i2],
            putative_corr_idxs_graph=putative_corr_idxs_dict[i1, i2],
            camera_intrinsics_i1=camera_intrinsics[i1],
            camera_intrinsics_i2=camera_intrinsics[i2],
            im_shape_i1=image_shapes[i1],
            im_shape_i2=image_shapes[i2],
        )

        # Store results.
        i2Ri1_graph_dict[(i1, i2)] = i2Ri1
        i2Ui1_graph_dict[(i1, i2)] = i2Ui1
        v_corr_idxs_graph_dict[(i1, i2)] = v_corr_idxs

    with dask.config.set(scheduler="single-threaded"):
        i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict = dask.compute(
            i2Ri1_graph_dict, i2Ui1_graph_dict, v_corr_idxs_graph_dict
        )

    return keypoints_list, i2Ri1_dict, i2Ui1_dict, v_corr_idxs_dict
