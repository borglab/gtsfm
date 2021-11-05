"""Base class for the M (matcher) stage of the front end.

Authors: Ayush Baid
"""
import abc
from typing import NamedTuple, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import Cal3Bundler, Pose3
from trimesh import Trimesh

import gtsfm.utils.metrics as metric_utils
from gtsfm.common.keypoints import Keypoints


class MatcherReport(NamedTuple):
    """Report containing information about the matcher result.

    Args:
        num_matches: number of putative matches.
        inlier_ratio_wrt_gt_model: # correct matches / # putative matches
    """

    num_matches: int
    inlier_ratio_gt_model: Optional[float] = None


class MatcherBase(metaclass=abc.ABCMeta):
    """Base class for all matchers.

    Matchers work on a pair of descriptors and match them by their distance.
    """

    @abc.abstractmethod
    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> np.ndarray:
        """Match descriptor vectors.

        # Some matcher implementations (such as SuperGlue) utilize keypoint coordinates as
        # positional encoding, so our matcher API provides them for optional use.

        Output format:
        1. Each row represents a match.
        2. First column represents keypoint index from image #i1.
        3. Second column represents keypoint index from image #i2.
        4. Matches are sorted in descending order of the confidence (score), if possible.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            descriptors_i1: descriptors corr. to keypoints_i1.
            descriptors_i2: descriptors corr. to keypoints_i2.
            im_shape_i1: shape of image #i1, as (height,width).
            im_shape_i2: shape of image #i2, as (height,width).


        Returns:
            Match indices (sorted by confidence), as matrix of shape (N, 2), where N < min(N1, N2).
        """
        # TODO(ayush): should I define matcher on descriptors or the distance matrices.
        # TODO(ayush): how to handle deep-matchers which might require the full image as input

    def evaluate(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        corr_idxs: np.ndarray,
        camera_intrinsics_i1: Cal3Bundler,
        camera_intrinsics_i2: Cal3Bundler,
        dist_threshold: float,
        gt_wTi1: Optional[Pose3],
        gt_wTi2: Optional[Pose3],
        gt_scene_mesh: Optional[Trimesh],
    ) -> MatcherReport:
        """Analyze the matcher result.

        We valuate the quality of the matches against ground truth, if available.

        Args:
            keypoints_i1: keypoints for image #i1, of length N1.
            keypoints_i2: keypoints for image #i2, of length N2.
            corr_idxs: correspondence indices, as array of shape (K,).
            camera_intrinsics_i1: intrinsics for camera i1.
            camera_intrinsics_i2: intrinsics for camera i2.
            dist_threshold: inlier threshold, in pixels.
            gt_wTi1: ground truth global pose for camera i1.
            gt_wTi2: ground truth global pose for camera i2.
            gt_scene_mesh: ground truth scene mesh.
        """

        # Note: ignoring reprojection error return argument.
        corr_idxs_inlier_mask_gt, _ = metric_utils.compute_correspondence_metrics(
            keypoints_i1,
            keypoints_i2,
            corr_idxs,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
            dist_threshold,
            gt_wTi1,
            gt_wTi2,
            gt_scene_mesh,
        )
        if corr_idxs_inlier_mask_gt is None:
            inlier_ratio_gt_model = None
        else:
            inlier_ratio_gt_model = np.mean(corr_idxs_inlier_mask_gt)
        # count how many matches between each image on average.
        return MatcherReport(
            num_matches=corr_idxs.shape[0],
            inlier_ratio_gt_model=inlier_ratio_gt_model,
        )

    def create_computation_graph(
        self,
        keypoints_i1_graph: Delayed,
        keypoints_i2_graph: Delayed,
        descriptors_i1_graph: Delayed,
        descriptors_i2_graph: Delayed,
        im_shape_i1_graph: Delayed,
        im_shape_i2_graph: Delayed,
        camera_intrinsics_i1_graph: Delayed,
        camera_intrinsics_i2_graph: Delayed,
        dist_threshold: Delayed,
        gt_wTi1_graph: Delayed,
        gt_wTi2_graph: Delayed,
        gt_scene_mesh_graph: Delayed,
    ) -> Delayed:
        """
        Generates computation graph for matched features using description graphs.

        Args:
            keypoints_i1_graph: keypoints for image #i1, wrapped in Delayed.
            keypoints_i2_graph: keypoints for image #i2, wrapped in Delayed.
            descriptors_i1_graph: descriptors corr. to keypoints_i1.
            descriptors_i2_graph: descriptors corr. to keypoints_i2.
            im_shape_i1_graph: Delayed with the (H,W) shape of image #i1.
            im_shape_i2_graph: Delayed with the (H,W) shape of image #i2.
            camera_intrinsics_i1_graph: intrinsics for camera i1.
            camera_intrinsics_i2_graph: intrinsics for camera i2.
            dist_threshold: inlier threshold, in pixels.
            gt_wTi1_graph: ground truth global pose for camera i1.
            gt_wTi2_graph: ground truth global pose for camera i2.
            gt_scene_mesh_graph: ground truth scene mesh.

        Returns:
            Delayed dask tasks for matching for input image pair.
            Delayed dask containing a MatcherReport for the input image pair.
        """
        corr_idxs_graph = dask.delayed(self.match)(
            keypoints_i1_graph,
            keypoints_i2_graph,
            descriptors_i1_graph,
            descriptors_i2_graph,
            im_shape_i1_graph,
            im_shape_i2_graph,
        )
        matcher_report = dask.delayed(self.evaluate)(
            keypoints_i1=keypoints_i1_graph,
            keypoints_i2=keypoints_i2_graph,
            corr_idxs=corr_idxs_graph,
            camera_intrinsics_i1=camera_intrinsics_i1_graph,
            camera_intrinsics_i2=camera_intrinsics_i2_graph,
            dist_threshold=dist_threshold,
            gt_wTi1=gt_wTi1_graph,
            gt_wTi2=gt_wTi2_graph,
            gt_scene_mesh=gt_scene_mesh_graph,
        )

        return corr_idxs_graph, matcher_report
