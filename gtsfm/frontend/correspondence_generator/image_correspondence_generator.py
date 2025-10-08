"""Correspondence generator that utilizes direct matching of keypoints across an image pair, without descriptors.

Authors: John Lambert
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dask.distributed import Client, Future

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.types import CALIBRATION_TYPE, CAMERA_TYPE
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.two_view_estimator import TwoViewEstimator, TwoViewOutput


class ImageCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Pair-wise direct matching of images (e.g. transformer-based)."""

    def __init__(self, matcher: ImageMatcherBase, deduplicate: bool = True) -> None:
        """
        Args:
            matcher: Matcher to use.
            deduplicate: Whether to de-duplicate with a single image the detections received from each image pair.
        """
        self._matcher = matcher

        self._aggregator: KeypointAggregatorBase = (
            KeypointAggregatorDedup() if deduplicate else KeypointAggregatorUnique()
        )

    def __repr__(self) -> str:
        return f"""
        ImageCorrespondenceGenerator:
           {self._matcher}
           {self._aggregator}
        """

    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        visibility_graph: VisibilityGraph,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            visibility_graph: The visibility graph defining which image pairs to process.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """

        def apply_image_matcher(image_matcher: ImageMatcherBase, **kwargs) -> Tuple[Keypoints, Keypoints]:
            return image_matcher.match(**kwargs)

        image_matcher_future = client.scatter(self._matcher, broadcast=False)
        pairwise_correspondence_futures = {
            (i1, i2): client.submit(
                apply_image_matcher,
                image_matcher_future,
                image_i1=images[i1],
                image_i2=images[i2],
            )
            for i1, i2 in visibility_graph
        }

        pairwise_correspondences: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]] = client.gather(
            pairwise_correspondence_futures
        )

        keypoints_list, putative_corr_idxs_dict = self._aggregator.aggregate(keypoints_dict=pairwise_correspondences)
        return keypoints_list, putative_corr_idxs_dict

    def generate_correspondences_and_estimate_two_view(
        self,
        client: Client,
        images: List[Image],
        visibility_graph: VisibilityGraph,
        camera_intrinsics: List[CALIBRATION_TYPE],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        gt_cameras: List[Optional[CAMERA_TYPE]],
        gt_scene_mesh: Optional[Any],
        two_view_estimator: TwoViewEstimator,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], TwoViewOutput]]:
        """Apply the correspondence generator to generate putative correspondences and subsequently process them with
        two view estimator to complete the front-end.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images.
            visibility_graph: The visibility graph defining which image pairs to process.
            camera_intrinsics: List of all camera intrinsics.
            relative_pose_priors: Priors on relative pose between two cameras.
            gt_cameras: GT cameras, used to evaluate metrics.
            gt_scene_mesh: GT mesh of the 3D scene, used to evaluate metrics.
            two_view_estimator: Two view estimator, which is used to verify correspondences and estimate pose.

        Returns:
            List of keypoints, one entry for each input images.
            Two view output for visibility graph pairs.
        """

        def apply_image_matcher(image_matcher: ImageMatcherBase, **kwargs) -> Tuple[Keypoints, Keypoints]:
            return image_matcher.match(**kwargs)

        def apply_two_view_estimator(two_view_estimator: TwoViewEstimator, **kwargs) -> TwoViewOutput:
            return two_view_estimator.run_2view(**kwargs)

        image_matcher_future = client.scatter(self._matcher, broadcast=False)
        two_view_estimator_future = client.scatter(two_view_estimator, broadcast=False)
        pairwise_correspondence_futures = {
            (i1, i2): client.submit(
                apply_image_matcher,
                image_matcher_future,
                image_i1=images[i1],
                image_i2=images[i2],
            )
            for i1, i2 in visibility_graph
        }

        pairwise_correspondences: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]] = client.gather(
            pairwise_correspondence_futures
        )

        keypoints_list, putative_corr_idxs_dict = self._aggregator.aggregate(keypoints_dict=pairwise_correspondences)

        two_view_output_futures = {
            (i1, i2): client.submit(
                apply_two_view_estimator,
                two_view_estimator_future,
                keypoints1=keypoints_list[i1],
                keypoints2=keypoints_list[i2],
                putative_corr_idxs=putative_corr_idxs_dict[(i1, i2)],
                camera_intrinsics_i1=camera_intrinsics[i1],
                camera_intrinsics_i2=camera_intrinsics[i2],
                i2Ti1_prior=relative_pose_priors.get((i1, i2)),
                gt_camera_i1=gt_cameras[i1],
                gt_camera_i2=gt_cameras[i2],
                gt_scene_mesh=gt_scene_mesh,
                i1=i1,
                i2=i2,
            )
            for (i1, i2) in visibility_graph
        }

        two_view_output_dict = client.gather(two_view_output_futures)
        return keypoints_list, two_view_output_dict
