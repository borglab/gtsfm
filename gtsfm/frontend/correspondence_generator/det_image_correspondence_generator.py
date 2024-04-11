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
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import (
    CorrespondenceGeneratorBase,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import (
    KeypointAggregatorBase,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)
from gtsfm.frontend.detector.detector_base import DetectorBase
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TwoViewEstimator


class DetImageCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Pair-wise direct matching of images (e.g. transformer-based)."""

    def __init__(
        self,
        detector: DetectorBase,
        matcher: ImageMatcherBase,
        aggregator: KeypointAggregatorBase = KeypointAggregatorUnique(),
    ) -> None:
        """
        Args:
            matcher: Matcher to use.
            deduplicate: Whether to de-duplicate with a single image the detections received from each image pair.
        """
        self._detector = detector
        self._matcher = matcher

    def __repr__(self) -> str:
        return f"""
        ImageCorrespondenceGenerator:
           {self._detector}
           {self._matcher}
        """

    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        image_pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            image_pairs: Indices of the pairs of images to estimate two-view pose and correspondences.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """

        def apply_detector(detector: DetectorBase, image: Image) -> Keypoints:
            return detector.detect(image)

        def apply_det_image_matcher(
            image_matcher: ImageMatcherBase,
            image_i1: Image,
            image_i2: Image,
            keypoints_i1: Keypoints,
            keypoints_i2: Keypoints,
        ) -> Tuple[Keypoints, Keypoints]:
            return image_matcher.match(image_i1, image_i2, keypoints_i1, keypoints_i2)

        # Apply detector.
        detector_future = client.scatter(self._detector, broadcast=False)
        keypoints_futures = [
            client.submit(apply_detector, detector_future, image) for image in images
        ]
        keypoints_list = client.gather(keypoints_futures)

        # Apply dense image matcher to detected keypoints.
        image_matcher_future = client.scatter(self._matcher, broadcast=False)
        putative_corr_idxs_futures = {
            (i1, i2): client.submit(
                apply_det_image_matcher,
                image_matcher_future,
                images[i1],
                images[i2],
                keypoints_list[i1],
                keypoints_list[i2],
            )
            for i1, i2 in image_pairs
        }

        putative_corr_idxs_dict = client.gather(putative_corr_idxs_futures)

        return keypoints_list, putative_corr_idxs_dict

    def generate_correspondences_and_estimate_two_view(
        self,
        client: Client,
        images: List[Image],
        image_pairs: List[Tuple[int, int]],
        camera_intrinsics: List[CALIBRATION_TYPE],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        gt_cameras: List[Optional[CAMERA_TYPE]],
        gt_scene_mesh: Optional[Any],
        two_view_estimator: TwoViewEstimator,
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], TWO_VIEW_OUTPUT]]:
        """Apply the correspondence generator to generate putative correspondences and subsequently process them with
        two view estimator to complete the front-end.

        Args:
            client: Dask client, used to execute the front-end as futures.
            images: List of all images.
            image_pairs: Indices of the pairs of images to estimate two-view pose and correspondences.
            camera_intrinsics: List of all camera intrinsics.
            relative_pose_priors: Priors on relative pose between two cameras.
            gt_cameras: GT cameras, used to evaluate metrics.
            gt_scene_mesh: GT mesh of the 3D scene, used to evaluate metrics.
            two_view_estimator: Two view estimator, which is used to verify correspondences and estimate pose.

        Returns:
            List of keypoints, one entry for each input images.
            Two view output for image_pairs.
        """

        def apply_image_matcher(
            image_matcher: ImageMatcherBase, image_i1: Image, image_i2: Image
        ) -> Tuple[Keypoints, Keypoints]:
            return image_matcher.match(image_i1=image_i1, image_i2=image_i2)

        def apply_two_view_estimator(
            two_view_estimator: TwoViewEstimator,
            keypoints_i1: Keypoints,
            keypoints_i2: Keypoints,
            putative_corr_idxs: np.ndarray,
            camera_intrinsics_i1: CALIBRATION_TYPE,
            camera_intrinsics_i2: CALIBRATION_TYPE,
            i2Ti1_prior: Optional[PosePrior],
            gt_camera_i1: Optional[CAMERA_TYPE],
            gt_camera_i2: Optional[CAMERA_TYPE],
            gt_scene_mesh: Optional[Any] = None,
        ) -> TWO_VIEW_OUTPUT:
            return two_view_estimator.run_2view(
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                putative_corr_idxs=putative_corr_idxs,
                camera_intrinsics_i1=camera_intrinsics_i1,
                camera_intrinsics_i2=camera_intrinsics_i2,
                i2Ti1_prior=i2Ti1_prior,
                gt_camera_i1=gt_camera_i1,
                gt_camera_i2=gt_camera_i2,
                gt_scene_mesh=gt_scene_mesh,
            )

        image_matcher_future = client.scatter(self._matcher, broadcast=False)
        two_view_estimator_future = client.scatter(two_view_estimator, broadcast=False)
        pairwise_correspondence_futures = {
            (i1, i2): client.submit(
                apply_image_matcher, image_matcher_future, images[i1], images[i2]
            )
            for i1, i2 in image_pairs
        }

        pairwise_correspondences: Dict[
            Tuple[int, int], Tuple[Keypoints, Keypoints]
        ] = client.gather(pairwise_correspondence_futures)

        keypoints_list, putative_corr_idxs_dict = self._aggregator.aggregate(
            keypoints_dict=pairwise_correspondences
        )

        two_view_output_futures = {
            (i1, i2): client.submit(
                apply_two_view_estimator,
                two_view_estimator_future,
                keypoints_list[i1],
                keypoints_list[i2],
                putative_corr_idxs_dict[(i1, i2)],
                camera_intrinsics[i1],
                camera_intrinsics[i2],
                relative_pose_priors.get((i1, i2)),
                gt_cameras[i1],
                gt_cameras[i2],
                gt_scene_mesh,
            )
            for (i1, i2) in image_pairs
        }

        two_view_output_dict = client.gather(two_view_output_futures)
        return keypoints_list, two_view_output_dict
