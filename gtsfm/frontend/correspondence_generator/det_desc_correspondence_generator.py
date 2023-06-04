"""Correspondence generator that utilizes explicit keypoint detection, following by descriptor matching, per image.

Authors: John Lambert
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from dask.distributed import Client

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.types import CALIBRATION_TYPE, CAMERA_TYPE
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import DetectorDescriptorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TwoViewEstimator


class DetDescCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Traditional pair-wise matching of descriptors."""

    def __init__(self, matcher: MatcherBase, detector_descriptor: DetectorDescriptorBase):
        self._detector_descriptor = detector_descriptor
        self._matcher = matcher

    def apply(
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
            client: dask client, used to execute the front-end as futures.
            images: list of all images.
            image_pairs: indices of the pairs of images to estimate two-view pose and correspondences.
            camera_intrinsics: list of all camera intrinsics.
            relative_pose_priors: priors on relative pose between two cameras.
            gt_cameras: GT cameras, used to evaluate metrics.
            gt_scene_mesh: GT mesh of the 3D scene, used to evaluate metrics.
            two_view_estimator: two view estimator, which is used to verify correspondences and estimate pose.

        Returns:
            List of keypoints, one entry for each input images.
            Two view output for image_pairs.
        """

        def apply_det_desc(det_desc: DetectorDescriptorBase, image: Image) -> Tuple[Keypoints, np.ndarray]:
            return det_desc.detect_and_describe(image)

        def apply_matcher_and_two_view_estimator(
            feature_matcher: MatcherBase,
            two_view_estimator: TwoViewEstimator,
            features_i1: Tuple[Keypoints, np.ndarray],
            features_i2: Tuple[Keypoints, np.ndarray],
            im_shape_i1: Tuple[int, int, int],
            im_shape_i2: Tuple[int, int, int],
            camera_intrinsics_i1: CALIBRATION_TYPE,
            camera_intrinsics_i2: CALIBRATION_TYPE,
            i2Ti1_prior: Optional[PosePrior],
            gt_camera_i1: Optional[CAMERA_TYPE],
            gt_camera_i2: Optional[CAMERA_TYPE],
            gt_scene_mesh: Optional[Any] = None,
        ) -> TWO_VIEW_OUTPUT:
            putative_corr_idxs = feature_matcher.match(
                features_i1[0], features_i2[0], features_i1[1], features_i2[1], im_shape_i1, im_shape_i2
            )

            return two_view_estimator.run_2view(
                features_i1[0],
                features_i2[0],
                putative_corr_idxs,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
                i2Ti1_prior,
                gt_camera_i1,
                gt_camera_i2,
                gt_scene_mesh,
            )

        det_desc_future = client.scatter(self._detector_descriptor, broadcast=False)
        feature_matcher_future = client.scatter(self._matcher, broadcast=False)
        two_view_estimator_future = client.scatter(two_view_estimator, broadcast=False)
        features_futures = [client.submit(apply_det_desc, det_desc_future, image) for image in images]

        two_view_output_futures = {
            (i1, i2): client.submit(
                apply_matcher_and_two_view_estimator,
                feature_matcher_future,
                two_view_estimator_future,
                features_futures[i1],
                features_futures[i2],
                images[i1].shape,
                images[i2].shape,
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
        keypoints_futures = [client.submit(lambda f: f[0], f) for f in features_futures]
        keypoints_list = client.gather(keypoints_futures)

        return keypoints_list, two_view_output_dict
