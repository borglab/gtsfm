"""Base class for correspondence generators.

Authors: John Lambert
"""
from abc import abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from dask.distributed import Client

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.types import CALIBRATION_TYPE, CAMERA_TYPE
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TwoViewEstimator


class CorrespondenceGeneratorBase:
    """Base class for correspondence generators."""

    @abstractmethod
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
