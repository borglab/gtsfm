"""Image retriever for front-end.

Authors: Travis Driver
"""

# import abc
from typing import List, Optional, Tuple, Dict
import itertools

import numpy as np
import dask
from dask.delayed import Delayed
from gtsam import Pose3, Cal3Bundler, Rot3, Unit3

import gtsfm.utils.logger as logger_utils
from gtsfm.two_view_estimator import TwoViewEstimator, TwoViewEstimationReport
from gtsfm.common.keypoints import Keypoints


logger = logger_utils.get_logger()


class RetirieverBase:
    """Base class for image retrieval.

    The Retriever proposes image pairs to conduct local feature matching.
    Defaults to exhaustive.
    """

    def __init__(
        self,
        two_view_estimator: TwoViewEstimator,
        intrinsics_list: List[Optional[Cal3Bundler]],
        image_shape_list: List[Tuple[int, int]],
        gt_pose_list: Optional[List[Pose3]] = None,
    ):
        """Initialize the Retriever.

        Keypoints and descriptors are empty until computed later by FeatureExtractor

        Args:
            two_view_estimator: performs local matching and computs relative pose.
        """
        # Store information for local feature matching by TwoViewEstimator.
        self._two_view_estimator = two_view_estimator
        self._intrinsics_list = intrinsics_list
        self._image_shape_list = image_shape_list
        self._gt_pose_list = gt_pose_list

        # Initialize keypoints and descriptors as None until computed by FeatureExtractor.
        self._keypoints_list: Optional[List[Keypoints]] = None
        self._descriptors_list: Optional[List[np.ndarray]] = None

        # Compile putative image pair combinations.
        # TODO (travisdriver): have the retriever compute the putative image pairs instead of considering every single
        #   combination. This was done primarily to preserve the current structure of the main program.
        self.putative_image_pair_ind = list(itertools.combinations(range(self.__len__()), 2))

    def add_features(self, keypoints_list: List[Delayed], descriptors_list: List[Delayed]) -> None:
        """Add keypoints and descriptors from the FeatureExtractor.

        Args:
            keypoints_list: list of Delayed keypoints (Keypoints) from the FeatureExtractor.
            descriptors_list: list of Delayed descriptors (np.ndarray) from the FeatureExtractor.
        """
        self._keypoints_list = keypoints_list
        self._descriptors_list = descriptors_list

    # ignored-abstractmethod
    # @abc.abstractmethod
    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """
        return len(self._image_shape_list)

    # ignored-abstractmethod
    # @abc.abstractmethod
    def _is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Default is exhaustive, i.e., all pairs are valid.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        return idx1 < idx2

    # ignored-abstractmethod
    # @abc.abstractmethod
    def _propose_pairs(self, idx1: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Default is exhaustive, i.e., all pairs are valid.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            validation result.
        """
        pass

    def _compute_local_matches(self, i1: int, i2: int) -> Tuple[Delayed, Delayed, Delayed, Optional[Delayed]]:
        # ) -> Tuple[Optional[Rot3], Optional[Unit3], Optional[np.ndarray], Optional[TwoViewEstimationReport]]:
        """Performs local matching and computes relativecrotations and relative (unit) translations for valid image
        pairs.
        """
        # Verify that keypoints and descriptors have been computed.
        if self._keypoints_list is None or self._descriptors_list is None:
            raise ValueError("Keypoints and/or descriptors never set.")

        # Check whether valid image pair and return if not.
        if not self._is_valid_pair(i1, i2):
            logger.debug(f"Invalid pair {i1} {i2}.")
            return None, None, None, None

        # Compute ground truth relative pose if available.
        gt_i2Ti1 = (
            dask.delayed(lambda x, y: x.between(y))(self._gt_pose_list[i2], self._gt_pose_list[i1])
            if self._gt_pose_list is not None
            else None
        )

        # Perform local feature matching.
        (i2Ri1, i2Ui1, v_corr_idxs, two_view_report) = self._two_view_estimator.create_computation_graph(
            self._keypoints_list[i1],
            self._keypoints_list[i2],
            self._descriptors_list[i1],
            self._descriptors_list[i2],
            self._intrinsics_list[i1],
            self._intrinsics_list[i2],
            self._image_shape_list[i1],
            self._image_shape_list[i2],
            gt_i2Ti1,
        )

        return (i2Ri1, i2Ui1, v_corr_idxs, two_view_report)

    # def create_computation_graph(
    #     self,
    #     i1: int,
    #     i2: int,
    #     camera_intrinsics_i1_graph: Delayed,
    #     camera_intrinsics_i2_graph: Delayed,
    #     im_shape_i1_graph: Delayed,
    #     im_shape_i2_graph: Delayed,
    #     i2Ti1_expected_graph: Optional[Delayed] = None,
    # ) -> Tuple[Delayed, Delayed, Delayed, Optional[Delayed]]:
    #     """Create delayed tasks for matching and verification.

    #     Args:
    #         i1: first image index
    #         i2: second image index
    #         camera_intrinsics_i1_graph: intrinsics for camera i1.
    #         camera_intrinsics_i2_graph: intrinsics for camera i2.
    #         im_shape_i1_graph: image shape for image i1.
    #         im_shape_i2_graph: image shape for image i2.
    #         i2Ti1_expected_graph (optional): ground truth relative pose, used for evaluation if available. Defaults to
    #             None.

    #     Returns:
    #         Computed relative rotation wrapped as Delayed.
    #         Computed relative translation direction wrapped as Delayed.
    #         Indices of verified correspondences wrapped as Delayed.
    #         TwoViewEstimatorReport wrapped as Delayed
    #     """
    #     # Get delayed object; cannot separate two arguments immediately.
    #     result_graph = dask.delayed(self._compute_local_matches)(
    #         self._keypoints_list[i1],
    #         self._keypoints_list[i2],
    #         self._descriptors_list[i1],
    #         self._descriptors_list[i2],
    #         camera_intrinsics_i1_graph,
    #         camera_intrinsics_i2_graph,
    #         im_shape_i1_graph,
    #         im_shape_i2_graph,
    #         i2Ti1_expected_graph,
    #     )
    #     i2Ri1, i2Ui1, v_corr_idxs, two_view_report = (
    #         result_graph[0],
    #         result_graph[1],
    #         result_graph[2],
    #         result_graph[3],
    #     )  # pylint: disable=W0631
    #     return i2Ri1, i2Ui1, v_corr_idxs, two_view_report

    def create_computation_graph(
        self, keypoints_graph_list: List[Delayed], descriptors_graph_list: List[Delayed]
    ) -> Dict[Tuple[int, int], Delayed]:
        """TODO(travisdriver)"""
        # Store the delayed keypoints and descriptors.
        self._keypoints_list = keypoints_graph_list
        self._descriptors_list = descriptors_graph_list
        return {(i1, i2): self._compute_local_matches(i1, i2) for (i1, i2) in self.putative_image_pair_ind}
