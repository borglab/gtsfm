"""Default image retriever for front-end.

Authors: Travis Driver
"""

# import abc
from typing import List, Optional, Tuple, Dict
import itertools

import numpy as np
import dask
from dask.delayed import Delayed
from gtsam import Pose3, Cal3Bundler

import gtsfm.utils.logger as logger_utils
from gtsfm.two_view_estimator import TwoViewEstimator
from gtsfm.common.keypoints import Keypoints


logger = logger_utils.get_logger()


class DefaultRetriever:
    """Default image retriever."""

    def __init__(
        self,
        two_view_estimator: TwoViewEstimator,
        max_frame_lookahead: Optional[int] = None,
    ):
        """Initialize the Retriever.

        Args:
            two_view_estimator: performs local matching and computs relative pose.
            max_frame_lookahead: maximum number of consecutive frames to consider for local matching. Any value less
                than the size of the dataset assumes data is sequentially captured
        """
        self._two_view_estimator = two_view_estimator
        self._max_frame_lookahead = max_frame_lookahead

        # Initialize data for the TwoViewEstimator as None until computed later. These variables are assigned as
        # attributes to avoid large inputs to delayed calls as per Dask documentation.
        # Ref: https://tinyurl.com/2y9nd9uu
        self._keypoints_list: Optional[List[Keypoints]] = None
        self._descriptors_list: Optional[List[np.ndarray]] = None
        self._intrinsics_list: Optional[List[Optional[Cal3Bundler]]] = None
        self._image_shape_list: Optional[List[Tuple[int, int]]] = None
        self._gt_pose_list: Optional[List[Pose3]] = None
        self.putative_image_pairs_ind: List[Tuple[int, int]] = []

    # ignored-abstractmethod
    # @abc.abstractmethod
    def __len__(self) -> int:
        """
        The number of images in the dataset.

        Returns:
            the number of images.
        """
        return len(self._image_shape_list) if self._image_shape_list is not None else 0

    # ignored-abstractmethod
    # @abc.abstractmethod
    def _is_valid_pair(self, idx1: int, idx2: int) -> bool:
        """Checks if (idx1, idx2) is a valid pair.

        Default is exhaustive, i.e., all pairs are valid.

        Args:
            idx1: first index of the pair.
            idx2: second index of the pair.

        Returns:
            Whether the pair is valid according to the image retrieval method.
        """
        if self._max_frame_lookahead is None:
            return idx1 < idx2
        return idx1 < idx2 and idx2 - idx1 <= self._max_frame_lookahead

    def _compute_local_matches(self, i1: int, i2: int) -> Tuple[Delayed, Delayed, Delayed, Optional[Delayed]]:
        """Performs local matching and computes relative rotations and relative (unit) translations for valid image
        pairs.

        Args:
            Indices of images in putative image pair.

        Returns:
            i2Ri1: Delayed (Rot3) relative roation.
            i2Ui1: Delayed (Unit3) relative unit translation.
            v_corr_idxs: Delayed (np.ndarray) verified correspondences.
            two_view_report: Delayed TwoViewEstimationReport

        Raises:
            ValueError if attributes not initialized.
        """
        # Verify that local matching data initialized. Have to be individually checked to appease mypy.
        if (
            self._keypoints_list is None
            or self._descriptors_list is None
            or self._intrinsics_list is None
            or self._image_shape_list is None
        ):
            raise ValueError("Image retrieval data not properly initialized.")

        # Check whether valid image pair, and return None if not.
        if not self._is_valid_pair(i1, i2):
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

    def create_computation_graph(
        self,
        keypoints_graph_list: List[Delayed],
        descriptors_graph_list: List[Delayed],
        intrinsics_graph_list: List[Optional[Cal3Bundler]],
        image_shape_graph_list: List[Tuple[int, int]],
        gt_pose_graph_list: Optional[List[Pose3]] = None,
    ) -> Dict[Tuple[int, int], Delayed]:
        """Create delayed tasks for local feature matching.

        Args:
            keypoints_graph_list: List of delayed keypoints from the FeatureExtractor.
            descriptors_graph_list: List of delayed descriptors from the FeatureExtractor.

        Returns:
            Dict with keys of Tuples of image pair indices for each pair in `self._putative_image_pairs_ind` and values
                of delayed local matching tasks for the respective pairs.
        """
        # Set atrributes.
        self._keypoints_list = keypoints_graph_list
        self._descriptors_list = descriptors_graph_list
        self._intrinsics_list = intrinsics_graph_list
        self._image_shape_list = image_shape_graph_list
        self._gt_pose_list = gt_pose_graph_list

        # Compile putative image pair combinations.
        # TODO (travisdriver): don't consider every combination. This was done primarily to preserve the current
        #   structure of the main program.
        self.putative_image_pairs_ind = list(itertools.combinations(range(self.__len__()), 2))

        # Return Delayed local matching tasks.
        return {(i1, i2): self._compute_local_matches(i1, i2) for (i1, i2) in self.putative_image_pairs_ind}
