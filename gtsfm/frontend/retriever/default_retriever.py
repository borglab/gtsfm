"""Default image retriever for front-end.

Authors: Travis Driver
"""

# import abc
from typing import Optional


from gtsfm.frontend.retriever.retriever_base import RetrieverBase
import gtsfm.utils.logger as logger_utils
from gtsfm.two_view_estimator import TwoViewEstimator


logger = logger_utils.get_logger()


class DefaultRetriever(RetrieverBase):
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
        super().__init__(two_view_estimator)
        self._max_frame_lookahead = max_frame_lookahead

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
