"""Sequential retriever, that uses a sliding window/fixed lookahead to propose image pairs.

Only useful for temporally ordered data.

Authors: John Lambert
"""
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.retriever.retriever_base import RetrieverBase, ImageMatchingRegime

logger = logger_utils.get_logger()


class SequentialRetriever(RetrieverBase):
    def __init__(self, max_frame_lookahead: int) -> None:
        """
        Args:
            max_frame_lookahead: Maximum number of consecutive frames to consider for matching/co-visibility.
        """
        super().__init__(matching_regime=ImageMatchingRegime.SEQUENTIAL)
        self._max_frame_lookahead = max_frame_lookahead

    def __repr__(self) -> str:
        return f"""
        SequentialRetriever:
           Max. frame lookahead {self._max_frame_lookahead}
        """

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],  # pylint: disable=unused-argument
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,  # pylint: disable=unused-argument
    ) -> List[Tuple[int, int]]:
        """Compute potential image pairs.

        Args:
            global_descriptors: the global descriptors for the retriever, if needed.
            image_fnames: file names of the images
            plots_output_dir: Directory to save plots to. If None, plots are not saved.

        Returns:
            List of (i1,i2) image pairs.
        """
        num_images = len(image_fnames)

        pairs = []
        for i1 in range(num_images):
            max_i2 = min(i1 + self._max_frame_lookahead + 1, num_images)
            for i2 in range(i1 + 1, max_i2):
                pairs.append((i1, i2))

        logger.info("Found %d pairs from the SequentialRetriever", len(pairs))
        return pairs
