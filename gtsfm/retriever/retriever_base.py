"""Base class for the Retriever, which provides a list of potential image pairs.

Authors: John Lambert
"""

import abc
from typing import List, Tuple

from gtsfm.loader.loader_base import LoaderBase


class RetrieverBase:
    def __init__(self, num_matched: int = 2) -> None:
        """
        Args:
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.
        """
        self._num_matched = num_matched

    @abc.abstractmethod
    def run(loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs.
        
        Args:
            loader: image loader. The length of this loader will provide the total number of images
                for exhaustive global descriptor matching.

        Return:
            pair_indices: (i1,i2) image pairs.
        """