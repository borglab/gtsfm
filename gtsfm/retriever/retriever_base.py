"""Base class for the Retriever, which provides a list of potential image pairs.

Authors: John Lambert
"""

import abc
from typing import List, Tuple

from gtsfm.loader.loader_base import LoaderBase

class RetrieverBase:
    def __init__(self) -> None:
        """ """
        pass

    @abc.abstractmethod
    def run(loader: LoaderBase, num_images: int, num_matched: int = 2) -> List[Tuple[int, int]]:
        """
        Args:
            loader: image loader.
            num_images: total number of images for exhaustive global descriptor matching.
            num_matched: number of K potential matches to provide per query. These are the top "K" matches per query.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
