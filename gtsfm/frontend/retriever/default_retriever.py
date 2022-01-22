"""Default image retriever for front-end.
Authors: Travis Driver, Jon Womack
"""
import abc
from typing import List, Optional, Tuple, Dict
import itertools

import numpy as np
import cbir
import dask
from dask.delayed import Delayed
from gtsam import Pose3, Cal3Bundler
import networkx as nx
import pickle

from gtsfm.frontend.retriever.retriever_base import RetrieverBase
import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints


logger = logger_utils.get_logger()


class DefaultRetriever(RetrieverBase):
    """Default class for image retrieval. Keeps all image pairs from Dataloader.
    """

    def __init__(self, image_pair_indices):
        """Initialize the Retriever.
        Args:
        """
        super().__init__(image_pair_indices)

    def retrieve_potential_matches(self, i1, i2, image_graph) -> Delayed:
        """

        :param i1:
        :param i2:
        :param image_graph:
        :return: Value between 0 and 1, with 1 representing a high likelyhood of match and 0 low likelyhood of match.
        """

        # Use Image Retriever to retrieve a subset of image pairs
        image1 = image_graph[i1]
        image2 = image_graph[i2]
        # embedding1 = dask.delayed(self.voc.embedding(image1))
        # embedding2 = dask.delayed(self.voc.embedding(image2))
        # if np.dot(embedding1, embedding2) > .5:
        #      return 1
        return 1

    def create_computation_graph(self, image_graph: Delayed) -> Dict[Tuple[int, int], Delayed]:
        """Given an image, create detection and descriptor generation tasks

        Args:
            image_graph: image wrapped up in Delayed

        Returns:
            Delayed object for detected keypoints.
            Delayed object for corr. descriptors.
        """
        return {(i1, i2): self.retrieve_potential_matches(i1, i2, image_graph) for (i1, i2) in self.image_pair_indices}