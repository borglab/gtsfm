"""Image retriever for front-end.
Authors: Travis Driver, Jon Womack
"""

import abc
from typing import List, Optional, Tuple, Dict
import itertools

import numpy as np
import dask
from dask.delayed import Delayed
from gtsam import Pose3, Cal3Bundler

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints


logger = logger_utils.get_logger()


# class RetrieverBase(metaclass=abc.ABCMeta):
class RetrieverBase:
    """Base class for image retrieval.
    The Retriever proposes image pairs to conduct local feature matching.
    """

    def __init__(self, image_pair_indices):
        """Initialize the Retriever.
        Args:
        """
        self.image_pair_indices = image_pair_indices

    # def _is_valid_pair(self, idx1: int, idx2: int) -> bool:
    #     """Checks if (idx1, idx2) is a valid pair.
    #     Default is exhaustive, i.e., all pairs are valid.
    #     Args:
    #         idx1: first index of the pair.
    #         idx2: second index of the pair.
    #     Returns:
    #         validation result.
    #     """

    def do_vocab(self, image_graph):
        # Load Vocabulary
        import cbir
        import networkx as nx
        import pickle
        orb_descriptor = cbir.descriptors.Orb()
        n_branches = 10
        depth = 4
        vocabulary_folder = "../frame_selection/data/"
        voc = cbir.encoders.VocabularyTree(n_branches=n_branches, depth=depth, descriptor=orb_descriptor)
        voc.graph = nx.read_gpickle(vocabulary_folder + "graph.pickle")
        with open(vocabulary_folder + 'nodes.pickle', 'rb') as f:
            voc.nodes = pickle.load(f)

        # Use Image Retriever to retrieve a subset of image pairs
        logger.info("HERE")
        image_indices = []
        for (i1, i2) in self.image_pair_indices:
            image_indices.append(i1)
            image_indices.append(i2)

        embeddings = []
        for image_id in set(image_indices):
            image = image_graph[image_id]
            logger.info(image)
            embedding = voc.embedding(image)
            embeddings.append(np.asarray(embedding))
            # print(np.dot(embedding, embeddings[0]))
        retrieved_image_pair_indices = []
        for embedding1 in embeddings:
            for embedding2 in embeddings:
                if np.dot(embedding1, embedding2) > .5:
                    retrieved_image_pair_indices.append((embedding1,embedding2))
        return retrieved_image_pair_indices

    def create_computation_graph(self, image_graph: Delayed) -> Tuple[Delayed, Delayed]:
        """Given an image, create detection and descriptor generation tasks

        Args:
            image_graph: image wrapped up in Delayed

        Returns:
            Delayed object for detected keypoints.
            Delayed object for corr. descriptors.
        """
        retrieved_image_pair_indices = dask.delayed(self.do_vocab, nout=1)(image_graph)
        return retrieved_image_pair_indices