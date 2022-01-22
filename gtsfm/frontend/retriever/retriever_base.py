"""Image retriever for front-end.
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

import gtsfm.utils.logger as logger_utils
from gtsfm.common.keypoints import Keypoints


logger = logger_utils.get_logger()


class RetrieverBase(metaclass=abc.ABCMeta):
    """Base class for image retrieval.
    The Retriever proposes image pairs to conduct local feature matching.
    """

    def __init__(self, image_pair_indices):
        """Initialize the Retriever

        :param image_pair_indices: Set of possible image pairs based on Dataloader
        """
        self.image_pair_indices = image_pair_indices
        orb_descriptor = cbir.descriptors.Orb()
        n_branches = 10
        depth = 4
        vocabulary_folder = "../frame_selection/data/"
        self.voc = cbir.encoders.VocabularyTree(n_branches=n_branches, depth=depth, descriptor=orb_descriptor)
        self.voc.graph = nx.read_gpickle(vocabulary_folder + "graph.pickle")
        with open(vocabulary_folder + 'nodes.pickle', 'rb') as f:
            self.voc.nodes = pickle.load(f)

    # ignored-abstractmethod
    @abc.abstractmethod
    def retrieve_potential_matches(self, i1, i2, image_graph) -> Tuple[Delayed, Delayed, Delayed, Optional[Delayed]]:
        """

        :param i1:
        :param i2:
        :param image_graph:
        :return:
        """

    def create_computation_graph(self, image_graph: Delayed) -> Dict[Tuple[int, int], Delayed]:
        """Given an image, create detection and descriptor generation tasks

        Args:
            image_graph: image wrapped up in Delayed

        Returns:
            Delayed object for detected keypoints.
            Delayed object for corr. descriptors.
        """
        return {(i1, i2): self.retrieve_potential_matches(i1, i2, image_graph) for (i1, i2) in self.image_pair_indices}