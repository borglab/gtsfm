"""Vocabulary Based image retriever for front-end.
Authors: Jon Womack
"""
from collections import defaultdict
from typing import List, Tuple

import numpy as np
import cbir
import dask
from dask.delayed import Delayed
import networkx as nx
import pickle
from pathlib import Path

from gtsfm.frontend.retriever.retriever_base import RetrieverBase
import gtsfm.utils.logger as logger_utils


logger = logger_utils.get_logger()


class VocabTreeRetriever(RetrieverBase):
    """Class for vocabulary based image retrieval. Selects a subset of the image pairs from Dataloader."""

    def __init__(self, image_pair_indices):
        """Initialize the Retriever.
        Args:
            image_pair_indices: All possible image pairs (image_id1, image_id2) as determined by Dataloader.
        """
        super().__init__(image_pair_indices)
        orb_descriptor = cbir.descriptors.Orb()
        n_branches = 10
        depth = 4
        vocabulary_folder = Path(__file__).resolve().parent.parent.parent.parent.parent / "frame_selection" / "data"
        vocabulary_graph = vocabulary_folder / "graph.pickle"
        vocabulary_nodes = vocabulary_folder / "nodes.pickle"
        self.voc = cbir.encoders.VocabularyTree(n_branches=n_branches, depth=depth, descriptor=orb_descriptor)
        self.voc.graph = nx.read_gpickle(vocabulary_graph)
        with open(vocabulary_nodes, "rb") as f:
            self.voc.nodes = pickle.load(f)

    @dask.delayed
    def retrieve_potential_matches(self, image_graph: Delayed, num_closest_images) -> Delayed:
        """

        Args:
            image_graph: Delayed List of GTSfM Image objects
            num_closest_images: parameter for selecting N-nearest neighbors (images) based on vocabulary.
        Returns:
            retrieved_image_pair_indices: List of retrieved image index pairs

        """

        # Use Image Retriever to retrieve a subset of image pairs
        image_pair_distances = defaultdict(list)  # Dict[i1] = [(i2, distance), (i4, distance), ...]
        for i1, i2 in self.image_pair_indices:
            image1 = image_graph[i1].value_array
            image2 = image_graph[i2].value_array
            embedding1 = self.voc.embedding(image1)
            embedding2 = self.voc.embedding(image2)
            distance = np.dot(embedding1, embedding2)
            image_pair_distances[i1].append((i2, distance))
            image_pair_distances[i2].append((i1, distance))

        # Select only the X most similar images to be pairs
        retrieved_image_pair_indices = []
        for image_id in image_pair_distances.keys():
            image_distances = image_pair_distances[image_id]
            image_distances.sort(key=lambda x: x[1])  # x[1] is distance between image i and image x[0]
            closest_images = image_distances[:num_closest_images]
            retrieved_image_pair_indices += [(image_id2, image_id) for image_id2, distance in closest_images]
        return retrieved_image_pair_indices

    def create_computation_graph(self, image_graph: Delayed) -> List[Tuple[int, int]]:
        """Retrieve potential image matches

        Args:
            image_graph: List of GTSfM Image objects wrapped up in Delayed

        Returns:
            List of image index pairs
        """
        num_closest_images = 20
        return dask.delayed(self.retrieve_potential_matches(image_graph, num_closest_images))
