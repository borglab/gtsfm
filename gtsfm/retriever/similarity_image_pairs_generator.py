"""Extended image pairs generator that returns a similarity matrix.

Authors: Zongyue Liu
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from dask.distributed import Client, Future
import logging

from gtsfm.retriever.image_pairs_generator import ImagePairsGenerator

logger = logging.getLogger(__name__)

class SimilarityImagePairsGenerator(ImagePairsGenerator):
    """Extension of ImagePairsGenerator that also returns a similarity matrix.
    
    This class extends the base image pairs generator to compute and return
    a similarity matrix between all images, which can be used for graph partitioning.
    """
    
    def generate_image_pairs_with_similarity(
        self,
        client: Client,
        images: List[Future],
        image_fnames: List[str],
        plots_output_dir=None,
    ) -> Tuple[List[Tuple[int, int]], np.ndarray]:
        """Generate image pairs and compute a similarity matrix.
        
        Args:
            client: Dask client for parallel computation.
            images: List of images as futures.
            image_fnames: List of image filenames.
            plots_output_dir: Directory to save plots (if needed).
            
        Returns:
            Tuple containing:
            - List of image pairs (i,j) where i < j
            - NxN similarity matrix where similarity_matrix[i,j] represents
              the similarity between images i and j
        """
        # First, generate the image pairs using the base implementation
        image_pairs = self.generate_image_pairs(
            client=client,
            images=images,
            image_fnames=image_fnames,
            plots_output_dir=plots_output_dir,
        )
        
        # Create a similarity matrix based on the retriever's results
        n = len(images)
        similarity_matrix = np.zeros((n, n))

        logger.info(f"=== Similarity Matrix Generation Summary ===")
        logger.info(f"Retriever type: {type(self._retriever).__name__}")

        # If the retriever provides similarity scores, use them
        if hasattr(self._retriever, "get_similarity_scores"):
            logger.warning("Method: Using similarity scores from retriever")
            similarity_scores = self._retriever.get_similarity_scores()
            
            # Fill the similarity matrix
            for (i, j), score in similarity_scores.items():
                similarity_matrix[i, j] = score
                similarity_matrix[j, i] = score  # Ensure matrix is symmetric
        else:
            logger.warning("Method: Using binary values (1.0) for connected image pairs")
            for i, j in image_pairs:
                similarity_matrix[i, j] = 1.0
                similarity_matrix[j, i] = 1.0
                
        return image_pairs, similarity_matrix   