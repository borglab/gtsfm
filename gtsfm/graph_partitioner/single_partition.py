"""Implementation of a graph partitioner that returns a single partition.

Authors: Zongyue Liu
"""

from typing import Dict, List, Tuple
import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase

logger = logger_utils.get_logger()


class SinglePartition(GraphPartitionerBase):
    """Graph partitioner that returns all edges as a single partition.
    
    This implementation doesn't actually partition the graph but serves as 
    a baseline implementation that maintains the original workflow.
    """
    
    def __init__(self):
        """Initialize the partitioner."""
        super().__init__(process_name="SinglePartition")
        
    def partition_image_pairs(
        self, image_pairs: list[tuple[int, int]]
    ) -> list[list[tuple[int, int]]]:
        """Return all image pairs as a single partition.
        
        Args:
            image_pairs: List of image pairs (i,j) where i < j.
                               
        Returns:
            A list containing a single subgraph with all valid edges.
        """
        logger.info(f"SinglePartition: returning all {len(image_pairs)} pairs as a single partition")
        return [image_pairs]