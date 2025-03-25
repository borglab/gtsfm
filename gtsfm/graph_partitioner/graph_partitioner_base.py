"""Base class for graph partitioners in GTSFM.

Authors: Zongyue Liu
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.ui.gtsfm_process import GTSFMProcess
from gtsfm.ui.gtsfm_process import UiMetadata

logger = logger_utils.get_logger()


class GraphPartitionerBase(GTSFMProcess):
    """Base class for all graph partitioners in GTSFM.

    Graph partitioners take a set of image pairs (or a similarity matrix) and 
    divide them into subgraphs to be processed independently.
    """

    def __init__(self, process_name: str = "GraphPartitioner"):
        """Initialize the base graph partitioner.
        
        Args:
            process_name: Name of the process, used for logging.
        """
        super().__init__()
        self.process_name = process_name

    @classmethod
    def get_ui_metadata(cls) -> UiMetadata:
        """Return metadata for UI display.
        
        Returns:
            UiMetadata object containing UI metadata for this process.
        """
        return UiMetadata(
            display_name="Graph Partitioner",
            input_products=("Image Pairs",),
            output_products=("Subgraphs",),
            parent_plate="Preprocessing"
        )

    @abstractmethod
    def partition_image_pairs(
        self, image_pairs: List[Tuple[int, int]]
    ) -> List[List[Tuple[int, int]]]:
        """Partition a set of image pairs into subgraphs.
        
        Args:
            image_pairs: List of image pairs (i,j) where i < j.
            
        Returns:
            List of subgraphs, where each subgraph is a list of image pairs.
        """
        pass
