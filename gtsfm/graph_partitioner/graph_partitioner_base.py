"""Base class for graph partitioners in GTSFM.

Authors: Zongyue Liu
"""

from abc import abstractmethod

import gtsfm.utils.logger as logger_utils
from gtsfm.common.types import ImagePairs
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

logger = logger_utils.get_logger()


class GraphPartitionerBase(GTSFMProcess):
    """Base class for all graph partitioners in GTSFM.

    Graph partitioners take a set of image pairs and
    divide them into subgraphs to be processed independently.
    """

    def __init__(self, process_name: str = "GraphPartitioner"):
        """Initialize the base graph partitioner.

        Args:
            process_name: Name of the process, used for logging.
        """
        super().__init__()
        self.process_name = process_name

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Return metadata for UI display.

        Returns:
            UiMetadata object containing UI metadata for this process.
        """
        return UiMetadata(
            display_name="Graph Partitioner",
            input_products=("Image Pair Indices",),
            output_products=("Subgraphs",),
            parent_plate="Preprocessing",
        )

    @abstractmethod
    def partition_image_pairs(self, image_pairs: ImagePairs) -> list[ImagePairs]:
        """Partition a set of image pairs into subgraphs.

        Args:
            image_pairs: List of image pairs (i,j) where i < j.
        Returns:
            List of subgraphs, where each subgraph is a list of image pairs.
        """
        pass
