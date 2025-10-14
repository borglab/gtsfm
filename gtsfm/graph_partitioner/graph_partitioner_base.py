"""Base class for graph partitioners in GTSFM.

Authors: Zongyue Liu
"""

from abc import abstractmethod

import gtsfm.utils.logger as logger_utils
from gtsfm.products.partition import Partition
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

logger = logger_utils.get_logger()


class GraphPartitionerBase(GTSFMProcess):
    """Base class for all graph partitioners in GTSFM.

    Graph partitioners take a set of visibility graph and
    partitions it into subgraphs to be processed independently.
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
            input_products=("Visibility Graph",),
            output_products=("Partition",),
            parent_plate="Preprocessing",
        )

    @abstractmethod
    def run(self, graph: VisibilityGraph) -> Partition:
        """Partition a set of visibility graph into subgraphs.

        Args:
            graph: a visibility graph.
        Returns:
            Partition: leaves and inter-partition edges map.
        """

    @staticmethod
    def log_partition_details(partition: Partition) -> None:
        """Log details of each partition for debugging.

        Args:
            partition: Partition object containing partition details.
        """
        logger.info("%d subgraphs found.", len(partition.subgraphs))
        for i, leaf in enumerate(partition.subgraphs):
            logger.info("Subgraph %d: keys (%d): %s", i, len(leaf.keys), leaf.keys)
            logger.info("Subgraph %d: num intra-partition edges: %d", i, len(leaf.edges))
            logger.debug("Subgraph %d: intra-partition edges: %s", i, leaf.edges)
