"""Base class for graph partitioners in GTSFM.

Authors: Zongyue Liu
"""

from __future__ import annotations

from abc import abstractmethod

import gtsfm.utils.logger as logger_utils
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

logger = logger_utils.get_logger()


class GraphPartitionerBase(GTSFMProcess):
    """Base class for all graph partitioners in GTSFM.

    Graph partitioners take a visibility graph and cluster it so that
    subsets of the problem can be processed independently.
    """

    def __init__(self, process_name: str = "GraphPartitioner"):
        """Initialize the base graph partitioner.

        Args:
            process_name: Name of the process, used for logging.
        """
        super().__init__()
        self.process_name = process_name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(process_name={self.process_name})"

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Return metadata for UI display.

        Returns:
            UiMetadata object containing UI metadata for this process.
        """
        return UiMetadata(
            display_name="Graph Partitioner",
            input_products=("Visibility Graph",),
            output_products=("ClusterTree",),
            parent_plate="Preprocessing",
        )

    @abstractmethod
    def run(self, graph: VisibilityGraph) -> ClusterTree | None:
        """Cluster a visibility graph.

        Args:
            graph: a visibility graph.
        Returns:
            ClusterTree describing the hierarchical structure.
        """

    @staticmethod
    def log_partition_details(cluster_tree: ClusterTree | None) -> None:
        """Log details of each cluster for debugging.

        Args:
            cluster_tree: ClusterTree object containing cluster details.
        """
        if cluster_tree is None:
            logger.info("0 leaf clusters found.")
            return

        leaves = cluster_tree.leaves()
        logger.info("%d leaf clusters found.", len(leaves))
        for i, leaf in enumerate(leaves, 1):
            leaf_keys = leaf.all_keys()
            logger.info("Leaf Cluster %d: keys (%d): %s", i, len(leaf_keys), list(map(int, leaf_keys)))
            logger.info("Leaf Cluster %d: num intra-cluster edges: %d", i, len(leaf.value))
            logger.debug("Leaf Cluster %d: intra-cluster edges: %s", i, leaf.value)
