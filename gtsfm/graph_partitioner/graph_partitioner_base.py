"""Base class for graph partitioners in GTSFM.

Authors: Zongyue Liu
"""

from __future__ import annotations

import pickle
from abc import abstractmethod

import gtsfm.utils.logger as logger_utils
from gtsfm.common.outputs import OutputPaths
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
    def log_partition_details(cluster_tree: ClusterTree | None, output_paths: OutputPaths) -> None:
        """Persist the cluster tree structure for later inspection and log summary stats."""
        if cluster_tree is None:
            logger.info("0 leaf clusters found.")
            return

        logger.info("Cluster Tree:\n%s", str(cluster_tree))

        try:
            with open(output_paths.results / "cluster_tree.pkl", "wb") as f:
                pickle.dump(cluster_tree, f)
        except Exception as exc:
            logger.warning("Failed to serialize cluster tree: %s", exc)
