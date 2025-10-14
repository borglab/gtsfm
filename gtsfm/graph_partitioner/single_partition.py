"""Implementation of a graph partitioner that returns a single cluster.

Authors: Zongyue Liu
"""

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.clustering import Cluster, Clustering
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


class SinglePartitioner(GraphPartitionerBase):
    """Graph partitioner that returns all edges as a single cluster.

    This implementation doesn't actually partition the graph but serves as
    a baseline implementation that maintains the original workflow.
    """

    def __init__(self):
        """Initialize the partitioner."""
        super().__init__(process_name="SinglePartitioner")

    def run(self, graph: VisibilityGraph) -> Clustering:
        """Return all edges as a single-leaf clustering.

        Args:
            graph: input visibility graph.

        Returns:
            Clustering with a single leaf cluster containing all edges.
        """
        if len(graph) == 0:
            logger.warning("SinglePartitioner: received empty visibility graph.")
            return Clustering(root=None)

        logger.info("SinglePartitioner: returning all %d pairs as a single cluster", len(graph))
        keys = set()
        for i, j in graph:
            keys.add(i)
            keys.add(j)
        cluster = Cluster(keys=frozenset(keys), edges=list(graph), children=())
        return Clustering(root=cluster)
