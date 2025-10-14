"""Implementation of a graph partitioner that returns a single partition.

Authors: Zongyue Liu
"""

import gtsfm.utils.logger as logger_utils
from gtsfm.graph_partitioner.graph_partitioner_base import GraphPartitionerBase
from gtsfm.products.partition import Partition, Subgraph
from gtsfm.products.visibility_graph import VisibilityGraph

logger = logger_utils.get_logger()


class SinglePartition(GraphPartitionerBase):
    """Graph partitioner that returns all edges as a single partition.

    This implementation doesn't actually partition the graph but serves as
    a baseline implementation that maintains the original workflow.
    """

    def __init__(self):
        """Initialize the partitioner."""
        super().__init__(process_name="SinglePartition")

    def run(self, graph: VisibilityGraph) -> Partition:
        """Return all visibility graph as a single partition.

        Args:
            graph: input visibility graph.

        Returns:
            Partition with a single leaf and empty inter-partition edges map.
        """
        logger.info(f"SinglePartition: returning all {len(graph)} pairs as a single partition")
        keys = set()
        for i, j in graph:
            keys.add(i)
            keys.add(j)
        return Partition(subgraphs=[Subgraph(keys=keys, edges=graph)], edge_cuts={})
