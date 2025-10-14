"""
Defining data structures for graph partitions and subgraphs.
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, TypeVar

from gtsfm.products.visibility_graph import AnnotatedGraph, VisibilityGraph

T = TypeVar("T")


@dataclass(frozen=True)
class Subgraph:
    keys: Set[int]  # set of image indices in this subgraph
    edges: VisibilityGraph  # edges within this subgraph

    def extract(self, annotated_graph: AnnotatedGraph[T]) -> AnnotatedGraph[T]:
        """Extract the portion of the annotated graph that corresponds to this subgraph.

        Args:
            annotated_graph: Dictionary mapping image pairs to results.

        Returns:
            Annotated graph containing only the edges present in this subgraph.
        """
        return {pair: annotated_graph[pair] for pair in self.edges if pair in annotated_graph}


@dataclass(frozen=True)
class Partition:
    subgraphs: List[Subgraph]  # list of subgraphs (partitions)
    edge_cuts: Dict[Tuple[int, int], VisibilityGraph]  # map of edges between subgraphs i and j

    def group_by_subgraph(self, annotated_graph: AnnotatedGraph[T]) -> List[AnnotatedGraph[T]]:
        """Group results by subgraph.

        Args:
            annotated_graph: Dictionary mapping image pairs to results.

        Returns:
            List of annotated graphs, where each contains results for one subgraph.
        """
        return [subgraph.extract(annotated_graph) for subgraph in self.subgraphs]
