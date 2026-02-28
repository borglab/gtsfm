"""Unit tests for graph partitioner utility helpers."""

import unittest

from gtsfm.graph_partitioner.partition_utils import (
    build_edges_for_keyset,
    canonical_edge,
    count_cross_edges,
    graph_adjacency,
    min_key,
    partition_local_keys,
)


class TestPartitionUtils(unittest.TestCase):
    def test_min_key(self) -> None:
        self.assertEqual(min_key({3, 1, 2}), 1)
        self.assertEqual(min_key(set()), 10**18)

    def test_canonical_edge(self) -> None:
        self.assertEqual(canonical_edge(1, 3), (1, 3))
        self.assertEqual(canonical_edge(3, 1), (1, 3))

    def test_count_cross_edges(self) -> None:
        graph = [(0, 1), (1, 2), (2, 3), (0, 3)]
        self.assertEqual(count_cross_edges({0, 1}, {2, 3}, graph), 2)
        self.assertEqual(count_cross_edges({0}, {1}, graph), 1)

    def test_partition_local_keys_balanced(self) -> None:
        bins = partition_local_keys([0, 1, 2, 3, 4], 2)
        self.assertEqual(bins, [{0, 1, 2}, {3, 4}])

    def test_graph_adjacency(self) -> None:
        graph = [(0, 1), (1, 2)]
        adjacency = graph_adjacency(graph)
        self.assertEqual(adjacency[0], {1})
        self.assertEqual(adjacency[1], {0, 2})
        self.assertEqual(adjacency[2], {1})

    def test_build_edges_for_keyset(self) -> None:
        graph = [(0, 1), (1, 2), (2, 3)]
        edges = build_edges_for_keyset({0, 1, 2, 3}, graph)
        self.assertEqual(edges, [(0, 1), (1, 2), (2, 3)])


if __name__ == "__main__":
    unittest.main()
