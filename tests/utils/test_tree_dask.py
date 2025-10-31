"""Tests for Dask-based tree computations."""

from __future__ import annotations

import pickle
import unittest
from pathlib import Path

from dask.distributed import Client

from gtsfm.utils.tree import Tree
from gtsfm.utils.tree_dask import gather_future_tree, submit_tree_fold, submit_tree_map

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


def _subtree_sum(value: int, child_sums: tuple[int, ...]) -> int:
    """Helper used in both synchronous and asynchronous tree folds."""
    return value + sum(child_sums)


def _double(value: int) -> int:
    """Simple mapper to exercise per-node transformations."""
    return 2 * value


class TestTreeDask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.client = Client(processes=False, threads_per_worker=1, dashboard_address=None)
        except Exception as exc:  # pragma: no cover - environment dependent.
            raise unittest.SkipTest(f"Unable to start Dask client: {exc}") from exc

    def setUp(self) -> None:
        """Make a small tree with known aggregation results."""
        self.tree = Tree(
            value=1,
            children=(
                Tree(value=2, children=(Tree(value=3), Tree(value=4))),
                Tree(value=5),
            ),
        )

    def get_client(self) -> Client:
        return self.client  # type: ignore

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_submit_tree_fold_matches_sequential_fold(self) -> None:
        expected = self.tree.fold(_subtree_sum)

        future = submit_tree_fold(self.get_client(), self.tree, _subtree_sum)
        result = future.result()

        self.assertEqual(result, expected)

    def test_submit_tree_map_produces_future_tree(self) -> None:
        expected_tree = self.tree.map(_double)

        future_tree = submit_tree_map(self.get_client(), self.tree, _double)
        result_tree = gather_future_tree(self.get_client(), future_tree)

        self.assertEqual(result_tree, expected_tree)

    def test_metis_partition_and_pickle(self) -> None:

        # Read a pickled tree from set1_lund_door
        with open(TEST_DATA_ROOT / "set1_lund_door" / "cluster_tree.pkl", "rb") as f:
            cluster_tree = pickle.load(f)
        self.assertTrue(cluster_tree is not None)

        def dummy_optimize_cluster(visibility_graph: list) -> int:
            """Dummy optimizer that just counts edges."""
            return len(visibility_graph)

        def dummy_merge(node_total: int, child_totals: tuple[int, ...]) -> int:
            """Dummy merge scheduled on Dask to accumulate edge counts."""
            return node_total + sum(child_totals)

        # Map: simulate cluster reconstruction for each cluster in the tree
        future_tree = submit_tree_map(self.get_client(), cluster_tree, dummy_optimize_cluster)

        # Fold: accumulate total edge count from all clusters
        total_edges_future = submit_tree_fold(self.get_client(), future_tree, dummy_merge)

        # Asking for the merged result blocks until all computations are done
        total_edges = total_edges_future.result()

        # There are 45 edges in the original visibility graph
        self.assertEqual(total_edges, 45)


if __name__ == "__main__":
    unittest.main()
