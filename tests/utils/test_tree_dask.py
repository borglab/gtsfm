"""Tests for Dask-based tree computations."""

from __future__ import annotations

import unittest

from dask.distributed import Client

from gtsfm.utils.tree import Tree
from gtsfm.utils.tree_dask import gather_future_tree, submit_tree_fold, submit_tree_map


def _subtree_sum(value: int, child_sums: tuple[int, ...]) -> int:
    """Helper used in both synchronous and asynchronous tree folds."""
    return value + sum(child_sums)


def _compute_subtree_sums_sync(tree: Tree[int]) -> Tree[int]:
    """Return a tree where each node stores the sum of its subtree."""
    child_summaries = tuple(_compute_subtree_sums_sync(child) for child in tree.children)
    node_sum = tree.value + sum(child.value for child in child_summaries)
    return Tree(value=node_sum, children=child_summaries)


@unittest.skipUnless(Client is not None, "Dask is required for Tree Dask tests.")
class TestTreeDask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert Client is not None  # for mypy
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

    @classmethod
    def tearDownClass(cls):
        cls.client.close()

    def test_submit_tree_fold_matches_sequential_fold(self):
        expected = self.tree.fold(_subtree_sum)

        future = submit_tree_fold(self.client, self.tree, _subtree_sum)
        result = future.result()

        self.assertEqual(result, expected)

    def test_submit_tree_map_produces_future_tree(self):
        expected_tree = _compute_subtree_sums_sync(self.tree)

        future_tree = submit_tree_map(self.client, self.tree, _subtree_sum)
        result_tree = gather_future_tree(self.client, future_tree)

        self.assertEqual(result_tree, expected_tree)


if __name__ == "__main__":
    unittest.main()
