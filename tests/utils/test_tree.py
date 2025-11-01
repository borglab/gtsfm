"""Tests for synchronous tree helpers."""

from __future__ import annotations

import unittest

from gtsfm.utils.tree import Tree


class TestTree(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = Tree(
            value=1,
            children=(
                Tree(value=2, children=(Tree(value=3), Tree(value=4))),
                Tree(value=5),
            ),
        )

    def test_map_with_path(self) -> None:
        """Ensure the generated tree mirrors paths from the root."""
        path_tree = self.tree.map_with_path(lambda path, value: (path, value))

        self.assertEqual(path_tree.value, ((), 1))
        first_child = path_tree.children[0]
        self.assertEqual(first_child.value, ((1,), 2))
        self.assertEqual(first_child.children[0].value, ((1, 1), 3))
        self.assertEqual(first_child.children[1].value, ((1, 2), 4))
        self.assertEqual(path_tree.children[1].value, ((2,), 5))

    def test_map_with_children(self) -> None:
        """Validate post-order mapping receives child results."""
        mapped_tree = self.tree.map_with_children(lambda value, child_sums: value + sum(child_sums))

        self.assertEqual(mapped_tree.value, 15)
        self.assertEqual(mapped_tree.children[0].value, 9)
        self.assertEqual(mapped_tree.children[1].value, 5)
        # Leaf nodes should equal their original value because they have no children.
        leaf_values = [child.value for child in mapped_tree.children[0].children]
        self.assertEqual(leaf_values, [3, 4])


if __name__ == "__main__":
    unittest.main()
