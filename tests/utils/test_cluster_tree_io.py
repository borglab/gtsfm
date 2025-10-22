"""
Unit tests for io utility functions.
Authors: Frank Dellaert.
"""

import unittest
from pathlib import Path

from gtsfm.utils.cluster_tree_io import ColmapScene, print_tree_cameras_and_points, read_colmap_hierarchy_as_tree

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# Test for cluster tree IO
class TestClusterTreeIO(unittest.TestCase):
    def test_read_colmap_hierarchy_as_tree_lund_door_binary(self) -> None:
        # Base directory containing the COLMAP hierarchy shown in the prompt.
        base_dir = TEST_DATA_ROOT / "lund_door_binary"

        # Execute
        tree = read_colmap_hierarchy_as_tree(str(base_dir))
        assert tree is not None

        # Sanity-check scene contents for all leaves
        for node in tree.traverse():
            path, scene = node.value
            if scene:
                self.assertIsInstance(path, Path)
                self.assertIsInstance(scene, ColmapScene)
                self.assertTrue(scene.is_valid())

        # Check that only the four expected leaf nodes have scene data
        for node in tree.traverse():
            if node.is_leaf():
                path, scene = node.value
                assert scene is not None
            else:
                path, scene = node.value
                self.assertIsNone(scene)

        # Check the data in the leaves
        stems_sizes: list[tuple[str, int]] = []
        for node in tree.traverse():
            if node.is_leaf():
                path, scene = node.value
                assert scene is not None
                stems_sizes.append((path.stem, scene.num_points()))
        self.assertEqual(stems_sizes, [("C_1_1", 1721), ("C_1_2", 1740), ("C_2_1", 1784), ("C_2_2", 1654)])

    @unittest.skip("slow")
    def test_read_palace_metis(self) -> None:
        # Base directory containing the COLMAP hierarchy shown in the prompt.
        base_dir = Path("/Users/dellaert/git/gtsfm/palace_metis")

        # Execute
        tree = read_colmap_hierarchy_as_tree(str(base_dir), "sparse")
        print_tree_cameras_and_points(tree)


if __name__ == "__main__":
    unittest.main()
