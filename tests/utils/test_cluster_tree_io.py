"""
Unit tests for io utility functions.
Authors: Frank Dellaert.
"""

import unittest
from pathlib import Path

import gtsfm.utils.cluster_tree_io as cluster_tree_io
from gtsfm.utils.cluster_tree_io import ColmapScene

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# Test for cluster tree IO
class TestClusterTreeIO(unittest.TestCase):
    def setUp(self) -> None:
        # Base directory containing the COLMAP hierarchy.
        self.base_dir = TEST_DATA_ROOT / "lund_door_binary"

    def test_read_dir_hierarchy_as_tree(self) -> None:
        """Test reading a directory hierarchy as a tree of Paths."""
        tree = cluster_tree_io.read_dir_hierarchy_as_tree(str(self.base_dir))
        assert tree is not None

        # assert all values are Paths
        self.assertTrue(tree.all(lambda path: isinstance(path, Path)))

        # Check stems of directories
        stems = [node.value.stem for node in tree]
        self.assertEqual(
            stems,
            [
                "lund_door_binary",
                "C_1",
                "C_1_1",
                "ba_output",
                "C_1_2",
                "ba_output",
                "C_2",
                "C_2_1",
                "ba_output",
                "C_2_2",
                "ba_output",
            ],
        )

    def test_read_colmap_hierarchy_as_tree_lund_door_binary(self) -> None:
        # Base directory containing the COLMAP hierarchy shown in the prompt.
        base_dir = TEST_DATA_ROOT / "lund_door_binary"

        # Execute
        tree = cluster_tree_io.read_colmap_hierarchy_as_tree(str(base_dir))
        assert tree is not None

        # Sanity-check scene contents for all leaves
        for node in tree:
            path, scene = node.value
            self.assertIsInstance(path, Path)
            if scene:
                self.assertIsInstance(scene, ColmapScene)
                self.assertTrue(scene.is_valid())

        # Check that only the four expected leaf nodes have scene data
        for node in tree:
            path, scene = node.value
            self.assertEqual(scene is not None, node.is_leaf())

        # Check the data in the leaves
        stems_sizes: list[tuple[str, int]] = []
        for node in tree:
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
        tree = cluster_tree_io.read_colmap_hierarchy_as_tree(str(base_dir), "sparse")
        cluster_tree_io.print_tree_cameras_and_points(tree)


if __name__ == "__main__":
    unittest.main()
