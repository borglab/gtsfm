"""
Unit tests for io utility functions.
Authors: Frank Dellaert.
"""

import tempfile
import unittest
from pathlib import Path

import gtsfm.utils.cluster_tree_io as cluster_tree_io
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.utils.tree import PreOrderIter

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
        stems = [node.value.stem for node in PreOrderIter(tree)]
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
        """Test reading the COLMAP hierarchy as a tree of (Path, GtsfmData)."""
        tree = cluster_tree_io.read_colmap_hierarchy_as_tree(str(self.base_dir))
        assert tree is not None

        # Sanity-check scene contents for all leaves
        for node in PreOrderIter(tree):
            path, scene = node.value
            self.assertIsInstance(path, Path)
            if scene:
                self.assertIsInstance(scene, GtsfmData)

        # Check that only the four expected leaf nodes have scene data
        for node in PreOrderIter(tree):
            path, scene = node.value
            self.assertEqual(scene is not None, node.is_leaf())

        # Check the data in the leaves
        stems_sizes: list[tuple[str, int]] = []
        for node in PreOrderIter(tree):
            if node.is_leaf():
                path, scene = node.value
                assert scene is not None
                stems_sizes.append((path.stem, scene.number_tracks()))
        self.assertEqual(stems_sizes, [("C_1_1", 1721), ("C_1_2", 1740), ("C_2_1", 1784), ("C_2_2", 1654)])

    def test_read_dir_hierarchy_as_tree_empty(self) -> None:
        """Test reading an empty directory hierarchy as a tree of Paths."""
        empty_dir = TEST_DATA_ROOT / "lund_door_binary" / "C_1" / "C_1_1" / "ba_output"
        tree = cluster_tree_io.read_colmap_hierarchy_as_tree(str(empty_dir))
        self.assertIsNone(tree)

    def test_read_colmap_hierarchy_as_tree_with_downsampling(self) -> None:
        """Test reading the COLMAP hierarchy as a tree of (Path, GtsfmData) with downsampling."""
        tree = cluster_tree_io.read_colmap_hierarchy_as_tree(str(self.base_dir))
        assert tree is not None

        seed = 42
        fraction_points_to_keep = 0.1
        downsampled_tree = cluster_tree_io.downsample(tree, fraction_points_to_keep=fraction_points_to_keep, seed=seed)

        # Check the number of sizes in the leaves
        sizes = []
        for node in PreOrderIter(downsampled_tree):
            if node.is_leaf():
                path, scene = node.value
                assert scene is not None
                sizes.append(scene.number_tracks())
        # Check exact sizes with this rng seed and fraction
        self.assertEqual(sizes, [172, 174, 178, 165])

    def test_write_colmap_hierarchy_as_tree(self) -> None:
        """Test writing the COLMAP hierarchy as a tree."""
        tree = cluster_tree_io.read_colmap_hierarchy_as_tree(str(self.base_dir))
        assert tree is not None

        image_shapes = [(800, 600)] * 10  # Dummy shapes
        image_filenames = [f"image_{i}.jpg" for i in range(10)]  # Dummy filenames

        with tempfile.TemporaryDirectory() as temp_dir:
            cluster_tree_io.write_colmap_hierarchy_as_tree(
                tree, temp_dir, image_shapes=image_shapes, image_filenames=image_filenames
            )
            # Read back the written tree and verify it matches the original
            read_back_tree = cluster_tree_io.read_colmap_hierarchy_as_tree(temp_dir)
            assert read_back_tree is not None, "Read back tree is None"
            # Compare the two trees
            for orig_node, read_node in zip(PreOrderIter(tree), PreOrderIter(read_back_tree)):
                orig_path, orig_scene = orig_node.value
                read_path, read_scene = read_node.value
                self.assertEqual(orig_path, read_path)
                if orig_scene is None:
                    self.assertIsNone(read_scene)
                else:
                    self.assertIsNotNone(read_scene)
                    self.assertEqual(orig_scene.number_tracks(), read_scene.number_tracks())


if __name__ == "__main__":
    unittest.main()
