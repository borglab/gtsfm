"""
Unit tests for io utility functions.
Authors: Frank Dellaert.
"""

import pickle
import unittest
from pathlib import Path

import gtsfm.products.scene_tree as scene_tree
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.scene_tree import SceneTree, Tree

TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data"


# Test for cluster tree IO
class TestClusterTreeIO(unittest.TestCase):
    def setUp(self) -> None:
        # Base directory containing the COLMAP hierarchy.
        self.base_dir = TEST_DATA_ROOT / "palace"

    @unittest.skip("Needs new downsampled Palace dataset with correct cluster tree.")
    def test_whole_enchilada(self) -> None:
        """Test reading the COLMAP hierarchy as a tree of (Path, GtsfmData)."""
        # TODO(frank): we need somehow to have images names and shapes if we want to save.
        # Add into GtsfmData?
        tree = scene_tree.read_colmap(str(self.base_dir))
        assert tree is not None

        with open(self.base_dir / "cluster_tree.pkl", "rb") as f:
            cluster_tree: ClusterTree = pickle.load(f)
        print("Loaded cluster_tree from", self.base_dir / "cluster_tree.pkl")
        print(cluster_tree)

        zipped = Tree.zip(tree, cluster_tree)
        print("Zipped tree and cluster_tree:", zipped)

        reordered_tree: SceneTree = scene_tree.reorder(tree, cluster_tree)
        self.assertEqual(scene_tree.number_tracks(reordered_tree), scene_tree.number_tracks(tree))

        # Given a cluster tree with *local* VGGT results, we want to output a single GtsfmData that is
        # the merger of all the cluster results.
        merged_scene = scene_tree.merge(reordered_tree)
        self.assertIsInstance(merged_scene, GtsfmData)
        print(merged_scene)

        # TODO(frank): these images names and shapes should be real.
        image_filenames = [f"San_Francisco_{i:04d}.jpg" for i in range(281)]  # Dummy filenames
        image_shapes = [(1936, 1296)] * len(image_filenames)  # Dummy shapes
        for idx, (name, shape) in enumerate(zip(image_filenames, image_shapes)):
            merged_scene.set_image_info(idx, name=name, shape=shape)
        merged_scene.export_as_colmap_text(TEST_DATA_ROOT / "palace_merged_colmap" / "color")

        # We only merge 10 of the 13 clusters because of overlap issues
        self.assertEqual(merged_scene.number_tracks(), 10000)


if __name__ == "__main__":
    unittest.main()
