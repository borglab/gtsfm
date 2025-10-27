"""Unit tests for VGGT glue.

Authors: Xinan Zhang and Frank Dellaert
"""

import pickle
import unittest
from pathlib import Path

import torch
from torchvision.transforms import v2 as transforms  # type: ignore

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.products.cluster_tree import ClusterTree
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.utils.tree import Tree  # PreOrderIter

LocalScene = tuple[Path, GtsfmData]
SceneTree = Tree[LocalScene]


def run_vggt(image_batch: torch.Tensor) -> GtsfmData:
    """Run VGGT on the given image keys and return GtsfmData."""
    # call run_vggt
    return GtsfmData(0, None, None)


def run_vggt_for_edges(vg: VisibilityGraph) -> GtsfmData:
    return GtsfmData(0, None, None)  # dummy
    # keys = visibility_graph_keys(vg)
    # TODO: load images using the loader and the call the above
    # return run_vggt(keys)


TEST_DATA = Path(__file__).parent.parent / "data"
PALACE = TEST_DATA / "palace"
DOOR = TEST_DATA / "set1_lund_door"


class TestVGGT(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def test_run_vggt_on_some_images(self):
        """Load four door images using Olsson loader and run vggt on them."""

        img_load_resolution = 1024
        loader = OlssonLoader(dataset_dir=str(DOOR), max_resolution=img_load_resolution)
        indices = [4, 11, 8, 2]

        resize_transform = None
        resize_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.from_numpy(x)),
                transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [H,W,C] → [C,H,W]
                transforms.Resize(size=(322, 322), antialias=True),  # Expects [C,H,W]
            ]
        )
        # Transform 2: Convert to float32 and normalize to [0, 1]
        batch_transform = transforms.Lambda(lambda x: x.type(torch.float32) / 255.0)

        image_batch = loader.load_image_batch(indices, resize_transform, batch_transform)

        gtsfm_data = run_vggt(image_batch)

        self.assertIsNotNone(gtsfm_data)
        self.assertEqual(gtsfm_data.number_images(), 4)

    @unittest.skip("Skipping VGGT on cluster tree test for now.")
    def test_vggt_on_cluster_tree(self) -> None:
        """Test VGGT on a small cluster tree."""
        data_path = PALACE / "cluster_tree.pkl"
        with open(data_path, "rb") as f:
            cluster_tree = pickle.load(f)

        self.assertIsNotNone(cluster_tree)
        self.assertIsInstance(cluster_tree, ClusterTree)

        assert cluster_tree is not None
        vggt_results: Tree[GtsfmData] = cluster_tree.map(run_vggt_for_edges)
        # for node in PreOrderIter(vggt_results):
        #     print(f"ClusterTree node with {node.value}")

        self.assertEqual(vggt_results.value.number_images(), 35)


if __name__ == "__main__":
    unittest.main()
