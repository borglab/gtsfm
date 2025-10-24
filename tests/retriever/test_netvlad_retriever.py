"""Unit tests for the NetVLAD retriever.

Authors: John Lambert
"""

import unittest
from pathlib import Path


from gtsfm.frontend.global_descriptor.netvlad_global_descriptor import NetVLADGlobalDescriptor
from gtsfm.loader.colmap_loader import ColmapLoader
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.retriever.netvlad_retriever import NetVLADRetriever

DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data"
DOOR_DATA_ROOT = DATA_ROOT_PATH / "set1_lund_door"
SKYDIO_DATA_ROOT = DATA_ROOT_PATH / "crane_mast_8imgs_colmap_output"


class TestNetVLADRetriever(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.global_descriptor = NetVLADGlobalDescriptor()

    def test_netvlad_retriever_crane_mast(self) -> None:
        """Test the NetVLAD retriever on 2 frames of the Skydio Crane-Mast dataset."""
        dataset_dir = SKYDIO_DATA_ROOT
        images_dir = SKYDIO_DATA_ROOT / "images"

        loader = ColmapLoader(
            dataset_dir=str(dataset_dir),
            images_dir=str(images_dir),
            max_resolution=760,
        )

        retriever = NetVLADRetriever(num_matched=2)
        resize_transform, batch_transform = self.global_descriptor.get_preprocessing_transforms()

        indices = list(range(len(loader)))
        batch_tensor = loader.load_image_batch(indices, resize_transform, batch_transform)

        descriptors = self.global_descriptor.describe_batch(batch_tensor)

        pairs = retriever.get_image_pairs(descriptors, loader.image_filenames(), plots_output_dir=None)

        # Only 1 pair possible between frame 0 and frame 1.
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs, [(0, 1)])

    def test_netvlad_retriever_door(self) -> None:
        """Test the NetVLAD retriever on 12 frames of the Lund Door Dataset."""
        loader = OlssonLoader(dataset_dir=str(DOOR_DATA_ROOT))
        retriever = NetVLADRetriever(num_matched=2)
        resize_transform, batch_transform = self.global_descriptor.get_preprocessing_transforms()

        indices = list(range(len(loader)))
        batch_tensor = loader.load_image_batch(indices, resize_transform, batch_transform)
        
        descriptors = self.global_descriptor.describe_batch(batch_tensor)

        pairs = retriever.get_image_pairs(descriptors, loader.image_filenames(), plots_output_dir=None)

        self.assertEqual(len(pairs), 21)

        for i1, i2 in pairs:
            self.assertTrue(i1 != i2)
            self.assertTrue(i1 < i2)

        # closest image is most similar for the Door dataset.
        expected_pairs = [
            (0, 1),
            (0, 2),
            (1, 2),
            (1, 3),
            (2, 3),
            (2, 4),
            (3, 4),
            (3, 5),
            (4, 5),
            (4, 6),
            (5, 6),
            (5, 7),
            (6, 7),
            (6, 8),
            (7, 8),
            (7, 9),
            (8, 9),
            (8, 10),
            (9, 10),
            (9, 11),
            (10, 11),
        ]
        self.assertEqual(pairs, expected_pairs)


if __name__ == "__main__":
    unittest.main()
