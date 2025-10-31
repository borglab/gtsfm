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
from gtsfm.utils.vggt import (
    VGGTReconstructionConfig,
    default_vggt_device,
    default_vggt_dtype,
    load_vggt_model,
    run_vggt_reconstruction,
)

LocalScene = tuple[Path, GtsfmData]
SceneTree = Tree[LocalScene]

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"


def run_vggt(
    image_batch: torch.Tensor,
    image_indices: list[int],
    original_coords,
    seed=42,
    use_ba=False,
    conf_threshold_value=5.0,
    vggt_fixed_resolution=518,
    img_load_resolution=1024,
    max_query_pts=1000,
    query_frame_num=4,
    fine_tracking=True,
    vis_thresh=0.2,
    max_reproj_error=8.0,
    camera_type="SIMPLE_PINHOLE",
    use_colmap_ba=False,
) -> GtsfmData:
    """Run VGGT on the given image keys and return GtsfmData."""

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Setting seed as: {seed}")

    device = default_vggt_device()
    dtype = default_vggt_dtype(device)
    print(f"Using device: {device.type}")
    print(f"Using dtype: {dtype}")

    model = load_vggt_model(device=device)
    print("Model loaded")

    image_batch = image_batch.to(device)
    print("image_batch: ", image_batch.shape)
    original_coords = original_coords.to(device)

    config = VGGTReconstructionConfig(
        use_ba=use_ba,
        vggt_fixed_resolution=vggt_fixed_resolution,
        img_load_resolution=img_load_resolution,
        max_query_pts=max_query_pts,
        query_frame_num=query_frame_num,
        fine_tracking=fine_tracking,
        vis_thresh=vis_thresh,
        max_reproj_error=max_reproj_error,
        confidence_threshold=conf_threshold_value,
        shared_camera=False,
        use_colmap_ba=use_colmap_ba,
        camera_type_ba=camera_type,
    )

    result = run_vggt_reconstruction(
        image_batch,
        image_indices=image_indices,
        image_names=[f"image_{idx}" for idx in image_indices],
        original_coords=original_coords,
        config=config,
        device=device,
        dtype=dtype,
        model=model,
        total_num_images=len(image_indices),
    )

    output_dir = DATA_ROOT_PATH / "vggt_test_output"
    if result.used_ba:
        output_subdir = f"sparse_w_ba_{query_frame_num}_{max_query_pts}_{use_colmap_ba}"
    else:
        output_subdir = "sparse_wo_ba"
    sparse_reconstruction_dir = output_dir / output_subdir
    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    sparse_reconstruction_dir.mkdir(parents=True, exist_ok=True)
    result.gtsfm_data.export_as_colmap_text(sparse_reconstruction_dir)

    if result.fallback_reason:
        print(result.fallback_reason)

    return result.gtsfm_data


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

        # resize_transform = None
        resize_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.from_numpy(x)),
                transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [H,W,C] → [C,H,W]
                transforms.Resize(size=(img_load_resolution, img_load_resolution), antialias=True),  # Expects [C,H,W]
            ]
        )
        # Transform 2: Convert to float32 and normalize to [0, 1]
        batch_transform = transforms.Lambda(lambda x: x.type(torch.float32) / 255.0)

        image_batch, original_coords = loader.load_image_batch_vggt(
            indices,
            img_load_resolution,
            resize_transform,
            batch_transform,
        )

        # image_batch, original_coords = loader.load_and_preprocess_images_square_vggt(indices, img_load_resolution)

        print("image_batch: ", image_batch.shape)

        with torch.no_grad():

            gtsfm_data = run_vggt(image_batch, indices, original_coords, use_ba=True)

        self.assertIsNotNone(gtsfm_data)
        self.assertEqual(gtsfm_data.number_images(), len(indices))
        self.assertGreaterEqual(len(gtsfm_data.get_valid_camera_indices()), len(indices))

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
