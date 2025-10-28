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

from gtsfm.utils.vggt import *

LocalScene = tuple[Path, GtsfmData]
SceneTree = Tree[LocalScene]


def run_vggt(image_batch: torch.Tensor, image_indices: list[int], original_coords, seed=42, use_ba=False, conf_thres_value=5.0, vggt_fixed_resolution=518) -> GtsfmData:
    """Run VGGT on the given image keys and return GtsfmData."""
    # call run_vggt
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
    print(f"Setting seed as: {seed}")
    
    # Set device and dtype
    device = default_vggt_device()
    dtype = default_vggt_dtype(device)
    print(f"Using device: {device.type}")
    print(f"Using dtype: {dtype}")
    
    # Run VGGT for camera and depth estimation
    model = load_vggt_model(device=device)
    
    print("Model loaded")
    
    image_batch = image_batch.to(device)
    print('image_batch: ', image_batch.shape)
    original_coords = original_coords.to(device)
    
    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, image_batch, dtype, 518)
    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)
    
    if use_ba:
        image_size = np.array(images.shape[-2:])
        scale = img_load_resolution / vggt_fixed_resolution
        shared_camera = args.shared_camera

        with Timer("Tracking inference"):
            with torch.cuda.amp.autocast(dtype=dtype):
                # Predicting Tracks
                # Using VGGSfM tracker instead of VGGT tracker for efficiency
                # VGGT tracker requires multiple backbone runs to query different frames (this is a problem caused by the training process)
                # Will be fixed in VGGT v2

                # You can also change the pred_tracks to tracks from any other methods
                # e.g., from COLMAP, from CoTracker, or by chaining 2D matches from Lightglue/LoFTR.
                pred_tracks, pred_vis_scores, pred_confs, points_3d, points_rgb = predict_tracks(
                    images,
                    conf=depth_conf,
                    points_3d=points_3d,
                    masks=None,
                    max_query_pts=args.max_query_pts,
                    query_frame_num=args.query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=args.fine_tracking,
                )

                torch.cuda.empty_cache()

        # rescale the intrinsic matrix from 518 to 1024
        intrinsic[:, :2, :] *= scale
        track_mask = pred_vis_scores > args.vis_thresh

        # TODO: radial distortion, iterative BA, masks
        reconstruction, valid_track_mask = batch_np_matrix_to_pycolmap(
            points_3d,
            extrinsic,
            intrinsic,
            pred_tracks,
            image_size,
            masks=track_mask,
            max_reproj_error=args.max_reproj_error,
            shared_camera=shared_camera,
            camera_type=args.camera_type,
            points_rgb=points_rgb,
            image_id_list=image_indices,
        )

        if reconstruction is None:
            raise ValueError("No reconstruction can be built with BA")

        if args.use_colmap_ba:

            # Bundle Adjustment w/ Colmap
            ba_options = pycolmap.BundleAdjustmentOptions()
            pycolmap.bundle_adjustment(reconstruction, ba_options)

        reconstruction_resolution = img_load_resolution
    else:
        conf_thres_value = conf_thres_value
        max_points_for_colmap = 100000  # randomly sample 3D points
        shared_camera = False  # in the feedforward manner, we do not support shared camera
        camera_type = "PINHOLE"  # in the feedforward manner, we only support PINHOLE camera

        image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        num_frames, height, width, _ = points_3d.shape

        points_rgb = F.interpolate(
            image_batch, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        )
        points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
        points_rgb = points_rgb.transpose(0, 2, 3, 1)

        # (S, H, W, 3), with x, y coordinates and frame indices
        points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

        conf_mask = depth_conf >= conf_thres_value
        # at most writing 100000 3d points to colmap reconstruction object
        conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

        points_3d = points_3d[conf_mask]
        points_xyf = points_xyf[conf_mask]
        points_rgb = points_rgb[conf_mask]

        print("Converting to COLMAP format")
        reconstruction = batch_np_matrix_to_pycolmap_wo_track(
            points_3d,
            points_xyf,
            points_rgb,
            extrinsic,
            intrinsic,
            image_size,
            shared_camera=shared_camera,
            camera_type=camera_type,
            image_id_list=image_indices,
        )

        reconstruction_resolution = vggt_fixed_resolution

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        image_indices,
        original_coords.cpu().numpy(),
        img_size=reconstruction_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
        image_id_list=image_indices,
    )

    if not use_ba:
        print(f"Saving reconstruction to ./sparse_wo_ba")
        sparse_reconstruction_dir = Path("./") / "sparse_wo_ba"
    else:
        print(
            f"Saving reconstruction to {args.output_dir}/{cluster_key}/sparse_w_ba_{args.query_frame_num}_{args.max_query_pts}_{args.use_colmap_ba}"
        )
        sparse_reconstruction_dir = os.path.join(
            args.output_dir,
            cluster_key,
            f"sparse_w_ba_{args.query_frame_num}_{args.max_query_pts}_{args.use_colmap_ba}",
        )
    sparse_reconstruction_dir.mkdir(parents=True, exist_ok=True)
    reconstruction.write(sparse_reconstruction_dir)
    reconstruction.write_text(sparse_reconstruction_dir)

    # Save point cloud for fast visualization
    trimesh.PointCloud(points_3d, colors=points_rgb).export(sparse_reconstruction_dir / "points.ply")
    
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

        # resize_transform = None
        # resize_transform = transforms.Compose(
        #     [
        #         transforms.Lambda(lambda x: torch.from_numpy(x)),
        #         transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [H,W,C] → [C,H,W]
        #         transforms.Resize(size=(518, 518), antialias=True),  # Expects [C,H,W]
        #     ]
        # )
        # # Transform 2: Convert to float32 and normalize to [0, 1]
        # batch_transform = transforms.Lambda(lambda x: x.type(torch.float32) / 255.0)

        # image_batch = loader.load_image_batch(indices, resize_transform, batch_transform)
        
        image_batch, original_coords = loader.load_and_preprocess_images_square_vggt(indices, img_load_resolution)

        gtsfm_data = run_vggt(image_batch, indices, original_coords)

        self.assertIsNotNone(gtsfm_data)
        # self.assertEqual(gtsfm_data.number_images(), 4)

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
