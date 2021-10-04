"""Inference methods for GtsfmData in PatchmatchNet

Authors: Ren Liu
"""
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.densify.mvs_base import MVSBase
import gtsfm.densify.mvs_utils as mvs_utils
from gtsfm.densify.patchmatchnet_data import PatchmatchNetData
from gtsfm.utils import logger as logger_utils
from thirdparty.patchmatchnet.models.net import PatchmatchNet
import thirdparty.patchmatchnet.utils as patchmatchnet_utils
import thirdparty.patchmatchnet.eval as patchmatchnet_eval_funcs

logger = logger_utils.get_logger()

torch.backends.cudnn.benchmark = True
PATCHMATCHNET_WEIGHTS_PATH = "thirdparty/patchmatchnet/checkpoints/model_000007.ckpt"

# all default values are assigned by Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/eval.py
DEFAULT_BATCH_SIZE = 1
DEFAULT_VIEW_NUMBER = 5
DEFAULT_WORKERS_NUMBER = 4

DEFAULT_GEOMETRIC_PIXEL_THRESH = 1.0
DEFAULT_GEOMETRIC_DEPTH_THRESH = 0.01
DEFAULT_PHOTOMETRIC_THRESH = 0.8

DEFAULT_INTERVAL_SCALE = [0.005, 0.0125, 0.025]
DEFAULT_PROPAGATION_RANGE = [6, 4, 2]
DEFAULT_ITERATION_NUMBER = [1, 2, 2]
DEFAULT_SAMPLE_NUMBER = [8, 8, 16]
DEFAULT_PROPAGATE_NEIGHBORS = [0, 8, 16]
DEFAULT_EVALUATE_NEIGHBORS = [9, 9, 9]

DEFAULT_MINIMUM_CONSISTENT_VIEW_NUMBER = 3


class MVSPatchmatchNet(MVSBase):
    """Inference methods for GtsfmData in densification using PatchmatchNet."""

    def densify(
        self,
        images: Dict[int, Image],
        sfm_result: GtsfmData,
        max_num_views: int = DEFAULT_VIEW_NUMBER,
        thresholds: List[float] = [
            DEFAULT_GEOMETRIC_PIXEL_THRESH,
            DEFAULT_GEOMETRIC_DEPTH_THRESH,
            DEFAULT_PHOTOMETRIC_THRESH,
        ],
    ) -> np.ndarray:
        """Get dense point cloud using PatchmatchNet from GtsfmData. The method implements the densify method in MVSBase
        Ref: Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/eval.py

        Args:
            images: image dictionary obtained from loaders
            sfm_result: result of GTSFM after bundle adjustment
            num_views: Defaults to DEFAULT_VIEW_NUMBER. maximum number of views,
                containing 1 reference view and (num_views-1) source views
            thresholds: geometric pixel threshold, geometric depth threshold, and photometric
                threshold for filtering inference results.
                Defaults to [DEFAULT_GEOMETRIC_PIXEL_THRESH, DEFAULT_GEOMETRIC_DEPTH_THRESH, DEFAULT_PHOTOMETRIC_THRESH]
                1. for geometric thresholds, small threshold means high accuracy and low completeness
                2. for photometric thresholds, large threshold means high accuracy and low completeness

        Returns:
            3D coordinates (in the world frame) of the dense point cloud, (point number, 3)
        """
        dataset = PatchmatchNetData(images=images, sfm_result=sfm_result, max_num_views=max_num_views)
        loader = DataLoader(
            dataset=dataset,
            batch_size=DEFAULT_BATCH_SIZE,
            shuffle=False,
            num_workers=DEFAULT_WORKERS_NUMBER,
            drop_last=False,
        )

        model = PatchmatchNet(
            patchmatch_interval_scale=DEFAULT_INTERVAL_SCALE,
            propagation_range=DEFAULT_PROPAGATION_RANGE,
            patchmatch_iteration=DEFAULT_ITERATION_NUMBER,
            patchmatch_num_sample=DEFAULT_SAMPLE_NUMBER,
            propagate_neighbors=DEFAULT_PROPAGATE_NEIGHBORS,
            evaluate_neighbors=DEFAULT_EVALUATE_NEIGHBORS,
        )
        model = nn.DataParallel(model)

        # Check if cuda devices is supported, and load the pretrained model
        #   the pretrained checkpoint should be pre-downloaded using gtsfm/download_model_weights.sh
        if torch.cuda.is_available():
            model.cuda()
            state_dict = torch.load(PATCHMATCHNET_WEIGHTS_PATH)
        else:
            state_dict = torch.load(PATCHMATCHNET_WEIGHTS_PATH, map_location=torch.device("cpu"))

        model.load_state_dict(state_dict["model"])
        model.eval()

        depth_est_list = {}
        confidence_est_list = {}
        with torch.no_grad():
            for batch_idx, sample in enumerate(loader):
                start_time = time.time()

                ids = sample["idx"]

                # Check if cuda devices is supported, and store the inference data to the target device
                if torch.cuda.is_available():
                    sample_device = patchmatchnet_utils.tocuda(sample)
                else:
                    sample_device = sample

                # Inference using PatchmatchNet
                outputs = model(
                    sample_device["imgs"],
                    sample_device["proj_matrices"],
                    sample_device["depth_min"],
                    sample_device["depth_max"],
                )

                outputs = patchmatchnet_utils.tensor2numpy(outputs)
                del sample_device

                # Save depth maps and confidence maps
                for idx, depth_est, photometric_confidence in zip(
                    ids, outputs["refined_depth"]["stage_0"], outputs["photometric_confidence"]
                ):

                    idx = idx.cpu().numpy().tolist()
                    depth_est_list[idx] = depth_est.copy()
                    confidence_est_list[idx] = photometric_confidence.copy()

                logger.info(
                    "[Densify::PatchMatchNet] Iter %d/%d, time = %.3f",
                    batch_idx + 1,
                    len(loader),
                    time.time() - start_time,
                )

        # Filter inference result with thresholds
        dense_point_cloud = self.filter_depth(
            dataset=dataset,
            depth_list=depth_est_list,
            confidence_list=confidence_est_list,
            geo_pixel_thresh=thresholds[0],
            geo_depth_thresh=thresholds[1],
            photo_thresh=thresholds[2],
        )

        return dense_point_cloud

    def filter_depth(
        self,
        dataset: PatchmatchNetData,
        depth_list: Dict[int, np.ndarray],
        confidence_list: Dict[int, np.ndarray],
        geo_pixel_thresh: float,
        geo_depth_thresh: float,
        photo_thresh: float,
    ) -> np.ndarray:
        """Create a dense point cloud by filtering depth maps based on estimated confidence maps and consistent geometry

        A 3D point is consistent in geometry between two views if:
            1. the location distance between the original pixel in one view and the corresponding pixel
                reprojected from the other view is less than geo_pixel_thresh;
            2. the distance between the estimated depth in one view and its reprojected depth from the other view
                is less than geo_depth_thresh.

        A 3D point is consistent in geometry in the output point cloud if it is consistent in geometry between the
            reference view and more than DEFAULT_MINIMUM_CONSISTENT_VIEW_NUMBER source views.


        Ref: Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/eval.py

        Args:
            dataset: an instance of PatchmatchData as the inference dataset
            depth_list: list of 2D depth map (H, W) from each view
            confidence_list: list of 2D confidence map (H, W) from each view
            geo_pixel_thresh: geometric pixel threshold
            geo_depth_thresh: geometric depth threshold
            photo_thresh: photometric threshold

        Returns:
            3D coordinates (in the world frame) of the dense point cloud, (point number, 3)
        """
        # coordinates of the final point cloud
        vertices = []
        # vertex colors of the final point cloud, used in generating colored mesh
        vertex_colors = []

        packed_pairs = dataset.get_packed_pairs()

        # For each reference view and the corresponding source views
        for pair in packed_pairs:
            ref_view = pair["ref_id"]
            src_views = pair["src_ids"]

            # Load the camera parameters
            ref_intrinsics, ref_extrinsics = dataset.get_camera_params(ref_view)

            # Load the reference image
            ref_img = dataset.get_image(ref_view)
            # Load the estimated depth of the reference view
            ref_depth_est = depth_list[ref_view][0]
            # Load the photometric mask of the reference view
            confidence = confidence_list[ref_view]
            # Filter the pixels that satisfy photometric consistancy among reference view and source views,
            #   by checking whether the confidence is larger than the pre-defined photometric threshold
            photo_mask = confidence > photo_thresh

            all_srcview_depth_ests = []

            # Compute the geometric mask, the value of geo_mask_sum means the number of source views where
            #   the reference depth is valid according to the geometric thresholds
            geo_mask_sum = 0
            for src_view in src_views:
                # camera parameters of the source view
                src_intrinsics, src_extrinsics = dataset.get_camera_params(src_view)

                # the estimated depth of the source view
                src_depth_est = depth_list[src_view][0]

                # Check geometric consistency
                geo_mask, depth_reprojected, _, _ = patchmatchnet_eval_funcs.check_geometric_consistency(
                    ref_depth_est,
                    ref_intrinsics,
                    ref_extrinsics,
                    src_depth_est,
                    src_intrinsics,
                    src_extrinsics,
                    geo_pixel_thresh,
                    geo_depth_thresh,
                )
                geo_mask_sum += geo_mask.astype(np.int32)
                all_srcview_depth_ests.append(depth_reprojected)

            depth_est_averaged = (sum(all_srcview_depth_ests) + ref_depth_est) / (geo_mask_sum + 1)
            # Valid points requires at least 3 source views validated under geometric threshoulds
            geo_mask = geo_mask_sum >= DEFAULT_MINIMUM_CONSISTENT_VIEW_NUMBER

            # Combine geometric mask and photometric mask
            final_mask = np.logical_and(photo_mask, geo_mask)

            # Initialize coordinate grids
            height, width = depth_est_averaged.shape[:2]
            u, v = np.meshgrid(np.arange(0, width), np.arange(0, height))

            # Get valid points filtered by photometric and geometric thresholds
            valid_points = final_mask
            u, v, depth = u[valid_points], v[valid_points], depth_est_averaged[valid_points]

            # Get the point coordinates inside the reference view's camera frame
            xyz_ref = np.linalg.inv(ref_intrinsics) @ mvs_utils.cart_to_homogenous(np.array([u, v])) * depth

            # Get the point coordinates inside the world frame
            xyz_world = (np.linalg.inv(ref_extrinsics) @ mvs_utils.cart_to_homogenous(xyz_ref))[:3]
            vertices.append(xyz_world.T)

            # Get the point colors for colored mesh
            color = ref_img[valid_points]
            vertex_colors.append((color * 255).astype(np.uint8))

            logger.info(
                "[Densify::PatchMatchNet] RefView: %03d Geometric: %.03f Photometric: %.03f Final: %.03f",
                ref_view,
                geo_mask.mean(),
                photo_mask.mean(),
                final_mask.mean(),
            )

        return np.concatenate(vertices, axis=0)
