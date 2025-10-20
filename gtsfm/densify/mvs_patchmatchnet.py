"""Inference methods for GtsfmData in PatchmatchNet

Authors: Ren Liu
"""

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import gtsfm.densify.mvs_utils as mvs_utils
import thirdparty.patchmatchnet.eval as patchmatchnet_eval
import thirdparty.patchmatchnet.utils as patchmatchnet_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.metrics_sink import MetricsSink
from gtsfm.densify.mvs_base import MVSBase
from gtsfm.densify.patchmatchnet_data import PatchmatchNetData
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.utils import logger as logger_utils
from thirdparty.patchmatchnet.models.net import PatchmatchNet

METRICS_GROUP = "multi_view_stereo"

logger = logger_utils.get_logger()

torch.backends.cudnn.benchmark = True
PATCHMATCHNET_WEIGHTS_PATH = (
    Path(__file__).resolve().parent.parent.parent / "thirdparty" / "patchmatchnet" / "checkpoints" / "model_000007.ckpt"
)

# all default values are assigned by Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/eval.py
BATCH_SIZE = 1
NUM_VIEWS = 5

# the reprojection error in pixel coordinates should be less than 1
MAX_GEOMETRIC_PIXEL_THRESH = 1.0
# the reprojection error along the camera's z(depth) axis should be less than 0.01
MAX_GEOMETRIC_DEPTH_THRESH = 0.01
MIN_CONFIDENCE_THRESH = 0.8

INTERVAL_SCALE = [0.005, 0.0125, 0.025]
PROPAGATION_RANGE = [6, 4, 2]
NUM_ITERS = [1, 2, 2]
NUM_SAMPLES = [8, 8, 16]
PROPAGATE_NEIGHBORS = [0, 8, 16]
EVALUATE_NEIGHBORS = [9, 9, 9]

# a reconstructed point is consistent in geometry if it satisfies all geometric thresholds in more than 1 source views
MIN_NUM_CONSISTENT_VIEWS = 1


class MVSPatchmatchNet(MVSBase):
    """Inference methods for GtsfmData in densification using PatchmatchNet.
    PyTorch DataLoader is used to load batched data for PatchmatchNet.
    """

    def densify(
        self,
        images: Dict[int, Image],
        sfm_result: GtsfmData,
        metrics_sink: Optional[MetricsSink] = None,
        **kwargs,
    ) -> Tuple[np.ndarray, np.ndarray]:
        return self._densify(images, sfm_result, metrics_sink=metrics_sink, **kwargs)

    def _densify(
        self,
        images: Dict[int, Image],
        sfm_result: GtsfmData,
        max_num_views: int = NUM_VIEWS,
        max_geo_pixel_thresh: float = MAX_GEOMETRIC_PIXEL_THRESH,
        max_geo_depth_thresh: float = MAX_GEOMETRIC_DEPTH_THRESH,
        min_conf_thresh: float = MIN_CONFIDENCE_THRESH,
        min_num_consistent_views: float = MIN_NUM_CONSISTENT_VIEWS,
        num_workers: int = 0,
        metrics_sink: Optional[MetricsSink] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get dense point cloud using PatchmatchNet from GtsfmData. The method implements the densify method in MVSBase
        Ref: Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/eval.py

        Args:
            images: image dictionary obtained from loaders
            sfm_result: result of GTSFM after bundle adjustment
            max_num_views: maximum number of views, containing 1 reference view and (num_views-1) source views
            max_geo_pixel_thresh: maximum reprojection error in pixel coordinates,
                small threshold means high accuracy and low completeness
            max_geo_depth_thresh: maximum reprojection error in depth from camera,
                small threshold means high accuracy and low completeness
            min_conf_thresh: minimum confidence required for a valid point,
                large threshold means high accuracy and low completeness
            min_num_consistent_views: a reconstructed point is consistent in geometry if it satisfies all geometric
                thresholds in more than min_num_consistent_views source views
            num_workers: number of workers when loading data

        Returns:
            dense_point_cloud: 3D coordinates (in the world frame) of the dense point cloud
                with shape (N, 3) where N is the number of points
            dense_point_colors: RGB color of each point in the dense point cloud
                with shape (N, 3) where N is the number of points
        """
        dataset = PatchmatchNetData(images=images, sfm_result=sfm_result, max_num_views=max_num_views)

        # TODO(johnwlambert): using Dask's LocalCluster with multiprocessing in Pytorch (i.e. num_workers>0)
        # will give -> "AssertionError('daemonic processes are not allowed to have children')" -> fix needed
        if num_workers != 0:
            raise ValueError("Using multiprocessing in Pytorch within Dask's LocalCluster is currently unsupported.")

        loader = DataLoader(
            dataset=dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            drop_last=False,
        )

        patchmatch_model = PatchmatchNet(
            patchmatch_interval_scale=INTERVAL_SCALE,
            propagation_range=PROPAGATION_RANGE,
            patchmatch_iteration=NUM_ITERS,
            patchmatch_num_sample=NUM_SAMPLES,
            propagate_neighbors=PROPAGATE_NEIGHBORS,
            evaluate_neighbors=EVALUATE_NEIGHBORS,
        )
        model = nn.DataParallel(patchmatch_model)

        # Check if cuda devices are available, and load the pretrained model
        #   the pretrained checkpoint should be pre-downloaded using gtsfm/download_model_weights.sh
        if not PATCHMATCHNET_WEIGHTS_PATH.exists():
            raise FileNotFoundError(
                f"PatchmatchNet weights not found at {PATCHMATCHNET_WEIGHTS_PATH}. "
                "Please run 'bash download_model_weights.sh' from the repo root."
            )

        logger.info("⏳ Loading PatchMatchNet model weights...")
        if torch.cuda.is_available():
            model.cuda()
            state_dict = torch.load(PATCHMATCHNET_WEIGHTS_PATH)
        else:
            state_dict = torch.load(PATCHMATCHNET_WEIGHTS_PATH, map_location=torch.device("cpu"))

        model.load_state_dict(state_dict["model"])
        model.eval()

        depth_est_list = {}
        confidence_est_list = {}

        batch_times: list[float] = []

        logger.info("Starting PatchMatchNet inference...")
        with torch.no_grad():
            for batch_idx, sample in enumerate(loader):
                start_time = time.time()

                pm_ids = sample["idx"]

                # Check if cuda devices are available
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

                # Save depth maps and confidence maps
                for pm_i, depth_est, photometric_confidence in zip(
                    pm_ids, outputs["refined_depth"]["stage_0"], outputs["photometric_confidence"]
                ):
                    pm_i = pm_i.cpu().numpy().tolist()
                    depth_est_list[pm_i] = depth_est.copy()
                    confidence_est_list[pm_i] = photometric_confidence.copy()

                time_elapsed = time.time() - start_time
                batch_times.append(time_elapsed)

                logger.debug(
                    "[Densify::PatchMatchNet] Iter %d/%d, time = %.3f",
                    batch_idx + 1,
                    len(loader),
                    time_elapsed,
                )

        # Filter inference result with thresholds
        dense_point_cloud, dense_point_colors, filtering_metrics = self.filter_depth(
            dataset=dataset,
            depth_list=depth_est_list,
            confidence_list=confidence_est_list,
            max_geo_pixel_thresh=max_geo_pixel_thresh,
            max_geo_depth_thresh=max_geo_depth_thresh,
            min_conf_thresh=min_conf_thresh,
            min_num_consistent_views=min_num_consistent_views,
        )

        # Initialize densify metrics, add elapsed time per batch to the metric list
        densify_metrics = GtsfmMetricsGroup(
            name=METRICS_GROUP,
            metrics=[
                GtsfmMetric(name="num_valid_reference_views", data=len(loader)),
                GtsfmMetric(name="elapsed_time_per_ref_img(sec)", data=np.array(batch_times)),
            ],
        )
        # merge filtering metrics to densify metrics
        densify_metrics.extend(filtering_metrics)

        if metrics_sink is not None:
            metrics_sink.record(densify_metrics)

        return dense_point_cloud, dense_point_colors

    def filter_depth(
        self,
        dataset: PatchmatchNetData,
        depth_list: Dict[int, np.ndarray],
        confidence_list: Dict[int, np.ndarray],
        max_geo_pixel_thresh: float,
        max_geo_depth_thresh: float,
        min_conf_thresh: float,
        min_num_consistent_views: float,
    ) -> Tuple[np.ndarray, np.ndarray, GtsfmMetricsGroup]:
        """Create a dense point cloud by filtering depth maps based on estimated confidence maps and consistent geometry

        A 3D point is consistent in geometry between two views if:
            1. the location distance between the original pixel in one view and the corresponding pixel
                reprojected from the other view is less than max_geo_pixel_thresh;
            2. the distance between the estimated depth in one view and its reprojected depth from the other view
                is less than max_geo_depth_thresh.

        A 3D point is consistent in geometry in the output point cloud if it is consistent in geometry between the
            reference view and more than MINIMUM_CONSISTENT_VIEW_NUMBER source views.

        Reference: Wang et al., https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/eval.py#L227
        Note: we rename the photometric threshold as a "confidence threshold".

        Args:
            dataset: an instance of PatchmatchData as the inference dataset
            depth_list: list of batched 2D depth maps (1, H, W) from each view
            confidence_list: list of 2D confidence maps (H, W) from each view
            max_geo_pixel_thresh: maximum reprojection error in pixel coordinates
            max_geo_depth_thresh: maximum reprojection error in depth from camera
            min_conf_thresh: minimum confidence required for a valid point
            min_num_consistent_views: a reconstructed point is consistent in geometry if it satisfies all geometric
                thresholds in more than min_num_consistent_views source views

        Returns:
            dense_points: 3D coordinates (in the world frame) of the dense point cloud
                with shape (N, 3) where N is the number of points
            dense_point_colors: RGB color of each point in the dense point cloud
                with shape (N, 3) where N is the number of points
            filter_metrics: Metrics for dense reconstruction while filtering points from depth maps
        """
        # coordinates of the final point cloud
        vertices = []
        # vertex colors of the final point cloud, used in generating colored mesh
        vertex_colors = []
        # depth maps from each view
        depths = []

        # record valid ratio of each kind of masks among images while filtering points from depth maps
        geo_mask_ratios = []
        conf_mask_ratios = []
        joint_mask_ratios = []

        reprojection_errors = []

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
            # Load the confidence mask of the reference view
            confidence = confidence_list[ref_view]
            # Filter the pixels that have enough confidence among reference view and source views,
            #   by checking whether the confidence is larger than the pre-defined confidence threshold
            confidence_mask = confidence > min_conf_thresh

            all_src_view_depth_estimates = []

            # Compute the geometric mask, the value of geo_mask_sum means the number of source views where
            #   the reference depth is valid according to the geometric thresholds
            geo_mask_sum = 0
            for src_view in src_views:
                # camera parameters of the source view
                src_intrinsics, src_extrinsics = dataset.get_camera_params(src_view)

                # the estimated depth of the source view
                src_depth_est = depth_list[src_view][0]

                # Check geometric consistency
                geo_mask, depth_reprojected, _, _ = patchmatchnet_eval.check_geometric_consistency(
                    ref_depth_est,
                    ref_intrinsics,
                    ref_extrinsics,
                    src_depth_est,
                    src_intrinsics,
                    src_extrinsics,
                    max_geo_pixel_thresh,
                    max_geo_depth_thresh,
                )
                geo_mask_sum += geo_mask.astype(np.int32)
                all_src_view_depth_estimates.append(depth_reprojected)

            depth_est_averaged = (sum(all_src_view_depth_estimates) + ref_depth_est) / (geo_mask_sum + 1)
            # Valid points requires at least 3 source views validated under geometric thresholds
            geo_mask = geo_mask_sum >= min_num_consistent_views

            # Combine geometric mask and confidence mask
            joint_mask = np.logical_and(confidence_mask, geo_mask)

            # Compute and record the reprojection errors
            reprojection_errors.extend(
                compute_filtered_reprojection_error(
                    dataset=dataset,
                    ref_view=ref_view,
                    src_views=src_views,
                    depth_list=depth_list,
                    max_reprojection_err=max_geo_pixel_thresh,
                    joint_mask=joint_mask,
                )
            )
            # Set the depths of invalid positions to 0
            depth_est_averaged[np.logical_not(joint_mask)] = 0
            # Append the depth map to the depth map list
            depths.append(depth_est_averaged)

            # Initialize coordinate grids
            height, width = depth_est_averaged.shape[:2]
            u, v = np.meshgrid(np.arange(0, width), np.arange(0, height))

            # Get valid points filtered by confidence and geometric thresholds
            valid_points = joint_mask
            u, v, depth = u[valid_points], v[valid_points], depth_est_averaged[valid_points]

            # Get the point coordinates inside the reference view's camera frame
            itj = np.linalg.inv(ref_intrinsics) @ mvs_utils.cart_to_homogenous(np.array([u, v])) * depth

            # Get the point coordinates inside the world frame
            wtj = (np.linalg.inv(ref_extrinsics) @ mvs_utils.cart_to_homogenous(itj))[:3]
            vertices.append(wtj.T)

            # Get the point colors for colored mesh
            color = ref_img[valid_points]
            vertex_colors.append((color * 255).astype(np.uint8))

            geo_mask_ratios.append(geo_mask.mean())
            conf_mask_ratios.append(confidence_mask.mean())
            joint_mask_ratios.append(joint_mask.mean())

            logger.debug(
                "[Densify::PatchMatchNet] RefView: %03d Geometric: %.03f Confidence: %.03f Joint: %.03f",
                ref_view,
                geo_mask.mean(),
                confidence_mask.mean(),
                joint_mask.mean(),
            )

        dense_points = np.concatenate(vertices, axis=0)
        dense_point_colors = np.concatenate(vertex_colors, axis=0)

        # compute and collect metrics during filtering points from depth maps
        filtering_metrics = []
        # compute the proportion of valid pixels in the geometric masks among all reference views
        filtering_metrics.append(GtsfmMetric(name="geometric_mask_valid_ratios", data=geo_mask_ratios))
        # compute the proportion of valid pixels in the confidence masks among all reference views
        filtering_metrics.append(GtsfmMetric(name="confidence_mask_valid_ratios", data=conf_mask_ratios))
        # compute the proportion of valid pixels in the joint masks among all reference views
        filtering_metrics.append(GtsfmMetric(name="joint_mask_valid_ratios", data=joint_mask_ratios))
        filtering_metrics.append(GtsfmMetric(name="reprojection_errors", data=np.array(reprojection_errors)))

        return dense_points, dense_point_colors, GtsfmMetricsGroup(name="filtering metrics", metrics=filtering_metrics)


def compute_filtered_reprojection_error(
    dataset: PatchmatchNetData,
    ref_view: int,
    src_views: List[int],
    depth_list: Dict[int, np.ndarray],
    max_reprojection_err: float,
    joint_mask: np.ndarray,
) -> List[float]:
    """Compute reprojection errors of reference view pixels among all source views, filtered by joint mask
    Detailed steps include:
        1. Compute coordinates of each pixel in reference depth map at reference camera frame by ref_intrinsics.
        2. Project these points to a source view by ref_extrinsics, src_extrinsics and src intrinsics.
        3. Use the source depth map to calculate the depths of reprojected pixel by interpolation.
        4. Compute coordinates of each reprojected pixel at source camera frame by src_intrinsics.
        5. Re-project these points to the reference view by src_extrinsics, ref_extrinsics, and ref intrinsics.
        6. Compute the reprojection errors by Euclidean distance and filter the errors by the joint mask.

    Args:
        dataset: an instance of PatchmatchData as the inference dataset
        ref_view: reference view pm_i
        src_views: list of source view pm_i
        depth_list: list of batched 2D depth maps (1, H, W) from each reference view
        max_reprojection_err: maximum reprojection error in pixels
        joint_mask: the union set of geometric mask and confidence mask, in shape of (H, W)

    Returns:
        list of filtered reprojection errors among all source views in the reference view
    """
    # initialize reprojection err list
    reprojection_errors = []

    # get reference view estimated depth map
    ref_depth_est = depth_list[ref_view][0]
    # get the resolution of reference view depth map
    height, width = ref_depth_est.shape[:2]
    # get camera parameters of the reference view
    ref_intrinsics, ref_extrinsics = dataset.get_camera_params(ref_view)

    # Compute reprojection error after filtering by joint mask
    # 1. generate reference coordinates for each pixel
    u_ref, v_ref = np.meshgrid(np.arange(0, width), np.arange(0, height))
    # 2. compute reprojection errors from each reference-source view pair
    for src_view in src_views:
        # camera parameters of the source view
        src_intrinsics, src_extrinsics = dataset.get_camera_params(src_view)

        # the estimated depth of the source view
        src_depth_est = depth_list[src_view][0]

        # compute reprojected coordinates
        (
            _,
            u_reprojected,
            v_reprojected,
            _,
            _,
        ) = patchmatchnet_eval.reproject_with_depth(
            ref_depth_est,
            ref_intrinsics,
            ref_extrinsics,
            src_depth_est,
            src_intrinsics,
            src_extrinsics,
        )

        # compute reprojection error
        reprojection_error = np.hypot((u_reprojected - u_ref), (v_reprojected - v_ref))[joint_mask]
        # record reprojection errors
        reprojection_errors.extend(reprojection_error[reprojection_error < max_reprojection_err].tolist())
    return reprojection_errors
