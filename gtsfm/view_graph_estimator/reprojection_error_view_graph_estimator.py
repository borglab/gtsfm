"""
In the spirit of "Growing Consensus" (Son16cvpr)
https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Son_Solving_Small-Piece_Jigsaw_CVPR_2016_paper.pdf

Authors: John Lambert
"""

import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple

import dask
import matplotlib.pyplot as plt
import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Pose3, Rot3, Unit3


import gtsfm.densify.mvs_utils as mvs_utils
import gtsfm.multi_view_optimizer as multi_view_optimizer
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.graph as graph_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metrics_utils
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.keypoints import Keypoints
from gtsfm.data_association.data_assoc import DataAssociation
from gtsfm.data_association.point3d_initializer import TriangulationOptions, TriangulationSamplingMode
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.two_view_estimator import TwoViewEstimationReport
from gtsfm.view_graph_estimator.view_graph_estimator_base import ViewGraphEstimatorBase


logger = logger_utils.get_logger()


MIN_ALLOWED_TRI_ANGLE_DEG = 2
MIN_ALLOWED_SUPPORT = 220
# simulated (from global) vs. local measurement discrepancy
MAX_ALLOWED_ROT_DISCREPANCY_DEG = 5
MAX_ALLOWED_TRANS_DISCREPANCY_DEG = 5

MAX_ALLOWED_ROT_CYCLE_ERROR_DEG = 5
SUPPORT_CUTOFF_REPROJ_THRESHOLD_PX = 20


@dataclass
class TripletOptimizationReport:
    """
    """
    triplet: Tuple[int,int,int]
    is_success: bool = False
    min_tri_angle: float = np.nan
    support: int = np.nan
    max_rot_discrepancy_deg: float = np.nan
    max_trans_discrepancy_deg: float = np.nan
    max_rot_error: float = np.nan
    max_trans_error: float = np.nan

    @property
    def i0(self) -> int:
        return triplet[0]

    @property
    def i1(self) -> int:
        return triplet[1]

    @property
    def i2(self) -> int:
        return triplet[2]


class ThreeViewRegistrationMethod(str, Enum):
    """Two supported modes for 3-view registration:
    1. PNP (a la incremental SfM) or
    2. Global rotation averaging + translation averaging.
         w/ no BA, and immediately check compare pose differences (or reprojection errors).

    Bundle adjustment is not necessary, as it will not stray from the TA solution.
    """

    PNP: str = "PNP"
    AVERAGING_NO_BA: str = "AVERAGING_NO_BA"


class ReprojectionErrorViewGraphEstimator(ViewGraphEstimatorBase):
    """Triangulates and bundle adjusts points in 3-views and uses their reprojection error to reason about outliers.

    Alternatively, could use triangulation + PNP.
    """

    def __init__(
        self, registration_method: ThreeViewRegistrationMethod = ThreeViewRegistrationMethod.AVERAGING_NO_BA
    ) -> None:
        """ """
        self._registration_method = registration_method
        self._rot_avg_module = ShonanRotationAveraging()
        self._trans_avg_module = TranslationAveraging1DSFM(robust_measurement_noise=True, perform_outlier_rejection=False)

        # TODO: could limit to length 3 tracks.
        self._data_association_module = DataAssociation(
            min_track_len=3,
            triangulation_options=TriangulationOptions(
                reproj_error_threshold=50000,
                mode=TriangulationSamplingMode.NO_RANSAC,
                max_num_hypotheses=100,
            ),
            save_track_patches_viz=False,
        )

    def run(
        self,
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        cameras_gt: Optional[List[PinholeCameraCal3Bundler]] = None,
        images = None,
    ) -> Set[Tuple[int, int]]:
        """Estimates the view graph using the rotation consistency constraint in a cycle of 3 edges.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints: keypoints for each images.
            two_view_reports: Dict from (i1, i2) to the TwoViewEstimationReport of the edge.
            cameras_gt:

        Returns:
            Edges of the view-graph, which are the subset of the image pairs in the input args.
        """
        
        triplets_graph = self.estimate_triplets(i2Ri1_dict=i2Ri1_dict)

        # TODO: cannot loop over triplets to parallelize this, because we do not know its size (single Delayed object).
        triplet_reports_dict = {}
        for triplet_idx, triplet in enumerate(triplets_graph):  # sort order guaranteed
            print(f"On Triplet {triplet_idx}/{len(triplets_graph)}") 
            triplet_reports_dict[triplet] = self.process_triplet(
                triplet=triplet,
                i2Ri1_dict=i2Ri1_dict,
                i2Ui1_dict=i2Ui1_dict,
                calibrations=calibrations,
                corr_idxs_i1i2=corr_idxs_i1i2,
                keypoints=keypoints,
                two_view_reports=two_view_reports,
                cameras_gt=cameras_gt,
                images=images
            )

        self.__save_plots(triplet_reports_dict)
        valid_edges = self.aggregate_valid_edges(triplet_reports_dict)
        return valid_edges


    def estimate_triplets(self, i2Ri1_dict: Dict[Tuple[int, int], Rot3]) -> List[Tuple[int,int,int]]:
        """ """
        logger.info("Input number of edges: %d" % len(i2Ri1_dict))
        input_edges: List[Tuple[int, int]] = i2Ri1_dict.keys()
        triplets: List[Tuple[int, int, int]] = graph_utils.extract_cyclic_triplets_from_edges(input_edges)

        N = len(triplets)
        logger.info("Number of triplets: %d" % N)

        return triplets


    def aggregate_valid_edges(self, triplet_reports_dict: Dict[Tuple[int,int,int], TripletOptimizationReport]):
        """
        """
        valid_edges = set()
        for (i0,i1,i2), report in triplet_reports_dict.items():
            if not report.is_success:
                continue
            valid_edges.add((i0, i1))
            valid_edges.add((i1, i2))
            valid_edges.add((i0, i2))

        return valid_edges


    def process_triplet(
        self,
        triplet: Tuple[int,int,int],
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1i2: Dict[Tuple[int, int], np.ndarray],
        keypoints: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        cameras_gt: Optional[List[PinholeCameraCal3Bundler]] = None,
        images = None,
    ) -> TripletOptimizationReport:
        """ """
        if triplet is None:
            return TripletOptimizationReport(triplet=triplet)
        i0, i1, i2 = triplet

        # immediately reject bad cycles
        cycle_error = comp_utils.compute_cyclic_rotation_error(
            i1Ri0=i2Ri1_dict[(i0, i1)], i2Ri1=i2Ri1_dict[(i1, i2)], i2Ri0=i2Ri1_dict[(i0, i2)]
        )
        if cycle_error > MAX_ALLOWED_ROT_CYCLE_ERROR_DEG:
            logger.info(f"Immediately rejected for cycle error {cycle_error:.1f} deg.")
            return TripletOptimizationReport(triplet=triplet)

        i2Ri1_dict_subscene, i2Ui1_dict_subscene, corr_idxs_i1_i2_subscene, _ = self._filter_with_edges(
            i2Ri1_dict=i2Ri1_dict,
            i2Ui1_dict=i2Ui1_dict,
            corr_idxs_i1i2=corr_idxs_i1i2,
            two_view_reports=two_view_reports,
            edges_to_select=[(i0, i1), (i1, i2), (i0, i2)],
        )
        if self._registration_method == ThreeViewRegistrationMethod.AVERAGING_NO_BA:
            logger.setLevel(logging.WARNING)
            start = time.time()
            triplet_report = self.optimize_three_views_averaging(
                triplet=(i0, i1, i2),
                i2Ri1_dict=i2Ri1_dict_subscene,
                i2Ui1_dict=i2Ui1_dict_subscene,
                calibrations=calibrations,
                corr_idxs_i1_i2=corr_idxs_i1_i2_subscene,
                keypoints_list=keypoints,
                two_view_reports=two_view_reports,
                cameras_gt=cameras_gt,
                images=images
            )
            end = time.time()
            duration = end - start
            logger.info(f"Took {duration:.1f} sec to optimize triplet ({i0},{i1},{i2})")
            logger.setLevel(logging.INFO)

        elif self._registration_method == ThreeViewRegistrationMethod.PNP:
            self.optimize_three_views_incremental()

        # form 3 edges e_i, e_j, e_k between fully connected subgraph (nodes i0,i1,i2)
        edges = [(i0, i1), (i1, i2), (i0, i2)]
        rot_errors = [two_view_reports[e].R_error_deg for e in edges]
        trans_errors = [two_view_reports[e].U_error_deg for e in edges]
        gt_known = all([err is not None for err in rot_errors])
        # if ground truth unknown, cannot estimate error w.r.t. GT
        triplet_report.max_rot_error = max(rot_errors) if gt_known else None
        triplet_report.max_trans_error = max(trans_errors) if gt_known else None

        return triplet_report


    def __save_plots(self, triplet_reports_dict: Dict[Tuple[int,int,int], TripletOptimizationReport]) -> None:
        """Save information about proxy error metric vs. GT error metric."""

        support_per_triplet = []
        reproj_error_per_triplet = []
        max_gt_rot_error_in_cycle = []
        max_gt_trans_error_in_cycle = []
        min_tri_angle_per_triplet = []
        max_rot_discrepancy_per_triplet = []
        max_trans_discrepancy_per_triplet = []
        is_success_per_triplet = []

        for triplet_report in triplet_reports_dict.values():
            min_tri_angle_per_triplet.append(triplet_report.min_tri_angle)
            support_per_triplet.append(triplet_report.support)
            max_rot_discrepancy_per_triplet.append(triplet_report.max_rot_discrepancy_deg)
            max_trans_discrepancy_per_triplet.append(triplet_report.max_trans_discrepancy_deg)
            is_success_per_triplet.append(triplet_report.is_success)
            
            max_gt_rot_error_in_cycle.append(triplet_report.max_rot_error)
            max_gt_trans_error_in_cycle.append(triplet_report.max_trans_error)

        pose_errors = np.maximum(np.array(max_gt_rot_error_in_cycle), np.array(max_gt_trans_error_in_cycle))

        xlabels = ["Support (#Inliers)", "Max Rot Discrepancy (deg)", "Max Trans Discrepancy (deg)", "Min Tri. Angle"]
        fnames = ["gt_error_vs_support.jpg", "gt_error_vs_max_rot_discrep.jpg", "gt_error_vs_max_trans_discrep.jpg", "gt_error_vs_mintriangle.jpg"]
        proxy_metrics = [
            np.array(support_per_triplet),
            np.array(max_rot_discrepancy_per_triplet),
            np.array(max_trans_discrepancy_per_triplet),
            np.array(min_tri_angle_per_triplet),
        ]

        inliers = np.array(is_success_per_triplet)

        for xlabel, fname, proxy_metric in zip(xlabels, fnames, proxy_metrics):
            plt.scatter(
                proxy_metric[inliers],
                pose_errors[inliers],
                10,
                color="g",
                marker=".",
                label="inlier by success criteria",
            )
            plt.scatter(
                proxy_metric[~inliers],
                pose_errors[~inliers],
                10,
                color="r",
                marker=".",
                label="outlier by success criteria",
            )
            plt.xlabel(xlabel)
            plt.ylabel("GT Pose Angular Error (deg.)")
            plt.legend(loc="upper right")
            plt.savefig(os.path.join("plots", fname), dpi=500)
            plt.close("all")

    def optimize_three_views_averaging(
        self,
        triplet: Tuple[int,int,int],
        i2Ri1_dict: Dict[Tuple[int, int], Rot3],
        i2Ui1_dict: Dict[Tuple[int, int], Unit3],
        calibrations: List[Cal3Bundler],
        corr_idxs_i1_i2: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
        two_view_reports: Dict[Tuple[int, int], TwoViewEstimationReport],
        cameras_gt: Optional[List[PinholeCameraCal3Bundler]] = None,
        images = None,
        visualize: bool = False
    ) -> Tuple[List[Optional[Pose3]], np.ndarray, Optional[GtsfmMetricsGroup], GtsfmMetricsGroup, GtsfmMetricsGroup]:
        """Use 3-view averaging to estimate global camera poses, and then compute per-point reprojection errors.

        Args:
            i2Ri1_dict: Dict from (i1, i2) to relative rotation of i1 with respect to i2.
            i2Ui1_dict: Dict from (i1, i2) to relative translation direction of i1 with respect to i2.
            calibrations: list of calibrations for each image.
            corr_idxs_i1i2: Dict from (i1, i2) to indices of verified correspondences from i1 to i2.
            keypoints_list: keypoints for each images.
            cameras_gt: cameras with GT intrinsics and GT extrinsics.

        Returns:
            wTi_list: estimated camera poses.
            reproj_errors: reprojection errors.
            ra_metrics: 3-view rotation averaging metrics.
            ta_metrics: 3-view translation averaging metrics.
            ba_metrics: 3-view bundle adjustment metrics.
        """
        if cameras_gt is not None:
            gt_wTi_list = [cam.pose() for cam in cameras_gt]
            num_images = len(cameras_gt)
        else:
            gt_wTi_list = None
            # num_images is arbitrary, as long as it is larger than the largest key
            num_images = max([max(i1, i2) for (i1, i2) in i2Ri1_dict.keys()]) + 1

        wRi_list = self._rot_avg_module.run(num_images, i2Ri1_dict)

        for (i1,i2) in i2Ri1_dict.keys():
            if wRi_list[i1] is None or wRi_list[i2] is None:
                print("Shonan missing an optimized pose!")
                return self._triplet_failure_result

        wti_list, ta_metrics = self._trans_avg_module.run(num_images, i2Ui1_dict, wRi_list, gt_wTi_list=gt_wTi_list)

        if gt_wTi_list:
            ra_metrics = metrics_utils.compute_global_rotation_metrics(wRi_list, wti_list, gt_wTi_list)
        else:
            ra_metrics = None

        init_cameras_dict = multi_view_optimizer.init_cameras(wRi_list, wti_list, calibrations)
        # `final_data` is the RA+TA+DA output.
        final_data, _ = self._data_association_module.run(
            num_images=num_images,
            cameras=init_cameras_dict,
            corr_idxs_dict=corr_idxs_i1_i2,
            keypoints_list=keypoints_list,
            images=images,
            cameras_gt=cameras_gt,
        )

        if final_data is None:
            return TripletOptimizationReport(triplet=triplet)

        reproj_errors = final_data.get_scene_reprojection_errors()
        wTi_list = final_data.get_camera_poses()

        for (i1,i2) in i2Ri1_dict.keys():
            if wTi_list[i1] is None or wTi_list[i2] is None:
                print("Missing an optimized pose!")
                return TripletOptimizationReport(triplet=triplet)

        if visualize:
            point_cloud = final_data.get_point_cloud()
            rgb = np.zeros_like(point_cloud).astype(np.uint8)
            args = SimpleNamespace(**{"point_rendering_mode": "point", "frustum_ray_len": 0.1, "sphere_radius": 0.1})
            import gtsfm.visualization.open3d_vis_utils as open3d_vis_utils
            open3d_vis_utils.draw_scene_open3d(point_cloud, rgb, wTi_list, calibrations, args)

        ta_change_metrics = metrics_utils.compute_translation_angle_metric(i2Ui1_dict, wTi_list)
        relative_rot_discrepancies = compute_relative_rotation_metric(i2Ri1_dict, wTi_list)

        if images is not None and visualize:
            for (i1,i2) in i2Ri1_dict.keys():
                import gtsfm.utils.viz as viz_utils
                two_view_report = SimpleNamespace(**{"v_corr_idxs_inlier_mask_gt": None})
                viz_utils.save_twoview_correspondences_viz(
                    images[i1],
                    images[i2],
                    keypoints_list[i1],
                    keypoints_list[i2],
                    corr_idxs_i1_i2[(i1,i2)],
                    two_view_report=two_view_report,
                    file_path=os.path.join("trifocal_correspondences", f"{i1}_{i2}.jpg"),
                )

        support = (reproj_errors < SUPPORT_CUTOFF_REPROJ_THRESHOLD_PX).sum()
        relative_trans_discrepancies = np.round(ta_change_metrics._data)


        # check how much each pose, in terms of translation directon and rotation angle

        tri_angle_percentiles = []
        assert len(i2Ri1_dict.keys()) == 3
        for (i1, i2) in i2Ri1_dict.keys():
            if final_data.get_camera(i1) is None or final_data.get_camera(i2) is None:
                tri_angle_percentiles.append(0)
                continue
            # TODO: try w/ filtered and unfiltered points.
            # Ref: https://github.com/colmap/colmap/blob/dev/src/sfm/incremental_mapper.cc#L1065
            tri_angles = mvs_utils.calculate_triangulation_angles_in_degrees(
                camera_1=final_data.get_camera(i1),
                camera_2=final_data.get_camera(i2),
                points_3d=final_data.get_point_cloud(),
            )
            repr_percentile = np.percentile(tri_angles, 75)
            tri_angle_percentiles.append(repr_percentile)

        # also check entropy of these histograms above
        min_tri_angle = min(tri_angle_percentiles)
        max_rot_discrepancy = max(relative_rot_discrepancies)
        max_trans_discrepancy = max(relative_trans_discrepancies)

        is_failure = min_tri_angle < MIN_ALLOWED_TRI_ANGLE_DEG \
                     or support < MIN_ALLOWED_SUPPORT \
                     or max_rot_discrepancy > MAX_ALLOWED_ROT_DISCREPANCY_DEG \
                     or max_trans_discrepancy > MAX_ALLOWED_TRANS_DISCREPANCY_DEG
        is_success = not is_failure
        if is_failure:
            print("REJECT!")
        else:
            print("ACCEPT!")

        print("\tSupport: ", support)
        print("\tRotation discrepancies: ", np.round(relative_rot_discrepancies))
        print("\tTA Change metrics: ", relative_trans_discrepancies)
        #print("Rotation euler angle norms: ", np.round(rotation_changes))

        print("\tRA Errors w.r.t. GT: ", ra_metrics._metrics[1].name, np.round(ra_metrics._metrics[1].data))
        print("\tTA Errors wr.r.t GT: ", ta_metrics._metrics[8].name, np.round(ta_metrics._metrics[8].data))
        print(f"\tMin Tri Angle: {min_tri_angle:.1f}")

        triplet_opt_report = TripletOptimizationReport(
            triplet=triplet,
            is_success=is_success,
            min_tri_angle=min_tri_angle,
            support=support,
            max_rot_discrepancy_deg=max_rot_discrepancy,
            max_trans_discrepancy_deg=max_trans_discrepancy
        )
        return triplet_opt_report

    def optimize_three_views_incremental(self) -> None:
        """
        Estimate global camera poses for 3-views using PNP / absolute pose estimation.
        """
        import pycolmap

        reconstructions = pycolmap.incremental_mapping(
            database_path, image_dir, models_path, num_threads=min(multiprocessing.cpu_count(), 16)
        )


def compute_relative_rotation_metric(i2Ri1_dict: Dict[Tuple[int, int], Rot3], wTi_list: List[Pose3]) -> List[float]:
    """ """
    errors = []
    import gtsfm.utils.geometry_comparisons as comp_utils
    for (i1,i2), i2Ri1 in i2Ri1_dict.items():
        i2Ri1_simulated = wTi_list[i2].between(wTi_list[i1]).rotation()
        error_deg = comp_utils.compute_relative_rotation_angle(i2Ri1, i2Ri1_simulated)
        errors.append(error_deg)
    return errors





