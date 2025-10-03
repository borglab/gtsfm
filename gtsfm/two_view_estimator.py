"""Estimator which operates on a pair of images to compute relative pose and verified indices.

Authors: Ayush Baid, John Lambert, Zongyue Liu
"""
import dataclasses
import logging
import timeit
from typing import Any, Dict, List, Optional, Tuple
import json
import socket
import time
from datetime import datetime

from dask.distributed import Client
import numpy as np
from gtsam import PinholeCameraCal3Bundler, Pose3, Rot3, SfmTrack, Unit3

import gtsfm.common.types as gtsfm_types
from gtsfm.common.dask_db_module_base import DaskDBModuleBase
import gtsfm.utils.geometry_comparisons as comp_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.metrics as metric_utils
from gtsfm.bundle.two_view_ba import TwoViewBundleAdjustment
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.data_association.point3d_initializer import Point3dInitializer, TriangulationOptions
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.frontend.inlier_support_processor import InlierSupportProcessor
from gtsfm.frontend.verifier.verifier_base import VerifierBase

logger = logger_utils.get_logger()

mpl_logger = logging.getLogger("matplotlib")
mpl_logger.setLevel(logging.WARNING)

pil_logger = logging.getLogger("PIL")
pil_logger.setLevel(logging.INFO)

PRE_BA_REPORT_TAG = "PRE_BA_2VIEW_REPORT"
POST_BA_REPORT_TAG = "POST_BA_2VIEW_REPORT"
POST_ISP_REPORT_TAG = "POST_INLIER_SUPPORT_PROCESSOR_2VIEW_REPORT"
VIEWGRAPH_REPORT_TAG = "VIEWGRAPH_2VIEW_REPORT"

TWO_VIEW_OUTPUT = Tuple[
    Optional[Rot3],
    Optional[Unit3],
    np.ndarray,
    TwoViewEstimationReport,
    TwoViewEstimationReport,
    TwoViewEstimationReport,
]


class TwoViewEstimator(DaskDBModuleBase):
    """Wrapper for running two-view relative pose estimation on image pairs in the dataset."""

    def __init__(
        self,
        verifier: VerifierBase,
        inlier_support_processor: InlierSupportProcessor,
        bundle_adjust_2view: bool,
        eval_threshold_px: float,
        triangulation_options: TriangulationOptions,
        bundle_adjust_2view_maxiters: int = 100,
        ba_reproj_error_thresholds: List[Optional[float]] = [0.5],
        allow_indeterminate_linear_system: bool = False,
        postgres_params=None,
    ) -> None:
        """Initializes the two-view estimator from verifier.

        Args:
            verifier: Verifier to use.
            inlier_support_processor: Post-processor that uses information about RANSAC support to filter out pairs.
            bundle_adjust_2view: Boolean flag indicating if bundle adjustment is to be run on the 2-view data.
            eval_threshold_px: Distance threshold for marking a correspondence pair as inlier during evaluation
                (not during estimation).
            bundle_adjust_2view_maxiters (optional): Max number of iterations for 2-view BA. Defaults to 100.
            ba_reproj_error_thresholds (optional): Reprojection thresholds used to filter features after each stage of
                2-view BA. The length of this list decides the number of BA stages. Defaults to [0.5] (single stage).
            allow_indeterminate_linear_system: Reject a two-view measurement if an indeterminate linear system is
                encountered during marginal covariance computation after 2-view bundle adjustment.
            postgres_params: PostgreSQL connection parameters
        """
        super().__init__(postgres_params=postgres_params)
        self._verifier = verifier
        self.processor = inlier_support_processor
        self._bundle_adjust_2view = bundle_adjust_2view
        self._corr_metric_dist_threshold = eval_threshold_px
        self._triangulation_options = triangulation_options
        self._ba_reproj_error_thresholds = ba_reproj_error_thresholds
        self._bundle_adjust_2view_maxiters = bundle_adjust_2view_maxiters
        self._allow_indeterminate_linear_system = allow_indeterminate_linear_system
        self._ba_optimizer = TwoViewBundleAdjustment(
            reproj_error_thresholds=ba_reproj_error_thresholds,
            robust_measurement_noise=True,
            max_iterations=bundle_adjust_2view_maxiters,
            allow_indeterminate_linear_system=allow_indeterminate_linear_system,
        )
        self.postgres_params = postgres_params  # save connection parameters for use on remote worker
        
        # Initialize database
        self.init_tables()

    def init_tables(self):
        """Initialize database tables for two-view estimation"""
        if not self.db:
            return
        
        if not self._initialize_two_view_schema():
            logger.warning("Failed to initialize two-view database schema")

    def _initialize_two_view_schema(self) -> bool:
        """Initialize two-view estimation database tables"""
        try:
            # Create two-view results table
            if not self.db.execute(self._get_two_view_results_table_ddl()):
                logger.error("Failed to create two_view_results table")
                return False
            
            # Create two-view reports table  
            if not self.db.execute(self._get_two_view_reports_table_ddl()):
                logger.error("Failed to create two_view_reports table")
                return False
            
            logger.info("Two-view database schema initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize two-view database schema: {e}")
            return False
    
    def _get_two_view_results_table_ddl(self) -> str:
        """Get DDL for two_view_results table"""
        return """
        CREATE TABLE IF NOT EXISTS two_view_results (
            id SERIAL PRIMARY KEY,
            i1 INTEGER NOT NULL,
            i2 INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            verified_corr_count INTEGER,
            inlier_ratio FLOAT,
            rotation_matrix TEXT,
            translation_direction TEXT,
            success BOOLEAN NOT NULL,
            computation_time FLOAT,
            worker_name TEXT
        );
        """
    
    def _get_two_view_reports_table_ddl(self) -> str:
        """Get DDL for two_view_reports table"""
        return """
        CREATE TABLE IF NOT EXISTS two_view_reports (
            id SERIAL PRIMARY KEY,
            i1 INTEGER NOT NULL,
            i2 INTEGER NOT NULL,
            timestamp TIMESTAMP NOT NULL,
            pre_ba_inlier_ratio FLOAT,
            post_ba_inlier_ratio FLOAT,
            post_isp_inlier_ratio FLOAT,
            report_data TEXT
        );
        """

    def __repr__(self) -> str:
        return f"""
        TwoViewEstimator:
            Verifier: {self._verifier}
            Bundle adjust 2-view: {self._bundle_adjust_2view}
            Correspondence metric dist. threshold: {self._corr_metric_dist_threshold}
            BA reproj. error thresholds: {self._ba_reproj_error_thresholds}
            BA 2-view max. iters: {self._bundle_adjust_2view_maxiters}
            allow 2-view BA indeterminate linear system: {self._allow_indeterminate_linear_system}
        """

    def triangulate_two_view_correspondences(
        self,
        cameras: Dict[int, gtsfm_types.CAMERA_TYPE],
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        corr_ind: np.ndarray,
    ) -> Tuple[List[int], List[SfmTrack]]:
        """Triangulate 2-view correspondences to form 3D tracks.

        Args:
            camera_i1: Camera for 1st view.
            camera_i2: Camera for 2nd view.
            keypoints_i1: Keypoints for 1st view.
            keypoints_i2: Keypoints for 2nd view.
            corr_ind: Indices of corresponding keypoints.

        Returns:
            Indices of successfully triangulated tracks.
            Triangulated 3D points as tracks.
        """
        assert len(cameras) == 2
        point3d_initializer = Point3dInitializer(cameras, self._triangulation_options)
        triangulated_indices: List[int] = []
        triangulated_tracks: List[SfmTrack] = []
        for j, (idx1, idx2) in enumerate(corr_ind):
            track2d = SfmTrack2d(
                [SfmMeasurement(0, keypoints_i1.coordinates[idx1]), SfmMeasurement(1, keypoints_i2.coordinates[idx2])]
            )
            track, _, _ = point3d_initializer.triangulate(track2d)
            if track is not None:
                triangulated_indices.append(j)
                triangulated_tracks.append(track)

        return triangulated_indices, triangulated_tracks

    def bundle_adjust(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        verified_corr_idxs: np.ndarray,
        camera_intrinsics_i1: gtsfm_types.CALIBRATION_TYPE,
        camera_intrinsics_i2: gtsfm_types.CALIBRATION_TYPE,
        i2Ri1_initial: Optional[Rot3],
        i2Ui1_initial: Optional[Unit3],
        i2Ti1_prior: Optional[PosePrior],
    ) -> Tuple[Optional[Rot3], Optional[Unit3], np.ndarray]:
        """Refine the relative pose using bundle adjustment on the 2-view scene.

        Args:
            keypoints_i1: Keypoints from image i1.
            keypoints_i2: Keypoints from image i2.
            verified_corr_idxs: Indices of verified correspondences between i1 and i2.
            camera_intrinsics_i1: Intrinsics for i1.
            camera_intrinsics_i2: Intrinsics for i2.
            i2Ri1_initial: The relative rotation to be used as initial rotation between cameras.
            i2Ui1_initial: The relative unit direction, to be used to initialize initial translation between cameras.
            i2Ti1_prior: Prior on the relative pose for cameras (i1, i2).

        Returns:
            Optimized relative rotation i2Ri1.
            Optimized unit translation i2Ui1.
            Optimized verified_corr_idxs.
        """
        # Choose initial pose estimate for triangulation and BA (prior gets priority).
        i2Ti1_initial = i2Ti1_prior.value if i2Ti1_prior is not None else None
        if i2Ti1_initial is None and i2Ri1_initial is not None and i2Ui1_initial is not None:
            i2Ti1_initial = Pose3(i2Ri1_initial, i2Ui1_initial.point3())
        if i2Ti1_initial is None:
            return None, None, verified_corr_idxs

        # Set the i1 camera pose as the global coordinate system.
        camera_class = gtsfm_types.get_camera_class_for_calibration(camera_intrinsics_i1)
        cameras = {
            0: camera_class(Pose3(), camera_intrinsics_i1),
            1: camera_class(i2Ti1_initial.inverse(), camera_intrinsics_i2),
        }

        # Triangulate!
        start_time = timeit.default_timer()
        triangulated_indices, triangulated_tracks = self.triangulate_two_view_correspondences(
            cameras, keypoints_i1, keypoints_i2, verified_corr_idxs
        )
        logger.debug("Performed DA in %.6f seconds.", timeit.default_timer() - start_time)
        logger.debug("Triangulated %d correspondences out of %d.", len(triangulated_tracks), len(verified_corr_idxs))

        if len(triangulated_tracks) == 0:
            return i2Ti1_initial.rotation(), Unit3(i2Ti1_initial.translation()), np.zeros(shape=(0, 2), dtype=np.int32)

        # Build BA inputs.
        start_time = timeit.default_timer()
        ba_input = GtsfmData(number_images=2, cameras=cameras, tracks=triangulated_tracks)
        relative_pose_prior_for_ba = {(0, 1): i2Ti1_prior} if i2Ti1_prior is not None else {}

        # Optimize!
        _, ba_output, valid_mask = self._ba_optimizer.run_ba(
            ba_input, absolute_pose_priors=[], relative_pose_priors=relative_pose_prior_for_ba, verbose=False
        )
        if ba_output is None:
            # Indeterminate linear system was met.
            return None, None, np.zeros((0,2), dtype=np.int32)

        # Unpack results.
        valid_corr_idxs = verified_corr_idxs[triangulated_indices][valid_mask]
        wTi1, wTi2 = ba_output.get_camera_poses()  # extract the camera poses
        if wTi1 is None or wTi2 is None:
            logger.warning("2-view BA failed...")
            return i2Ri1_initial, i2Ui1_initial, valid_corr_idxs
        i2Ti1_optimized = wTi2.between(wTi1)
        logger.debug("Performed 2-view BA in %.6f seconds.", timeit.default_timer() - start_time)

        return i2Ti1_optimized.rotation(), Unit3(i2Ti1_optimized.translation()), valid_corr_idxs

    def __get_2view_report_from_results(
        self,
        i2Ri1_computed: Optional[Rot3],
        i2Ui1_computed: Optional[Unit3],
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        verified_corr_idxs: np.ndarray,
        inlier_ratio_wrt_estimate: float,
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
        gt_scene_mesh: Optional[Any],
    ) -> TwoViewEstimationReport:
        """Generate a TwoViewEstimationReport from the results of the two-view estimation.

        Currently metrics wrt GT camera only supports cases where gt_camera is PinholeCameraCal3Bundler.

        Args:
            i2Ri1_computed: Computed relative rotation.
            i2Ui1_computed: Computed relative unit translation.
            keypoints_i1: Keypoints from image i1.
            keypoints_i2: Keypoints from image i2.
            verified_corr_idxs: Indices of verified correspondences between i1 and i2.
            inlier_ratio_wrt_estimate: Inlier ratio w.r.t. the estimated relative pose.
            gt_camera_i1: Ground truth camera for i1.
            gt_camera_i2: Ground truth camera for i2.
            gt_scene_mesh: Ground truth scene mesh.

        Returns:
            TwoViewEstimationReport object, some fields may be None if either gt_camera are None.
        """
        if gt_camera_i1 and gt_camera_i2:
            # if we have the expected GT data, evaluate the computed relative pose
            R_error_deg, U_error_deg = compute_relative_pose_metrics(
                i2Ri1_computed, i2Ui1_computed, gt_camera_i1.pose(), gt_camera_i2.pose()
            )
            # TODO: add support for other camera models
            if isinstance(gt_camera_i1, PinholeCameraCal3Bundler) and isinstance(
                gt_camera_i2, PinholeCameraCal3Bundler
            ):
                inlier_mask_wrt_gt, reproj_error_wrt_gt = metric_utils.compute_correspondence_metrics(
                    keypoints_i1=keypoints_i1,
                    keypoints_i2=keypoints_i2,
                    corr_idxs_i1i2=verified_corr_idxs,
                    dist_threshold=self._corr_metric_dist_threshold,
                    gt_camera_i1=gt_camera_i1,
                    gt_camera_i2=gt_camera_i2,
                    gt_scene_mesh=gt_scene_mesh,
                )
            else:
                inlier_mask_wrt_gt, reproj_error_wrt_gt = None, None
        else:
            R_error_deg, U_error_deg, inlier_mask_wrt_gt, reproj_error_wrt_gt = None, None, None, None

        return generate_two_view_report(
            inlier_ratio_wrt_estimate,
            verified_corr_idxs,
            R_error_deg=R_error_deg,
            U_error_deg=U_error_deg,
            v_corr_idxs_inlier_mask_gt=inlier_mask_wrt_gt,
            reproj_error_gt_model=reproj_error_wrt_gt,
        )

    def get_corr_metric_dist_threshold(self) -> float:
        """Getter for the distance threshold used in the metric for correct correspondences."""
        return self._corr_metric_dist_threshold

    def run_2view(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: Optional[gtsfm_types.CALIBRATION_TYPE],
        camera_intrinsics_i2: Optional[gtsfm_types.CALIBRATION_TYPE],
        i2Ti1_prior: Optional[PosePrior],
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
        gt_scene_mesh: Optional[Any] = None,
        i1: Optional[int] = None,
        i2: Optional[int] = None,
    ) -> TWO_VIEW_OUTPUT:
        """Estimate the relative pose between two images, along with inlier correspondences.

        Args:
            keypoints_i1: Detected keypoints for image i1.
            keypoints_i2: Detected keypoints for image i2.
            putative_corr_idxs: Putative correspondences as indices of keypoint matches.
            camera_intrinsics_i1: Intrinsics for image i1.
            camera_intrinsics_i2: Intrinsics for image i2.
            i2Ti1_prior: Prior on relative pose between two cameras.
            gt_camera_i1: ground truth camera for image i1.
            gt_camera_i2: ground truth camera for image i2.
            gt_scene_mesh: scene mesh for evaluation.
            i1: Image index for first image (for database storage)
            i2: Image index for second image (for database storage)

        Returns:
            Estimated relative rotation, unit translation, verified correspondences, and two-view report.
        """
        
        # Record start time for computation measurement
        start_time = time.time()
        
        # verification on putative correspondences to obtain relative pose and verified correspondences
        (pre_ba_i2Ri1, pre_ba_i2Ui1, pre_ba_v_corr_idxs, pre_ba_inlier_ratio_wrt_estimate) = self._verifier.verify(
            keypoints_i1,
            keypoints_i2,
            putative_corr_idxs,
            camera_intrinsics_i1,
            camera_intrinsics_i2,
        )

        pre_ba_report = self.__get_2view_report_from_results(
            i2Ri1_computed=pre_ba_i2Ri1,
            i2Ui1_computed=pre_ba_i2Ui1,
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            verified_corr_idxs=pre_ba_v_corr_idxs,
            inlier_ratio_wrt_estimate=pre_ba_inlier_ratio_wrt_estimate,
            gt_camera_i1=gt_camera_i1,
            gt_camera_i2=gt_camera_i2,
            gt_scene_mesh=gt_scene_mesh,
        )

        # Optionally, do two-view bundle adjustment
        if self._bundle_adjust_2view and len(pre_ba_v_corr_idxs) >= self.processor._min_num_inliers_est_model:
            post_ba_i2Ri1, post_ba_i2Ui1, post_ba_v_corr_idxs = self.bundle_adjust(
                keypoints_i1,
                keypoints_i2,
                pre_ba_v_corr_idxs,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
                pre_ba_i2Ri1,
                pre_ba_i2Ui1,
                i2Ti1_prior,
            )
            post_ba_inlier_ratio_wrt_estimate = float(len(post_ba_v_corr_idxs)) / len(putative_corr_idxs)

            # TODO: Remove this hack once we can handle the lower post_ba_inlier_ratio_wrt_estimate downstream.
            post_ba_inlier_ratio_wrt_estimate = pre_ba_inlier_ratio_wrt_estimate

            post_ba_report = self.__get_2view_report_from_results(
                i2Ri1_computed=post_ba_i2Ri1,
                i2Ui1_computed=post_ba_i2Ui1,
                keypoints_i1=keypoints_i1,
                keypoints_i2=keypoints_i2,
                verified_corr_idxs=post_ba_v_corr_idxs,
                inlier_ratio_wrt_estimate=post_ba_inlier_ratio_wrt_estimate,
                gt_camera_i1=gt_camera_i1,
                gt_camera_i2=gt_camera_i2,
                gt_scene_mesh=gt_scene_mesh,
            )
        else:
            post_ba_i2Ri1 = pre_ba_i2Ri1
            post_ba_i2Ui1 = pre_ba_i2Ui1
            post_ba_v_corr_idxs = pre_ba_v_corr_idxs
            post_ba_report = dataclasses.replace(pre_ba_report)

        (
            post_isp_i2Ri1,
            post_isp_i2Ui1,
            post_isp_v_corr_idxs,
            post_isp_report,
        ) = self.processor.run_inlier_support(post_ba_i2Ri1, post_ba_i2Ui1, post_ba_v_corr_idxs, post_ba_report)

        # Store computation results in database (pass image indices and start time)
        self.store_computation_results(
            keypoints_i1, keypoints_i2, post_isp_i2Ri1, post_isp_i2Ui1, 
            post_isp_v_corr_idxs, pre_ba_report, post_ba_report, post_isp_report, 
            start_time, i1, i2
        )

        return post_isp_i2Ri1, post_isp_i2Ui1, post_isp_v_corr_idxs, pre_ba_report, post_ba_report, post_isp_report

    def store_computation_results(self, keypoints_i1, keypoints_i2, post_isp_i2Ri1, post_isp_i2Ui1, 
                                  post_isp_v_corr_idxs, pre_ba_report, post_ba_report, post_isp_report, 
                                  start_time, i1=None, i2=None):
        """Store computation results in database
        
        Args:
            keypoints_i1: Keypoints for first image
            keypoints_i2: Keypoints for second image  
            post_isp_i2Ri1: Estimated rotation after ISP
            post_isp_i2Ui1: Estimated translation after ISP
            post_isp_v_corr_idxs: Verified correspondences after ISP
            pre_ba_report: Report before bundle adjustment
            post_ba_report: Report after bundle adjustment  
            post_isp_report: Report after ISP
            start_time: Start time of computation
            i1: Index of first image
            i2: Index of second image
        """
        if not self.db:
            logger.warning(f"No database connection available for pair ({i1}, {i2})")
            return
            
        logger.debug(f"Storing results for image pair ({i1}, {i2})")
        
        # Store main results with image indices
        try:
            self._store_main_results(keypoints_i1, keypoints_i2, post_isp_i2Ri1, post_isp_i2Ui1, 
                                     post_isp_v_corr_idxs, post_isp_report, start_time, i1, i2)
            logger.debug(f"Main results stored successfully for pair ({i1}, {i2})")
        except Exception as e:
            logger.error(f"Failed to store main results for pair ({i1}, {i2}): {e}")
        
        # Store detailed reports with image indices  
        try:
            self._store_detailed_reports(keypoints_i1, keypoints_i2, pre_ba_report, 
                                         post_ba_report, post_isp_report, i1, i2)
            logger.debug(f"Detailed reports stored successfully for pair ({i1}, {i2})")
        except Exception as e:
            logger.error(f"Failed to store detailed reports for pair ({i1}, {i2}): {e}")

    def _store_main_results(self, keypoints_i1, keypoints_i2, post_isp_i2Ri1, post_isp_i2Ui1, 
                            post_isp_v_corr_idxs, post_isp_report, computation_time, i1, i2):
        """Store main computation results in two_view_results table"""
        if not self.db:
            return
            
        logger.debug(f"Storing main results for pair ({i1}, {i2})")
        
        worker_name = socket.gethostname()
        success = (post_isp_i2Ri1 is not None and post_isp_i2Ui1 is not None)
        verified_corr_count = len(post_isp_v_corr_idxs) if post_isp_v_corr_idxs is not None else 0
        
        # Convert numpy scalar to Python float to ensure PostgreSQL compatibility
        # psycopg2 cannot properly serialize numpy.float64, causing "schema np does not exist" errors
        inlier_ratio = post_isp_report.inlier_ratio_est_model if post_isp_report else None
        if inlier_ratio is not None:
            inlier_ratio = float(inlier_ratio)
        
        rotation_matrix = self._serialize_rotation(post_isp_i2Ri1) if post_isp_i2Ri1 else None
        translation_direction = self._serialize_translation(post_isp_i2Ui1) if post_isp_i2Ui1 else None
        
        insert_query = """
            INSERT INTO two_view_results
            (i1, i2, timestamp, verified_corr_count, inlier_ratio, rotation_matrix, 
             translation_direction, success, computation_time, worker_name)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            
        # Ensure all numeric parameters are Python native types (not numpy scalars)
        success = self.db.execute(
            insert_query,
            (int(i1), int(i2), datetime.now(), int(verified_corr_count),
             inlier_ratio, rotation_matrix, translation_direction, success, float(computation_time), worker_name)
        )
        
        if success:
            logger.info(f"Successfully stored results for image pair ({i1}, {i2})")
        else:
            logger.error(f"Failed to store results for image pair ({i1}, {i2})")
            
    def _store_detailed_reports(self, keypoints_i1, keypoints_i2, pre_ba_report, 
                                post_ba_report, post_isp_report, i1, i2):
        """Store detailed reports in two_view_reports table
        
        Args:
            keypoints_i1: Keypoints for first image
            keypoints_i2: Keypoints for second image
            pre_ba_report: Report before bundle adjustment
            post_ba_report: Report after bundle adjustment
            post_isp_report: Report after ISP  
            i1: Index of first image
            i2: Index of second image
        """
        logger.debug(f"Storing detailed reports for pair ({i1}, {i2})")
        
        # Extract inlier ratios and convert numpy scalars to Python floats
        # This prevents PostgreSQL serialization errors with numpy types
        pre_ba_inlier_ratio = pre_ba_report.inlier_ratio_est_model if pre_ba_report else None
        if pre_ba_inlier_ratio is not None:
            pre_ba_inlier_ratio = float(pre_ba_inlier_ratio)
            
        post_ba_inlier_ratio = post_ba_report.inlier_ratio_est_model if post_ba_report else None
        if post_ba_inlier_ratio is not None:
            post_ba_inlier_ratio = float(post_ba_inlier_ratio)
            
        post_isp_inlier_ratio = post_isp_report.inlier_ratio_est_model if post_isp_report else None
        if post_isp_inlier_ratio is not None:
            post_isp_inlier_ratio = float(post_isp_inlier_ratio)
        
        # Serialize report data
        report_data = {
            "pre_ba": self._serialize_report(pre_ba_report),
            "post_ba": self._serialize_report(post_ba_report),
            "post_isp": self._serialize_report(post_isp_report)
        }
        report_data_json = json.dumps(report_data)
            
        # Insert into database
        report_query = """
            INSERT INTO two_view_reports
            (i1, i2, timestamp, pre_ba_inlier_ratio, post_ba_inlier_ratio, post_isp_inlier_ratio, report_data)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            
        # Ensure image indices are Python int type
        success = self.db.execute(
            report_query,
            (int(i1), int(i2), datetime.now(),
             pre_ba_inlier_ratio, post_ba_inlier_ratio, post_isp_inlier_ratio, report_data_json)
        )
        
        if success:
            logger.debug(f"Successfully stored detailed reports for pair ({i1}, {i2})")
        else:
            logger.error(f"Failed to store detailed reports for pair ({i1}, {i2})")

    def _serialize_rotation(self, rotation: Optional[Rot3]) -> Optional[str]:
        """Helper method to serialize rotation matrix"""
        if rotation is None:
            return None
        return self.serialize_matrix(rotation.matrix())
    
    def _serialize_translation(self, translation: Optional[Unit3]) -> Optional[str]:
        """Helper method to serialize translation direction"""
        if translation is None:
            return None
        return self.serialize_matrix(translation.point3())

    def _serialize_report(self, report):
        """Helper method to serialize report objects"""
        if report is None:
            return None
        return self.serialize_matrix(report.__dict__)


def generate_two_view_report(
    inlier_ratio_est_model: float,
    v_corr_idxs: np.ndarray,
    R_error_deg: Optional[float] = None,
    U_error_deg: Optional[float] = None,
    v_corr_idxs_inlier_mask_gt: Optional[np.ndarray] = None,
    reproj_error_gt_model: Optional[np.ndarray] = None,
) -> TwoViewEstimationReport:
    """Wrapper around class constructor for Dask."""
    # Compute ground truth metrics.
    if v_corr_idxs_inlier_mask_gt is not None and reproj_error_gt_model is not None:
        num_inliers_gt_model = np.count_nonzero(v_corr_idxs_inlier_mask_gt)
        inlier_ratio_gt_model = (
            np.count_nonzero(v_corr_idxs_inlier_mask_gt) / v_corr_idxs.shape[0] if len(v_corr_idxs) > 0 else 0.0
        )
        inlier_avg_reproj_error_gt_model = np.mean(reproj_error_gt_model[v_corr_idxs_inlier_mask_gt])
        outlier_avg_reproj_error_gt_model = np.nanmean(
            reproj_error_gt_model[np.logical_not(v_corr_idxs_inlier_mask_gt)]
        )
    else:
        num_inliers_gt_model = 0
        inlier_ratio_gt_model = float("Nan")
        inlier_avg_reproj_error_gt_model = float("Nan")
        outlier_avg_reproj_error_gt_model = float("Nan")

    # Generate report.
    two_view_report = TwoViewEstimationReport(
        inlier_ratio_est_model=inlier_ratio_est_model,
        num_inliers_est_model=v_corr_idxs.shape[0],
        num_inliers_gt_model=num_inliers_gt_model,
        inlier_ratio_gt_model=inlier_ratio_gt_model,
        v_corr_idxs_inlier_mask_gt=v_corr_idxs_inlier_mask_gt,
        v_corr_idxs=v_corr_idxs,
        R_error_deg=R_error_deg,
        U_error_deg=U_error_deg,
        reproj_error_gt_model=reproj_error_gt_model,
        inlier_avg_reproj_error_gt_model=inlier_avg_reproj_error_gt_model,
        outlier_avg_reproj_error_gt_model=outlier_avg_reproj_error_gt_model,
    )
    return two_view_report


def compute_relative_pose_metrics(
    i2Ri1_computed: Optional[Rot3],
    i2Ui1_computed: Optional[Unit3],
    wTi1_expected: Optional[Pose3],
    wTi2_expected: Optional[Pose3],
) -> Tuple[Optional[float], Optional[float]]:
    """Compute the metrics on relative camera pose.

    Args:
        i2Ri1_computed: Computed relative rotation.
        i2Ui1_computed: Computed relative translation direction.
        i2Ti1_expected: Expected relative pose.

    Returns:
        Rotation error, in degrees
        Unit translation error, in degrees
    """
    if wTi1_expected is not None and wTi2_expected is not None:
        i2Ti1_expected = wTi2_expected.between(wTi1_expected)
        R_error_deg = comp_utils.compute_relative_rotation_angle(i2Ri1_computed, i2Ti1_expected.rotation())
        U_error_deg = comp_utils.compute_relative_unit_translation_angle(
            i2Ui1_computed, Unit3(i2Ti1_expected.translation())
        )
    else:
        return (None, None)

    return (R_error_deg, U_error_deg)


def aggregate_frontend_metrics(
    two_view_reports_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    angular_err_threshold_deg: float,
    metric_group_name: str,
) -> GtsfmMetricsGroup:
    """Aggregate the front-end metrics to log summary statistics.

    We define "pose error" as the maximum of the angular errors in rotation and translation, per:
        SuperGlue, CVPR 2020: https://arxiv.org/pdf/1911.11763.pdf
        Learning to find good correspondences. CVPR 2018:
        OA-Net, ICCV 2019:
        NG-RANSAC, ICCV 2019:

    Args:
        two_view_report_dict: Report containing front-end metrics for each image pair.
        angular_err_threshold_deg: Threshold for classifying angular error metrics as success.
        metric_group_name: Name we will assign to the GtsfmMetricGroup returned by this fn.
    """
    num_image_pairs = len(two_view_reports_dict.keys())

    # All rotational errors in degrees.
    rot3_angular_errors_list: List[float] = []
    trans_angular_errors_list: List[float] = []

    inlier_ratio_gt_model_all_pairs = []
    inlier_ratio_est_model_all_pairs = []
    num_inliers_gt_model_all_pairs = []
    num_inliers_est_model_all_pairs = []
    # Populate the distributions.
    for report in two_view_reports_dict.values():
        if report.R_error_deg is not None:
            rot3_angular_errors_list.append(report.R_error_deg)
        if report.U_error_deg is not None:
            trans_angular_errors_list.append(report.U_error_deg)

        inlier_ratio_gt_model_all_pairs.append(report.inlier_ratio_gt_model)
        inlier_ratio_est_model_all_pairs.append(report.inlier_ratio_est_model)
        num_inliers_gt_model_all_pairs.append(report.num_inliers_gt_model)
        num_inliers_est_model_all_pairs.append(report.num_inliers_est_model)

    rot3_angular_errors = np.array(rot3_angular_errors_list, dtype=float)
    trans_angular_errors = np.array(trans_angular_errors_list, dtype=float)
    # Count number of rot3 errors which are not None. Should be same in rot3/unit3.
    num_valid_image_pairs = np.count_nonzero(~np.isnan(rot3_angular_errors))

    # Compute pose errors by picking the max error from rot3 and unit3 errors.
    pose_errors = np.maximum(rot3_angular_errors, trans_angular_errors)

    # Check errors against the threshold.
    success_count_rot3 = np.sum(rot3_angular_errors < angular_err_threshold_deg)
    success_count_unit3 = np.sum(trans_angular_errors < angular_err_threshold_deg)
    success_count_pose = np.sum(pose_errors < angular_err_threshold_deg)

    # Count image pair entries where inlier ratio w.r.t. GT model == 1.
    all_correct = np.count_nonzero(
        [report.inlier_ratio_gt_model == 1.0 for report in two_view_reports_dict.values() if report is not None]
    )

    logger.debug(
        "[Two view optimizer] [Summary] Rotation success: %d/%d/%d",
        success_count_rot3,
        num_valid_image_pairs,
        num_image_pairs,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Translation success: %d/%d/%d",
        success_count_unit3,
        num_valid_image_pairs,
        num_image_pairs,
    )

    logger.debug(
        "[Two view optimizer] [Summary] Pose success: %d/%d/%d",
        success_count_pose,
        num_valid_image_pairs,
        num_image_pairs,
    )

    logger.debug(
        "[Two view optimizer] [Summary] # Image pairs with 100%% inlier ratio:: %d/%d", all_correct, num_image_pairs
    )

    # TODO(akshay-krishnan): Move angular_err_threshold_deg and num_total_image_pairs to metadata.
    frontend_metrics = GtsfmMetricsGroup(
        metric_group_name,
        [
            GtsfmMetric("angular_err_threshold_deg", angular_err_threshold_deg),
            GtsfmMetric("num_input_image_pairs", int(num_image_pairs)),
            GtsfmMetric("num_valid_image_pairs", int(num_valid_image_pairs)),
            GtsfmMetric("rotation_success_count", int(success_count_rot3)),
            GtsfmMetric("translation_success_count", int(success_count_unit3)),
            GtsfmMetric("pose_success_count", int(success_count_pose)),
            GtsfmMetric("num_all_inlier_correspondences_wrt_gt_model", int(all_correct)),
            GtsfmMetric("rot3_angular_errors_deg", rot3_angular_errors),
            GtsfmMetric("trans_angular_errors_deg", trans_angular_errors),
            GtsfmMetric("pose_errors_deg", pose_errors),
            GtsfmMetric("inlier_ratio_wrt_gt_model", inlier_ratio_gt_model_all_pairs),
            GtsfmMetric("inlier_ratio_wrt_est_model", inlier_ratio_est_model_all_pairs),
            GtsfmMetric("num_inliers_est_model", num_inliers_est_model_all_pairs),
            GtsfmMetric("num_inliers_gt_model", num_inliers_gt_model_all_pairs),
        ],
    )
    return frontend_metrics


def run_two_view_estimator_as_futures(
    client: Client,
    two_view_estimator: TwoViewEstimator,
    keypoints_list: List[Keypoints],
    putative_corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    camera_intrinsics: List[gtsfm_types.CALIBRATION_TYPE],
    relative_pose_priors: Dict[Tuple[int, int], PosePrior],
    gt_cameras: List[Optional[gtsfm_types.CAMERA_TYPE]],
    gt_scene_mesh: Optional[Any],
) -> Dict[Tuple[int, int], TWO_VIEW_OUTPUT]:
    """Run two-view estimator for all image pairs."""

    def apply_two_view_estimator(
        two_view_estimator: TwoViewEstimator,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        putative_corr_idxs: np.ndarray,
        camera_intrinsics_i1: gtsfm_types.CALIBRATION_TYPE,
        camera_intrinsics_i2: gtsfm_types.CALIBRATION_TYPE,
        i2Ti1_prior: Optional[PosePrior],
        gt_camera_i1: Optional[gtsfm_types.CAMERA_TYPE],
        gt_camera_i2: Optional[gtsfm_types.CAMERA_TYPE],
        gt_scene_mesh: Optional[Any] = None,
        i1: Optional[int] = None,
        i2: Optional[int] = None,
    ) -> TWO_VIEW_OUTPUT:
        return two_view_estimator.run_2view(
            keypoints_i1=keypoints_i1,
            keypoints_i2=keypoints_i2,
            putative_corr_idxs=putative_corr_idxs,
            camera_intrinsics_i1=camera_intrinsics_i1,
            camera_intrinsics_i2=camera_intrinsics_i2,
            i2Ti1_prior=i2Ti1_prior,
            gt_camera_i1=gt_camera_i1,
            gt_camera_i2=gt_camera_i2,
            gt_scene_mesh=gt_scene_mesh,
            i1=i1,
            i2=i2,
        )

    print("Distributing TwoViewEstimator to all workers...")
    
    try:
        two_view_estimator_future = client.scatter(two_view_estimator, broadcast=False)
        
        import time
        time.sleep(2)
        
        print("TwoViewEstimator distributed successfully")
        
    except Exception as e:
        print(f"Failed to scatter TwoViewEstimator: {e}")
        two_view_estimator_future = two_view_estimator

    # Submit tasks with image indices passed as separate parameters
    two_view_output_futures = {
        (i1, i2): client.submit(
            apply_two_view_estimator,
            two_view_estimator_future,
            keypoints_list[i1],
            keypoints_list[i2],
            putative_corr_idxs,
            camera_intrinsics[i1],
            camera_intrinsics[i2],
            relative_pose_priors.get((i1, i2)),
            gt_cameras[i1] if gt_cameras else None,
            gt_cameras[i2] if gt_cameras else None,
            gt_scene_mesh,
            i1,
            i2,
        )
        for (i1, i2), putative_corr_idxs in putative_corr_idxs_dict.items()
    }

    print(f"Submitted {len(two_view_output_futures)} tasks to workers")
    
    try:
        two_view_output_dict = client.gather(two_view_output_futures)
        return two_view_output_dict
    except Exception as e:
        print(f"Error during gather: {e}")
        two_view_output_dict = {}
        for (i1, i2), future in two_view_output_futures.items():
            try:
                result = future.result(timeout=300)  
                two_view_output_dict[(i1, i2)] = result
                print(f"Successfully processed pair ({i1}, {i2})")
            except Exception as pair_error:
                print(f"Failed to process pair ({i1}, {i2}): {pair_error}")
                continue
        
        return two_view_output_dict




def get_two_view_reports_summary(
    two_view_report_dict: Dict[Tuple[int, int], TwoViewEstimationReport],
    images: List[Image],
) -> List[Dict[str, Any]]:
    """Converts the TwoViewEstimationReports to a summary dict for each image pair.

    Args:
        two_view_report_dict: Front-end metrics for pairs of images.
        images: List of all images for this scene, in order of image/frame index.

    Returns:
        List of dictionaries, where each dictionary contains the metrics for an image pair.
    """

    def round_fn(x: Optional[float]) -> Optional[float]:
        return round(x, 2) if x else None

    metrics_list = []

    for (i1, i2), report in two_view_report_dict.items():
        # Note: if GT is unknown, then R_error_deg, U_error_deg, and inlier_ratio_gt_model will be None
        metrics_list.append(
            {
                "i1": int(i1),
                "i2": int(i2),
                "i1_filename": images[i1].file_name,
                "i2_filename": images[i2].file_name,
                "rotation_angular_error": round_fn(report.R_error_deg),
                "translation_angular_error": round_fn(report.U_error_deg),
                "num_inliers_gt_model": int(report.num_inliers_gt_model)
                if report.num_inliers_gt_model is not None
                else None,
                "inlier_ratio_gt_model": round_fn(report.inlier_ratio_gt_model),
                "inlier_avg_reproj_error_gt_model": round_fn(
                    np.nanmean(report.reproj_error_gt_model[report.v_corr_idxs_inlier_mask_gt])
                )
                if report.reproj_error_gt_model is not None and report.v_corr_idxs_inlier_mask_gt is not None
                else None,
                "outlier_avg_reproj_error_gt_model": round_fn(
                    np.nanmean(report.reproj_error_gt_model[np.logical_not(report.v_corr_idxs_inlier_mask_gt)])
                )
                if report.reproj_error_gt_model is not None and report.v_corr_idxs_inlier_mask_gt is not None
                else None,
                "inlier_ratio_est_model": round_fn(report.inlier_ratio_est_model),
                "num_inliers_est_model": int(report.num_inliers_est_model)
                if report.num_inliers_est_model is not None
                else None,
            }
        )
    return metrics_list
