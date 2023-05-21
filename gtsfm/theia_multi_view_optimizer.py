"""Optimizer which performs averaging and bundle adjustment on all images in the scene.

Authors: Ayush Baid
"""
from typing import Any, Dict, List, Optional, Tuple

import dask
import numpy as np
import os
from dask.delayed import Delayed
from pathlib import Path
from gtsam import Pose3, Cal3Bundler, Rot3, Unit3, PinholeCameraCal3Bundler, SfmTrack
import pytheia as pt

import gtsfm.common.types as gtsfm_types
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d, SfmMeasurement
from gtsfm.evaluation.metrics import GtsfmMetricsGroup
from gtsfm.common.two_view_estimation_report import TwoViewEstimationReport
from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.bundle.bundle_adjustment import evaluate as evaluate_ba
from gtsfm.data_association.data_assoc import DataAssociation


class TheiaMultiViewOptimizer:
    def __init__(self, data_association_module: DataAssociation) -> None:
        self._output_reproj_error_thresh = 3
        self._da = data_association_module

    def run(
        self,
        images: List[Image],
        num_images: int,
        keypoints: List[Keypoints],
        i2Ri1: Dict[Tuple[int, int], Rot3],
        i2Ui1: Dict[Tuple[int, int], Unit3],
        v_corr_idxs: Dict[Tuple[int, int], np.ndarray],
        all_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        two_views_reports_dict: Optional[Dict[Tuple[int, int], TwoViewEstimationReport]],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_wTi_list: List[Optional[Pose3]],
        output_root: Optional[Path] = None,
    ) -> Tuple[GtsfmData, GtsfmData, Dict[Any, Any], List[GtsfmMetricsGroup]]:
        # create debug directory
        debug_output_dir = None
        if output_root:
            debug_output_dir = output_root / "debug"
            os.makedirs(debug_output_dir, exist_ok=True)

        recon = init_recon(num_images, camera_intrinsics=all_intrinsics)
        assert two_views_reports_dict is not None
        view_graph = init_viewgraph(two_views_reports_dict, recon)
        tracks2d = get_2d_tracks(v_corr_idxs, keypoints, recon)

        cameras_for_da = {i: camera for i, camera in enumerate(cameras_gt) if camera is not None}

        # Run DA for metrics
        sfm_tracks, avg_track_reprojection_errors, triangulation_exit_codes = self._da.run_triangulation(
            cameras_for_da, tracks2d
        )
        _, da_metrics = self._da.run_da(
            num_images,
            cameras=cameras_for_da,
            tracks_2d=tracks2d,
            sfm_tracks=sfm_tracks,
            avg_track_reproj_errors=avg_track_reprojection_errors,
            triangulation_exit_codes=triangulation_exit_codes,
            cameras_gt=cameras_gt,
            relative_pose_priors={},
            images=images,
        )

        pre_ba_gtsfm_data = to_gtsfm_data(recon, num_images=num_images, use_unestimated=True)

        # Run the global reconstruction
        options = pt.sfm.ReconstructionEstimatorOptions()
        options.num_threads = 7
        options.rotation_filtering_max_difference_degrees = 10.0
        options.bundle_adjustment_robust_loss_width = 3.0
        options.bundle_adjustment_loss_function_type = pt.sfm.LossFunctionType(1)
        options.subsample_tracks_for_bundle_adjustment = True
        options.filter_relative_translations_with_1dsfm = True

        reconstruction_estimator = pt.sfm.GlobalReconstructionEstimator(options)
        reconstruction_summary = reconstruction_estimator.Estimate(view_graph, recon)

        print("Reconstruction summary message: {}".format(reconstruction_summary.message))

        post_ba_gtsfm_data_unfiltered = to_gtsfm_data(recon, num_images=num_images)
        print(
            f"Num tracks: {post_ba_gtsfm_data_unfiltered.number_tracks()}, Num cameras: {post_ba_gtsfm_data_unfiltered.number_images()}"
        )
        post_ba_gtsfm_data_filtered, _ = post_ba_gtsfm_data_unfiltered.filter_landmarks(
            self._output_reproj_error_thresh
        )

        ba_metrics = evaluate_ba(post_ba_gtsfm_data_unfiltered, post_ba_gtsfm_data_filtered, cameras_gt)

        return (
            pre_ba_gtsfm_data,
            post_ba_gtsfm_data_filtered,
            two_views_reports_dict,
            [da_metrics, ba_metrics],
        )

    def create_computation_graph(
        self,
        images_graph: List[Delayed],
        num_images: int,
        keypoints_graph: List[Keypoints],
        i2Ri1_graph: Dict[Tuple[int, int], Delayed],
        i2Ui1_graph: Dict[Tuple[int, int], Delayed],
        v_corr_idxs_graph: Dict[Tuple[int, int], Delayed],
        all_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        absolute_pose_priors: List[Optional[PosePrior]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        two_view_reports_dict: Optional[Dict[Tuple[int, int], Delayed]],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        gt_wTi_list: List[Optional[Pose3]],
        output_root: Optional[Path] = None,
    ) -> Tuple[Delayed, Delayed, Delayed, list]:
        """Creates a computation graph for multi-view optimization.

        Args:
            num_images: number of images in the scene.
            keypoints_graph: keypoints for images, each wrapped up as Delayed.
            i2Ri1_graph: relative rotations for image pairs, each value wrapped up as Delayed.
            i2Ui1_graph: relative unit-translations for image pairs, each value wrapped up as Delayed.
            v_corr_idxs_graph: indices of verified correspondences for image pairs, wrapped up as Delayed.
            all_intrinsics: intrinsics for images.
            absolute_pose_priors: priors on the camera poses (not delayed).
            relative_pose_priors: priors on the pose between camera pairs (not delayed)
            two_view_reports_dict: Dict of TwoViewEstimationReports after inlier support processor.
            cameras_gt: list of GT cameras (if they exist), ordered by camera index.
            gt_wTi_list: list of GT poses of the camera.
            output_root: path where output should be saved.

        Returns:
            The GtsfmData input to bundle adjustment, aligned to GT (if provided), wrapped up as Delayed.
            The final output GtsfmData, wrapped up as Delayed.
            Dict of TwoViewEstimationReports after view graph estimation.
            List of GtsfmMetricGroups from different modules, wrapped up as Delayed.
        """

        ba_input, ba_output, two_view_report_post_view_graph_estimation, delayed_metrics = dask.delayed(
            self.run, nout=4
        )(
            images_graph,
            num_images,
            keypoints_graph,
            i2Ri1_graph,
            i2Ui1_graph,
            v_corr_idxs_graph,
            all_intrinsics,
            absolute_pose_priors,
            relative_pose_priors,
            two_view_reports_dict,
            cameras_gt,
            gt_wTi_list,
            output_root,
        )

        metrics = [delayed_metrics[idx] for idx in range(2)]

        return ba_input, ba_output, two_view_report_post_view_graph_estimation, metrics


def init_viewgraph(
    two_view_report_dict: Dict[Tuple[int, int], TwoViewEstimationReport], recon: pt.sfm.Reconstruction
) -> pt.sfm.ViewGraph:
    view_graph = pt.sfm.ViewGraph()

    for (i1, i2), two_view_report in two_view_report_dict.items():
        view_id1 = recon.ViewIdFromName(f"{i1}")
        view_id2 = recon.ViewIdFromName(f"{i2}")

        two_view_info = pt.sfm.TwoViewInfo()

        if two_view_report.theia_twoview_info is None:
            continue

        two_view_info.focal_length_1 = two_view_report.theia_twoview_info.focal_length_1
        two_view_info.focal_length_2 = two_view_report.theia_twoview_info.focal_length_2
        two_view_info.position_2 = two_view_report.theia_twoview_info.position_2
        two_view_info.rotation_2 = two_view_report.theia_twoview_info.rotation_2
        two_view_info.num_verified_matches = two_view_report.theia_twoview_info.num_verified_matches
        two_view_info.num_homography_inliers = two_view_report.theia_twoview_info.num_homography_inliers
        two_view_info.visibility_score = two_view_report.theia_twoview_info.visibility_score

        view_graph.AddEdge(view_id2, view_id1, two_view_info)

    return view_graph


def init_recon(
    num_images: int, camera_intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]]
) -> pt.sfm.Reconstruction:
    recon = pt.sfm.Reconstruction()

    gtsfm_intrinsics = camera_intrinsics[0]
    assert gtsfm_intrinsics is not None
    prior = to_theia_intrinsics_prior(gtsfm_intrinsics)
    camera = pt.sfm.Camera(pt.sfm.CameraIntrinsicsModelType(1))  # pinhole radial tangential camera
    camera.SetFromCameraIntrinsicsPriors(prior)

    # add all images using a dummy name
    for i in range(num_images):
        image_name = f"{i}"
        view_id = recon.AddView(image_name, 0, i)  # fixing all images to be from the same camera group
        print(f"Added image {i} with view id {view_id}")

        # TODO(Ayush): maybe this disables shared cameras?
        c = recon.MutableView(view_id).MutableCamera()
        c.DeepCopy(camera)
        recon.MutableView(view_id).SetCameraIntrinsicsPrior(prior)

    return recon


def to_gtsfm_data(recon: pt.sfm.Reconstruction, num_images: int, use_unestimated: bool = False) -> GtsfmData:
    view_ids: List[int] = recon.ViewIds()
    gtsfm_data = GtsfmData(number_images=num_images)

    # Extract cameras
    for view_id in view_ids:
        view: pt.sfm.View = recon.View(view_id)
        if not use_unestimated and not view.IsEstimated():
            continue
        i = int(view.Name())

        camera: pt.sfm.Camera = view.Camera()
        wTi = Pose3(Rot3(camera.GetOrientationAsRotationMatrix()).inverse(), camera.GetPosition())
        intrinsics = Cal3Bundler(
            fx=camera.FocalLength(),
            k1=0,
            k2=0,
            u0=camera.CameraIntrinsics().PrincipalPointX(),
            v0=camera.CameraIntrinsics().PrincipalPointY(),
        )
        gtsfm_camera = PinholeCameraCal3Bundler(wTi, intrinsics)

        gtsfm_data.add_camera(i, gtsfm_camera)

    # Extract tracks
    track_ids: List[int] = recon.TrackIds()

    for track_id in track_ids:
        track: pt.sfm.Track = recon.Track(track_id)
        if True:  # use_unestimated or (track.IsEstimated() and track.NumViews() >= 2):
            track_2d = to_gtsfm_track(track_id=track_id, recon=recon)

            point_4d: np.ndarray = track.Point()
            gtsam_track = SfmTrack(point_4d[:3].reshape(3, 1) / (point_4d[3]))

            if gtsam_track is not None:
                for i, uv in track_2d.measurements:
                    gtsam_track.addMeasurement(i, uv)
                gtsfm_data.add_track(gtsam_track)
        else:
            pass
            # recon.RemoveTrack(track_id)

    return gtsfm_data


def to_theia_intrinsics_prior(intrinsics: Optional[gtsfm_types.CALIBRATION_TYPE]) -> pt.sfm.CameraIntrinsicsPrior:
    assert isinstance(intrinsics, Cal3Bundler)

    prior = pt.sfm.CameraIntrinsicsPrior()
    prior.focal_length.value = [intrinsics.fx()]
    prior.aspect_ratio.value = [1.0]  # [intrinsics.fy() / intrinsics.fx()]
    prior.principal_point.value = [intrinsics.px(), intrinsics.py()]
    prior.radial_distortion.value = [intrinsics.k1(), intrinsics.k2(), 0, 0]
    prior.tangential_distortion.value = [0, 0]
    prior.skew.value = [0]
    # TODO: unfix this
    # prior.image_width = int(760)
    # prior.image_height = int(1135)
    # 'PINHOLE_RADIAL_TANGENTIAL', 'DIVISION_UNDISTORTION', 'DOUBLE_SPHERE', 'FOV', 'EXTENDED_UNIFIED', 'FISHEYE
    prior.camera_intrinsics_model_type = "PINHOLE_RADIAL_TANGENTIAL"

    return prior


def to_gtsfm_track(track_id, recon: pt.sfm.Reconstruction) -> SfmTrack2d:
    track: pt.sfm.Track = recon.Track(track_id)
    view_ids = track.ViewIds()
    measurements: List[SfmMeasurement] = []
    for view_id in view_ids:
        view = recon.View(view_id)
        feature_coordinates = view.GetFeature(track_id).point
        i = int(view.Name())
        measurements.append(SfmMeasurement(i, feature_coordinates))

    return SfmTrack2d(measurements)


def get_2d_tracks(
    corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
    keypoints_list: List[Keypoints],
    recon: pt.sfm.Reconstruction,
) -> List[SfmTrack2d]:
    track_builder = pt.sfm.TrackBuilder(3, 30)

    for (i1, i2), v_corr_idxs in corr_idxs_dict.items():
        keypoints_i1 = keypoints_list[i1]
        keypoints_i2 = keypoints_list[i2]

        for keypoint_idx_i1, keypoint_idx_i2 in v_corr_idxs:
            feature_i1 = pt.sfm.Feature(keypoints_i1.coordinates[keypoint_idx_i1])
            feature_i2 = pt.sfm.Feature(keypoints_i2.coordinates[keypoint_idx_i2])

            track_builder.AddFeatureCorrespondence(
                recon.ViewIdFromName(f"{i1}"), feature_i1, recon.ViewIdFromName(f"{i2}"), feature_i2
            )

    track_builder.BuildTracks(recon)

    track_ids = recon.TrackIds()

    gtsfm_tracks: List[SfmTrack2d] = []
    for id in track_ids:
        gtsfm_track = to_gtsfm_track(track_id=id, recon=recon)

        gtsfm_tracks.append(gtsfm_track)

    return gtsfm_tracks
