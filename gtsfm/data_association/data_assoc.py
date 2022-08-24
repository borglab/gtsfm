""" Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Estimates 3D landmark for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

References: 
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2,
   November, pp. 146–157, 1997

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert
"""
import os
from collections import Counter
from typing import Dict, List, NamedTuple, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import SfmTrack

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.data_association.dsf_tracks_estimator import DsfTracksEstimator
from gtsfm.data_association.point3d_initializer import Point3dInitializer, TriangulationExitCode, TriangulationOptions
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup

logger = logger_utils.get_logger()


class DataAssociation(NamedTuple):
    """Class to form feature tracks; for each track, call LandmarkInitializer.

    Args:
        min_track_len: min length required for valid feature track / min nb of supporting views required for a landmark
                       to be valid.
        triangulation_options: options for triangulating points.
        save_track_patches_viz: whether to save a mosaic of individual patches associated with each track.
    """

    min_track_len: int
    triangulation_options: TriangulationOptions
    save_track_patches_viz: Optional[bool] = False

    def __validate_track(self, sfm_track: Optional[SfmTrack]) -> bool:
        """Validate the track by checking its length."""
        return sfm_track is not None and sfm_track.numberMeasurements() >= self.min_track_len

    def run(
        self,
        num_images: int,
        cameras: Dict[int, gtsfm_types.CAMERA_TYPE],
        corr_idxs_dict: Dict[Tuple[int, int], np.ndarray],
        keypoints_list: List[Keypoints],
        cameras_gt: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        relative_pose_priors: Dict[Tuple[int, int], Optional[PosePrior]],
        images: Optional[List[Image]] = None,
    ) -> Tuple[GtsfmData, GtsfmMetricsGroup]:
        """Perform the data association.

        Args:
            num_images: Number of images in the scene.
            cameras: dictionary, with image index -> camera mapping.
            corr_idxs_dict: dictionary, with key as image pair (i1,i2) and value as matching keypoint indices.
            keypoints_list: keypoints for each image.
            cameras_gt: list of GT cameras, to be used for benchmarking the tracks.
            images: a list of all images in scene (optional and only for track patch visualization)

        Returns:
            A tuple of GtsfmData with cameras and tracks, and a GtsfmMetricsGroup with data association metrics
        """
        # generate tracks for 3D points using pairwise correspondences
        tracks_estimator = DsfTracksEstimator()
        tracks_2d = tracks_estimator.run(corr_idxs_dict, keypoints_list)

        if self.save_track_patches_viz and images is not None:
            io_utils.save_track_visualizations(tracks_2d, images, save_dir=os.path.join("plots", "tracks_2d"))

        # track lengths w/o triangulation check
        track_lengths_2d = np.array(list(map(lambda x: int(x.number_measurements()), tracks_2d)), dtype=np.uint32)

        logger.debug("[Data association] input number of tracks: %s", len(tracks_2d))
        logger.debug("[Data association] input avg. track length: %s", np.mean(track_lengths_2d))

        # Initialize 3D landmark for each track
        point3d_initializer = Point3dInitializer(cameras, self.triangulation_options)

        # form GtsfmData object after triangulation
        triangulated_data = GtsfmData(num_images)

        # add all cameras
        for i, camera in cameras.items():
            triangulated_data.add_camera(i, camera)

        exit_codes_wrt_gt = track_utils.classify_tracks2d_with_gt_cameras(tracks=tracks_2d, cameras_gt=cameras_gt)

        # add valid tracks where triangulation is successful
        exit_codes_wrt_computed: List[TriangulationExitCode] = []
        per_accepted_track_avg_errors = []
        per_rejected_track_avg_errors = []
        for track_2d in tracks_2d:
            # triangulate and filter based on reprojection error
            sfm_track, avg_track_reproj_error, triangulation_exit_code = point3d_initializer.triangulate(track_2d)
            exit_codes_wrt_computed.append(triangulation_exit_code)
            if triangulation_exit_code == TriangulationExitCode.CHEIRALITY_FAILURE:
                continue

            if sfm_track is not None and self.__validate_track(sfm_track):
                triangulated_data.add_track(sfm_track)
                per_accepted_track_avg_errors.append(avg_track_reproj_error)
            else:
                per_rejected_track_avg_errors.append(avg_track_reproj_error)

        # aggregate the exit codes to get the distribution w.r.t each triangulation exit
        # get the exit codes distribution w.r.t. the camera params computed by the upstream modules of GTSFM
        exit_codes_wrt_computed_distribution = Counter(exit_codes_wrt_computed)
        # compute the exit codes distribution w.r.t. a tuple of exit codes: the exit code when triangulated with the
        # ground truth cameras and the exit code when triangulated with the computed cameras.
        exit_codes_wrt_gt_and_computed_distribution = None
        if exit_codes_wrt_gt is not None:
            exit_codes_wrt_gt_and_computed_distribution = Counter(zip(exit_codes_wrt_gt, exit_codes_wrt_computed))

        track_cheirality_failure_ratio = exit_codes_wrt_computed_distribution[
            TriangulationExitCode.CHEIRALITY_FAILURE
        ] / len(tracks_2d)

        # pick only the largest connected component
        # TODO(Ayush): remove this for hilti as disconnected components not an issue?
        cam_edges_from_prior = [k for k, v in relative_pose_priors.items() if v is not None]
        connected_data = triangulated_data.select_largest_connected_component(extra_camera_edges=cam_edges_from_prior)
        num_accepted_tracks = connected_data.number_tracks()
        accepted_tracks_ratio = num_accepted_tracks / len(tracks_2d)

        mean_3d_track_length, median_3d_track_length = connected_data.get_track_length_statistics()
        track_lengths_3d = connected_data.get_track_lengths()

        logger.debug("[Data association] output number of tracks: %s", num_accepted_tracks)
        logger.debug("[Data association] output avg. track length: %.2f", mean_3d_track_length)

        data_assoc_metrics = GtsfmMetricsGroup(
            "data_association_metrics",
            [
                GtsfmMetric(
                    "2D_track_lengths",
                    track_lengths_2d,
                    store_full_data=False,
                    plot_type=GtsfmMetric.PlotType.HISTOGRAM,
                ),
                GtsfmMetric("accepted_tracks_ratio", accepted_tracks_ratio),
                GtsfmMetric("track_cheirality_failure_ratio", track_cheirality_failure_ratio),
                GtsfmMetric("num_accepted_tracks", num_accepted_tracks),
                GtsfmMetric(
                    "3d_tracks_length",
                    track_lengths_3d,
                    store_full_data=False,
                    plot_type=GtsfmMetric.PlotType.HISTOGRAM,
                ),
                GtsfmMetric("accepted_track_avg_errors_px", per_accepted_track_avg_errors, store_full_data=False),
                GtsfmMetric(
                    "rejected_track_avg_errors_px",
                    np.array(per_rejected_track_avg_errors).astype(np.float32),
                    store_full_data=False,
                ),
                GtsfmMetric(name="number_cameras", data=len(connected_data.get_valid_camera_indices())),
            ],
        )

        if exit_codes_wrt_gt_and_computed_distribution is not None:
            for (gt_exit_code, computed_exit_code), count in exit_codes_wrt_gt_and_computed_distribution.items():
                # Each track has 2 associated exit codes: the triangulation exit codes w.r.t ground truth cameras
                # and w.r.t cameras computed by upstream modules of GTSFM. We get the distribution of the number of
                # tracks for each pair of (triangulation exit code w.r.t GT cams, triangulation exit code w.r.t
                # computed cams)
                metric_name = "#tracks triangulated with GT cams: {}, computed cams: {}".format(
                    gt_exit_code.name, computed_exit_code.name
                )

                data_assoc_metrics.add_metric(GtsfmMetric(name=metric_name, data=count))

        return connected_data, data_assoc_metrics

    def create_computation_graph(
        self,
        num_images: int,
        cameras: Delayed,
        corr_idxs_graph: Delayed,
        keypoints_graph: List[Delayed],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        images_graph: Optional[List[Delayed]] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Creates a computation graph for performing data association.

        Args:
            num_images: number of images in the scene.
            cameras: list of cameras wrapped up as Delayed.
            corr_idxs_graph: dictionary of correspondence indices, each value wrapped up as Delayed.
            keypoints_graph: list of wrapped up keypoints for each image.
            cameras_gt: a list of cameras with ground truth params, if they exist.
            relative_pose_priors: pose priors on the relative pose between camera poses.
            images_graph: a list of all images in scene (optional and only for track patch visualization)

        Returns:
            ba_input_graph: GtsfmData object wrapped up using dask.delayed
            data_assoc_metrics_graph: dictionary with different statistics about the data
                association result
        """
        ba_input_graph, data_assoc_metrics_graph = dask.delayed(self.run, nout=2)(
            num_images, cameras, corr_idxs_graph, keypoints_graph, cameras_gt, relative_pose_priors, images_graph
        )

        return ba_input_graph, data_assoc_metrics_graph
