"""Create 2D-3D data association as a precursor to Bundle Adjustment.
1. Forms feature tracks from verified correspondences and global poses.
2. Estimates 3D landmark for each track (Ransac and simple triangulation modes available)
3. Filters tracks based on reprojection error.

References:
1. Richard I. Hartley and Peter Sturm. Triangulation. Computer Vision and Image Understanding, Vol. 68, No. 2,
   November, pp. 146-157, 1997

Authors: Sushmita Warrier, Xiaolong Wu, John Lambert, Travis Driver
"""

import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import dask
import gtsam  # type: ignore
import numpy as np
from dask.delayed import Delayed
from gtsam import SfmTrack

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
import gtsfm.utils.tracks as track_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.data_association.point3d_initializer import Point3dInitializer, TriangulationExitCode, TriangulationOptions
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata

logger = logger_utils.get_logger()

# Heuristically set to limit the number of delayed tasks, as recommended by Dask:
# https://docs.dask.org/en/stable/delayed-best-practices.html#avoid-too-many-tasks
MAX_DELAYED_TRIANGULATION_CALLS = 1e3


@dataclass(frozen=True)
class DataAssociation(GTSFMProcess):
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

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Data Association",
            input_products=(
                "View-Graph Correspondences",
                "Global Rotations",
                "Global Translations",
                "Camera Intrinsics",
            ),
            output_products=("3D Tracks",),
            parent_plate="Sparse Reconstruction",
        )

    def __validate_track(self, sfm_track: Optional[SfmTrack]) -> bool:
        """Validate the track by checking its length."""
        return sfm_track is not None and sfm_track.numberMeasurements() >= self.min_track_len

    def assemble_gtsfm_data_from_tracks(
        self,
        num_images: int,
        cameras: Dict[int, gtsfm_types.CAMERA_TYPE],
        tracks_2d: List[SfmTrack2d],
        sfm_tracks: List[Optional[SfmTrack]],
        avg_track_reproj_errors: List[Optional[float]],
        triangulation_exit_codes: List[TriangulationExitCode],
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        relative_pose_priors: Dict[Tuple[int, int], Optional[PosePrior]],
        images: Optional[List[Image]] = None,
    ) -> Tuple[GtsfmData, GtsfmMetricsGroup]:
        """Validates 3D triangulated tracks, assembles GtsfmData, and computes DA metrics.

        Only the largest connected component of cameras, represented in 3d tracks, is retained.

        Args:
            num_images: Number of images in the scene.
            cameras: Dictionary, with image index -> camera mapping.
            tracks_2d: List of 2D tracks.
            sfm_tracks: List of triangulated tracks.
            avg_track_reproj_errors: List of average reprojection errors per track.
            triangulation_exit_codes: Exit codes for each triangulation call.
            cameras_gt: List of GT cameras, to be used for benchmarking the tracks.
            images: A list of all images in scene (optional and only for track patch visualization).

        Returns:
            A tuple of GtsfmData with cameras and tracks, and a GtsfmMetricsGroup with data association metrics.
        """
        if self.save_track_patches_viz and images is not None:
            io_utils.save_track_visualizations(tracks_2d, images, save_dir=os.path.join("plots", "tracks_2d"))

        # Track lengths w/o triangulation check.
        track_lengths_2d = np.array(list(map(lambda x: int(x.number_measurements()), tracks_2d)), dtype=np.uint32)

        logger.debug("Input number of tracks: %s", len(tracks_2d))
        logger.debug("Input avg. track length: %s", np.mean(track_lengths_2d))

        # Form GtsfmData object after triangulation.
        triangulated_data = GtsfmData(num_images)

        # Add all cameras.
        for i, camera in cameras.items():
            triangulated_data.add_camera(i, camera)

        # If GT cameras are available and all are PinholeCameraCal3Bundler, compute the exit codes w.r.t. GT cameras.
        exit_codes_wrt_gt = None
        pinhole_bundler_cams = [cam for cam in cameras_gt if isinstance(cam, gtsam.PinholeCameraCal3Bundler)]
        if len(pinhole_bundler_cams) == len(cameras_gt):
            exit_codes_wrt_gt = track_utils.classify_tracks2d_with_gt_cameras(tracks_2d, pinhole_bundler_cams)

        # Add valid tracks where triangulation was successful.
        exit_codes_wrt_computed: List[TriangulationExitCode] = []
        per_accepted_track_avg_errors = []
        per_rejected_track_avg_errors = []
        assert len(tracks_2d) == len(sfm_tracks)
        for j in range(len(tracks_2d)):
            # Filter triangulated points based on reprojection error and exit code.
            sfm_track = sfm_tracks[j]
            avg_track_reproj_error = avg_track_reproj_errors[j]
            triangulation_exit_code = triangulation_exit_codes[j]
            exit_codes_wrt_computed.append(triangulation_exit_code)
            if triangulation_exit_code == TriangulationExitCode.CHEIRALITY_FAILURE:
                continue

            if sfm_track is not None and self.__validate_track(sfm_track):
                triangulated_data.add_track(sfm_track)
                per_accepted_track_avg_errors.append(avg_track_reproj_error)
            else:
                per_rejected_track_avg_errors.append(avg_track_reproj_error)

        # Aggregate the exit codes to get the distribution w.r.t each triangulation exit
        # Get the exit codes distribution w.r.t. the camera params computed by the upstream modules of GTSFM
        exit_codes_wrt_computed_distribution = Counter(exit_codes_wrt_computed)
        # Compute the exit codes distribution w.r.t. a tuple of exit codes: the exit code when triangulated with the
        # Ground truth cameras and the exit code when triangulated with the computed cameras.
        exit_codes_wrt_gt_and_computed_distribution = None
        if exit_codes_wrt_gt is not None:
            exit_codes_wrt_gt_and_computed_distribution = Counter(zip(exit_codes_wrt_gt, exit_codes_wrt_computed))

        track_cheirality_failure_ratio = exit_codes_wrt_computed_distribution[
            TriangulationExitCode.CHEIRALITY_FAILURE
        ] / len(tracks_2d)

        # Pick only the largest connected component.
        # TODO(Ayush): remove this for hilti as disconnected components not an issue?
        cam_edges_from_prior = [k for k, v in relative_pose_priors.items() if v is not None]
        connected_data = triangulated_data.select_largest_connected_component(extra_camera_edges=cam_edges_from_prior)
        num_accepted_tracks = connected_data.number_tracks()
        accepted_tracks_ratio = num_accepted_tracks / len(tracks_2d)

        mean_3d_track_length, median_3d_track_length = connected_data.get_track_length_statistics()
        track_lengths_3d = connected_data.get_track_lengths()

        logger.debug("output number of tracks: %s", num_accepted_tracks)
        logger.debug("output avg. track length: %.2f", mean_3d_track_length)

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

    def run_triangulation(
        self,
        cameras: Dict[int, gtsfm_types.CAMERA_TYPE],
        tracks_2d: List[SfmTrack2d],
    ) -> Tuple[List[Delayed], List[Delayed], List[Delayed]]:
        """Performs triangulation of batched 2D tracks in parallel.

        Refs:
        - https://docs.dask.org/en/stable/delayed-best-practices.html#compute-on-lots-of-computation-at-once
        - https://docs.dask.org/en/stable/delayed-best-practices.html#avoid-too-many-tasks

        Args:
            cameras: List of cameras wrapped up as Delayed.
            tracks_2d: List of tracks wrapped up as Delayed.

        Returns:
            sfm_tracks: List of triangulated tracks.
            avg_track_reproj_errors: List of average reprojection errors per track.
            triangulation_exit_codes: Exit codes for each triangulation call.
        """

        def triangulate_batch(
            point3d_initializer: Point3dInitializer, tracks_2d: List[SfmTrack2d]
        ) -> List[Tuple[Optional[SfmTrack], Optional[float], TriangulationExitCode]]:
            """Triangulates a batch of 2D tracks sequentially."""
            batch_results = []
            for track_2d in tracks_2d:
                batch_results.append(point3d_initializer.triangulate(track_2d))
            return batch_results

        # Initialize 3D landmark for each track.
        point3d_initializer = Point3dInitializer(cameras, self.triangulation_options)

        # Loop through tracks and and generate delayed triangulation tasks.
        batch_size = int(np.ceil(len(tracks_2d) / MAX_DELAYED_TRIANGULATION_CALLS))
        triangulation_results = []
        if batch_size == 1:
            for track_2d in tracks_2d:
                triangulation_results.append(dask.delayed(point3d_initializer.triangulate)(track_2d))
        else:
            for j in range(0, len(tracks_2d), batch_size):
                triangulation_results.append(
                    dask.delayed(triangulate_batch)(point3d_initializer, tracks_2d[j : j + batch_size])
                )

        # Perform triangulation in parallel.
        triangulation_results = dask.compute(*triangulation_results)

        # Unpack results.
        sfm_tracks, avg_track_reproj_errors, triangulation_exit_codes = [], [], []
        if batch_size == 1:
            for sfm_track, avg_track_reproj_error, exit_code in triangulation_results:
                sfm_tracks.append(sfm_track)
                avg_track_reproj_errors.append(avg_track_reproj_error)
                triangulation_exit_codes.append(exit_code)
        else:
            for batch_results in triangulation_results:
                for sfm_track, avg_track_reproj_error, exit_code in batch_results:
                    sfm_tracks.append(sfm_track)
                    avg_track_reproj_errors.append(avg_track_reproj_error)
                    triangulation_exit_codes.append(exit_code)

        return sfm_tracks, avg_track_reproj_errors, triangulation_exit_codes

    def run_triangulation_and_evaluate(
        self,
        num_images: int,
        cameras: Dict[int, gtsfm_types.CAMERA_TYPE],
        tracks_2d: List[SfmTrack2d],
        cameras_gt: Sequence[Optional[gtsfm_types.CAMERA_TYPE]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        images: Optional[List[Delayed]] = None,
    ) -> Tuple[GtsfmData, GtsfmMetricsGroup]:
        """Runs triangulation, evaluates results, and forms metrics group."""
        start_time = time.time()

        # Triangulate 2D tracks to form 3D tracks.
        sfm_tracks, avg_track_reproj_errors, triangulation_exit_codes = self.run_triangulation(
            cameras=cameras, tracks_2d=tracks_2d
        )
        triangulation_runtime_sec = time.time() - start_time

        # Validate 3D tracks, create BA input, and compute metrics.
        gtsfm_data_creation_start_time = time.time()
        ba_input, data_assoc_metrics = self.assemble_gtsfm_data_from_tracks(
            num_images=num_images,
            cameras=cameras,
            tracks_2d=tracks_2d,
            sfm_tracks=sfm_tracks,
            avg_track_reproj_errors=avg_track_reproj_errors,
            triangulation_exit_codes=triangulation_exit_codes,
            cameras_gt=cameras_gt,
            relative_pose_priors=relative_pose_priors,
            images=images,
        )
        gtsfm_data_creation_runtime = time.time() - gtsfm_data_creation_start_time

        end_time = time.time()
        total_duration_sec = end_time - start_time
        data_assoc_metrics.add_metric(GtsfmMetric("triangulation_runtime_sec", triangulation_runtime_sec))
        data_assoc_metrics.add_metric(GtsfmMetric("gtsfm_data_creation_runtime", gtsfm_data_creation_runtime))
        data_assoc_metrics.add_metric(GtsfmMetric("total_duration_sec", total_duration_sec))
        logger.info("🚀 runtime duration: %.2f sec.", total_duration_sec)
        return ba_input, data_assoc_metrics

    def create_computation_graph(
        self,
        num_images: int,
        cameras: Delayed,
        tracks_2d: Delayed,
        cameras_gt: List[Optional[gtsfm_types.CAMERA_TYPE]],
        relative_pose_priors: Dict[Tuple[int, int], PosePrior],
        images: Optional[List[Delayed]] = None,
    ) -> Tuple[Delayed, Delayed]:
        """Creates a computation graph for performing data association.

        Args:
            num_images: Number of images in the scene.
            cameras: List of cameras wrapped up as Delayed.
            tracks_2d: List of tracks wrapped up as Delayed.
            cameras_gt: A list of cameras with ground truth params, if they exist.
            relative_pose_priors: Pose priors on the relative pose between camera poses.
            images: A list of all images in scene (optional and only for track patch visualization).

        Returns:
            ba_input_graph: GtsfmData object wrapped up using dask.delayed.
            data_assoc_metrics_graph: Dictionary with different statistics about the data
                association result.
        """
        ba_input_graph, data_assoc_metrics_graph = dask.delayed(self.run_triangulation_and_evaluate, nout=2)(
            num_images=num_images,
            cameras=cameras,
            tracks_2d=tracks_2d,
            cameras_gt=cameras_gt,
            relative_pose_priors=relative_pose_priors,
            images=images,
        )

        return ba_input_graph, data_assoc_metrics_graph
