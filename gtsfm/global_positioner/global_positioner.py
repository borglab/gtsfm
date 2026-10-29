"""Global positioning of cameras and 3D points via direction constraints.

Implements the global positioning step from GLOMAP (Pan et al., ECCV 2024),
which jointly estimates camera positions and 3D point positions using
unit direction constraints from 2D feature tracks.

This module replaces BOTH translation averaging AND data association
(triangulation) in the GTSFM pipeline. It outputs GtsfmData directly —
cameras with full poses + 3D tracks with landmarks — ready for bundle
adjustment.

Pipeline integration:
    BEFORE:  rot_avg → trans_avg → data_association → BA
    AFTER:   rot_avg → GlobalPositioner → BA

Uses gtsam.GlobalPositioner, which solves the bipartite camera+landmark
estimation problem directly using BilinearAngleTranslationFactor (BATA).

References:
    Pan, L., Barath, D., Pollefeys, M., Schonberger, J.L.
    "Global Structure-from-Motion Revisited." ECCV 2024.

Authors: Kathir Gounder
"""

import os
import pickle
import time
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import gtsam
import numpy as np
from gtsam import (
    BinaryMeasurementUnit3,
    BinaryMeasurementsUnit3,
    Point3,
    Pose3,
    Rot3,
    SfmTrack,
    Unit3,
    Values,
)
from gtsam.symbol_shorthand import A as C  # Camera position variables
from gtsam.symbol_shorthand import B as L  # Landmark position variables

import gtsfm.common.types as gtsfm_types
import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmTrack2d
from gtsfm.evaluation.metrics import GtsfmMetric, GtsfmMetricsGroup
from gtsfm.utils.tracks import compute_world_directions, filter_tracks_by_measurements

logger = logger_utils.get_logger()

MIN_TRACK_MEASUREMENTS = 3
DEFAULT_HUBER_LOSS_SCALE = 0.1
DEFAULT_DIRECTION_NOISE_SIGMA = 0.01
DEFAULT_MAX_ITERATIONS = 100
DEFAULT_MAX_REPROJ_ERROR = 5.0
DEFAULT_MAX_TRACKS = 0


def _build_inputs(
    observations: List[Tuple[int, int, np.ndarray]],
    noise_sigma: float,
    huber_loss_scale: float,
) -> Tuple[BinaryMeasurementsUnit3, Set[int], Set[int]]:
    """Build direction measurements and explicit camera/landmark key sets."""
    noise_model = gtsam.noiseModel.Isotropic.Sigma(2, noise_sigma)
    huber = gtsam.noiseModel.mEstimator.Huber.Create(huber_loss_scale)
    robust_noise = gtsam.noiseModel.Robust.Create(huber, noise_model)

    measurements = BinaryMeasurementsUnit3()
    # GlobalPositioner's std::set arguments use pybind11's Python-set caster,
    # rather than the separately wrapped gtsam.KeySet container.
    camera_keys: Set[int] = set()
    landmark_keys: Set[int] = set()

    for track_idx, cam_idx, direction in observations:
        measurements.append(BinaryMeasurementUnit3(
            C(cam_idx), L(track_idx), Unit3(Point3(*direction)), robust_noise,
        ))
        camera_keys.add(C(cam_idx))
        landmark_keys.add(L(track_idx))

    return measurements, camera_keys, landmark_keys


def build_and_solve(
    observations: List[Tuple[int, int, np.ndarray]],
    noise_sigma: float,
    huber_loss_scale: float,
    max_iterations: int,
) -> Tuple[Values, Set[int], Set[int]]:
    """Solve via gtsam.GlobalPositioner — graph, gauge, init, and LM in one call."""
    measurements, camera_keys, landmark_keys = _build_inputs(observations, noise_sigma, huber_loss_scale)
    anchor_key = min(camera_keys)

    logger.info("GlobalPositioner: %d measurements, %d cameras, %d landmarks",
                len(observations), len(camera_keys), len(landmark_keys))

    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(max_iterations)
    lm_params.setRelativeErrorTol(1e-5)
    lm_params.setVerbosityLM("SUMMARY")

    gp = gtsam.GlobalPositioner(lm_params)
    result = gp.run(measurements, camera_keys, landmark_keys, anchor_key)

    cam_indices = {gtsam.Symbol(k).index() for k in camera_keys}
    track_indices = {gtsam.Symbol(k).index() for k in landmark_keys}
    return result, cam_indices, track_indices


def build_and_solve_with_trace(
    observations: List[Tuple[int, int, np.ndarray]],
    noise_sigma: float,
    huber_loss_scale: float,
    max_iterations: int,
) -> Tuple[Values, Set[int], Set[int], List[Values], List[float]]:
    """Solve with per-iteration capture for convergence visualization."""
    measurements, camera_keys, landmark_keys = _build_inputs(observations, noise_sigma, huber_loss_scale)
    anchor_key = min(camera_keys)

    logger.info("GlobalPositioner (trace): %d measurements, %d cameras, %d landmarks",
                len(observations), len(camera_keys), len(landmark_keys))

    lm_params = gtsam.LevenbergMarquardtParams()
    lm_params.setMaxIterations(max_iterations)
    lm_params.setRelativeErrorTol(1e-5)
    lm_params.setVerbosityLM("SILENT")

    gp = gtsam.GlobalPositioner(lm_params)
    graph = gp.buildGraph(measurements, True)
    gp.addAnchorPrior(anchor_key, graph)
    initial = gp.initializeRandomly(camera_keys, landmark_keys, measurements)

    initial_error = graph.error(initial)
    logger.info("GlobalPositioner (trace): initial error = %.2f", initial_error)

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, lm_params)
    values_trace = [Values(initial)]
    error_trace = [initial_error]

    for iteration in range(1, max_iterations + 1):
        t0 = time.time()
        optimizer.iterate()
        dt = time.time() - t0
        error = optimizer.error()
        # values() exposes optimizer-owned state that subsequent iterations replace.
        # Preserve an owning copy for each frame and for use after optimizer teardown.
        values_trace.append(Values(optimizer.values()))
        error_trace.append(error)
        if iteration <= 5 or iteration % 5 == 0:
            logger.info("  iter %3d: error=%.2f  (%.2fs)", iteration, error, dt)
        # Early stopping on convergence.
        if len(error_trace) >= 2:
            relative_change = abs(error_trace[-2] - error) / max(error_trace[-2], 1e-12)
            if relative_change < 1e-5:
                logger.info("  Converged at iteration %d (relative change %.2e)", iteration, relative_change)
                break

    result = Values(optimizer.values())
    cam_indices = {gtsam.Symbol(k).index() for k in camera_keys}
    track_indices = {gtsam.Symbol(k).index() for k in landmark_keys}
    logger.info("GlobalPositioner (trace): final error = %.4f (%d frames)", error_trace[-1], len(values_trace))
    return result, cam_indices, track_indices, values_trace, error_trace


def build_gtsfm_data(
    result_values: Values,
    num_images: int,
    wRi_list: List[Optional[Rot3]],
    intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
    tracks_2d: List[SfmTrack2d],
    camera_indices: Set[int],
    track_indices: Set[int],
    valid_cameras: Set[int],
    max_reproj_error: float = DEFAULT_MAX_REPROJ_ERROR,
    max_angle_error_deg: float = 1.0,
) -> Tuple[GtsfmData, int, int]:
    """Build GtsfmData from optimized camera + landmark positions."""
    gtsfm_data = GtsfmData(number_images=num_images)
    camera_class = gtsfm_types.get_camera_class_for_calibration(intrinsics[0])

    cameras_added = 0
    for cam_idx in camera_indices:
        wRi, Ki = wRi_list[cam_idx], intrinsics[cam_idx]
        if wRi is None or Ki is None:
            continue
        try:
            pose = Pose3(wRi, result_values.atPoint3(C(cam_idx)))
            gtsfm_data.add_camera(cam_idx, camera_class(pose, Ki))
            cameras_added += 1
        except RuntimeError:
            logger.warning("GlobalPositioner: failed to build camera %d.", cam_idx)

    num_tracks_before = 0
    num_tracks_after = 0
    estimated_camera_set = set(gtsfm_data.get_valid_camera_indices())

    for track_idx in sorted(track_indices):
        try:
            point_3d = result_values.atPoint3(L(track_idx))
        except RuntimeError:
            continue
        if track_idx >= len(tracks_2d):
            continue

        sfm_track = SfmTrack(point_3d)
        num_valid = 0
        for m_idx in range(tracks_2d[track_idx].number_measurements()):
            m = tracks_2d[track_idx].measurement(m_idx)
            if m.i in estimated_camera_set:
                sfm_track.addMeasurement(m.i, m.uv)
                num_valid += 1
        if num_valid < 2:
            continue

        num_tracks_before += 1
        # Per-observation angular error filter (GLOMAP FilterTracksByAngle).
        # Remove observations where camera ray disagrees with 3D point direction.
        cos_angle_thresh = np.cos(np.radians(max_angle_error_deg))
        filtered_track = SfmTrack(point_3d)
        for m_idx in range(sfm_track.numberMeasurements()):
            cam_idx_m, uv_m = sfm_track.measurement(m_idx)
            camera_m = gtsfm_data.get_camera(cam_idx_m)
            if camera_m is None:
                continue
            try:
                # Compute normalized ray from camera to 3D point.
                pt_cam = camera_m.pose().transformTo(point_3d)
                if pt_cam[2] < 1e-12:
                    continue  # Behind camera.
                pt_cam_dir = pt_cam / np.linalg.norm(pt_cam)
                # Compute observed feature ray direction.
                Ki = intrinsics[cam_idx_m]
                uv_norm = Ki.calibrate(uv_m)
                obs_dir = np.array([uv_norm[0], uv_norm[1], 1.0])
                obs_dir /= np.linalg.norm(obs_dir)
                # Angular consistency check.
                if pt_cam_dir.dot(obs_dir) < cos_angle_thresh:
                    continue  # Angle too large, skip this observation.
                # Also check reprojection error.
                proj = camera_m.project(point_3d)
                if np.sqrt((proj[0] - uv_m[0])**2 + (proj[1] - uv_m[1])**2) > max_reproj_error:
                    continue
                filtered_track.addMeasurement(cam_idx_m, uv_m)
            except RuntimeError:
                continue
        if filtered_track.numberMeasurements() >= 2:
            gtsfm_data.add_track(filtered_track)
            num_tracks_after += 1

    logger.info("GlobalPositioner: %d cameras, %d/%d tracks (%.1f%% kept).",
                cameras_added, num_tracks_after, num_tracks_before,
                100 * num_tracks_after / max(num_tracks_before, 1))
    return gtsfm_data, num_tracks_before, num_tracks_after


class GlobalPositioner:
    """Joint camera + landmark position estimation via BATA direction constraints.

    Uses gtsam.GlobalPositioner to solve the bipartite estimation problem.

    Pipeline:  rot_avg → GlobalPositioner.run() → BA
    """

    def __init__(
        self,
        noise_sigma: float = DEFAULT_DIRECTION_NOISE_SIGMA,
        huber_loss_scale: float = DEFAULT_HUBER_LOSS_SCALE,
        max_iterations: int = DEFAULT_MAX_ITERATIONS,
        min_track_measurements: int = MIN_TRACK_MEASUREMENTS,
        max_reproj_error: float = DEFAULT_MAX_REPROJ_ERROR,
        max_tracks: int = DEFAULT_MAX_TRACKS,
        save_iteration_visualization: bool = False,
    ) -> None:
        self._noise_sigma = noise_sigma
        self._huber_loss_scale = huber_loss_scale
        self._max_iterations = max_iterations
        self._min_track_measurements = min_track_measurements
        self._max_reproj_error = max_reproj_error
        self._max_tracks = max_tracks
        self._save_iteration_visualization = save_iteration_visualization

    def run(
        self,
        num_images: int,
        wRi_list: List[Optional[Rot3]],
        tracks_2d: List[SfmTrack2d],
        intrinsics: List[Optional[gtsfm_types.CALIBRATION_TYPE]],
        output_root: Optional[Union[str, Path]] = None,
    ) -> Tuple[GtsfmData, GtsfmMetricsGroup]:
        """Run global positioning and produce GtsfmData for BA."""
        start_time = time.time()

        if not tracks_2d:
            logger.error("GlobalPositioner: no 2D tracks provided.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])
        if intrinsics is None:
            logger.error("GlobalPositioner: no intrinsics provided.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        valid_cameras = {i for i, wRi in enumerate(wRi_list) if wRi is not None}
        if len(valid_cameras) < 2:
            logger.error("GlobalPositioner: need >= 2 cameras with rotations.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        logger.info("GlobalPositioner: %d valid cameras, %d input tracks.", len(valid_cameras), len(tracks_2d))

        filtered_tracks = filter_tracks_by_measurements(
            tracks_2d, valid_cameras, self._min_track_measurements, self._max_tracks,
        )
        logger.info("GlobalPositioner: %d tracks after filtering.", len(filtered_tracks))
        if not filtered_tracks:
            logger.error("GlobalPositioner: no tracks survived filtering.")
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        t0 = time.time()
        observations = compute_world_directions(filtered_tracks, intrinsics, wRi_list, valid_cameras)
        dt_dir = time.time() - t0
        logger.info("GlobalPositioner: %d direction observations (%.2fs).", len(observations), dt_dir)

        if not observations:
            return GtsfmData(number_images=num_images), GtsfmMetricsGroup("global_positioning_metrics", [])

        t0 = time.time()
        if self._save_iteration_visualization:
            result_values, cam_indices, track_indices, values_trace, error_trace = build_and_solve_with_trace(
                observations, self._noise_sigma, self._huber_loss_scale, self._max_iterations,
            )
            # Extract positions as numpy arrays (avoids GTSAM pickling issues)
            sorted_cams = sorted(cam_indices)
            sorted_tracks = sorted(track_indices)
            positions_trace = []
            for vals in values_trace:
                cam_pos = {ci: np.array(vals.atPoint3(C(ci))) for ci in sorted_cams}
                lmk_pos = {}
                for ti in sorted_tracks:
                    try:
                        lmk_pos[ti] = np.array(vals.atPoint3(L(ti)))
                    except RuntimeError:
                        pass
                positions_trace.append({"cameras": cam_pos, "landmarks": lmk_pos})

            # Convert wRi to rotation matrices (pure numpy, no GTSAM objects)
            wRi_matrices = {}
            for ci in sorted_cams:
                wRi = wRi_list[ci]
                if wRi is not None:
                    wRi_matrices[ci] = wRi.matrix()

            # Save trace under the dataset-specific output root so multiple
            # runs don't clobber each other's traces.
            if output_root is not None:
                trace_path = str(Path(output_root) / "results" / "gp_convergence_trace.pkl")
            else:
                trace_path = "results/gp_convergence_trace.pkl"
            os.makedirs(os.path.dirname(trace_path), exist_ok=True)
            with open(trace_path, "wb") as f:
                pickle.dump({
                    "positions_trace": positions_trace, "error_trace": error_trace,
                    "camera_indices": cam_indices, "track_indices": track_indices,
                    "wRi_matrices": wRi_matrices,
                    "filtered_tracks": filtered_tracks,
                }, f)
            logger.info("GlobalPositioner: saved trace (%d frames) to %s", len(positions_trace), trace_path)
        else:
            result_values, cam_indices, track_indices = build_and_solve(
                observations, self._noise_sigma, self._huber_loss_scale, self._max_iterations,
            )
        dt_solve = time.time() - t0
        logger.info("GlobalPositioner: optimization took %.2fs.", dt_solve)

        t0 = time.time()
        gtsfm_data, tracks_before, tracks_after = build_gtsfm_data(
            result_values, num_images, wRi_list, intrinsics, filtered_tracks,
            cam_indices, track_indices, valid_cameras, self._max_reproj_error,
        )
        dt_build = time.time() - t0
        total_duration = time.time() - start_time

        metrics = GtsfmMetricsGroup("global_positioning_metrics", [
            GtsfmMetric("num_valid_cameras", len(valid_cameras)),
            GtsfmMetric("num_tracks_used", len(filtered_tracks)),
            GtsfmMetric("num_direction_observations", len(observations)),
            GtsfmMetric("num_camera_variables", len(cam_indices)),
            GtsfmMetric("num_landmark_variables", len(track_indices)),
            GtsfmMetric("num_cameras_in_output", len(gtsfm_data.get_valid_camera_indices()) if gtsfm_data else 0),
            GtsfmMetric("num_tracks_before_filter", tracks_before),
            GtsfmMetric("num_tracks_after_filter", tracks_after),
            GtsfmMetric("direction_computation_sec", dt_dir),
            GtsfmMetric("optimization_sec", dt_solve),
            GtsfmMetric("gtsfm_data_build_sec", dt_build),
            GtsfmMetric("total_duration_sec", total_duration),
        ])

        logger.info(
            "GlobalPositioner: done in %.2fs — %d cameras, %d tracks.",
            total_duration,
            len(gtsfm_data.get_valid_camera_indices()) if gtsfm_data else 0,
            gtsfm_data.number_tracks() if gtsfm_data else 0,
        )
        return gtsfm_data, metrics
