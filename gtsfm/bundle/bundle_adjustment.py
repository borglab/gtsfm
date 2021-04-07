"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""
from typing import NamedTuple

import dask
import gtsam
from dask.delayed import Delayed
from gtsam import GeneralSFMFactorCal3Bundler, SfmTrack, Values, symbol_shorthand

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData

# TODO: any way this goes away?
C = symbol_shorthand.C
P = symbol_shorthand.P

PINHOLE_CAM_CAL3BUNDLER_DOF = 9  # 6 dof for pose, and 3 dof for f, k1, k2
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logger = logger_utils.get_logger()


class BundleAdjustmentOptimizer(NamedTuple):
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given
    tracks from triangulation."""

    output_reproj_error_thresh: float

    def run(self, initial_data: GtsfmData) -> GtsfmData:
        """Run the bundle adjustment by forming factor graph and optimizing using Levenberg–Marquardt optimization.

        Args:
            initial_data: initialized cameras, tracks w/ their 3d landmark from triangulation.

        Results:
            optimized camera poses, 3D point w/ tracks, and error metrics.
        """
        logger.info(
            f"Input: {initial_data.number_tracks()} tracks on {len(initial_data.get_valid_camera_indices())} cameras\n"
        )

        # noise model for measurements -- one pixel in u and v
        measurement_noise = gtsam.noiseModel.Isotropic.Sigma(IMG_MEASUREMENT_DIM, 1.0)

        # Create a factor graph
        graph = gtsam.NonlinearFactorGraph()

        # Add measurements to the factor graph
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)  # SfmTrack
            # retrieve the SfmMeasurement objects
            for m_idx in range(track.number_measurements()):
                # i represents the camera index, and uv is the 2d measurement
                i, uv = track.measurement(m_idx)
                # note use of shorthand symbols C and P
                graph.add(GeneralSFMFactorCal3Bundler(uv, measurement_noise, C(i), P(j)))

        # get all the valid camera indices, which need to be added to the graph.
        valid_camera_indices = initial_data.get_valid_camera_indices()

        # Add a prior on first pose. This indirectly specifies where the origin is.
        graph.push_back(
            gtsam.PriorFactorPinholeCameraCal3Bundler(
                C(valid_camera_indices[0]),
                initial_data.get_camera(valid_camera_indices[0]),
                gtsam.noiseModel.Isotropic.Sigma(PINHOLE_CAM_CAL3BUNDLER_DOF, 0.1),
            )
        )
        # Also add a prior on the position of the first landmark to fix the scale
        graph.push_back(
            gtsam.PriorFactorPoint3(
                P(0), initial_data.get_track(0).point3(), gtsam.noiseModel.Isotropic.Sigma(POINT3_DOF, 0.1),
            )
        )

        # Create initial estimate
        initial = gtsam.Values()

        # add each PinholeCameraCal3Bundler
        for i in valid_camera_indices:
            camera = initial_data.get_camera(i)
            initial.insert(C(i), camera)

        # add each SfmTrack
        for j in range(initial_data.number_tracks()):
            track = initial_data.get_track(j)
            initial.insert(P(j), track.point3())

        # Optimize the graph and print results
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosityLM("ERROR")
            lm = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result_values = lm.optimize()
        except Exception:
            logger.exception("LM Optimization failed")
            # as we did not perform the bundle adjustment, we skip computing the total reprojection error
            return GtsfmData(initial_data.number_images())

        final_error = graph.error(result_values)

        # Error drops from ~2764.22 to ~0.046
        logger.info(f"initial error: {graph.error(initial):.2f}")
        logger.info(f"final error: {final_error:.2f}")

        # construct the results
        optimized_data = values_to_gtsfm_data(result_values, initial_data)

        logger.info("[Result] Number of tracks before filtering %d", optimized_data.number_tracks())

        # filter the largest errors
        filtered_result = optimized_data.filter_landmarks(self.output_reproj_error_thresh)

        logger.info("[Result] Number of tracks after filtering: %d", filtered_result.number_tracks())
        mean_track_length, median_track_length = filtered_result.get_track_length_statistics()
        logger.info("[Result] Mean track length %.3f", mean_track_length)
        logger.info("[Result] Median track length %.3f", median_track_length)
        filtered_result.log_scene_reprojection_error_stats()

        return filtered_result

    def create_computation_graph(self, sfm_data_graph: Delayed) -> Delayed:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: an GtsfmData object wrapped up using dask.delayed

        Returns:
            GtsfmData wrapped up using dask.delayed
        """
        return dask.delayed(self.run)(sfm_data_graph)


def values_to_gtsfm_data(values: Values, initial_data: GtsfmData) -> GtsfmData:
    """Cast results from the optimization to GtsfmData object.

    Args:
        values: results of factor graph optimization.
        initial_data: data used to generate the factor graph; used to extract information about poses and 3d points in
                      the graph.

    Returns:
        optimized poses and landmarks.
    """
    result = GtsfmData(initial_data.number_images())

    # add cameras
    for i in initial_data.get_valid_camera_indices():
        result.add_camera(i, values.atPinholeCameraCal3Bundler(C(i)))

    # add tracks
    for j in range(initial_data.number_tracks()):
        input_track = initial_data.get_track(j)

        # populate the result with optimized 3D point
        result_track = SfmTrack(values.atPoint3(P(j)),)

        for measurement_idx in range(input_track.number_measurements()):
            i, uv = input_track.measurement(measurement_idx)
            result_track.add_measurement(i, uv)

        result.add_track(result_track)

    return result
