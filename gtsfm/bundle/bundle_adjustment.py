"""Factor-graph based formulation of Bundle adjustment and optimization.

Authors: Xiaolong Wu, John Lambert, Ayush Baid
"""

import logging
import sys

import dask
from dask.delayed import Delayed
from gtsam import (
    GeneralSFMFactor2Cal3Bundler,
    LevenbergMarquardtOptimizer,
    LevenbergMarquardtParams,
    NonlinearFactorGraph,
    PinholeCameraCal3Bundler,
    PriorFactorCal3Bundler,
    PriorFactorPose3,
    SfmData,
    SfmTrack,
    Values,
    symbol_shorthand,
)
from gtsam.noiseModel import Isotropic

from gtsfm.common.sfm_result import SfmResult

# TODO: any way this goes away?
P = symbol_shorthand.P  # 3d point
X = symbol_shorthand.X  # camera pose
K = symbol_shorthand.K  # calibration

CAM_POSE3_DOF = 6  # 6 dof for pose of camera
CAM_CAL3BUNDLER_DOF = 3  # 3 dof for f, k1, k2 for intrinsics of camera
IMG_MEASUREMENT_DIM = 2  # 2d measurements (u,v) have 2 dof
POINT3_DOF = 3  # 3d points have 3 dof

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class BundleAdjustmentOptimizer:
    """Bundle adjustment using factor-graphs in GTSAM.

    This class refines global pose estimates and intrinsics of cameras, and also refines 3D point cloud structure given tracks from triangulation."""

    def __init__(self, shared_calib: bool = False):
        """Initializes the parameters for bundle adjustment module.

        Args:
            shared_calib (optional): Flag to enable shared calibration across
                                     all cameras. Defaults to False.
        """

        self._shared_calib = shared_calib

    def __add_camera_prior_and_initial_value(
        self,
        graph: NonlinearFactorGraph,
        initial_values: Values,
        camera: PinholeCameraCal3Bundler,
        camera_idx: int,
    ) -> None:
        """Add a prior factor in the factor graph and initial values for the
        camera parameters.

        Args:
            graph: factor graph for the problem.
            initial_values: object holding the initial values for camera
                            parameters and 3d points.
            camera: the camera object to add to graph and to initial values.
            camera_idx: index of the camera.
        """

        # add prior factor for pose
        graph.push_back(
            PriorFactorPose3(
                X(camera_idx),
                camera.pose(),
                Isotropic.Sigma(CAM_POSE3_DOF, 0.1),
            )
        )

        # add initial value for pose
        initial_values.insert(X(camera_idx), camera.pose())

        # process just one camera if shared calibration; otherwise process all
        if camera_idx == 0 or not self._shared_calib:
            # add prior factor for calibration
            graph.push_back(
                PriorFactorCal3Bundler(
                    K(camera_idx),
                    camera.calibration(),
                    Isotropic.Sigma(CAM_CAL3BUNDLER_DOF, 0.1),
                )
            )

            # add initial value for calibration
            initial_values.insert(K(camera_idx), camera.calibration())

    def __add_measurement_factors_and_initial_values(
        self,
        graph: NonlinearFactorGraph,
        initial_values: Values,
        track: SfmTrack,
        track_idx: int,
        measurement_noise: Isotropic,
    ) -> None:
        """Add prior factor for each 2D measurement and initial values for each
        3d point.

        Args:
            graph: factor graph for the problem.
            initial_values: object holding the initial values for camera
                            parameters and 3d points.
            track: [description]
            measurement_noise (Isotropic): [description]
        """

        # Add measurements to the factor graph
        for m_idx in range(track.number_measurements()):
            # i represents the camera index, and uv is the 2d measurement
            i, uv = track.measurement(m_idx)
            graph.add(
                GeneralSFMFactor2Cal3Bundler(
                    uv,
                    measurement_noise,
                    X(i),
                    P(track_idx),
                    K(0 if self._shared_calib else i),
                )
            )

        # add initial value for 3d point
        initial_values.insert(P(track_idx), track.point3())

    def run(self, initial_data: SfmData) -> SfmResult:
        """Run the bundle adjustment by forming factor graph and optimizing using Levenberg–Marquardt optimization.

        Args:
            initial_data: initialized cameras, tracks w/ their 3d landmark from triangulation.

        Results:
            optimized camera poses, 3D point w/ tracks, and error metrics.
        """
        logging.info(
            f"Input: {initial_data.number_tracks()} tracks on {initial_data.number_cameras()} cameras\n"
        )

        # noise model for measurements -- one pixel in u and v
        measurement_noise = Isotropic.Sigma(IMG_MEASUREMENT_DIM, 1.0)

        # Create a factor graph
        graph = NonlinearFactorGraph()

        # Create initial estimate
        initial_values = Values()

        # adding factors for measurements and 3D points's initial values
        for j in range(initial_data.number_tracks()):
            self.__add_measurement_factors_and_initial_values(
                graph,
                initial_values,
                initial_data.track(j),
                j,
                measurement_noise,
            )

        # add prior on all camera poses
        for i in range(initial_data.number_cameras()):
            initialized_cam = initial_data.camera(i)

            self.__add_camera_prior_and_initial_value(
                graph, initial_values, initialized_cam, i
            )

        # Optimize the graph and print results
        try:
            params = LevenbergMarquardtParams()
            params.setVerbosityLM("ERROR")
            lm = LevenbergMarquardtOptimizer(graph, initial_values, params)
            result_values = lm.optimize()
        except RuntimeError:
            logging.exception("LM Optimization failed")
            return

        initial_error = graph.error(initial_values)
        final_error = graph.error(result_values)

        # Error drops from ~2764.22 to ~0.046
        logging.info(f"initial error: {initial_error:.2f}")
        logging.info(f"final error: {final_error:.2f}")

        # construct the results
        optimized_data = self.__values_to_sfm_data(result_values, initial_data)
        sfm_result = SfmResult(optimized_data, final_error)

        return sfm_result

    def create_computation_graph(self, sfm_data_graph: Delayed) -> Delayed:
        """Create the computation graph for performing bundle adjustment.

        Args:
            sfm_data_graph: an SfmData object wrapped up using dask.delayed

        Returns:
            SfmResult wrapped up using dask.delayed
        """
        return dask.delayed(self.run)(sfm_data_graph)

    def __values_to_sfm_data(
        self, values: Values, initial_data: SfmData
    ) -> SfmData:
        """Cast results from the optimization to SfmData object.

        Args:
            values: results of factor graph optimization.
            initial_data: data used to generate the factor graph; used to
                          extract information about poses and 3d points in the
                          graph.

        Returns:
            optimized poses and landmarks.
        """
        result = SfmData()

        # add cameras
        for i in range(initial_data.number_cameras()):
            result.add_camera(
                PinholeCameraCal3Bundler(
                    values.atPose3(X(i)),
                    values.atCal3Bundler(K(0 if self._shared_calib else i)),
                )
            )

        # add tracks
        for j in range(initial_data.number_tracks()):
            input_track = initial_data.track(j)

            # populate the result with optimized 3D point
            result_track = SfmTrack(
                values.atPoint3(P(j)),
            )

            for measurement_idx in range(input_track.number_measurements()):
                i, uv = input_track.measurement(measurement_idx)
                result_track.add_measurement(i, uv)

            result.add_track(result_track)

        return result
