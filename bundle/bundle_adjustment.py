import logging
import sys

import dask
import gtsam
import numpy as np

import matplotlib.pyplot as plt

from dask.delayed import Delayed
from gtsam import (GeneralSFMFactorCal3Bundler, PinholeCameraCal3Bundler,
                   PriorFactorPinholeCameraCal3Bundler, SfmData,
                   symbol_shorthand)

from common.sfmresult import SfmResult

C = symbol_shorthand.C
P = symbol_shorthand.P


logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)


class BundleAdjustmentBase:
    """Base class for bundle adjustment.
    
    This class generates global ratation, translation estimates 
    of cameras, and 3D point clouds structured tracks from 
    triangulation.
    """
    def __init__(self) -> None:
        """ Initialization of the Bundle Adjuster """

    def run(self, scene_data: SfmData) -> SfmResult:
        """ Run LM optimization with input data and report resulting cal3bunder, points, and error

        Args:
            scene_data: structured tracks (SfmData) after Triangularization and Outlier rejection. 
        Results:
            sfm_result: optimized global camera poses, point clouds, and error of optimization
        """
        logging.info(
            f"Input: {scene_data.number_tracks()} tracks on {scene_data.number_cameras()} cameras\n")

        # Create a factor graph
        graph = gtsam.NonlinearFactorGraph()

        # We share *one* noiseModel between all projection factors
        noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v

        # Add measurements to the factor graph
        j = 0
        for t_idx in range(scene_data.number_tracks()):
            track = scene_data.track(t_idx)  # SfmTrack
            # retrieve the SfmMeasurement objects
            for m_idx in range(track.number_measurements()):
                # i represents the camera index, and uv is the 2d measurement
                i, uv = track.measurement(m_idx)
                # note use of shorthand symbols C and P
                graph.add(GeneralSFMFactorCal3Bundler(uv, noise, C(i), P(j)))
            j += 1

        # Add a prior on pose x1. This indirectly specifies where the origin is.
        graph.push_back(
            gtsam.PriorFactorPinholeCameraCal3Bundler(
                C(0), scene_data.camera(0), gtsam.noiseModel.Isotropic.Sigma(9, 0.1)
            )
        )
        # Also add a prior on the position of the first landmark to fix the scale
        graph.push_back(
            gtsam.PriorFactorPoint3(
                P(0), scene_data.track(0).point3(),
                gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
            )
        )

        # Create initial estimate
        initial = gtsam.Values()

        i = 0
        # add each PinholeCameraCal3Bundler
        for cam_idx in range(scene_data.number_cameras()):
            camera = scene_data.camera(cam_idx)
            initial.insert(C(i), camera)
            i += 1

        j = 0
        # add each SfmTrack
        for t_idx in range(scene_data.number_tracks()):
            track = scene_data.track(t_idx)
            initial.insert(P(j), track.point3())
            j += 1

        # Optimize the graph and print results
        try:
            params = gtsam.LevenbergMarquardtParams()
            params.setVerbosityLM("ERROR")
            lm = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
            result = lm.optimize()
        except Exception as e:
            logging.exception("LM Optimization failed")
            return
        
        # Error drops from ~2764.22 to ~0.046
        logging.info(f"initial error: {graph.error(initial)}")
        logging.info(f"final error: {graph.error(result)}")

        # initialize sfmResult container
        sfm_result = SfmResult([], [], graph.error(result))
        
        # read pose
        for key in result.keys():
            try:
                sfm_result.cameras.append(
                    result.atPinholeCameraCal3Bundler(key)
                )
            except RuntimeError:
                continue
        
        # read points
        for key in result.keys():
            try:
                sfm_result.points3d.append(
                    result.atPoint3(key)
                )
            except RuntimeError:
                continue
        
        return sfm_result

    def create_computation_graph(self, scene_data: Delayed) -> Delayed:
        """ Create the computation graph for performing bundle adjustment
         
        Args:
            scene_data: structured tracks (SfmData) after Triangularization and Outlier rejection. 
        Results:
            sfm_result: optimized global camera poses, point clouds, and error of optimization
        """
        return dask.delayed(self.run)(scene_data)
