"""
Base class for bundle adjustment module.

Accepts intrinsics, point-correspondence constraints, and initialization
of global rotations and translations, to optimize final camera poses
and point locations.

L and X represent landmark and vehicle/camera poses respectively

Authors: John Lambert
"""
import abc
from typing import Dict, List, Tuple

import numpy as np

import gtsam
from gtsam import symbol_shorthand_L as L
from gtsam import symbol_shorthand_X as X
from gtsam.gtsam import (
	Cal3_S2,
	DoglegOptimizer,
	GenericProjectionFactorCal3_S2,
	Marginals,
	NonlinearFactorGraph,
	PinholeCameraCal3_S2,
	Point3,
	Pose3,
	PriorFactorPoint3,
	PriorFactorPose3,
	Rot3,
	SimpleCamera,
	Values
)

MEASUREMENT_NOISE = gtsam.noiseModel_Isotropic.Sigma(2, 1.0)  # one pixel in u and v


class BundleAdjustmentBase(metaclass=abc.ABCMeta):
	"""Base class for all rotation averaging algorithms."""

	@abc.abstractmethod
	def run(
		self,
		intrinsics: List[np.ndarray],
		global_rotations: List[gtsam.Rot3],
		global_translations: List[gtsam.Point3],
		correspondence_dict: Dict[Tuple[int,int],np.ndarray],
		) -> Tuple[List[gtsam.Rot3], np.ndarray]:
		"""
		Based off of the example in:
		https://github.com/borglab/gtsam/blob/develop/cython/gtsam/examples/SFMExample.py
		gtsam/cython/gtsam/examples/VisualISAM2Example.py

		Args:
			intrinsics: Length N list with 3x3 matrices for each camera ID
				[fx,s,px]
				[0,fy,py]
				[0, 0, 1]
			global_rotations: Length N list of global rotations
				For each image index, we have an
				estimated of its global rotation to world frame
				TODO: decide if its camera_R_world, world_R_camera
			global_translations: Length N list of globbal translations
				list index is image index
			correspondence_dict: dictionary from (i,j) camera
				pair to Nx4 array, with each matrix row
				representing x_i,y_i,x_j,y_j

		Returns:
			optimized_poses: output of BA. poses are indexed in the list
				by their image index
			point_cloud: Numpy array of shape Nx3
		"""
		assert len(intrinsics) == len(global_rotations)
		assert len(global_rotations) == len(global_translations)

		# Create a factor graph
		graph = NonlinearFactorGraph()

		# TODO: the sigma parameters seem like magic numbers? should make module-level constants?
		# Add a prior on pose x1. This indirectly specifies where the origin is.
		# 0.3 rad std on roll,pitch,yaw and 0.1m on x,y,z
		pose_noise = gtsam.noiseModel_Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
		factor = PriorFactorPose3(X(0), poses[0], pose_noise)
		graph.push_back(factor)

		# Create the data structure to hold the initial estimate to the solution
		initial_estimate = Values()

		# Simulated measurements from each camera pose, adding them to the factor graph
		for k, K in enumerate(intrinsics):
			camera_R_world = global_rotations[k]
			camera_t_world = global_translations[k]

			camera_SE3_world = gtsam.Pose3(
				gtsam.Rot3(camera_R_world),
				gtsam.Point3(camera_t_world.flatten())
			)
			initial_estimate.insert(X(k), camera_SE3_world)

			camera = PinholeCameraCal3_S2(camera_SE3_world, K)

			# TODO: vectorize data structure so no double-for loop here
			# TODO: Will the same measurement appear at (i,j) and at (j,i), or only once?
			for (i,j), correspondences in correspondence_dict.items():
				for (x_i, y_i, x_j, y_j) in correspondences:
					if i == k:
						measurement = np.array([x_i, y_i])
					if j == k:
						measurement = np.array([x_j, y_j])

					factor = GenericProjectionFactorCal3_S2(
						measurement,
						MEASUREMENT_NOISE,
						X(k),
						L(l_idx), # how should we index the landmarks? Need a counter?
						K
					)
					graph.push_back(factor)
					# initial_estimate.insert(L(l_idx), transformed_point) # Get initial triangulation?

		# Because the structure-from-motion problem has a scale ambiguity, the problem is still under-constrained
		# Here we add a prior on the position of the first landmark. This fixes the scale by indicating the distance
		# between the first camera and the first landmark. All other landmark positions are interpreted using this scale.
		point_noise = gtsam.noiseModel_Isotropic.Sigma(3, 0.1)
		factor = PriorFactorPoint3(L(0), points[0], point_noise)
		graph.push_back(factor)
		graph.print_('Factor Graph:\n')

		initial_estimate.print_('Initial Estimates:\n')

		# TODO: Frank -- Is Dogleg preferred? vs L.M.
		# Optimize the graph and print results
		params = gtsam.DoglegParams()
		params.setVerbosity('TERMINATION')
		optimizer = DoglegOptimizer(graph, initial_estimate, params)
		print('Optimizing:')
		result = optimizer.optimize()
		result.print_('Final results:\n')
		print('initial error = {}'.format(graph.error(initial_estimate)))
		print('final error = {}'.format(graph.error(result)))

		optimized_poses = None
		point_cloud = None
		return optimized_poses, point_cloud



