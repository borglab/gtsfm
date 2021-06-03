
from typing import Tuple

import cv2
import numpy as np

from gtsfm.common.keypoints import Keypoints

#from gtsfm.frontend.verifier.verifier_base import TwoViewEstimationReport

import gtsfm.utils.logger as logger_utils

logger = logger_utils.get_logger()

"""
Verification that determines whether the scene is planar or the camera motion is a pure rotation.

COLMAP also checks degeneracy of structure here:
    https://github.com/colmap/colmap/blob/dev/src/estimators/two_view_geometry.cc#L277
"""

MIN_PTS_HOMOGRAPHY = 4
REPROJECTION_THRESHOLD_PX = 0.5
DEFAULT_RANSAC_PROB = 0.999


class HomographyEstimator:

	def estimate(
		self,
		keypoints_i1: Keypoints,
		keypoints_i2: Keypoints,
		match_indices: np.ndarray
	) -> Tuple[float, int]:
		"""Estimate to what extent the correspondences agree with an estimated homography.

		We provide statistics of the RANSAC result, like COLMAP does here for LORANSAC:
	        https://github.com/colmap/colmap/blob/dev/src/optim/loransac.h
		
		Args:
	        keypoints_i1: detected features in image #i1.
	        keypoints_i2: detected features in image #i2.
	        match_indices: matches as indices of features from both images, of shape (N3, 2), where N3 <= min(N1, N2).

		Returns:
		    inlier_ratio: i.e. ratio of correspondences which approximately agree with planar geometry
		    num_inliers: number of correspondence consistent with estimated homography H
		"""
		if match_indices.shape[0] < MIN_PTS_HOMOGRAPHY:
			num_inliers = 0
			inlier_ratio = 0.0
			return num_inliers, inlier_ratio

		uv_i1 = keypoints_i1.coordinates
		uv_i2 = keypoints_i2.coordinates

		# TODO: cast as np.float32?
		H, inlier_mask = cv2.findHomography(
			srcPoints=uv_i1[match_indices[:, 0]],
			dstPoints=uv_i2[match_indices[:, 1]],
			method=cv2.RANSAC,
			ransacReprojThreshold=REPROJECTION_THRESHOLD_PX,
			#maxIters=10000,
			confidence=DEFAULT_RANSAC_PROB
		)

		inlier_idxs = np.where(inlier_mask.ravel() == 1)[0]
		inlier_ratio = inlier_mask.mean()

		num_inliers = inlier_mask.sum()
		return num_inliers, inlier_ratio
		# report = TwoViewEstimationReport(
		# 	success=num_inliers >= MIN_PTS_HOMOGRAPHY,
		# 	num_inliers=num_inliers,
		# 	inlier_mask=inlier_mask.squeeze(),
		# 	i2Ri1=None,
		# 	i2Ui1=None,
		# 	v_corr_idxs=match_indices[inlier_idxs]
		# )

		# return report


def test_estimate_homography_inliers_minimalset() -> None:
	"""Fit homography on minimal set of 4 correspondences, w/ no outliers."""

	# fmt: off
	uv_i1 = np.array(
		[
			[0,0],
			[1,0],
			[1,1],
			[0,1]
		]
	)

	uv_i2 = np.array(
		[
			[0,0],
			[2,0],
			[2,2],
			[0,2]
		]
	)

	# fmt: on
	keypoints_i1 = Keypoints(coordinates=uv_i1)
	keypoints_i2 = Keypoints(coordinates=uv_i2)
	# fmt: off
	match_indices = np.array(
		[
			[0,0],
			[1,1],
			[2,2],
			[3,3]
		]
	)

	# fmt: on
	estimator = HomographyEstimator()
	num_inliers, inlier_ratio = estimator.estimate(
		keypoints_i1,
		keypoints_i2,
		match_indices
	)
	assert inlier_ratio == 1.0
	assert num_inliers == 4


def test_estimate_homography_inliers_corrupted() -> None:
	"""Fit homography on set of 6 correspondences, w/ 2 outliers."""

	# fmt: off
	uv_i1 = np.array(
		[
			[0,0],
			[1,0],
			[1,1],
			[0,1],
			[0,1000], # outlier
			[0,2000], # outlier
		]
	)

	# 2x multiplier on uv_i1
	uv_i2 = np.array(
		[
			[0,0],
			[2,0],
			[2,2],
			[0,2],
			[500,0], # outlier
			[1000,0] # outlier
		]
	)

	# fmt: on
	keypoints_i1 = Keypoints(coordinates=uv_i1)
	keypoints_i2 = Keypoints(coordinates=uv_i2)
	# fmt: off
	match_indices = np.array(
		[
			[0,0],
			[1,1],
			[2,2],
			[3,3],
			[4,4],
			[5,5]
		]
	)
	
	# fmt: on
	estimator = HomographyEstimator()
	num_inliers, inlier_ratio = estimator.estimate(
		keypoints_i1,
		keypoints_i2,
		match_indices
	)
	
	assert inlier_ratio == 4/6
	assert num_inliers == 4

# TODO: add case from virtual plane, and real camera geometry
