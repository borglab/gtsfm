"""
Unit tests on homography estimation.

Author: John Lambert
"""

import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.homography import HomographyEstimator


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
    num_inliers, inlier_ratio = estimator.estimate(keypoints_i1, keypoints_i2, match_indices)
    assert inlier_ratio == 1.0
    assert num_inliers == 4


# def test_estimate_homography_inliers_corrupted() -> None:
#     """Fit homography on set of 6 correspondences, w/ 2 outliers."""

#     # fmt: off
#     uv_i1 = np.array(
#     	[
#     		[0,0],
#     		[1,0],
#     		[1,1],
#     		[0,1],
#     		[0,1000], # outlier
#     		[0,2000] # outlier
#     	]
#     )

#     # # 2x multiplier on uv_i1
#     # uv_i2 = np.array(
#     # 	[
#     # 		[0,0],
#     # 		[2,0],
#     # 		[2,2],
#     # 		[0,2],
#     # 		[500,0], # outlier
#     # 		[1000,0] # outlier
#     # 	]
#     # )

#  #    # fmt: on
# 	# keypoints_i1 = Keypoints(coordinates=uv_i1)
# 	# keypoints_i2 = Keypoints(coordinates=uv_i2)
# 	# # fmt: off
# 	# match_indices = np.array(
# 	# 	[
# 	# 		[0,0],
# 	# 		[1,1],
# 	# 		[2,2],
# 	# 		[3,3],
# 	# 		[4,4],
# 	# 		[5,5]
# 	# 	]
# 	# )

# 	# fmt: on
# 	estimator = HomographyEstimator()
# 	num_inliers, inlier_ratio = estimator.estimate(
# 		keypoints_i1,
# 		keypoints_i2,
# 		match_indices
# 	)

# 	assert inlier_ratio == 4/6
# 	assert num_inliers == 4

#     # # TODO: add case from virtual plane, and real camera geometry