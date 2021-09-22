
"""

Test cases come from COLMAP https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc
which in turn take values from OpenCV.

Author: John Lambert (Python)
"""

import numpy as np

import gtsfm.utils.homography_decomposition as homography_utils


def test_decompose_homography_matrix() -> None:
	"""
	See https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc#L44

	# Note that the test case values are obtained from OpenCV.
	"""
	H = np.array(
	  	[
		  	[ 2.649157564634028, 4.583875997496426, 70.694447785121326 ],
		    [  -1.072756858861583, 3.533262150437228, 1513.656999614321649 ],
		    [  0.001303887589576, 0.003042206876298, 1 ]
	    ]
	)
  	H *= 3

	K = np.array(
		[
			[640, 0, 320],
			[0, 640, 240],
			[0, 0, 1]
		]
	)

	R_cmbs, t_cmbs, n_cmbs = decompose_homography_matrix(
	    H, K1=K, K2=K
	)

	assert len(R) == 4
	assert len(t) == 4
	assert len(n) == 4

	R_ref = np.array(
		[
			[ 0.43307983549125, 0.545749113549648, -0.717356090899523 ],
			[ -0.85630229674426, 0.497582023798831, -0.138414255706431 ],
			[ 0.281404038139784, 0.67421809131173, 0.682818960388909 ]
     	]
    )
	t_ref = np.array([1.826751712278038, 1.264718492450820, 0.195080809998819])
	n_ref = np.array([-0.244875830334816, -0.480857890778889, -0.841909446789566])

	ref_solution_exists = false

	kEps = 1e-6
	for i in range(4):
		if ((R[i] - R_ref).norm() < kEps and (t[i] - t_ref).norm() < kEps and (n[i] - n_ref).norm() < kEps):
			ref_solution_exists = True

	assert ref_solution_exists



def test_pose_from_homography_matrix() -> None:
	"""

	See: https://github.com/colmap/colmap/blob/dev/src/base/homography_matrix_test.cc#L120
	"""
	K1 = np.eye(3)
	K2 = np.eye(3)
	R_ref = np.eye(3)
	t_ref = np.array([1, 0, 0])
	n_ref = np.array([-1, 0, 0])
	d_ref = 1
	const Eigen::Matrix3d H = HomographyMatrixFromPose(K1, K2, R_ref, t_ref, n_ref, d_ref)

	# fmt: off
	points1 = np.array(
		[
			[0.1, 0.4],
			[0.2, 0.3],
			[0.3, 0.2],
			[0.4, 0.1]
		]
	)
	# fmt: on

	points2 = np.zeros((0,2))
	for point1 in points1:
		const Eigen::Vector3d point2 = H * point1.homogeneous()
		points2.push_back(point2.hnormalized())

	R, t, n, points3D = pose_from_homography_matrix(
	    H,
	    K1,
	    K2,
	    points1,
	    points2,
	)

	np.testing.assert_almost_equal(R, R_ref)
	np.testing.assert_almost_equal(t, t_ref)
	np.testing.assert_almost_equal(n, n_ref)
	assert points3D.shape == points1.shape




def test_check_cheirality() -> None:
	""" """
	R = ""
	t = ""
	points1 = ""
	points2 = ""
	points3D = check_cheirality(R, t, points1, points2)

	assert False


def test_compute_opposite_of_minor() -> None:
	""" """
	matrix = np.zeros()
	row = -1
	col = -1
	value = compute_opposite_of_minor(matrix, row, col)

	assert False