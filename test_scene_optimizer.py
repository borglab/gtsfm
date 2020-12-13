# """Unit tests for the the main driver class.

# Authors: John Lambert, Ayush Baid
# """
import pdb
import unittest
from pathlib import Path
# from typing import List, Optional

# import dask
import cv2
import imageio
import numpy as np
# from gtsam import EssentialMatrix, Pose3, Rot3, Unit3

# import utils.geometry_comparisons as geometry_comparisons
# from averaging.rotation.shonan import ShonanRotationAveraging
# from averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from frontend.matcher.twoway_matcher import TwoWayMatcher
# from frontend.verifier.degensac import Degensac
from frontend.verifier.ransac import Ransac

# from scene_optimizer import SceneOptimizer
from loader.folder_loader import FolderLoader



DATA_ROOT_PATH = Path(__file__).resolve().parent / 'tests' / 'data'




def show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2, line_colors=None):
	"""
	Visualizes corresponding points between two images by drawing a line segment
	between the two images for each (x1,y1) (x2,y2) pair.

	Args:
	- imgA: A numpy array of shape (M,N,3)
	- imgB: A numpy array of shape (D,E,3)
	- x1: A numpy array of shape (k,) containing x-locations of keypoints in imgA
	- y1: A numpy array of shape (k,) containing y-locations of keypoints in imgA
	- x2: A numpy array of shape (j,) containing x-locations of keypoints in imgB
	- y2: A numpy array of shape (j,) containing y-locations of keypoints in imgB
	- line_colors: A numpy array of shape (N x 3) with colors of correspondence lines (optional)

	Returns:
	- newImg: A numpy array of shape (max(M,D), N+E, 3)
	"""
	newImg = hstack_images(imgA, imgB)
	shiftX = imgA.shape[1]
	X1 = X1.astype(np.int)
	Y1 = Y1.astype(np.int)
	X2 = X2.astype(np.int)
	Y2 = Y2.astype(np.int)

	dot_colors = np.random.rand(len(X1), 3)
	if line_colors is None:
		line_colors = dot_colors

	for x1, y1, x2, y2, dot_color, line_color in zip(X1, Y1, X2, Y2, dot_colors,line_colors):
		newImg = cv2.circle(newImg, (x1, y1), 10, dot_color, -1)
		newImg = cv2.circle(newImg, (x2+shiftX, y2), 10, dot_color, -1)
		newImg = cv2.line(newImg, (x1, y1), (x2+shiftX, y2), line_color, 5, cv2.LINE_AA)
	return newImg


def hstack_images(imgA, imgB):
    """
    Stacks 2 images side-by-side and creates one combined image.

    Args:
    - imgA: A numpy array of shape (M,N,3) representing rgb image
    - imgB: A numpy array of shape (D,E,3) representing rgb image

    Returns:
    - newImg: A numpy array of shape (max(M,D), N+E, 3)
    """
    Height = max(imgA.shape[0], imgB.shape[0])
    Width  = imgA.shape[1] + imgB.shape[1]

    newImg = np.zeros((Height, Width, 3), dtype=imgA.dtype)
    newImg[:imgA.shape[0], :imgA.shape[1], :] = imgA
    newImg[:imgB.shape[0], imgA.shape[1]:, :] = imgB

    return newImg






class TestSceneOptimizer(unittest.TestCase):
	"""[summary]

	Args:
	unittest ([type]): [description]
	"""

	def setUp(self) -> None:

		self.loader = FolderLoader(
			str(DATA_ROOT_PATH / "argoverse/train1/273c1883-673a-36bf-b124-88311b1a80be/ring_front_center"), image_extension='jpg'
		)
		assert len(self.loader)
		# self.obj = SceneOptimizer(
		# 	,
		# 	,
		# 	verifier=Degensac(),
		# 	rot_avg_module=ShonanRotationAveraging(),
		# 	trans_avg_module=TranslationAveraging1DSFM()
		# )
		self.det_desc = SIFTDetectorDescriptor()
		self.matcher = TwoWayMatcher()
		self.verifier = Ransac()

	def test_create_computation_graph(self):
		""" """
		
		exact_intrinsics_flag = True
		images = [self.loader.get_image(i) for i in range(2)]
		
		joint_graph = [self.det_desc.detect_and_describe(x) for x in images]

		keypoints_list = [x[0] for x in joint_graph]
		descriptors_list = [x[1] for x in joint_graph]

		# perform matching and verification on valid image pairs in the loader
		verification_function = self.verifier.verify_with_exact_intrinsics \
			if exact_intrinsics_flag else \
		self.verifier.verify_with_approximate_intrinsics

		for (i1, i2) in [(0,1)]:
			match_correspondence_indices = self.matcher.match(
				descriptors_list[i1],
				descriptors_list[i2]
			)

		i2Ri1, i2Ui1, verified_correspondence_indices = \
		verification_function(
			keypoints_list[i1],
			keypoints_list[i2],
			match_correspondence_indices,
			self.loader.get_camera_intrinsics(i1),
			self.loader.get_camera_intrinsics(i2),
		)

		euler_angle_err_tol = 1.4
		translation_err_tol = 0.02

		# Ground truth is provided in inverse format, so invert SE(3) object
		i2Ti1 = Pose3(i2Ri1, i2Ui1.point3())
		i1Ti2 = i2Ti1.inverse()
		i1ti2 = i1Ti2.translation()
		i1Ri2 = i1Ti2.rotation().matrix()

		euler_angles = Rotation.from_matrix(i1Ri2).as_euler('zyx', degrees=True)
		gt_euler_angles = np.array([-0.37, 32.47, -0.42])
		assert np.allclose(gt_euler_angles, euler_angles, atol=euler_angle_err_tol)

		gt_i1ti2 = np.array([ 0.21, -0.0024, 0.976])
		assert np.allclose(gt_i1ti2, i1ti2, atol=translation_err_tol)

		pdb.set_trace()

		X1 = ''
		Y1 = ''
		X2 = ''
		Y2 = ''
		imgA = images[0]
		imgB = images[1]
		show_correspondence_lines(imgA, imgB, X1, Y1, X2, Y2)



		#return detection_graph, description_graph


		# # run normally without dask
		# expected_keypoints_list, \
		# expected_global_rotations, \
		# expected_global_translations, \
		# expected_verified_corr_indices = self.obj.run(
		# self.loader, exact_intrinsics=exact_intrinsics_flag)

#         expected_wTi_list = [Pose3(wRi, wti) if wti is not None else None for (
#             wRi, wti) in zip(expected_global_rotations, expected_global_translations)]

#         # generate the dask computation graph
#         keypoints_graph, \
#             global_rotations_graph, \
#             global_translations_graph, \
#             verified_corr_graph = self.obj.create_computation_graph(
#                 len(self.loader),
#                 self.loader.get_valid_pairs(),
#                 self.loader.create_computation_graph_for_images(),
#                 self.loader.create_computation_graph_for_intrinsics(),
#                 exact_intrinsics=exact_intrinsics_flag
#             )

#         with dask.config.set(scheduler='single-threaded'):
#             computed_keypoints_list = dask.compute(keypoints_graph)[0]
#             computed_global_rotations = dask.compute(
#                 global_rotations_graph)[0]
#             computed_global_translations = dask.compute(
#                 global_translations_graph)[0]
#             computed_verified_corr_indices = dask.compute(
#                 verified_corr_graph)[0]

#         computed_wTi_list = [Pose3(wRi, wti) if wti is not None else None for (
#             wRi, wti) in zip(computed_global_rotations, computed_global_translations)]

#         # compute the number of length of lists and dictionaries
#         self.assertEqual(len(computed_keypoints_list),
#                          len(expected_keypoints_list))
#         self.assertEqual(len(computed_global_rotations),
#                          len(expected_global_rotations))
#         self.assertEqual(len(computed_global_translations),
#                          len(expected_global_translations))
#         self.assertEqual(len(computed_verified_corr_indices),
#                          len(expected_verified_corr_indices))

#         # compare keypoints for all indices
#         self.assertListEqual(computed_keypoints_list, expected_keypoints_list)

#         # assert global rotations and translations
#         self.assertTrue(geometry_comparisons.compare_rotations(
#             computed_global_rotations, expected_global_rotations))
#         self.assertTrue(geometry_comparisons.compare_global_poses(
#             computed_wTi_list, expected_wTi_list))


#     def test_find_largest_connected_component(self):
#         """Tests the function to prune the scene graph to its largest connected
#         component."""

#         # create a graph with two connected components of length 4 and 3.
#         input_essential_matrices = {
#             (0, 1): generate_random_essential_matrix(),
#             (1, 5): None,
#             (1, 3): generate_random_essential_matrix(),
#             (3, 2): generate_random_essential_matrix(),
#             (2, 7): None,
#             (4, 6): generate_random_essential_matrix(),
#             (6, 7): generate_random_essential_matrix()
#         }

#         # generate Rot3 and Unit3 inputs
#         input_relative_rotations = dict()
#         input_relative_unit_translations = dict()
#         for (i1, i2), i2Ei1 in input_essential_matrices.items():
#             if i2Ei1 is None:
#                 input_relative_rotations[(i1, i2)] = None
#                 input_relative_unit_translations[(i1, i2)] = None
#             else:
#                 input_relative_rotations[(i1, i2)] = i2Ei1.rotation()
#                 input_relative_unit_translations[(i1, i2)] = i2Ei1.direction()

#         expected_edges = [(0, 1), (3, 2), (1, 3)]

#         computed_relative_rotations, computed_relative_unit_translations = \
#             GTSFM.select_largest_connected_component(
#                 input_relative_rotations, input_relative_unit_translations)

#         # check the edges in the pruned graph
#         self.assertCountEqual(
#             list(computed_relative_rotations.keys()), expected_edges)
#         self.assertCountEqual(
#             list(computed_relative_unit_translations.keys()), expected_edges)

#         # check the actual Rot3 and Unit3 values
#         for (i1, i2) in expected_edges:
#             self.assertTrue(
#                 computed_relative_rotations[(i1, i2)].equals(
#                     input_relative_rotations[(i1, i2)], 1e-2))
#             self.assertTrue(
#                 computed_relative_unit_translations[(i1, i2)].equals(
#                     input_relative_unit_translations[(i1, i2)], 1e-2))


# def generate_random_essential_matrix() -> EssentialMatrix:
#     rotation_angles = np.random.uniform(
#         low=0.0, high=2*np.pi, size=(3,))
#     R = Rot3.RzRyRx(
#         rotation_angles[0], rotation_angles[1], rotation_angles[2])
#     t = np.random.uniform(
#         low=-1.0, high=1.0, size=(3, ))

#     return EssentialMatrix(R, Unit3(T))


if __name__ == "__main__":
    unittest.main()
