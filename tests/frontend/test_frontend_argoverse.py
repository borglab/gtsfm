# """Unit tests for the the main driver class.

# Authors: John Lambert, Ayush Baid
# """
import pdb
import unittest
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import dask
import numpy as np
from dask.delayed import Delayed
from gtsam import EssentialMatrix, Pose3, Rot3, Unit3
from gtsam import Rot3, Unit3, Cal3Bundler, Pose3 # PinholeCameraCal3Bundler, 
from scipy.spatial.transform import Rotation

# import utils.geometry_comparisons as geometry_comparisons
# from averaging.rotation.shonan import ShonanRotationAveraging
# from averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
# from frontend.verifier.degensac import Degensac
from gtsfm.frontend.verifier.ransac import Ransac

# from scene_optimizer import SceneOptimizer
from gtsfm.loader.folder_loader import FolderLoader


# from averaging.rotation.rotation_averaging_base import RotationAveragingBase
# from averaging.translation.translation_averaging_base import \
#     TranslationAveragingBase
# from bundle.bundle_adjustment import BundleAdjustmentOptimizer
# from data_association.dummy_da import DummyDataAssociation
from gtsfm.frontend.detector_descriptor.detector_descriptor_base import \
DetectorDescriptorBase
from gtsfm.frontend.matcher.matcher_base import MatcherBase
from gtsfm.frontend.verifier.verifier_base import VerifierBase


DATA_ROOT_PATH = Path(__file__).resolve().parent / 'tests' / 'data'



class TestFrontend(unittest.TestCase):
	"""Tests a combined FeatureExtractor and TwoViewEstimator using Argoverse image pair
	"""

	def setUp(self) -> None:
		""" """
		self.loader = FolderLoader(
			str(DATA_ROOT_PATH / "argoverse" / "train1" / "273c1883-673a-36bf-b124-88311b1a80be" / "ring_front_center"), image_extension='jpg'
		)
		assert len(self.loader)
		det_desc = SIFTDetectorDescriptor()
		self.feature_extractor = FeatureExtractor(det_desc)
		self.two_view_estimator = TwoViewEstimator(
			matcher=TwoWayMatcher(),
			verifier=Ransac()
		)


	def test_frontend_result(self):
		""" Compare recovered relative rotation and translation with ground truth."""
		num_images = len(self.loader)
		image_pair_indices = self.loader.get_valid_pairs()

		image_graph = self.loader.create_computation_graph_for_images()
		camera_intrinsics_graph = self.loader.create_computation_graph_for_intrinsics()
		use_intrinsics_in_verification = True

		####### copied from scene optimizer ############
		# detection and description graph
		keypoints_graph_list = []
		descriptors_graph_list = []
		for delayed_image in image_graph:
			(
			delayed_dets,
			delayed_descs,
			) = self.feature_extractor.create_computation_graph(delayed_image)
			keypoints_graph_list += [delayed_dets]
			descriptors_graph_list += [delayed_descs]

		# estimate two-view geometry and get indices of verified correspondences.
		i2Ri1_graph_dict = {}
		i2Ui1_graph_dict = {}
		v_corr_idxs_graph_dict = {}
		for (i1, i2) in image_pair_indices:
			(
			i2Ri1,
			i2Ui1,
			v_corr_idxs,
			) = self.two_view_estimator.create_computation_graph(
				keypoints_graph_list[i1],
				keypoints_graph_list[i2],
				descriptors_graph_list[i1],
				descriptors_graph_list[i2],
				camera_intrinsics_graph[i1],
				camera_intrinsics_graph[i2],
				use_intrinsics_in_verification,
			)
			i2Ri1_graph_dict[(i1, i2)] = i2Ri1
			i2Ui1_graph_dict[(i1, i2)] = i2Ui1
			v_corr_idxs_graph_dict[(i1, i2)] = v_corr_idxs
			####### copied from scene optimizer ############

		pdb.set_trace()
		with dask.config.set(scheduler='single-threaded'):
			i2Ri1_list = dask.compute(i2Ri1_graph)[0]
			i2ti1_list = dask.compute(i2Ui1_graph)[0]
			v_corr_idxs_list = dask.compute(v_corr_idxs_graph)[0]

		pdb.set_trace()

		i2Ri1 = i2Ri1_list[(0,1)]
		i2Ui1 = i2ti1_list[(0,1)]

		euler_angle_err_tol = 1.4
		translation_err_tol = 0.026

		# Ground truth is provided in inverse format, so invert SE(3) object
		i2Ti1 = Pose3(i2Ri1, i2Ui1.point3())
		i1Ti2 = i2Ti1.inverse()
		i1ti2 = i1Ti2.translation()
		i1Ri2 = i1Ti2.rotation().matrix()

		pdb.set_trace()

		euler_angles = Rotation.from_matrix(i1Ri2).as_euler('zyx', degrees=True)
		gt_euler_angles = np.array([-0.37, 32.47, -0.42])
		assert np.allclose(gt_euler_angles, euler_angles, atol=euler_angle_err_tol)

		gt_i1ti2 = np.array([ 0.21, -0.0024, 0.976])
		assert np.allclose(gt_i1ti2, i1ti2, atol=translation_err_tol)


if __name__ == "__main__":
	unittest.main()
