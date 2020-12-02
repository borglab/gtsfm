


import pickle
import random
import unittest
from typing import Any, Tuple

import dask
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Pose3, Rot3, Unit3

from common.keypoints import Keypoints
from frontend.verifier.degensac import Degensac

ARGOVERSE_TEST_DATA_ROOT = Path(__file__).parent.parent.resolve() / "argoverse"

# def plot_argoverse_epilines_from_annotated_correspondences(img1: np.ndarray, img2: np.ndarray, K: np.ndarray):
# 	""" """

# 	cam2_E_cam1, inlier_mask = cv2.findEssentialMat(img1_kpts, img2_kpts, K, method=cv2.RANSAC, threshold=0.1)
	
# 	print('Num inliers: ', inlier_mask.sum())
# 	cam2_F_cam1 = get_fmat_from_emat(cam2_E_cam1, K1=K, K2=K)
# 	_num_inlier, cam2_R_cam1, cam2_t_cam1, _ = cv2.recoverPose(cam2_E_cam1, img1_kpts, img2_kpts, mask=inlier_mask)

# 	r = Rotation.from_matrix(cam2_R_cam1)
# 	print('cam2_R_cam1 recovered from correspondences', r.as_euler('zyx', degrees=True))
# 	print('cam2_t_cam1: ', np.round(cam2_t_cam1.squeeze(), 2))

# 	cam2_SE3_cam1 = SE3(cam2_R_cam1, cam2_t_cam1.squeeze() )
# 	cam1_SE3_cam2 = cam2_SE3_cam1.inverse()
# 	cam1_R_cam2 = cam1_SE3_cam2.rotation
# 	cam1_t_cam2 = cam1_SE3_cam2.translation

# 	r = Rotation.from_matrix(cam1_R_cam2)
# 	print('cam1_R_cam2: ', r.as_euler('zyx', degrees=True)) ## prints "[-0.32  33.11 -0.45]"
# 	print('cam1_t_cam2: ', np.round(cam1_t_cam2,2)) ## [0.21 0.   0.98]

# 	pdb.set_trace()
# 	draw_epilines(img1_kpts, img2_kpts, img1, img2, cam2_F_cam1)
# 	plt.show()

# 	draw_epipolar_lines(cam2_F_cam1, img1, img2, img1_kpts, img2_kpts)
# 	plt.show()



# def main():

# 	log_id = '273c1883-673a-36bf-b124-88311b1a80be'
# 	dataset_dir = '/Users/johnlambert/Downloads/visual-odometry-tutorial/train1'

# 	# img_names = [
# 	# 	'ring_front_center_315975640448534784.jpg',
# 	# 	'ring_front_center_315975643412234000.jpg'
# 	# ]
# 	img_dir = '/Users/johnlambert/Downloads/visual-odometry-tutorial/train1/273c1883-673a-36bf-b124-88311b1a80be/ring_front_center'
# 	dataset_name = 'argoverse'

# 	ts1 = 315975640448534784 # nano-second timestamp
# 	ts2 = 315975643412234000

# 	img1_fpath = f'{img_dir}/ring_front_center_{ts1}.jpg'
# 	img2_fpath = f'{img_dir}/ring_front_center_{ts2}.jpg'

# 	img1 = imageio.imread(img1_fpath).astype(np.float32) / 255
# 	img2 = imageio.imread(img2_fpath).astype(np.float32) / 255
# 	# plt.imshow(img)
# 	# plt.show()

# 	if dataset_name == 'argoverse':
# 		calib_fpath = '/Users/johnlambert/Downloads/visual-odometry-tutorial/train1/273c1883-673a-36bf-b124-88311b1a80be/vehicle_calibration_info.json'
# 		calib_dict = load_calib(calib_fpath)
# 		K = calib_dict['ring_front_center'].K[:3,:3]


# 	plot_argoverse_epilines_from_annotated_correspondences(img1, img2, K)


# if __name__ == '__main__':
# 	main()

def load_pickle_file(pkl_fpath: str) -> Any:
    """ Loads data serialized using the pickle library """
    with open(str(pkl_fpath), 'rb') as f:
        d = pickle.load(f)
    return d


class TestVerifierBase(unittest.TestCase):
    """Unit tests for the Base Verifier class.

    Should be inherited by all verifier unit tests.
    """

    def setUp(self):
        super().setUp()

        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)

        self.verifier = Degensac()


    def load_annotated_correspondences(self):
        """" """
        fname = 'argoverse_315975640448534784__315975643412234000.pkl'
        pkl_fpath = ARGOVERSE_TEST_DATA_ROOT / f'labeled_correspondences/{fname}'
        d = load_pickle_file(pkl_fpath)

        X1 = np.array(d['x1'])
        Y1 = np.array(d['y1'])
        X2 = np.array(d['x2'])
        Y2 = np.array(d['y2'])

        img1_uv = np.hstack([ X1.reshape(-1,1), Y1.reshape(-1,1) ]).astype(np.int32)
        img2_uv = np.hstack([ X2.reshape(-1,1), Y2.reshape(-1,1) ]).astype(np.int32)
        
        keypoints_i1 = Keypoints(img1_uv)
        keypoints_i2 = Keypoints(img2_uv)

        return keypoints_i1, keypoints_i2


    def test_with_annotated_correspondences(self):
        """
        """
        keypoints_i1, keypoints_i2 = self.load_annotated_correspondences()

        pdb.set_trace()
        # match keypoints row by row
        match_indices = np.vstack(
            [
                np.arange(len(keypoints_i1)),
                np.arange(len(keypoints_i1))
            ]).T

        computed_i2Ri1, computed_i2ti1, verified_indices = self.verifier.verify_with_approximate_intrinsics(
            keypoints_i1,
            keypoints_i2,
            match_indices,
            Cal3Bundler(),
            Cal3Bundler()
        )
        pdb.set_trace()
        # self.assertTrue(computed_i2Ei1.equals(
        #     expected_i2Ei1, 1e-2))
        # np.testing.assert_array_equal(verified_indices, match_indices)


    # def test_verify_empty_matches(self):
    #     """Tests the output when there are no match indices."""

    #     keypoints_i1 = generate_random_keypoints(10, [250, 300])
    #     keypoints_i2 = generate_random_keypoints(12, [400, 300])
    #     match_indices = np.array([], dtype=np.int32)
    #     intrinsics_i1 = Cal3Bundler()
    #     intrinsics_i2 = Cal3Bundler()

    #     i2Ei1, verified_indices = self.verifier.verify_with_exact_intrinsics(
    #         keypoints_i1, keypoints_i2, match_indices, intrinsics_i1, intrinsics_i2
    #     )

    #     self.assertIsNone(i2Ei1)
    #     self.assertEqual(0, verified_indices.size)

    # def test_create_computation_graph(self):
    #     """Checks that the dask computation graph produces the same results as
    #     direct APIs."""

    #     # Set up 3 pairs of inputs to the verifier
    #     num_images = 6
    #     image_indices = [(0, 1), (4, 3), (2, 5)]

    #     # creating inputs for verification and use GTSFM's direct API to get
    #     # expected results
    #     keypoints_list = [None]*num_images
    #     matches_dict = dict()
    #     intrinsics_list = [None]*num_images

    #     expected_results = dict()
    #     for (i1, i2) in image_indices:
    #         keypoints_i1, keypoints_i2, matches_i1i2, \
    #             intrinsics_i1, intrinsics_i2 = \
    #             generate_random_input_for_verifier()

    #         keypoints_list[i1] = keypoints_i1
    #         keypoints_list[i2] = keypoints_i2

    #         matches_dict[(i1, i2)] = matches_i1i2

    #         intrinsics_list[i1] = intrinsics_i1
    #         intrinsics_list[i2] = intrinsics_i2

    #         verification_result_i1i2 = \
    #             self.verifier.verify_with_exact_intrinsics(
    #                 keypoints_i1,
    #                 keypoints_i2,
    #                 matches_i1i2,
    #                 intrinsics_i1,
    #                 intrinsics_i2
    #             )

    #         expected_results[(i1, i2)] = verification_result_i1i2

    #     # Convert the inputs to computation graphs
    #     detection_graph = [dask.delayed(x) for x in keypoints_list]
    #     matcher_graph = {image_indices: dask.delayed(match) for
    #                      (image_indices, match) in matches_dict.items()}
    #     intrinsics_graph = [dask.delayed(x) for x in intrinsics_list]

    #     # generate the computation graph for the verifier
    #     computation_graph = self.verifier.create_computation_graph(
    #         detection_graph,
    #         matcher_graph,
    #         intrinsics_graph,
    #         exact_intrinsics_flag=True
    #     )

    #     with dask.config.set(scheduler='single-threaded'):
    #         dask_results = dask.compute(computation_graph)[0]

    #     # compare the lengths of two results dictionaries
    #     self.assertEqual(len(expected_results), len(dask_results))

    #     # compare the values in two dictionaries
    #     for indices_i1i2 in dask_results.keys():
    #         i2Ei1_dask, verified_indices_i1i2_dask = dask_results[indices_i1i2]

    #         i2Ei1_expected, verified_indices_i1i2_expected = \
    #             expected_results[indices_i1i2]

    #         if i2Ei1_expected is None:
    #             self.assertIsNone(i2Ei1_dask)
    #         else:
    #             self.assertTrue(i2Ei1_expected.equals(i2Ei1_dask, 1e-2))

    #         np.testing.assert_array_equal(
    #             verified_indices_i1i2_expected, verified_indices_i1i2_dask)







