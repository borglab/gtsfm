


import pickle
import pdb
import random
import unittest
from pathlib import Path
from typing import Any, Tuple

import dask
import numpy as np
from gtsam import Cal3Bundler, EssentialMatrix, Pose3, Rot3, Unit3
from scipy.spatial.transform import Rotation

from common.keypoints import Keypoints
# from frontend.verifier.degensac import Degensac
from frontend.verifier.ransac import Ransac

ARGOVERSE_TEST_DATA_ROOT = Path(__file__).parent.parent.parent.resolve() / "data" / "argoverse"

RANDOM_SEED = 0

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

        #self.verifier = Degensac()
        self.verifier = Ransac()


    def load_annotated_correspondences(self):
        """" """
        fname = 'argoverse_315975640448534784__315975643412234000.pkl'
        pkl_fpath = ARGOVERSE_TEST_DATA_ROOT / f'labeled_correspondences/{fname}'
        d = load_pickle_file(pkl_fpath)

        X1 = np.array(d['x1'])
        Y1 = np.array(d['y1'])
        X2 = np.array(d['x2'])
        Y2 = np.array(d['y2'])

        img1_uv = np.hstack([ X1.reshape(-1,1), Y1.reshape(-1,1) ]).astype(np.float32)
        img2_uv = np.hstack([ X2.reshape(-1,1), Y2.reshape(-1,1) ]).astype(np.float32)
        
        keypoints_i1 = Keypoints(img1_uv)
        keypoints_i2 = Keypoints(img2_uv)

        return keypoints_i1, keypoints_i2

    def load_intrinsics(self):
        """ """
        fx = 1392.1069298937407 # also fy
        px = 980.1759848618066
        py = 604.3534182680304

        k1 = 0
        k2 = 0
        return fx, px, py, k1, k2

    def test_with_annotated_correspondences(self):
        """
        """
        fx, px, py, k1, k2 = self.load_intrinsics()
        keypoints_i1, keypoints_i2 = self.load_annotated_correspondences()

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
            Cal3Bundler(fx, k1, k2, px, py),
            Cal3Bundler(fx, k1, k2, px, py)
        )
        # Ground truth is provided in inverse format, so invert SE(3) object
        i2Ti1 = Pose3(computed_i2Ri1, computed_i2ti1.point3())
        i1Ti2 = i2Ti1.inverse()
        i1_t_i2 = i1Ti2.translation()
        i1Ri2 = i1Ti2.rotation().matrix()
        euler_angles = Rotation.from_matrix(i1Ri2).as_euler('zyx', degrees=True)
        gt_euler_angles = np.array([-0.37, 32.47, -0.42])
        pdb.set_trace()
        assert np.allclose(gt_euler_angles, euler_angles, atol=1.0)
        print(euler_angles)
        # print('Euler: ', np.round(euler_angles, 2))
        gt_i1_t_i2 = np.array([ 0.21, -0.0024, 0.976])
        print('t: ', i1_t_i2)

        assert np.allclose(gt_i1_t_i2, i1_t_i2, atol=1e-2)
        

        # self.assertTrue(computed_i2Ei1.equals(
        #     expected_i2Ei1, 1e-2))
        # np.testing.assert_array_equal(verified_indices, match_indices)

