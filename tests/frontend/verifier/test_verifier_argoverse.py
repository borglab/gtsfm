


import pickle
import pdb
import random
import unittest
from pathlib import Path
from typing import Any, Tuple

import dask
import numpy as np
from gtsam import Cal3Bundler, Pose3, Rot3, Unit3
from scipy.spatial.transform import Rotation

from common.keypoints import Keypoints
from frontend.verifier.degensac import Degensac
from frontend.verifier.ransac import Ransac

ARGOVERSE_TEST_DATA_ROOT = Path(__file__).parent.parent.parent.resolve() / "data" / "argoverse"

RANDOM_SEED = 0


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
        # self.verifier = Ransac()


    def load_annotated_correspondences(self):
        """"
        Annotated from Argoverse
        'train1/273c1883-673a-36bf-b124-88311b1a80be/ring_front_center'

        Image pair annotated at the following timestamps:
            ts1 = 315975640448534784 # nano-second timestamp
            ts2 = 315975643412234000

        with img_names:
          'ring_front_center_315975640448534784.jpg',
          'ring_front_center_315975643412234000.jpg'
        """
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
        
        pdb.set_trace()
        euler_angles = Rotation.from_matrix(i1Ri2).as_euler('zyx', degrees=True)
        gt_euler_angles = np.array([-0.37, 32.47, -0.42])
        assert np.allclose(gt_euler_angles, euler_angles, atol=1.0)

        gt_i1_t_i2 = np.array([ 0.21, -0.0024, 0.976])
        assert np.allclose(gt_i1_t_i2, i1_t_i2, atol=1e-2)
        
