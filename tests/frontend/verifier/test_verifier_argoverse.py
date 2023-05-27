"""Ensure that Verifier classes can compute relative pose for Argoverse image pairs.

Authors: John Lambert
"""

import pickle
import random
import unittest
from pathlib import Path
from typing import Any, Tuple

import numpy as np
from gtsam import Cal3Bundler, Pose3
from scipy.spatial.transform import Rotation

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.verifier.ransac import Ransac
from gtsfm.frontend.verifier.verifier_base import VerifierBase

ARGOVERSE_TEST_DATA_ROOT = Path(__file__).parent.parent.parent.resolve() / "data" / "argoverse"

RANDOM_SEED = 0


def load_pickle_file(pkl_fpath: str) -> Any:
    """Loads data serialized using the pickle library"""
    with open(str(pkl_fpath), "rb") as f:
        d = pickle.load(f)
    return d


def load_argoverse_log_annotated_correspondences():
    """Annotated from Argoverse, ring front-center camera, from vehicle log subdir:
    'train1/273c1883-673a-36bf-b124-88311b1a80be/ring_front_center'

    Image pair annotated at the following timestamps:
        ts1 = 315975640448534784 # nano-second timestamp
        ts2 = 315975643412234000

    with img_names:
      'ring_front_center_315975640448534784.jpg',
      'ring_front_center_315975643412234000.jpg'
    """
    fname = "argoverse_315975640448534784__315975643412234000.pkl"
    pkl_fpath = ARGOVERSE_TEST_DATA_ROOT / f"labeled_correspondences/{fname}"
    d = load_pickle_file(pkl_fpath)

    X1 = np.array(d["x1"])
    Y1 = np.array(d["y1"])
    X2 = np.array(d["x2"])
    Y2 = np.array(d["y2"])

    img1_uv = np.hstack([X1.reshape(-1, 1), Y1.reshape(-1, 1)]).astype(np.float32)
    img2_uv = np.hstack([X2.reshape(-1, 1), Y2.reshape(-1, 1)]).astype(np.float32)

    keypoints_i1 = Keypoints(img1_uv)
    keypoints_i2 = Keypoints(img2_uv)

    return keypoints_i1, keypoints_i2


def load_log_front_center_intrinsics() -> Tuple[float, float, float, float, float]:
    """Provide camera parameters for front-center camera Vehicle log ID: 273c1883-673a-36bf-b124-88311b1a80be"""
    fx = 1392.1069298937407  # also fy
    px = 980.1759848618066
    py = 604.3534182680304

    k1 = 0.0
    k2 = 0.0
    return fx, px, py, k1, k2


def check_verifier_output_error(verifier: VerifierBase, euler_angle_err_tol: float, translation_err_tol: float) -> None:
    """Check error using annotated correspondences as input, instead of noisy detector-descriptor matches."""
    fx, px, py, k1, k2 = load_log_front_center_intrinsics()
    keypoints_i1, keypoints_i2 = load_argoverse_log_annotated_correspondences()

    # match keypoints row by row
    match_indices = np.vstack([np.arange(len(keypoints_i1)), np.arange(len(keypoints_i1))]).T

    i2Ri1, i2ti1, _, _ = verifier.apply(
        keypoints_i1, keypoints_i2, match_indices, Cal3Bundler(fx, k1, k2, px, py), Cal3Bundler(fx, k1, k2, px, py)
    )

    # Ground truth is provided in inverse format, so invert SE(3) object
    i2Ti1 = Pose3(i2Ri1, i2ti1.point3())
    i1Ti2 = i2Ti1.inverse()
    i1ti2 = i1Ti2.translation()
    i1Ri2 = i1Ti2.rotation().matrix()

    euler_angles = Rotation.from_matrix(i1Ri2).as_euler("zyx", degrees=True)
    gt_euler_angles = np.array([-0.37, 32.47, -0.42])
    gt_euler_angles_str = str(np.round(gt_euler_angles, 1))
    euler_angles_str = str(np.round(euler_angles, 1))
    assert np.allclose(
        gt_euler_angles, euler_angles, atol=euler_angle_err_tol
    ), f"GT {gt_euler_angles_str} vs. Est. {euler_angles_str} w/ tol {euler_angle_err_tol:.1f}"

    gt_i1ti2 = np.array([0.21, -0.0024, 0.976])
    gt_i1ti2_str = str(np.round(gt_i1ti2, 1))
    i1ti2_str = str(np.round(i1ti2, 1))
    assert np.allclose(
        gt_i1ti2, i1ti2, atol=translation_err_tol
    ), f"t: GT {gt_i1ti2_str} vs. Est. {i1ti2_str} w/ tol {translation_err_tol:.2f}"


class TestRansacVerifierArgoverse(unittest.TestCase):
    def setUp(self):
        super().setUp()

        np.random.seed(RANDOM_SEED)
        random.seed(RANDOM_SEED)
        self.verifier = Ransac(use_intrinsics_in_verification=True, estimation_threshold_px=0.5)

        self.euler_angle_err_tol = 1.0
        self.translation_err_tol = 0.01

    def testRecoveredPoseError(self):
        check_verifier_output_error(self.verifier, self.euler_angle_err_tol, self.translation_err_tol)

    def test_5pt_algo_5correspondences(self) -> None:
        """ """
        fx, px, py, k1, k2 = load_log_front_center_intrinsics()
        keypoints_i1, keypoints_i2 = load_argoverse_log_annotated_correspondences()

        # match keypoints row by row
        match_indices = np.vstack([np.arange(len(keypoints_i1)), np.arange(len(keypoints_i1))]).T

        intrinsics_i1 = Cal3Bundler(fx, k1, k2, px, py)
        intrinsics_i2 = Cal3Bundler(fx, k1, k2, px, py)

        match_indices = match_indices[:5]

        i2Ri1, i2ti1, _, _ = self.verifier.apply(
            keypoints_i1, keypoints_i2, match_indices, intrinsics_i1, intrinsics_i2
        )


# TODO(johnlambert): fix this test, as it seems to be consistently failing.
# class TestDegensacVerifierArgoverse(unittest.TestCase):
#     def setUp(self):
#         super().setUp()
#
#         np.random.seed(RANDOM_SEED)
#         random.seed(RANDOM_SEED)
#         self.verifier = Degensac(use_intrinsics_in_verification=False, estimation_threshold_px=0.5)
#
#         self.euler_angle_err_tol = 2.0
#         self.translation_err_tol = 0.02
#
#     def testRecoveredPoseError(self):
#         check_verifier_output_error(self.verifier, self.euler_angle_err_tol, self.translation_err_tol)
#         assert True
