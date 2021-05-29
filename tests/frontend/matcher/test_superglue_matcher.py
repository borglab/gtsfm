
import numpy as np

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.superglue_matcher import SuperGlueMatcher


def test_SuperGlueMatcher() -> None:
    """Ensure the SuperGlue matcher returns output of the correct shape, for random input."""
    # image height and width
    H, W = 20, 20

    im_shape_i1 = (H,W)
    im_shape_i2 = (H,W)

    num_kps_i1 = 50
    kps_i1 = Keypoints(coordinates=np.random.randint(0, img_height, size=(num_kps_i1, 2)), responses=np.random.rand(50))
    descs_i1 = np.random.randn(num_kps_i1, 256)

    num_kps_i2 = 100
    kps_i2 = Keypoints(coordinates=np.random.randint(0, img_height, size=(num_kps_i2, 2)), responses=np.random.rand(100))
    descs_i2 = np.random.randn(num_kps_i2, 256)

    matcher = SuperGlueMatcher()

    match_indices = matcher.match(kps_i1, kps_i2, descs_i1, descs_i2, im_shape_i1, im_shape_i2)
    assert isinstance(match_indices, np.ndarray)
    assert match_indices.dtype == np.int64
