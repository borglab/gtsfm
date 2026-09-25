"""A/B: TorchTwoWayMatcher must reproduce the OpenCV TwoWayMatcher on realistic descriptors."""

import numpy as np

from gtsfm.frontend.matcher.torch_twoway_matcher import TorchTwoWayMatcher
from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher


def _sift_like(n, seed):
    rng = np.random.RandomState(seed)
    d = rng.rand(n, 128).astype(np.float32)
    return d / np.linalg.norm(d, axis=1, keepdims=True) * 512.0  # SIFT-scale magnitudes


def _match_both(d1, d2, ratio=0.8):
    kwargs = dict(keypoints_i1=None, keypoints_i2=None, im_shape_i1=(100, 100), im_shape_i2=(100, 100))
    a = TwoWayMatcher(ratio_test_threshold=ratio).match(descriptors_i1=d1, descriptors_i2=d2, **kwargs)
    b = TorchTwoWayMatcher(ratio_test_threshold=ratio).match(descriptors_i1=d1, descriptors_i2=d2, **kwargs)
    return a, b


def test_matches_opencv_on_correlated_descriptors() -> None:
    d1 = _sift_like(600, 0)
    noise = _sift_like(600, 1) * 0.02
    d2 = np.vstack([d1[:300] + noise[:300], _sift_like(300, 2)]).astype(np.float32)
    a, b = _match_both(d1, d2)
    set_a = set(map(tuple, a.tolist()))
    set_b = set(map(tuple, b.tolist()))
    assert len(set_a) > 100  # sanity: real matches exist
    jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
    assert jaccard > 0.99, f"jaccard {jaccard}: torch and opencv matchers disagree"


def test_no_ratio_test_mode() -> None:
    d1 = _sift_like(200, 3)
    d2 = np.vstack([d1[:150], _sift_like(50, 4)]).astype(np.float32)
    a, b = _match_both(d1, d2, ratio=None)
    set_a = set(map(tuple, a.tolist()))
    set_b = set(map(tuple, b.tolist()))
    jaccard = len(set_a & set_b) / max(len(set_a | set_b), 1)
    assert jaccard > 0.99


def test_degenerate_single_descriptor() -> None:
    d1 = _sift_like(50, 5)
    d2 = _sift_like(1, 6)
    _, b = _match_both(d1, d2)
    assert b.size == 0 or b.shape[1] == 2
