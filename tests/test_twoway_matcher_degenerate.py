"""Regression: ratio-test matching must not crash when one side has <2 usable descriptors
(near-featureless internet images make knnMatch(k=2) return short candidate lists)."""

import numpy as np

from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher


def _match(matcher, d1, d2):
    return matcher.match(
        keypoints_i1=None,
        keypoints_i2=None,
        descriptors_i1=d1,
        descriptors_i2=d2,
        im_shape_i1=(100, 100),
        im_shape_i2=(100, 100),
    )


def test_single_train_descriptor_does_not_crash() -> None:
    matcher = TwoWayMatcher(ratio_test_threshold=0.8)
    rng = np.random.RandomState(0)
    d1 = rng.rand(50, 128).astype(np.float32)
    d2 = rng.rand(1, 128).astype(np.float32)  # 1 descriptor -> knnMatch returns 1-tuples
    result = _match(matcher, d1, d2)
    assert result.size == 0 or result.shape[1] == 2


def test_normal_matching_still_works() -> None:
    matcher = TwoWayMatcher(ratio_test_threshold=0.8)
    rng = np.random.RandomState(1)
    d1 = rng.rand(60, 128).astype(np.float32)
    # copy some rows so genuine matches exist
    d2 = np.vstack([d1[:20] + 0.001, rng.rand(40, 128)]).astype(np.float32)
    result = _match(matcher, d1, d2)
    assert len(result) >= 10
