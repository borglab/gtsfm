"""FLANN preserves mutual matching and ratio filtering while using approximate Euclidean search."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import tests.frontend.matcher.test_twoway_withratiotest_matcher as ratio_tests
import tests.frontend.matcher.test_twoway_withoutratiotest_matcher as no_ratio_tests
from gtsfm.frontend.matcher.twoway_matcher import MatchingDistanceType, TwoWayMatcher


class TestFlannWithRatio(ratio_tests.TestTwoWayWithRatioTestMatcher):
    def setUp(self):
        super().setUp()
        self.matcher = TwoWayMatcher(ratio_test_threshold=0.8)


class TestFlannWithoutRatio(no_ratio_tests.TestTwoWayWithoutRatioTestMatcher):
    def setUp(self):
        super().setUp()
        self.matcher = TwoWayMatcher()


def _match(matcher, first, second):
    return matcher.match(None, None, first, second, (100, 100, 3), (100, 100, 3))


@pytest.mark.parametrize("count", [0, 1])
def test_ratio_test_handles_too_few_descriptors(count):
    matcher = TwoWayMatcher(search_method="flann", ratio_test_threshold=0.8)
    first = np.ones((count, 128), dtype=np.float32)
    second = np.ones((3, 128), dtype=np.float32)
    assert _match(matcher, first, second).size == 0
    assert _match(matcher, second, first).size == 0


def test_single_descriptor_without_ratio_test():
    matcher = TwoWayMatcher(search_method="flann")
    descriptor = np.ones((1, 128), dtype=np.float32)
    np.testing.assert_array_equal(_match(matcher, descriptor, descriptor), [[0, 0]])


def test_brute_force_binary_matching_and_older_serialized_matchers_remain_supported():
    matcher = TwoWayMatcher(
        distance_type=MatchingDistanceType.HAMMING, ratio_test_threshold=0.8, search_method="brute_force"
    )
    # An in-flight Dask task or old pickle may have been created before these options existed.
    del matcher._search_method
    del matcher._flann_checks
    del matcher._flann_trees
    descriptors = np.random.default_rng(3).integers(0, 256, size=(20, 32), dtype=np.uint8)
    result = _match(matcher, descriptors, descriptors)
    assert set(map(tuple, result)) == {(i, i) for i in range(20)}
    assert matcher.cache_key == "TwoWayMatcher"


def test_filters_invalid_rows_and_remaps_noncontiguous_float64_descriptors():
    matcher = TwoWayMatcher(search_method="flann", ratio_test_threshold=0.8)
    # Clear nearest neighbors; NaN/Inf rows must not enter the index or shift returned indices.
    first = np.array([[np.nan, 0], [0.0, 0], [np.inf, 0], [10.0, 0], [20.0, 0]])[:, :1]
    second = np.array([[20.1, 0], [np.nan, 0], [0.1, 0], [10.1, 0]])[:, :1]
    result = _match(matcher, first, second)
    assert set(map(tuple, result)) == {(1, 2), (3, 3), (4, 0)}
    assert _match(matcher, first[:1], second).size == 0


def test_same_matcher_can_be_used_by_concurrent_pairs():
    matcher = TwoWayMatcher(search_method="flann", ratio_test_threshold=0.8)
    rng = np.random.default_rng(14)
    descriptors = rng.normal(size=(100, 128)).astype(np.float32)
    order = rng.permutation(len(descriptors))
    other = descriptors[order] + 0.001
    expected = {(int(original), int(shuffled)) for shuffled, original in enumerate(order)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: _match(matcher, descriptors, other), range(8)))
    assert all(set(map(tuple, result)) == expected for result in results)


@pytest.mark.parametrize("kwargs", [
    {"search_method": "unknown"},
    {"search_method": "flann", "distance_type": MatchingDistanceType.HAMMING},
    {"flann_checks": 0},
    {"flann_trees": 0},
])
def test_rejects_invalid_search_configuration(kwargs):
    with pytest.raises(ValueError):
        TwoWayMatcher(**kwargs)


def test_rejects_binary_descriptors_for_kd_tree_search():
    matcher = TwoWayMatcher(search_method="flann", ratio_test_threshold=0.8)
    descriptors = np.ones((3, 32), dtype=np.uint8)
    with pytest.raises(ValueError, match="floating-point"):
        _match(matcher, descriptors, descriptors)


def test_approximate_caches_are_separate_and_include_search_settings(tmp_path, monkeypatch):
    import gtsfm.frontend.cacher.matcher_cacher as cache_module
    from gtsfm.common.keypoints import Keypoints

    monkeypatch.setattr(cache_module, "CACHE_ROOT_PATH", tmp_path)
    descriptors = np.array([[0.0], [10.0], [20.0]], dtype=np.float32)
    keypoints = Keypoints(coordinates=np.zeros((3, 2)))
    matchers = [
        TwoWayMatcher(search_method="brute_force", ratio_test_threshold=0.8),
        TwoWayMatcher(ratio_test_threshold=0.8),
        TwoWayMatcher(search_method="flann", ratio_test_threshold=0.8, flann_checks=128),
        TwoWayMatcher(search_method="flann", ratio_test_threshold=0.8, flann_trees=8),
        TwoWayMatcher(search_method="flann", ratio_test_threshold=0.7),
    ]
    for matcher in matchers:
        cached = cache_module.MatcherCacher(matcher)
        args = (keypoints, keypoints, descriptors, descriptors, (100, 100, 3), (100, 100, 3))
        first = cached.match(*args)

        def unexpected_match(*args, **kwargs):
            pytest.fail("The second call should reuse this matcher's cache")

        monkeypatch.setattr(matcher, "match", unexpected_match)
        np.testing.assert_array_equal(cached.match(*args), first)
    assert len(list((tmp_path / "matcher").glob("*.pbz2"))) == len(matchers)
    assert matchers[0].cache_key == "TwoWayMatcher"  # Existing brute-force caches remain reusable.


@pytest.mark.parametrize("preset", ["megaloc_sift_metis", "megaloc_sift_single", "megaloc_sift_gp_metis"])
def test_presets_use_tested_flann_defaults(preset):
    from hydra import compose, initialize_config_module
    from hydra.utils import instantiate

    with initialize_config_module(config_module="gtsfm.configs", version_base=None):
        config = compose(config_name=preset)
    cached_matcher = instantiate(config.cluster_optimizer.correspondence_generator.matcher)
    assert cached_matcher._matcher.cache_key == "TwoWayMatcher_flann_trees4_checks256_ratio0.8"


@pytest.mark.parametrize("preset", ["orb", "brisk"])
def test_binary_presets_explicitly_use_brute_force(preset):
    from pathlib import Path
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    config = OmegaConf.load(Path(__file__).resolve().parents[3] / "gtsfm/configs/correspondence" / f"{preset}.yaml")
    matcher = instantiate(config.CorrespondenceGenerator.matcher)._matcher
    descriptors = np.random.default_rng(3).integers(0, 256, size=(20, 32), dtype=np.uint8)
    result = _match(matcher, descriptors, descriptors)
    assert set(map(tuple, result)) == {(i, i) for i in range(20)}
    assert matcher.cache_key == "TwoWayMatcher"
