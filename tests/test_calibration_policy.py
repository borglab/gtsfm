"""Regression tests for common camera assumptions across full-scene BA stages."""

from dataclasses import replace
from unittest.mock import Mock

import pytest

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions
from gtsfm.bundle.global_ba import GlobalBundleAdjustment
from gtsfm.cluster_merging import MergingOptions
from gtsfm.cluster_optimizer.cluster_anysplat import ClusterAnySplat
from gtsfm.cluster_optimizer.cluster_mvo import ClusterMVO
from gtsfm.cluster_optimizer.cluster_optimizer_cacher import ClusterOptimizerCacher
from gtsfm.cluster_optimizer.cluster_vggt import ClusterVGGT
from gtsfm.cluster_optimizer.cluster_vggt_with_frontend import ClusterVGGTWithFrontend
from gtsfm.multi_view_optimizer import MultiViewOptimizer
from gtsfm.scene_optimizer import SceneOptimizer


def _make_mvo(shared_calib: bool) -> ClusterMVO:
    """Build a classical optimizer with nondefault calibration prior strengths."""
    ba = GlobalBundleAdjustment(
        shared_calib=shared_calib,
        use_calibration_prior=True,
        calibration_prior_focal_sigma=7.0,
        calibration_prior_dist_sigma=(0.2, 0.3),
        calibration_prior_pp_sigma=0.001,
    )
    return ClusterMVO(Mock(), Mock(), MultiViewOptimizer(None, None, None, ba))


@pytest.mark.parametrize("shared_calib", [True, False])
@pytest.mark.parametrize("cached", [True, False])
def test_classical_policy_reaches_merge_and_retriangulation(shared_calib: bool, cached: bool) -> None:
    """Classical sharing and all prior strengths survive the separate merge defaults."""
    cluster = _make_mvo(shared_calib)
    policy = cluster.multiview_optimizer.ba_optimizer.calibration_policy
    if cached:
        cluster = ClusterOptimizerCacher(cluster)
    merging = MergingOptions(ba_options=BundleAdjustmentOptions(use_pose_prior_all_cameras=True))
    scene = SceneOptimizer(None, None, cluster, merging_options=merging)
    final_options = scene._merging_options.ba_options
    assert final_options.to_optimizer().calibration_policy == policy
    # Calibration inheritance must preserve stage-specific pose and solver choices.
    assert final_options.use_pose_prior_all_cameras
    assert final_options.robust_ba_mode == merging.ba_options.robust_ba_mode
    assert not merging.ba_options.use_calibration_prior  # caller-owned configuration is unchanged


@pytest.mark.parametrize("cluster_type", [ClusterVGGT, ClusterVGGTWithFrontend, ClusterAnySplat])
@pytest.mark.parametrize("explicit", [True, False])
@pytest.mark.parametrize("cached", [True, False])
def test_feedforward_policy_is_consistent(cluster_type, explicit: bool, cached: bool) -> None:
    """Explicit cluster calibration wins; otherwise merge calibration also reaches cluster BA."""
    options = BundleAdjustmentOptions(shared_calib=False, use_calibration_prior=True, calibration_prior_focal_sigma=3.0)
    kwargs = {"ba_options": options} if explicit else {}
    if cluster_type is ClusterVGGTWithFrontend:
        kwargs.update(correspondence_generator=Mock(), two_view_estimator=Mock())
    cluster = cluster_type(**kwargs)
    wrapped = ClusterOptimizerCacher(cluster) if cached else cluster
    merging = MergingOptions(ba_options=BundleAdjustmentOptions(shared_calib=True))
    expected = (options if explicit else merging.ba_options).calibration_policy
    scene = SceneOptimizer(None, None, wrapped, merging_options=merging)
    assert cluster.ba_options.to_optimizer().calibration_policy == expected
    assert scene._merging_options.ba_options.to_optimizer().calibration_policy == expected


def test_cache_key_tracks_effective_inherited_calibration() -> None:
    """A calibration change cannot reuse geometry produced under different camera assumptions."""
    cluster = ClusterOptimizerCacher(ClusterVGGT())
    policy = BundleAdjustmentOptions().calibration_policy
    cluster.resolve_calibration_policy(policy)
    first = cluster._optimizer_hash
    cluster.resolve_calibration_policy(replace(policy, shared_calib=True))
    second = cluster._optimizer_hash
    assert first != second
    cluster.resolve_calibration_policy(replace(policy, shared_calib=True))
    assert cluster._optimizer_hash == second
