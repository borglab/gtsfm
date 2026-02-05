"""Unit tests for EdgeQualityScore dataclass."""

import math

from gtsfm.products.edge_quality import EdgeQualityScore


def test_is_bad_high_reproj_error():
    """Edge with mean reprojection error exceeding threshold is bad."""
    score = EdgeQualityScore(num_supporting_tracks=5, mean_reproj_error_px=6.0, max_reproj_error_px=8.0, track_coverage_ratio=0.5)
    assert score.is_bad(max_reproj_error_px=5.0, min_track_coverage=0.1)


def test_is_bad_low_track_coverage():
    """Edge with track coverage below threshold is bad."""
    score = EdgeQualityScore(num_supporting_tracks=1, mean_reproj_error_px=1.0, max_reproj_error_px=2.0, track_coverage_ratio=0.05)
    assert score.is_bad(max_reproj_error_px=5.0, min_track_coverage=0.1)


def test_is_good_passes_both():
    """Edge passing both thresholds is not bad."""
    score = EdgeQualityScore(num_supporting_tracks=10, mean_reproj_error_px=2.0, max_reproj_error_px=4.0, track_coverage_ratio=0.5)
    assert not score.is_bad(max_reproj_error_px=5.0, min_track_coverage=0.1)


def test_is_bad_boundary_error_not_exceeded():
    """Edge with error exactly at threshold is NOT bad (strictly >)."""
    score = EdgeQualityScore(num_supporting_tracks=5, mean_reproj_error_px=5.0, max_reproj_error_px=5.0, track_coverage_ratio=0.5)
    assert not score.is_bad(max_reproj_error_px=5.0, min_track_coverage=0.1)


def test_is_bad_boundary_coverage_not_below():
    """Edge with coverage exactly at threshold is NOT bad (strictly <)."""
    score = EdgeQualityScore(num_supporting_tracks=5, mean_reproj_error_px=2.0, max_reproj_error_px=3.0, track_coverage_ratio=0.1)
    assert not score.is_bad(max_reproj_error_px=5.0, min_track_coverage=0.1)


def test_is_bad_both_fail():
    """Edge failing both signals is bad."""
    score = EdgeQualityScore(num_supporting_tracks=1, mean_reproj_error_px=10.0, max_reproj_error_px=15.0, track_coverage_ratio=0.01)
    assert score.is_bad(max_reproj_error_px=5.0, min_track_coverage=0.1)


def test_is_bad_inf_error():
    """Edge with infinite reprojection error (no valid projections) is bad."""
    score = EdgeQualityScore(num_supporting_tracks=0, mean_reproj_error_px=float("inf"), max_reproj_error_px=float("inf"), track_coverage_ratio=0.0)
    assert score.is_bad()


def test_is_bad_custom_thresholds():
    """Stricter custom thresholds can flag edges that pass defaults."""
    score = EdgeQualityScore(num_supporting_tracks=5, mean_reproj_error_px=2.0, max_reproj_error_px=3.0, track_coverage_ratio=0.3)
    # Passes default thresholds
    assert not score.is_bad(max_reproj_error_px=5.0, min_track_coverage=0.1)
    # Fails stricter thresholds
    assert score.is_bad(max_reproj_error_px=1.0, min_track_coverage=0.1)


def test_is_bad_default_thresholds():
    """Verify default threshold values work correctly."""
    good = EdgeQualityScore(num_supporting_tracks=5, mean_reproj_error_px=2.0, max_reproj_error_px=3.0, track_coverage_ratio=0.5)
    assert not good.is_bad()

    bad = EdgeQualityScore(num_supporting_tracks=1, mean_reproj_error_px=6.0, max_reproj_error_px=8.0, track_coverage_ratio=0.5)
    assert bad.is_bad()


def test_frozen_dataclass():
    """EdgeQualityScore is immutable."""
    score = EdgeQualityScore(num_supporting_tracks=5, mean_reproj_error_px=2.0, max_reproj_error_px=3.0, track_coverage_ratio=0.5)
    try:
        score.num_supporting_tracks = 10  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # Expected
