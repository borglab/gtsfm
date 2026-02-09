"""Unit tests for edge quality computation utilities.

Tests build synthetic GtsfmData with cameras and tracks using GTSAM primitives,
following the same patterns as tests/utils/test_reprojection.py.
"""

import json
from pathlib import Path

import numpy as np
import pytest
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point2, Point3, Pose3, Rot3, SfmTrack

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.edge_quality import EdgeQualityGraph, EdgeQualityScore
from gtsfm.products.visibility_graph import prune_edges
from gtsfm.utils.edge_quality import (
    aggregate_edge_quality,
    compute_edge_quality,
    export_edge_quality_to_json,
    identify_bad_edges,
    load_bad_edges_from_json,
    merge_edge_quality,
)

# ---------------------------------------------------------------------------
# Helpers: build synthetic cameras, tracks, and GtsfmData for testing.
# Following the pattern from tests/utils/test_reprojection.py.
# ---------------------------------------------------------------------------

# Shared calibration: focal=10, k1=k2=0, principal point (3, 4)
_K = Cal3Bundler(10, 0, 0, 3, 4)


def _make_camera(x: float, y: float, z: float) -> PinholeCameraCal3Bundler:
    """Create a camera at the given position with identity rotation."""
    return PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(x, y, z)), _K)


def _make_track(point3d: Point3, cameras: dict[int, PinholeCameraCal3Bundler], noise: dict[int, Point2] | None = None) -> SfmTrack:
    """Create an SfmTrack by projecting a 3D point into the given cameras.

    Args:
        point3d: The 3D landmark.
        cameras: Dict of camera_idx -> camera to project into.
        noise: Optional dict of camera_idx -> pixel noise to add (for creating bad measurements).

    Returns:
        SfmTrack with measurements.
    """
    track = SfmTrack(point3d)
    for idx in sorted(cameras.keys()):
        uv = cameras[idx].project(point3d)
        if noise is not None and idx in noise:
            uv = uv + noise[idx]
        track.addMeasurement(idx, uv)
    return track


def _make_scene_3_cameras() -> tuple[GtsfmData, dict[int, PinholeCameraCal3Bundler]]:
    """Build a 3-camera scene with 3 tracks and known geometry.

    Camera layout:
        cam0 at (0,0,0), cam1 at (2,-2,0), cam2 at (4,0,0)

    Tracks (all with perfect measurements -> 0 reprojection error):
        Track 0: pt (1,2,1) seen by [cam0, cam1]       -> supports edge (0,1)
        Track 1: pt (3,1,1) seen by [cam1, cam2]       -> supports edge (1,2)
        Track 2: pt (2,1,1) seen by [cam0, cam1, cam2] -> supports edges (0,1), (0,2), (1,2)
    """
    cameras = {
        0: _make_camera(0, 0, 0),
        1: _make_camera(2, -2, 0),
        2: _make_camera(4, 0, 0),
    }

    track0 = _make_track(Point3(1, 2, 1), {0: cameras[0], 1: cameras[1]})
    track1 = _make_track(Point3(3, 1, 1), {1: cameras[1], 2: cameras[2]})
    track2 = _make_track(Point3(2, 1, 1), cameras)

    scene = GtsfmData(number_images=3, cameras=cameras, tracks=[track0, track1, track2])
    return scene, cameras


# ===========================================================================
# Tests for compute_edge_quality()
# ===========================================================================


class TestComputeEdgeQuality:
    """Tests for the main compute_edge_quality function."""

    def test_none_reconstruction(self):
        """None GtsfmData -> all edges get inf error and 0 tracks."""
        vis_graph = [(0, 1), (1, 2)]
        result = compute_edge_quality(None, vis_graph)

        assert len(result) == 2
        for edge in vis_graph:
            score = result[edge]
            assert score.num_supporting_tracks == 0
            assert score.mean_reproj_error_px == float("inf")

    def test_empty_reconstruction_no_tracks(self):
        """GtsfmData with cameras but no tracks -> same as None."""
        cameras = {0: _make_camera(0, 0, 0), 1: _make_camera(1, 0, 0)}
        scene = GtsfmData(number_images=2, cameras=cameras)

        vis_graph = [(0, 1)]
        result = compute_edge_quality(scene, vis_graph)

        assert result[(0, 1)].num_supporting_tracks == 0
        assert result[(0, 1)].mean_reproj_error_px == float("inf")

    def test_good_edge_low_error(self):
        """Perfect measurements -> ~0 reprojection error."""
        scene, cameras = _make_scene_3_cameras()
        vis_graph = [(0, 1)]

        result = compute_edge_quality(scene, vis_graph)
        score = result[(0, 1)]

        # Track 0 and Track 2 both span edge (0,1)
        assert score.num_supporting_tracks == 2
        # Perfect projection -> 0 error
        np.testing.assert_allclose(score.mean_reproj_error_px, 0.0, atol=1e-6)
        np.testing.assert_allclose(score.max_reproj_error_px, 0.0, atol=1e-6)

    def test_bad_edge_high_error(self):
        """Noisy measurements -> high reprojection error."""
        cameras = {
            0: _make_camera(0, 0, 0),
            1: _make_camera(2, -2, 0),
        }
        # Add 20px noise to cam1 measurement
        noisy_track = _make_track(
            Point3(1, 2, 1),
            cameras,
            noise={1: Point2(20.0, -10.0)},
        )
        scene = GtsfmData(number_images=2, cameras=cameras, tracks=[noisy_track])

        vis_graph = [(0, 1)]
        result = compute_edge_quality(scene, vis_graph)
        score = result[(0, 1)]

        assert score.num_supporting_tracks == 1
        # cam0 has 0 error, cam1 has ~22.36px error -> mean should be ~11.18
        assert score.mean_reproj_error_px > 5.0
        assert score.max_reproj_error_px > 10.0

    def test_edge_with_no_supporting_tracks(self):
        """Edge in visibility graph but no track spans it -> inf error, 0 tracks."""
        # Create a scene where no track spans (0,2) by only using tracks 0 and 1
        cameras = {
            0: _make_camera(0, 0, 0),
            1: _make_camera(2, -2, 0),
            2: _make_camera(4, 0, 0),
        }
        track0 = _make_track(Point3(1, 2, 1), {0: cameras[0], 1: cameras[1]})
        track1 = _make_track(Point3(3, 1, 1), {1: cameras[1], 2: cameras[2]})
        scene_no_02 = GtsfmData(number_images=3, cameras=cameras, tracks=[track0, track1])

        vis_graph = [(0, 1), (0, 2), (1, 2)]
        result = compute_edge_quality(scene_no_02, vis_graph)

        # Edge (0,2) has no supporting tracks
        assert result[(0, 2)].num_supporting_tracks == 0
        assert result[(0, 2)].mean_reproj_error_px == float("inf")
        # Edges (0,1) and (1,2) do have tracks
        assert result[(0, 1)].num_supporting_tracks == 1
        assert result[(1, 2)].num_supporting_tracks == 1

    def test_multi_camera_track_supports_multiple_edges(self):
        """A track spanning 3 cameras supports C(3,2)=3 edges."""
        scene, _ = _make_scene_3_cameras()
        # Track 2 spans all 3 cameras: supports (0,1), (0,2), (1,2)
        vis_graph = [(0, 1), (0, 2), (1, 2)]

        result = compute_edge_quality(scene, vis_graph)

        # Edge (0,1): supported by Track 0 (2-camera) + Track 2 (3-camera) = 2 tracks
        assert result[(0, 1)].num_supporting_tracks == 2
        # Edge (0,2): supported by Track 2 only = 1 track
        assert result[(0, 2)].num_supporting_tracks == 1
        # Edge (1,2): supported by Track 1 + Track 2 = 2 tracks
        assert result[(1, 2)].num_supporting_tracks == 2

    def test_empty_visibility_graph(self):
        """Empty visibility graph -> empty result."""
        scene, _ = _make_scene_3_cameras()
        result = compute_edge_quality(scene, [])
        assert result == {}


# ===========================================================================
# Tests for identify_bad_edges()
# ===========================================================================


class TestIdentifyBadEdges:
    """Tests for the bad edge identification function."""

    def test_no_bad_edges(self):
        """All edges pass thresholds -> empty set."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.0, 2.0),
            (1, 2): EdgeQualityScore(3, 2.0, 3.0),
        }
        bad = identify_bad_edges(quality, max_reproj_error_px=5.0)
        assert bad == set()

    def test_all_bad_edges(self):
        """All edges fail -> full set."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(1, 10.0, 15.0),
            (1, 2): EdgeQualityScore(0, float("inf"), float("inf")),
        }
        bad = identify_bad_edges(quality)
        assert bad == {(0, 1), (1, 2)}

    def test_mixed_edges(self):
        """Some good, some bad -> correct subset."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.0, 2.0),  # good
            (1, 2): EdgeQualityScore(1, 8.0, 12.0),  # bad: high error
            (0, 2): EdgeQualityScore(1, 2.0, 3.0),   # good
        }
        bad = identify_bad_edges(quality, max_reproj_error_px=5.0)
        assert bad == {(1, 2)}

    def test_empty_input(self):
        """Empty quality dict -> empty set."""
        assert identify_bad_edges({}) == set()

    def test_custom_thresholds(self):
        """Stricter thresholds catch more edges."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 2.0, 3.0),
            (1, 2): EdgeQualityScore(3, 4.0, 5.0),
        }
        # Default thresholds: both pass
        assert identify_bad_edges(quality) == set()
        # Strict error threshold: (1,2) fails
        assert identify_bad_edges(quality, max_reproj_error_px=3.0) == {(1, 2)}


# ===========================================================================
# Tests for merge_edge_quality()
# ===========================================================================


class TestMergeEdgeQuality:
    """Tests for merging edge quality from multiple clusters."""

    def test_single_score_passthrough(self):
        """Single score returns identical values."""
        score = EdgeQualityScore(5, 2.0, 3.0)
        merged = merge_edge_quality([score])
        assert merged == score

    def test_merge_worst_case(self):
        """Multiple scores: max error, sum tracks."""
        scores = [
            EdgeQualityScore(3, 1.0, 2.0),
            EdgeQualityScore(5, 3.0, 4.0),
        ]
        merged = merge_edge_quality(scores)
        assert merged.num_supporting_tracks == 8  # 3 + 5
        assert merged.mean_reproj_error_px == 3.0  # max(1.0, 3.0)
        assert merged.max_reproj_error_px == 4.0  # max(2.0, 4.0)

    def test_empty_raises(self):
        """Empty list raises ValueError."""
        with pytest.raises(ValueError, match="Cannot merge empty list"):
            merge_edge_quality([])

    def test_merge_three_scores(self):
        """Merging 3 scores uses worst-case across all."""
        scores = [
            EdgeQualityScore(2, 1.0, 2.0),
            EdgeQualityScore(3, 5.0, 6.0),
            EdgeQualityScore(1, 2.0, 3.0),
        ]
        merged = merge_edge_quality(scores)
        assert merged.num_supporting_tracks == 6
        assert merged.mean_reproj_error_px == 5.0
        assert merged.max_reproj_error_px == 6.0


# ===========================================================================
# Tests for aggregate_edge_quality()
# ===========================================================================


class TestAggregateEdgeQuality:
    """Tests for aggregating edge quality across clusters."""

    def test_non_overlapping_clusters(self):
        """Disjoint edges from different clusters -> union unchanged."""
        cluster_a: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.0, 2.0),
        }
        cluster_b: EdgeQualityGraph = {
            (2, 3): EdgeQualityScore(3, 2.0, 3.0),
        }
        result = aggregate_edge_quality([cluster_a, cluster_b])
        assert len(result) == 2
        assert result[(0, 1)] == cluster_a[(0, 1)]
        assert result[(2, 3)] == cluster_b[(2, 3)]

    def test_overlapping_edge(self):
        """Same edge in 2 clusters -> merged via worst-case."""
        cluster_a: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(3, 1.0, 2.0),
        }
        cluster_b: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 3.0, 4.0),
        }
        result = aggregate_edge_quality([cluster_a, cluster_b])
        assert len(result) == 1
        merged = result[(0, 1)]
        assert merged.num_supporting_tracks == 8  # 3 + 5
        assert merged.mean_reproj_error_px == 3.0  # max

    def test_empty_clusters(self):
        """Empty list of clusters -> empty result."""
        result = aggregate_edge_quality([])
        assert result == {}

    def test_single_cluster(self):
        """Single cluster -> scores unchanged."""
        cluster: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.0, 2.0),
            (1, 2): EdgeQualityScore(3, 2.0, 3.0),
        }
        result = aggregate_edge_quality([cluster])
        assert result == cluster


# ===========================================================================
# Tests for prune_edges()
# ===========================================================================


class TestPruneEdges:
    """Tests for visibility graph pruning."""

    def test_prune_some(self):
        """Removes specified edges, keeps rest in order."""
        graph = [(0, 1), (1, 2), (2, 3), (3, 4)]
        bad = {(1, 2), (3, 4)}
        result = prune_edges(graph, bad)
        assert result == [(0, 1), (2, 3)]

    def test_prune_none(self):
        """Empty bad_edges -> graph unchanged."""
        graph = [(0, 1), (1, 2)]
        result = prune_edges(graph, set())
        assert result == graph

    def test_prune_all(self):
        """All edges bad -> empty graph."""
        graph = [(0, 1), (1, 2)]
        bad = {(0, 1), (1, 2)}
        result = prune_edges(graph, bad)
        assert result == []

    def test_prune_nonexistent_edge(self):
        """Bad edges not in graph -> no effect."""
        graph = [(0, 1), (1, 2)]
        bad = {(5, 6)}
        result = prune_edges(graph, bad)
        assert result == graph


# ===========================================================================
# Tests for export_edge_quality_to_json()
# ===========================================================================


class TestExportEdgeQualityToJson:
    """Tests for JSON export."""

    def test_export_creates_file(self, tmp_path: Path):
        """File is created with valid JSON."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.5, 2.5),
        }
        bad_edges: set[tuple[int, int]] = set()
        output_path = tmp_path / "edge_quality.json"

        export_edge_quality_to_json(quality, bad_edges, output_path)

        assert output_path.exists()
        data = json.loads(output_path.read_text())
        assert "metadata" in data
        assert "edge_quality" in data
        assert "bad_edges" in data

    def test_export_content(self, tmp_path: Path):
        """Verify metadata counts and edge entries."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.5, 2.5),
            (1, 2): EdgeQualityScore(0, float("inf"), float("inf")),
        }
        bad_edges = {(1, 2)}
        output_path = tmp_path / "edge_quality.json"

        export_edge_quality_to_json(quality, bad_edges, output_path)

        data = json.loads(output_path.read_text())
        assert data["metadata"]["total_edges"] == 2
        assert data["metadata"]["bad_edge_count"] == 1
        assert data["metadata"]["edges_with_no_tracks"] == 1
        assert "(0,1)" in data["edge_quality"]
        assert "(1,2)" in data["edge_quality"]
        assert data["edge_quality"]["(0,1)"]["is_bad"] is False
        assert data["edge_quality"]["(1,2)"]["is_bad"] is True
        assert "(1,2)" in data["bad_edges"]

    def test_export_creates_parent_dirs(self, tmp_path: Path):
        """Export creates parent directories if they don't exist."""
        quality: EdgeQualityGraph = {(0, 1): EdgeQualityScore(5, 1.5, 2.5)}
        output_path = tmp_path / "nested" / "dir" / "edge_quality.json"

        export_edge_quality_to_json(quality, set(), output_path)

        assert output_path.exists()

    def test_export_with_image_filenames(self, tmp_path: Path):
        """Image filenames are included in JSON when provided."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.5, 2.5),
            (1, 2): EdgeQualityScore(0, float("inf"), float("inf")),
        }
        fnames = ["img_000.jpg", "img_001.jpg", "img_002.jpg"]
        output_path = tmp_path / "edge_quality.json"

        export_edge_quality_to_json(quality, {(1, 2)}, output_path, image_filenames=fnames)

        data = json.loads(output_path.read_text())
        # Top-level filename list
        assert data["image_filenames"] == fnames
        # Per-edge filenames
        assert data["edge_quality"]["(0,1)"]["image_i"] == "img_000.jpg"
        assert data["edge_quality"]["(0,1)"]["image_j"] == "img_001.jpg"
        assert data["edge_quality"]["(1,2)"]["image_i"] == "img_001.jpg"
        assert data["edge_quality"]["(1,2)"]["image_j"] == "img_002.jpg"

    def test_export_without_image_filenames(self, tmp_path: Path):
        """Without image_filenames, JSON has no filename fields (backward compat)."""
        quality: EdgeQualityGraph = {(0, 1): EdgeQualityScore(5, 1.5, 2.5)}
        output_path = tmp_path / "edge_quality.json"

        export_edge_quality_to_json(quality, set(), output_path)

        data = json.loads(output_path.read_text())
        assert "image_filenames" not in data
        assert "image_i" not in data["edge_quality"]["(0,1)"]


# ===========================================================================
# Tests for load_bad_edges_from_json
# ===========================================================================


class TestLoadBadEdgesFromJson:
    """Tests for loading bad edges from a previously exported JSON report."""

    def test_load_bad_edges(self, tmp_path: Path):
        """Bad edges are correctly parsed from JSON report."""
        quality: EdgeQualityGraph = {
            (0, 1): EdgeQualityScore(5, 1.5, 2.5),
            (2, 3): EdgeQualityScore(0, float("inf"), float("inf")),
            (4, 5): EdgeQualityScore(3, 8.0, 12.0),
        }
        bad_edges = {(2, 3), (4, 5)}
        output_path = tmp_path / "edge_quality.json"
        export_edge_quality_to_json(quality, bad_edges, output_path)

        loaded = load_bad_edges_from_json(output_path)
        assert loaded == bad_edges

    def test_load_no_bad_edges(self, tmp_path: Path):
        """Empty bad_edges list returns empty set."""
        quality: EdgeQualityGraph = {(0, 1): EdgeQualityScore(5, 1.5, 2.5)}
        output_path = tmp_path / "edge_quality.json"
        export_edge_quality_to_json(quality, set(), output_path)

        loaded = load_bad_edges_from_json(output_path)
        assert loaded == set()


# ===========================================================================
# Integration test: full pipeline from synthetic scene to bad edge detection
# ===========================================================================


class TestEdgeQualityIntegration:
    """End-to-end test: synthetic scene -> compute quality -> identify bad edges."""

    def test_good_scene_no_bad_edges(self):
        """Scene with perfect measurements -> all edges pass."""
        scene, _ = _make_scene_3_cameras()
        vis_graph = [(0, 1), (0, 2), (1, 2)]

        quality = compute_edge_quality(scene, vis_graph)
        bad = identify_bad_edges(quality)

        assert bad == set()
        for score in quality.values():
            assert not score.is_bad()

    def test_scene_with_one_bad_edge(self):
        """Scene where one edge has noisy measurements -> correctly flagged."""
        cameras = {
            0: _make_camera(0, 0, 0),
            1: _make_camera(2, -2, 0),
            2: _make_camera(4, 0, 0),
        }
        # Track 0: good, between cam0 and cam1
        good_track = _make_track(Point3(1, 2, 1), {0: cameras[0], 1: cameras[1]})
        # Track 1: bad, between cam1 and cam2 (large noise on cam2)
        bad_track = _make_track(
            Point3(3, 1, 1),
            {1: cameras[1], 2: cameras[2]},
            noise={2: Point2(50.0, -30.0)},
        )

        scene = GtsfmData(number_images=3, cameras=cameras, tracks=[good_track, bad_track])
        vis_graph = [(0, 1), (1, 2)]

        quality = compute_edge_quality(scene, vis_graph)
        bad = identify_bad_edges(quality, max_reproj_error_px=5.0)

        # Edge (0,1) should be good, (1,2) should be bad
        assert (0, 1) not in bad
        assert (1, 2) in bad

    def test_prune_bad_edge_from_graph(self):
        """Full flow: compute quality -> identify bad -> prune -> verify."""
        cameras = {
            0: _make_camera(0, 0, 0),
            1: _make_camera(2, -2, 0),
            2: _make_camera(4, 0, 0),
        }
        good_track = _make_track(Point3(1, 2, 1), {0: cameras[0], 1: cameras[1]})
        bad_track = _make_track(
            Point3(3, 1, 1),
            {1: cameras[1], 2: cameras[2]},
            noise={2: Point2(50.0, -30.0)},
        )

        scene = GtsfmData(number_images=3, cameras=cameras, tracks=[good_track, bad_track])
        vis_graph = [(0, 1), (1, 2)]

        quality = compute_edge_quality(scene, vis_graph)
        bad = identify_bad_edges(quality, max_reproj_error_px=5.0)
        pruned = prune_edges(vis_graph, bad)

        assert pruned == [(0, 1)]
