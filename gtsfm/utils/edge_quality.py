"""Utilities for computing and analyzing visibility graph edge quality.

This module provides functions to evaluate the quality of visibility graph edges
based on the reconstruction quality of the tracks that span them.
"""

from __future__ import annotations

import itertools
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.edge_quality import EdgeQualityGraph, EdgeQualityScore
from gtsfm.products.visibility_graph import AnnotatedGraph, ImageIndexPair, VisibilityGraph
from gtsfm.utils.reprojection import compute_track_reprojection_errors


def compute_edge_quality(
    gtsfm_data: GtsfmData,
    visibility_graph: VisibilityGraph,
    expected_tracks_per_edge: Optional[AnnotatedGraph[int]] = None,
) -> EdgeQualityGraph:
    """Compute quality scores for each edge based on reconstruction quality.

    An edge (i, j) represents a connection between cameras i and j. The quality
    of an edge is determined by analyzing the 3D tracks that span both cameras.

    Args:
        gtsfm_data: Reconstruction result containing cameras and tracks.
        visibility_graph: List of image pairs (edges) to evaluate.
        expected_tracks_per_edge: Optional dict mapping edge to expected number
            of tracks (e.g., from two-view inlier count). Used for computing
            track_coverage_ratio.

    Returns:
        EdgeQualityGraph mapping each edge to its EdgeQualityScore.
    """
    if gtsfm_data is None or gtsfm_data.number_tracks() == 0:
        # No reconstruction, return empty scores for all edges
        return {
            edge: EdgeQualityScore(
                num_supporting_tracks=0,
                mean_reproj_error_px=float("inf"),
                max_reproj_error_px=float("inf"),
                track_coverage_ratio=0.0,
            )
            for edge in visibility_graph
        }

    # Step 1: Build edge -> track indices mapping
    edge_to_track_indices = _build_edge_to_tracks_mapping(gtsfm_data)

    # Step 2: Get cameras dict for reprojection computation
    cameras = gtsfm_data.cameras()

    # Step 3: Compute quality for each edge in visibility graph
    edge_quality: EdgeQualityGraph = {}
    for edge in visibility_graph:
        i, j = edge
        # Ensure canonical ordering (i < j)
        if i > j:
            edge = (j, i)

        track_indices = edge_to_track_indices.get(edge, [])
        num_tracks = len(track_indices)

        if num_tracks == 0:
            # No tracks span this edge
            edge_quality[edge] = EdgeQualityScore(
                num_supporting_tracks=0,
                mean_reproj_error_px=float("inf"),
                max_reproj_error_px=float("inf"),
                track_coverage_ratio=0.0,
            )
            continue

        # Compute reprojection errors for measurements from cameras i and j in supporting tracks
        reproj_errors = _compute_edge_reproj_errors(gtsfm_data, cameras, track_indices, edge)

        if len(reproj_errors) == 0 or np.all(np.isnan(reproj_errors)):
            mean_error = float("inf")
            max_error = float("inf")
        else:
            mean_error = float(np.nanmean(reproj_errors))
            max_error = float(np.nanmax(reproj_errors))

        # Compute track coverage ratio
        if expected_tracks_per_edge is not None and edge in expected_tracks_per_edge:
            expected = expected_tracks_per_edge[edge]
            coverage_ratio = num_tracks / expected if expected > 0 else 1.0
        else:
            # If no expected count, assume coverage is based on having any tracks
            coverage_ratio = 1.0 if num_tracks > 0 else 0.0

        edge_quality[edge] = EdgeQualityScore(
            num_supporting_tracks=num_tracks,
            mean_reproj_error_px=mean_error,
            max_reproj_error_px=max_error,
            track_coverage_ratio=coverage_ratio,
        )

    return edge_quality


def _build_edge_to_tracks_mapping(gtsfm_data: GtsfmData) -> dict[ImageIndexPair, list[int]]:
    """Build a mapping from edges to the track indices that span them.

    A track spans edge (i, j) if it has measurements in both cameras i and j.

    Args:
        gtsfm_data: Reconstruction with tracks.

    Returns:
        Dict mapping each edge (i, j) to list of track indices.
    """
    edge_to_tracks: dict[ImageIndexPair, list[int]] = defaultdict(list)

    for track_idx in range(gtsfm_data.number_tracks()):
        track = gtsfm_data.get_track(track_idx)
        # Get all camera indices that observe this track
        camera_indices = [track.measurement(m)[0] for m in range(track.numberMeasurements())]

        # Generate all edges covered by this track
        for i, j in itertools.combinations(sorted(set(camera_indices)), 2):
            edge_to_tracks[(i, j)].append(track_idx)

    return dict(edge_to_tracks)


def _compute_edge_reproj_errors(
    gtsfm_data: GtsfmData,
    cameras: dict,
    track_indices: list[int],
    edge: ImageIndexPair,
) -> np.ndarray:
    """Compute reprojection errors for measurements from edge cameras in tracks.

    Args:
        gtsfm_data: Reconstruction with tracks.
        cameras: Dict of cameras.
        track_indices: Indices of tracks spanning the edge.
        edge: The edge (i, j) to compute errors for.

    Returns:
        Array of reprojection errors for measurements from cameras i and j.
    """
    i, j = edge
    errors = []

    for track_idx in track_indices:
        track = gtsfm_data.get_track(track_idx)
        track_errors, _ = compute_track_reprojection_errors(cameras, track)

        # Get errors only for measurements from cameras i and j
        for m_idx in range(track.numberMeasurements()):
            cam_idx, _ = track.measurement(m_idx)
            if cam_idx == i or cam_idx == j:
                if not np.isnan(track_errors[m_idx]):
                    errors.append(track_errors[m_idx])

    return np.array(errors) if errors else np.array([])


def identify_bad_edges(
    edge_quality: EdgeQualityGraph,
    max_reproj_error_px: float = 5.0,
    min_track_coverage: float = 0.1,
) -> set[ImageIndexPair]:
    """Identify edges that fail quality thresholds.

    An edge is considered bad if:
    - mean_reproj_error_px > max_reproj_error_px, OR
    - track_coverage_ratio < min_track_coverage

    Quality threshold guidelines:
    - Good: < 1.0 px reprojection error
    - Acceptable: 1.0-3.0 px
    - Poor: 3.0-5.0 px
    - Bad: > 5.0 px

    Args:
        edge_quality: Dict mapping edges to their quality scores.
        max_reproj_error_px: Maximum allowed mean reprojection error.
        min_track_coverage: Minimum required track coverage ratio.

    Returns:
        Set of edges that fail the quality thresholds.
    """
    bad_edges: set[ImageIndexPair] = set()

    for edge, score in edge_quality.items():
        if score.is_bad(max_reproj_error_px, min_track_coverage):
            bad_edges.add(edge)

    return bad_edges


def merge_edge_quality(scores: list[EdgeQualityScore]) -> EdgeQualityScore:
    """Merge quality scores for an edge appearing in multiple clusters.

    When METIS partitions a graph, some edges may appear in multiple clusters
    (separator edges). This function merges their quality scores using a
    worst-case approach to be conservative.

    Args:
        scores: List of EdgeQualityScore from different clusters.

    Returns:
        Merged EdgeQualityScore using worst-case values.
    """
    if not scores:
        raise ValueError("Cannot merge empty list of scores")

    if len(scores) == 1:
        return scores[0]

    return EdgeQualityScore(
        num_supporting_tracks=sum(s.num_supporting_tracks for s in scores),
        mean_reproj_error_px=max(s.mean_reproj_error_px for s in scores),  # worst case
        max_reproj_error_px=max(s.max_reproj_error_px for s in scores),
        track_coverage_ratio=min(s.track_coverage_ratio for s in scores),  # worst case
    )


def aggregate_edge_quality(
    cluster_edge_qualities: list[EdgeQualityGraph],
) -> EdgeQualityGraph:
    """Aggregate edge quality scores from multiple clusters.

    Handles edges that appear in multiple clusters by merging their scores.

    Args:
        cluster_edge_qualities: List of EdgeQualityGraph from each cluster.

    Returns:
        Aggregated EdgeQualityGraph with merged scores for overlapping edges.
    """
    # Collect all scores per edge
    edge_to_scores: dict[ImageIndexPair, list[EdgeQualityScore]] = defaultdict(list)

    for cluster_quality in cluster_edge_qualities:
        for edge, score in cluster_quality.items():
            edge_to_scores[edge].append(score)

    # Merge scores for edges appearing in multiple clusters
    aggregated: EdgeQualityGraph = {}
    for edge, scores in edge_to_scores.items():
        aggregated[edge] = merge_edge_quality(scores)

    return aggregated


def export_edge_quality_to_json(
    edge_quality: EdgeQualityGraph,
    bad_edges: set[ImageIndexPair],
    output_path: Path,
) -> None:
    """Export edge quality analysis to JSON for debugging.

    Args:
        edge_quality: Dict mapping edges to their quality scores.
        bad_edges: Set of edges identified as bad.
        output_path: Path to write the JSON file.
    """
    # Sort edges by quality (worst first)
    sorted_edges = sorted(
        edge_quality.items(),
        key=lambda x: x[1].mean_reproj_error_px,
        reverse=True,
    )

    data = {
        "metadata": {
            "total_edges": len(edge_quality),
            "bad_edge_count": len(bad_edges),
            "edges_with_no_tracks": sum(1 for s in edge_quality.values() if s.num_supporting_tracks == 0),
        },
        "edge_quality": {
            f"({i},{j})": {
                "num_tracks": score.num_supporting_tracks,
                "mean_reproj_error_px": round(score.mean_reproj_error_px, 3)
                if score.mean_reproj_error_px != float("inf")
                else "inf",
                "max_reproj_error_px": round(score.max_reproj_error_px, 3)
                if score.max_reproj_error_px != float("inf")
                else "inf",
                "track_coverage": round(score.track_coverage_ratio, 3),
                "is_bad": (i, j) in bad_edges,
            }
            for (i, j), score in sorted_edges
        },
        "bad_edges": [f"({i},{j})" for i, j in sorted(bad_edges)],
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2))
