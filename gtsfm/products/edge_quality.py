"""Edge quality metrics for visibility graph edges.

This module defines data structures for tracking the quality of visibility graph edges
based on the reconstruction quality of clusters that contain them.
"""

from dataclasses import dataclass
from typing import Optional

from gtsfm.products.visibility_graph import AnnotatedGraph, ImageIndexPair


@dataclass(frozen=True)
class EdgeQualityScore:
    """Quality metrics for a single visibility graph edge (i, j).

    An edge connects two images that should share common 3D points. The quality
    of an edge is determined by analyzing the tracks (3D points) that span both
    cameras after reconstruction.

    Attributes:
        num_supporting_tracks: Number of 3D tracks that span both cameras i and j.
        mean_reproj_error_px: Mean reprojection error (in pixels) of measurements
            from cameras i and j in supporting tracks.
        max_reproj_error_px: Maximum reprojection error (outlier indicator).
        track_coverage_ratio: Ratio of actual tracks to expected tracks. A value
            of 1.0 means all expected matches triangulated successfully.
    """

    num_supporting_tracks: int
    mean_reproj_error_px: float
    max_reproj_error_px: float
    track_coverage_ratio: float

    def is_bad(
        self,
        max_reproj_error_px: float = 5.0,
        min_track_coverage: float = 0.1,
    ) -> bool:
        """Check if this edge fails quality thresholds.

        Quality thresholds:
        - Good: < 1.0 px reprojection error
        - Acceptable: 1.0-3.0 px
        - Poor: 3.0-5.0 px
        - Bad: > 5.0 px

        Args:
            max_reproj_error_px: Maximum allowed mean reprojection error.
            min_track_coverage: Minimum required track coverage ratio.

        Returns:
            True if the edge fails either threshold.
        """
        if self.mean_reproj_error_px > max_reproj_error_px:
            return True
        if self.track_coverage_ratio < min_track_coverage:
            return True
        return False


# Type alias: maps edge (i,j) to its quality score
EdgeQualityGraph = AnnotatedGraph[EdgeQualityScore]
