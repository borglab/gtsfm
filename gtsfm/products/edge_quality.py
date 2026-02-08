"""Edge quality metrics for visibility graph edges.

This module defines data structures for tracking the quality of visibility graph edges
based on the reconstruction quality of clusters that contain them.
"""

from dataclasses import dataclass

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
    """

    num_supporting_tracks: int
    mean_reproj_error_px: float
    max_reproj_error_px: float

    def is_bad(
        self,
        max_reproj_error_px: float = 5.0,
    ) -> bool:
        """Check if this edge fails quality thresholds.

        An edge is bad if:
        - It has zero supporting tracks (no 3D geometry between the cameras), OR
        - Its mean reprojection error exceeds the threshold.

        Quality thresholds:
        - Good: < 1.0 px reprojection error
        - Acceptable: 1.0-3.0 px
        - Poor: 3.0-5.0 px
        - Bad: > 5.0 px

        Args:
            max_reproj_error_px: Maximum allowed mean reprojection error.

        Returns:
            True if the edge has no supporting tracks or exceeds the error threshold.
        """
        if self.num_supporting_tracks == 0:
            return True
        return self.mean_reproj_error_px > max_reproj_error_px


# Type alias: maps edge (i,j) to its quality score
EdgeQualityGraph = AnnotatedGraph[EdgeQualityScore]
