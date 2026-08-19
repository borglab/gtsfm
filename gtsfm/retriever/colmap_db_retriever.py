"""Retriever that reads image pairs from a COLMAP database's verified two-view geometries.

For paper-scale scenes (e.g. the large 1DSfM / Phototourism sets) we build the frontend offline with
COLMAP (GPU SIFT extraction + vocab-tree matching), which is far faster and more memory-efficient than
GTSFM's Dask SIFT frontend. This retriever returns exactly the image pairs COLMAP geometrically verified
(the ``two_view_geometries`` table), so the verified pipeline partitions on the db's verified view graph.
Pair it with ``ColmapCorrespondenceGenerator``, which reads the keypoints + inlier matches for these pairs.

Authors: GTSFM
"""

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

import gtsfm.utils.logger as logger_utils
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()

# COLMAP encodes a pair as pair_id = image_id1 * MAX + image_id2 (image_ids are 1-indexed).
_COLMAP_MAX_IMAGE_ID = 2147483647


def _pair_id_to_image_ids(pair_id: int) -> Tuple[int, int]:
    """Decode a COLMAP pair_id back into its two image ids."""
    image_id2 = pair_id % _COLMAP_MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) // _COLMAP_MAX_IMAGE_ID
    return int(image_id1), int(image_id2)


class ColmapDBRetriever(RetrieverBase):
    """Return the geometrically-verified image pairs from a COLMAP ``database.db``."""

    def __init__(self, database_path: str) -> None:
        """
        Args:
            database_path: path to the COLMAP database.db (features + verified two-view geometries).
        """
        self._database_path = Path(database_path)

    def __repr__(self) -> str:
        return f"ColmapDBRetriever(database_path={self._database_path})"

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Read verified pairs from the db and map COLMAP image ids -> GTSFM image indices by filename."""
        del global_descriptors, plots_output_dir  # unused: pairs come from the db, not descriptors
        if not self._database_path.exists():
            raise FileNotFoundError(f"COLMAP database not found: {self._database_path}")

        # GTSFM index keyed by basename. Loader filenames may be relative paths; the db stores names matching
        # the image_path the db was built on — basename is the robust common denominator.
        idx_by_basename: Dict[str, int] = {os.path.basename(f): i for i, f in enumerate(image_fnames)}

        conn = sqlite3.connect(str(self._database_path))
        try:
            # COLMAP image_id -> GTSFM index (by basename match).
            colmap_id_to_gtsfm: Dict[int, int] = {}
            for colmap_id, name in conn.execute("SELECT image_id, name FROM images"):
                idx = idx_by_basename.get(os.path.basename(str(name)))
                if idx is not None:
                    colmap_id_to_gtsfm[int(colmap_id)] = idx

            pairs = set()
            n_verified = 0
            for pair_id, rows, config in conn.execute("SELECT pair_id, rows, config FROM two_view_geometries"):
                # config 2 = CALIBRATED (E), 3 = UNCALIBRATED (F), 4 = PLANAR, 5 = PANORAMIC,
                # 6 = PLANAR_OR_PANORAMIC. Homography-type pairs (4-6) carry real inlier
                # correspondences (flat facades, rotation-heavy viewpoints) and GLOMAP consumes
                # them — dropping them starved head-to-head same-graph comparisons (~1.3k of
                # Wilson's ToL EG pairs are H-verified). Low-parallax structure is handled
                # downstream by the triangulation/angle filters. rows>0 means inliers exist.
                if rows is None or int(rows) == 0 or int(config) not in (2, 3, 4, 5, 6):
                    continue
                n_verified += 1
                cid1, cid2 = _pair_id_to_image_ids(int(pair_id))
                i, j = colmap_id_to_gtsfm.get(cid1), colmap_id_to_gtsfm.get(cid2)
                if i is None or j is None or i == j:
                    continue
                pairs.add((i, j) if i < j else (j, i))
        finally:
            conn.close()

        visibility_graph: VisibilityGraph = sorted(pairs)
        logger.info(
            "🗄️  ColmapDBRetriever: %d verified pairs in db; %d/%d loader images matched by name → %d GTSFM pairs.",
            n_verified,
            len(colmap_id_to_gtsfm),
            len(image_fnames),
            len(visibility_graph),
        )
        if len(colmap_id_to_gtsfm) < len(image_fnames):
            logger.warning(
                "ColmapDBRetriever: %d loader images had NO match in the db by basename — confirm the db was "
                "built on the same images/ directory.",
                len(image_fnames) - len(colmap_id_to_gtsfm),
            )
        return visibility_graph
