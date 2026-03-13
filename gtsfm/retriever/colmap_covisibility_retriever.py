"""Retriever that builds a visibility graph from COLMAP ground-truth covisibility.

For each 3D point in a COLMAP reconstruction, all pairs of images that observe
that point are considered covisible. Useful for ablation studies.

Authors: GTSfM Authors
"""

from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np

import gtsfm.utils.logger as logger_utils
import thirdparty.colmap.scripts.python.read_write_model as colmap_io
from gtsfm.products.visibility_graph import VisibilityGraph
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()


class ColmapCovisibilityRetriever(RetrieverBase):
    """Retriever that extracts covisibility from a COLMAP reconstruction's 3D point tracks."""

    def __init__(self, colmap_data_dir: str, min_shared_points: int = 1) -> None:
        """
        Args:
            colmap_data_dir: Path to directory containing COLMAP model files
                (cameras.{txt,bin}, images.{txt,bin}, points3D.{txt,bin}).
            min_shared_points: Minimum number of shared 3D points for two images
                to be considered covisible.
        """
        self._colmap_data_dir = colmap_data_dir
        self._min_shared_points = min_shared_points

    def __repr__(self) -> str:
        return (
            f"ColmapCovisibilityRetriever("
            f"colmap_data_dir={self._colmap_data_dir}, "
            f"min_shared_points={self._min_shared_points})"
        )

    def get_image_pairs(
        self,
        global_descriptors: Optional[List[np.ndarray]],
        image_fnames: List[str],
        plots_output_dir: Optional[Path] = None,
    ) -> VisibilityGraph:
        """Build visibility graph from COLMAP covisibility.

        Args:
            global_descriptors: Ignored (not needed for GT covisibility).
            image_fnames: File names of images from the loader, in loader order.
            plots_output_dir: Unused.

        Returns:
            List of (i, j) pairs where i < j, representing covisible image pairs.
        """
        colmap_dir = Path(self._colmap_data_dir)
        if (colmap_dir / "images.txt").exists():
            ext = ".txt"
        elif (colmap_dir / "images.bin").exists():
            ext = ".bin"
        else:
            raise FileNotFoundError(f"No COLMAP images file found in {colmap_dir}")

        _, images, points3d = colmap_io.read_model(path=str(colmap_dir), ext=ext)

        # Map COLMAP image ID -> loader index via filename basename.
        fname_to_loader_idx = {fname: idx for idx, fname in enumerate(image_fnames)}
        colmap_id_to_loader_idx = {}
        for colmap_img in images.values():
            basename = Path(colmap_img.name).name
            if basename in fname_to_loader_idx:
                colmap_id_to_loader_idx[colmap_img.id] = fname_to_loader_idx[basename]

        logger.info(
            "Mapped %d / %d COLMAP images to loader indices.",
            len(colmap_id_to_loader_idx),
            len(images),
        )

        # Count shared 3D points per image pair.
        pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
        for point3d in points3d.values():
            loader_indices = set()
            for img_id in point3d.image_ids:
                lid = colmap_id_to_loader_idx.get(int(img_id))
                if lid is not None:
                    loader_indices.add(lid)

            for idx_i, idx_j in combinations(sorted(loader_indices), 2):
                pair_counts[(idx_i, idx_j)] += 1

        # Filter by minimum shared points.
        pairs: VisibilityGraph = [
            edge for edge, count in pair_counts.items() if count >= self._min_shared_points
        ]
        pairs.sort()

        logger.info(
            "ColmapCovisibilityRetriever: %d pairs (min_shared_points=%d) from %d 3D points.",
            len(pairs),
            self._min_shared_points,
            len(points3d),
        )

        return pairs
