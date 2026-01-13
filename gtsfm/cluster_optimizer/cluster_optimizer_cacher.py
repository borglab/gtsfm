"""Decorator which caches cluster optimizer outputs on disk."""

from __future__ import annotations

import hashlib
import typing
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
from dask.delayed import Delayed, delayed

import gtsfm.utils.cache as cache_utils
import gtsfm.utils.io as io_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.cluster_optimizer.cluster_optimizer_base import (
    ClusterComputationGraph,
    ClusterContext,
    ClusterOptimizerBase,
)
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.products.visibility_graph import visibility_graph_keys
from gtsfm.ui.gtsfm_process import UiMetadata

if TYPE_CHECKING:
    from gtsfm.products.one_view_data import OneViewData

# Keep cache location consistent with other cachers.
CACHE_ROOT_PATH = Path(__file__).resolve().parent.parent.parent / "cache"

logger = logger_utils.get_logger()


class ClusterOptimizerCacher(ClusterOptimizerBase):
    """Caches the delayed bundle result produced by a cluster optimizer."""

    def __init__(self, optimizer: ClusterOptimizerBase) -> None:
        super().__init__(
            pose_angular_error_thresh=optimizer.pose_angular_error_thresh,
            output_worker=optimizer._output_worker,
        )
        self._optimizer = optimizer
        self._optimizer_hash = hashlib.sha1(repr(optimizer).encode()).hexdigest()

    def __repr__(self) -> str:
        return repr(self._optimizer)

    def __getattr__(self, name: str):
        """Delegate public attribute access to the wrapped optimizer.

        Avoid forwarding private/dunder attributes to prevent recursive lookups during pickling.
        """
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._optimizer, name)

    def __getstate__(self) -> dict[str, object]:
        """Provide a minimal pickleable state."""
        return {
            "_optimizer": self._optimizer,
            "_optimizer_hash": self._optimizer_hash,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and keep worker routing consistent."""
        self._optimizer = typing.cast(ClusterOptimizerBase, state["_optimizer"])
        self._optimizer_hash = typing.cast(str, state["_optimizer_hash"])
        # Re-initialize the base class to mimic the constructor.
        super().__init__(
            pose_angular_error_thresh=self._optimizer.pose_angular_error_thresh,
            output_worker=self._optimizer._output_worker,
        )

    def _get_cache_path(self, cache_key: str) -> Path:
        return CACHE_ROOT_PATH / "cluster_optimizer" / f"{cache_key}.pbz2"

    def _hash_one_view_data(self, one_view_data: Optional["OneViewData"]) -> str:
        """Compute a stable hash for OneViewData contents."""
        if one_view_data is None:
            return "missing"

        components = [
            one_view_data.image_fname,
            repr(one_view_data.intrinsics),
            repr(one_view_data.absolute_pose_prior),
            repr(one_view_data.camera_gt),
            repr(one_view_data.pose_gt),
        ]
        return hashlib.sha1("|".join(components).encode()).hexdigest()

    def _generate_cache_key(self, context: ClusterContext) -> str:
        """Generate a stable key from optimizer config and cluster definition."""
        edges = np.array(sorted(context.visibility_graph), dtype=np.int64)
        edges_hash = cache_utils.generate_hash_for_numpy_array(edges)
        path_hash = hashlib.sha1("_".join(map(str, context.cluster_path)).encode()).hexdigest()
        indices = sorted(visibility_graph_keys(context.visibility_graph))
        image_hashes = [self._hash_one_view_data(context.one_view_data_dict.get(idx)) for idx in indices]
        images_hash = hashlib.sha1("|".join(image_hashes).encode()).hexdigest() if image_hashes else "no_images"
        return f"{self._optimizer_hash}_{context.num_images}_{path_hash}_{edges_hash}_{images_hash}"

    @staticmethod
    def get_ui_metadata() -> UiMetadata:
        """UI metadata for the cacher node; mirrors the wrapped optimizer outputs."""

        return UiMetadata(
            display_name="ClusterOptimizerCacher",
            input_products=("Visibility Graph", "Images", "OneViewData"),
            output_products=("Cluster Reconstruction",),
            parent_plate="Cluster Optimization",
        )

    def _load_result_from_cache(self, context: ClusterContext) -> Optional[GtsfmData]:
        cache_path = self._get_cache_path(self._generate_cache_key(context))
        cached = io_utils.read_from_bz2_file(cache_path)
        if isinstance(cached, GtsfmData):
            logger.info("Loaded cached cluster result for %s", context.label)
            return cached
        return None

    def _save_result_to_cache(self, result: GtsfmData, cache_path: Path) -> GtsfmData:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        io_utils.write_to_bz2_file(result, cache_path)
        return result

    def create_computation_graph(self, context: ClusterContext) -> ClusterComputationGraph | None:
        cached_result = self._load_result_from_cache(context)
        if cached_result is not None:
            cached_graph: Delayed = delayed(lambda r: r, pure=False)(cached_result)
            return ClusterComputationGraph(io_tasks=tuple(), metric_tasks=tuple(), sfm_result=cached_graph)

        computation = self._optimizer.create_computation_graph(context)
        if computation is None or computation.sfm_result is None:
            return computation

        cache_path = self._get_cache_path(self._generate_cache_key(context))
        sfm_result_with_cache = delayed(self._save_result_to_cache, pure=False)(computation.sfm_result, cache_path)

        return ClusterComputationGraph(
            io_tasks=computation.io_tasks,
            metric_tasks=computation.metric_tasks,
            sfm_result=sfm_result_with_cache,
        )
