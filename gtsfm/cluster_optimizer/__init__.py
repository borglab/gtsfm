"""Cluster optimizer package exports with lazy loading for public classes."""

import importlib
from typing import TYPE_CHECKING

from .cluster_optimizer_base import logger, save_metrics_reports

__all__ = [
    "Base",
    "Multiview",
    "Vggt",
    "FastVggt",
    "Anysplat",
    "Cacher",
    "logger",
    "save_metrics_reports",
]

# Provide symbols to type checkers/IDEs without incurring runtime imports.
if TYPE_CHECKING:
    from .cluster_anysplat import ClusterAnySplat as Anysplat
    from .cluster_mvo import ClusterMVO as Multiview
    from .cluster_optimizer_base import ClusterOptimizerBase as Base
    from .cluster_optimizer_cacher import ClusterOptimizerCacher as Cacher
    from .cluster_vggt import ClusterVGGT as Vggt
    from .cluster_fast_vggt import ClusterFastVGGT as FastVggt

# Short name → (module, class) for lazy attribute access.
_MOD_MAP = {
    "Base": ("gtsfm.cluster_optimizer.cluster_optimizer_base", "ClusterOptimizerBase"),
    "Multiview": ("gtsfm.cluster_optimizer.cluster_mvo", "ClusterMVO"),
    "Vggt": ("gtsfm.cluster_optimizer.cluster_vggt", "ClusterVGGT"),
    "FastVggt": ("gtsfm.cluster_optimizer.cluster_fast_vggt", "ClusterFastVGGT"),
    "Anysplat": ("gtsfm.cluster_optimizer.cluster_anysplat", "ClusterAnySplat"),
    "Cacher": ("gtsfm.cluster_optimizer.cluster_optimizer_cacher", "ClusterOptimizerCacher"),
}


def __getattr__(name: str):
    """Lazily import cluster optimizer classes on first access."""
    try:
        module_name, class_name = _MOD_MAP[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def __dir__():
    # Ensure help(), dir(), and IDEs display the short names.
    return sorted(list(globals().keys()) + __all__)
