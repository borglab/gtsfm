
from yacs.config import CfgNode

"""
YACS requires specifying a schema for the YAML config files. In this Python file,
we provide such a schema. Every GTSFM YAML config file should include 4
sections -- parameters for the SceneOptimizer, FeatureExtractor,
TwoViewEstimator, and MultiViewOptimizer. YAML files should be hierarchical.
"""

_Cfg = CfgNode(new_allowed=True)

_Cfg.SceneOptimizer = CfgNode(new_allowed=True)

_Cfg.FeatureExtractor = CfgNode(new_allowed=True)

_Cfg.TwoViewEstimator = CfgNode(new_allowed=True)

_Cfg.MultiViewOptimizer = CfgNode(new_allowed=True)


def get_cfg_defaults() -> CfgNode:
    """Provides a YACS object with default modules for GTSFM."""
    return _Cfg.clone()
