from yacs.config import CfgNode as YACS

"""
YACS requires specifying a schema for the YAML config files. In this Python file,
we provide such a schema. Every GTSFM YAML config file should include 4
sections -- parameters for the SceneOptimizer, FeatureExtractor,
TwoViewEstimator, and MultiViewOptimizer. YAML files should be hierarchical.
"""

_Cfg = YACS(new_allowed=True)

_Cfg.SceneOptimizer = YACS(new_allowed=True)

_Cfg.FeatureExtractor = YACS(new_allowed=True)

_Cfg.TwoViewEstimator = YACS(new_allowed=True)

_Cfg.MultiViewOptimizer = YACS(new_allowed=True)


def get_cfg_defaults() -> YACS:
    """Provides a YACS object with default modules for GTSFM."""
    return _Cfg.clone()
