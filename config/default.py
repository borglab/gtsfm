from yacs.config import CfgNode as YACS

"""
This file is to initialize all modules inside 
this hierarchical configuration tree for GTSFM.
Pls feel free to add new modules or submodules 
when needed.
"""

_Cfg = YACS(new_allowed=True)

_Cfg.SceneOptimizer = YACS(new_allowed=True)

_Cfg.FeatureExtractor = YACS(new_allowed=True)

_Cfg.TwoViewEstimator = YACS(new_allowed=True)

_Cfg.MultiViewOptimizer = YACS(new_allowed=True)

def get_cfg_defaults():
  """Get YACS object with default modules for GTSFM."""
  return _Cfg.clone()