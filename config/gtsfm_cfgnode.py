import argparse
import os
from typing import List, Optional

from omegaconf import DictConfig, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

from gtsfm_defaults import SceneOptimizer, FeatureExtractor, TwoViewEstimator, MultiViewOptimizer

import logging

"""
Provides 2 classes for GTSFM users to interface with command-line
input parameters and YAML files. Both classes wrap around YACS' CfgNode.

CfgNode is designed to be used without any command-line input, and
ArgsCfgNode is designed to merge argparse data with YAML data.

Our main requirement is that users will not be allowed to modify
these parameters after they are initialized here, i.e. "frozen".
"""

# A logger for this file
log = logging.getLogger(__name__)


# Create config group `config`
cs = ConfigStore.instance()
cs.store(group="schema/modules", name="sceneOptimizer", node=SceneOptimizer, package="SceneOptimizer")
cs.store(group="schema/modules", name="featureExtractor", node=FeatureExtractor, package="FeatureExtractor")
cs.store(group="schema/modules", name="twoViewEstimator", node=TwoViewEstimator, package="TwoViewEstimator")
cs.store(group="schema/modules", name="multiViewOptimizer", node=MultiViewOptimizer, package="MultiViewOptimizer")

@hydra.main(config_name='defaults')
def cfgnode(cfg: DictConfig) -> None:
    log.info(OmegaConf.to_yaml(cfg))
    log.info(OmegaConf.get_type(cfg.FeatureExtractor.submodule1.param_bool))
    gtsfm_cfgnode = instantiate(cfg.FeatureExtractor)
    print(gtsfm_cfgnode.submodule1.param_bool)
    
if __name__ == "__main__":
    cfg = cfgnode()
    print(cfg)
    

