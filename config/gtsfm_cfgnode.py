from omegaconf import MISSING, DictConfig, OmegaConf

import hydra
from hydra.core.config_store import ConfigStore

from dataclasses import dataclass

from gtsfm_default import GTSFM_CfgNode

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

cs = ConfigStore.instance()
cs.store(name="gtsfm_default", node=GTSFM_CfgNode)

@hydra.main(config_name='gtsfm_default')
def cfgnode(cfg: GTSFM_CfgNode) -> None:
    print(cfg)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(OmegaConf.get_type(cfg.featureExtractor.submodule1.param_bool))
    
if __name__ == "__main__":
    cfgnode()
    

