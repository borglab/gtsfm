import os
from pathlib import Path

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.experimental import compose, initialize_config_module
from hydra.utils import instantiate
from omegaconf import DictConfig

import gtsfm
from gtsfm.common.sfm_result import SfmResult
from gtsfm.loader.folder_loader import FolderLoader
from gtsfm.scene_optimizer import SceneOptimizer

# densify
from gtsfm.densify.mvsnets.mvsnets import MVSNets 


DATA_ROOT = Path(__file__).resolve().parent.parent.parent / "tests" / "data"


def run_scene_optimizer() -> None:
    """ """
    with initialize_config_module(config_module="gtsfm.configs"):
        # config is relative to the gtsfm module
        cfg = compose(config_name="default_lund_door_set1_config.yaml")
        scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)
        
        # ------ test from result ------
        import gtsam
        sfm_result = gtsam.readBal('./results/ba_output.bal')
        print(DATA_ROOT)
        MVSNets.densify(sfm_result, image_path=os.path.join(DATA_ROOT, "set1_1_lund_door"), image_extension="JPG", use_gt_cam=True)
        


if __name__ == "__main__":
    run_scene_optimizer()
