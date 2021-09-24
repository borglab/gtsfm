from gtsfm.loader.loader_base import LoaderBase
from pathlib import Path

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.scene_optimizer import SceneOptimizer

ROOT_PATH = Path(__file__).resolve().parent.parent.parent

logger = logger_utils.get_logger()


@hydra.main(config_path=str(ROOT_PATH / "gtsfm" / "configs"), config_name="defaults")
def run_scene_optimizer(
    cfg: DictConfig,
) -> None:
    """Runs the scene optimizer with the loader, scene_optimizer, and dask config from the hydra config."""

    logger.info(OmegaConf.to_yaml(cfg))

    scene_optimizer: SceneOptimizer = instantiate(cfg.scene_optimizer)

    loader: LoaderBase = instantiate(cfg.loader)

    sfm_result_graph = scene_optimizer.create_computation_graph(
        num_images=len(loader),
        image_pair_indices=loader.get_valid_pairs(),
        image_graph=loader.create_computation_graph_for_images(),
        camera_intrinsics_graph=loader.create_computation_graph_for_intrinsics(),
        image_shape_graph=loader.create_computation_graph_for_image_shapes(),
        gt_pose_graph=loader.create_computation_graph_for_poses(),
    )

    # create dask client
    cluster = LocalCluster(n_workers=cfg.num_workers, threads_per_worker=cfg.threads_per_worker)

    with Client(cluster), performance_report(filename="dask-report.html"):
        sfm_result = sfm_result_graph.compute()

    assert isinstance(sfm_result, GtsfmData)


if __name__ == "__main__":
    run_scene_optimizer()
