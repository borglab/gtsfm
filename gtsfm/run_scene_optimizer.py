
import os
from pathlib import Path
from types import SimpleNamespace

import hydra
from dask.distributed import Client, LocalCluster, performance_report
from hydra.utils import instantiate
from omegaconf import DictConfig

import gtsfm
from gtsfm.common.sfm_result import SfmResult
from gtsfm.loader.folder_loader import FolderLoader
from gtsfm.scene_optimizer import SceneOptimizer


TEST_ROOT = Path(__file__).resolve().parent.parent / "tests"


@hydra.main(config_name="config")
def run_scene_optimizer(cfg: DictConfig) -> None:
	scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

	loader = FolderLoader(
		os.path.join( TEST_ROOT, "data", "set1_lund_door"), image_extension="JPG"
	)

	sfm_result_graph = scene_optimizer.create_computation_graph(
		len(loader),
		loader.get_valid_pairs(),
		loader.create_computation_graph_for_images(),
		loader.create_computation_graph_for_intrinsics(),
		use_intrinsics_in_verification=True,
		gt_pose_graph=loader.create_computation_graph_for_poses(),
	)

	# create dask client
	cluster = LocalCluster(n_workers=2, threads_per_worker=4)

	with Client(cluster), performance_report(filename="dask-report.html"):
	    sfm_result = sfm_result_graph.compute()

	assert isinstance(sfm_result, SfmResult)


if __name__ == "__main__":
	run_scene_optimizer()


