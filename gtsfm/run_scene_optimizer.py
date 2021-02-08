
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
def run_scene_optimizer_hydra(cfg: DictConfig) -> None:
	scene_optimizer: SceneOptimizer = instantiate(cfg.SceneOptimizer)

	loader = FolderLoader(
		os.path.join( TEST_ROOT, "data", "set1_lund_door"), image_extension="JPG"
	)

	# import pdb
	# pdb.set_trace()
	# # bad support for Enum's currently
	# scene_optimizer.multiview_optimizer.data_association_module.mode = eval(
	# 	scene_optimizer.multiview_optimizer.data_association_module.mode
	# )

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


def run_scene_optimizer():
	""" """

	from gtsfm.averaging.translation.averaging_1dsfm import (
	    TranslationAveraging1DSFM,
	)
	from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
	from gtsfm.frontend.matcher.twoway_matcher import TwoWayMatcher
	from gtsfm.frontend.verifier.ransac import Ransac
	from gtsfm.frontend.detector_descriptor.sift import SIFTDetectorDescriptor

	from gtsfm.feature_extractor import FeatureExtractor
	from gtsfm.two_view_estimator import TwoViewEstimator
	from gtsfm.multi_view_optimizer import MultiViewOptimizer
	from gtsfm.data_association.data_assoc import DataAssociation

	from gtsfm.data_association.data_assoc import TriangulationParam

	feature_extractor = FeatureExtractor(
		detector_descriptor=SIFTDetectorDescriptor()
	)
	two_view_estimator = TwoViewEstimator(
		matcher=TwoWayMatcher(),
		verifier=Ransac()
	)
	data_association_module = DataAssociation(
      reproj_error_thresh=5,
      min_track_len=2,
      mode=TriangulationParam.NO_RANSAC,
      num_ransac_hypotheses=20

	)
	multiview_optimizer = MultiViewOptimizer(
    	rot_avg_module=ShonanRotationAveraging(),
    	trans_avg_module=TranslationAveraging1DSFM(),
    	data_association_module=data_association_module

	)
	scene_optimizer = SceneOptimizer(
		feature_extractor=feature_extractor,
		two_view_estimator=two_view_estimator,
		multiview_optimizer=multiview_optimizer,
		save_bal_files=True,
		save_viz=True
	)

	loader = FolderLoader(
		os.path.join("..","tests", "data", "set1_lund_door"), image_extension="JPG"
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
	run_scene_optimizer_hydra()
	#run_scene_optimizer()


