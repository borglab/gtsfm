"""Correspondence generator that utilizes direct matching of keypoints across an image pair, without descriptors.

Authors: John Lambert
"""
import tempfile
from typing import Any, Dict, List, Optional, Tuple

from dask.distributed import Client, Future
import numpy as np
import open3d

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.types import CALIBRATION_TYPE, CAMERA_TYPE
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase
from gtsfm.two_view_estimator import TWO_VIEW_OUTPUT, TwoViewEstimator
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader


class SyntheticCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Pair-wise direct matching of images (e.g. transformer-based)."""

    def __init__(self, dataset_root: str, scene_name: str, deduplicate: bool = True) -> None:
        """
        Args:
            dataset_root: str
            scene_name
            deduplicate: whether to de-duplicate with a single image the detections received from each image pair.
        """
        self._dataset_root = dataset_root
        self._scene_name = scene_name
        self._aggregator: KeypointAggregatorBase = (
            KeypointAggregatorDedup() if deduplicate else KeypointAggregatorUnique()
        )

    def generate_correspondences(
        self,
        client: Client,
        images: List[Future],
        image_pairs: List[Tuple[int, int]],
    ) -> Tuple[List[Keypoints], Dict[Tuple[int, int], np.ndarray]]:
        """Apply the correspondence generator to generate putative correspondences.

        Args:
            client: dask client, used to execute the front-end as futures.
            images: list of all images, as futures.
            image_pairs: indices of the pairs of images to estimate two-view pose and correspondences.

        Returns:
            List of keypoints, one entry for each input images.
            Putative correspondence as indices of keypoints, for pairs of images.
        """
        dataset_root = self._dataset_root
        scene_name = self._scene_name

        img_dir = f'{dataset_root}/{scene_name}'
        poses_fpath = f'{dataset_root}/{scene_name}_COLMAP_SfM.log'
        lidar_ply_fpath = f'{dataset_root}/{scene_name}.ply'
        colmap_ply_fpath = f'{dataset_root}/{scene_name}_COLMAP.ply'
        ply_alignment_fpath = f'{dataset_root}/{scene_name}_trans.txt'
        bounding_polyhedron_json_fpath = f'{dataset_root}/{scene_name}.json'
        loader = TanksAndTemplesLoader(
            img_dir=img_dir,
            poses_fpath=poses_fpath,
            lidar_ply_fpath=lidar_ply_fpath,
            ply_alignment_fpath=ply_alignment_fpath,
            bounding_polyhedron_json_fpath=bounding_polyhedron_json_fpath,
            colmap_ply_fpath=colmap_ply_fpath,
        )

        loader_future = client.scatter(loader, broadcast=False)
        mesh = loader.reconstruct_mesh()
        # TODO(jolambert): File Open3d bug to add pickle support for TriangleMesh.
        open3d_mesh_path = tempfile.NamedTemporaryFile(suffix='.obj').name
        open3d.io.write_triangle_mesh(filename=open3d_mesh_path, mesh=mesh)

        # def apply_image_matcher(
        #     image_matcher: ImageMatcherBase, image_i1: Image, image_i2: Image
        # ) -> Tuple[Keypoints, Keypoints]:
        #     return image_matcher.match(image_i1=image_i1, image_i2=image_i2)

        
        pairwise_correspondence_futures = {
            (i1, i2): client.submit(loader_future.generate_synthetic_correspondences_for_image_pair,
                loader.get_camera(index=i1),
                loader.get_camera(index=i2),
                open3d_mesh_path
            )
            for i1, i2 in image_pairs
        }

        pairwise_correspondences: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]] = client.gather(
            pairwise_correspondence_futures
        )

        keypoints_list, putative_corr_idxs_dict = self._aggregator.aggregate(keypoints_dict=pairwise_correspondences)

        return keypoints_list, putative_corr_idxs_dict
