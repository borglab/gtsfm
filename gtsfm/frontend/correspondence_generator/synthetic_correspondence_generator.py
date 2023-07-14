"""Correspondence generator that creates synthetic keypoint correspondences using a 3d mesh.

Authors: John Lambert
"""
import tempfile
from typing import Any, Dict, List, Tuple

from dask.distributed import Client, Future
import numpy as np
import open3d

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.common.types import CALIBRATION_TYPE, CAMERA_TYPE
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import CorrespondenceGeneratorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.loader.tanks_and_temples_loader import TanksAndTemplesLoader


class SyntheticCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Pair-wise synthetic keypoint correspondence generator."""

    def __init__(self, dataset_root: str, scene_name: str, deduplicate: bool = True) -> None:
        """
        Args:
            dataset_root: str
            scene_name: Name of scene from Tanks & Temples dataset.
            deduplicate: Whether to de-duplicate with a single image the detections received from each image pair.
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
            client: Dask client, used to execute the front-end as futures.
            images: List of all images, as futures.
            image_pairs: Indices of the pairs of images to estimate two-view pose and correspondences.

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

        
        mesh = loader.reconstruct_mesh()
        # TODO(jolambert): File Open3d bug to add pickle support for TriangleMesh.
        open3d_mesh_path = tempfile.NamedTemporaryFile(suffix='.obj').name
        open3d.io.write_triangle_mesh(filename=open3d_mesh_path, mesh=mesh)

        loader_future = client.scatter(loader, broadcast=False)
        def apply_synthetic_corr_generator(
            loader_: LoaderBase, camera_i1, camera_i2, open3d_mesh_fpath: str, i1: int, i2: int
        ) -> Tuple[Keypoints, Keypoints]:
            keypoints_i1, keypoints_i2 = loader_.generate_synthetic_correspondences_for_image_pair(camera_i1, camera_i2, open3d_mesh_fpath)

            ###########
            import numpy as np
            from gtsam import Unit3
            from gtsfm.frontend.verifier.loransac import LoRansac
            import gtsfm.utils.geometry_comparisons as geom_comp_utils
            num_kpts = len(keypoints_i1)
            match_indices = np.stack([np.arange(num_kpts), np.arange(num_kpts)], axis=-1)
            wTi1 = loader_.get_camera_pose(index=i1)
            wTi2 = loader_.get_camera_pose(index=i2)

            i2Ti1 = wTi2.between(wTi1)
            i2Ri1_expected = i2Ti1.rotation()
            i2Ui1_expected = Unit3(i2Ti1.translation())

            camera_intrinsics_i1 = loader_.get_camera_intrinsics_full_res(index=i1)
            camera_intrinsics_i2 = loader_.get_camera_intrinsics_full_res(index=i2)

            verifier = LoRansac(use_intrinsics_in_verification=True, estimation_threshold_px=0.5)
            i2Ri1_computed, i2Ui1_computed, verified_indices_computed, _ = verifier.verify(
                keypoints_i1,
                keypoints_i2,
                match_indices,
                camera_intrinsics_i1,
                camera_intrinsics_i2,
            )
            if i2Ri1_computed is not None and i2Ui1_computed is not None:
                rot_angular_err = geom_comp_utils.compute_relative_rotation_angle(i2Ri1_expected, i2Ri1_computed)
                direction_angular_err = geom_comp_utils.compute_relative_unit_translation_angle(i2Ui1_expected, i2Ui1_computed)
                print(f"Errors ({i1},{i2}): rotation {rot_angular_err:.4f}, direction {direction_angular_err:.4f}")
            ################
            return keypoints_i1, keypoints_i2


        pairwise_correspondence_futures = {
            (i1, i2): client.submit(apply_synthetic_corr_generator,
                loader_future,
                loader.get_camera(index=i1),
                loader.get_camera(index=i2),
                open3d_mesh_path,
                i1,
                i2
            )
            for i1, i2 in image_pairs
        }

        pairwise_correspondences: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]] = client.gather(
            pairwise_correspondence_futures
        )

        keypoints_list, putative_corr_idxs_dict = self._aggregator.aggregate(keypoints_dict=pairwise_correspondences)
        return keypoints_list, putative_corr_idxs_dict
