"""Interface class from gtsfmData to patchmatchnetData

Authors: Ren Liu
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.densify.mvs_math import piecewise_gaussian, to_camera_coordinates


class PatchmatchNetData(Dataset):
    """PatchmatchNetData class for Patchmatch Net. It contains the interface from GtsfmData"""

    def __init__(self, images: Dict[int, Image], sfm_result: GtsfmData, num_views: int = 5) -> None:
        """Initialize method for PatchmatchnetData

        Args:
            images: input images (H, W, C) to GTSFM
            sfm_result: sfm results calculated by GTSFM
            num_views: number of views, containing 1 reference view and (num_views-1) source views
        """

        assert images is not None and len(images) > 1

        # cache sfm result
        self.sfm_result = sfm_result

        # Patchmatch Net meta
        self.num_views = num_views
        self.num_stages = 4

        # Test data preparation
        self.keys = sorted(self.sfm_result.get_valid_camera_indices())
        self.num_images = len(self.keys)
        self.keys_map = {}
        for i in range(self.num_images):
            self.keys_map[self.keys[i]] = i
        self.images = images
        self.image_w = images[self.keys[0]].width
        self.image_h = images[self.keys[0]].height

        self.pairs, self.depth_metas = self.configure()

    def configure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Configure pairs and depth_metas for each view from sfm_result

        Returns:
            pairs: np.ndarray in shape of (num_images, num_views-1). Each row_id indicates the index of reference view
                in self.keys, with (num_views-1) values indicating the indices of source views in self.keys
            depth_metas: np.ndarray in shape of (num_images, 2). Each row_id indicates the index of reference view
                in self.keys, with 2 values indicating [min_depth, max_depth]
        """
        num_images = self.num_images
        num_tracks = self.sfm_result.number_tracks()

        pair_scores = np.zeros((num_images, num_images))
        for i in range(num_images):
            pair_scores[i, i] = -np.inf

        depth_metas = np.zeros((num_images, 2))
        depth_collection_views: List[List[float]] = [[] for _ in range(num_images)]

        for i in range(num_tracks):
            track_i = self.sfm_result.get_track(i)
            num_measurements_i = track_i.number_measurements()
            measurements = [track_i.measurement(j) for j in range(num_measurements_i)]
            position_3d = track_i.point3()
            for cam_a in range(num_measurements_i):
                for cam_b in range(cam_a + 1, num_measurements_i):
                    cam_a_id = measurements[cam_a][0]
                    cam_b_id = measurements[cam_b][0]

                    # if both cameras are valid cameras
                    if cam_a_id in self.keys_map and cam_b_id in self.keys_map:

                        key_a_id = self.keys_map[cam_a_id]
                        key_b_id = self.keys_map[cam_b_id]

                        cam_a_pos_i = to_camera_coordinates(
                            p=position_3d, camera_pose=self.sfm_result.get_camera(cam_a_id).pose().matrix()
                        )
                        cam_b_pos_i = to_camera_coordinates(
                            p=position_3d, camera_pose=self.sfm_result.get_camera(cam_b_id).pose().matrix()
                        )

                        score_a_b = piecewise_gaussian(p_a=cam_a_pos_i, p_b=cam_b_pos_i)

                        pair_scores[key_a_id, key_b_id] += score_a_b
                        pair_scores[key_b_id, key_a_id] += score_a_b

                        depth_collection_views[key_a_id].append(cam_a_pos_i[-1])
                        depth_collection_views[key_b_id].append(cam_b_pos_i[-1])

        depth_metas[:, 0] = np.array([np.floor(np.min(depth_collection_views[i])) for i in range(num_images)])
        depth_metas[:, 1] = np.array([np.ceil(np.max(depth_collection_views[i])) for i in range(num_images)])
        pairs = np.argsort(pair_scores, axis=0)[:, -self.num_views + 1 :][:, ::-1]

        return pairs, depth_metas

    def __len__(self) -> int:
        """Get the number of images

        Returns:
            length of image dictionary's keys
        """

        return self.num_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get one test input to Patchmatch Net, Wang's work in https://github.com/FangjinhuaWang/PatchmatchNet.git is
            referred to in this method

        Args:
            index: index of yield item

        Returns:
            return dictionary contains:
                "idx" test image index: int
                "imgs" source and reference images: (num_views, image_channel, image_h, image_w)
                "proj_matrices" projection matrices: (num_views, 4, 4)
                "depth_min" minimum depth: int
                "depth_max" maximum depth: int
                "filename" output filename pattern: string
        """
        ref_key = self.keys[index]
        src_keys = [self.keys[src_index] for src_index in self.pairs[index]]

        cam_keys = [ref_key] + src_keys

        imgs_0 = []
        imgs_1 = []
        imgs_2 = []
        imgs_3 = []
        proj_matrices_0 = []
        proj_matrices_1 = []
        proj_matrices_2 = []
        proj_matrices_3 = []

        for cam_key in cam_keys:
            img = self.images[cam_key].value_array
            np_img = np.array(img, dtype=np.float32) / 255.0
            np_img = cv2.resize(np_img, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

            h, w, _ = np_img.shape

            imgs_0.append(np_img)
            imgs_1.append(cv2.resize(np_img, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR))
            imgs_2.append(cv2.resize(np_img, (w // 4, h // 4), interpolation=cv2.INTER_LINEAR))
            imgs_3.append(cv2.resize(np_img, (w // 8, h // 8), interpolation=cv2.INTER_LINEAR))

            # intrinsics, extrinsics = self.data["cameras"][vid]
            intrinsics = self.sfm_result.get_camera(cam_key).calibration().K()
            extrinsics = np.linalg.inv(self.sfm_result.get_camera(cam_key).pose().matrix())

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 0.125
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_3.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_2.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_1.append(proj_mat)

            proj_mat = extrinsics.copy()
            intrinsics[:2, :] *= 2
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices_0.append(proj_mat)

        imgs_0 = np.stack(imgs_0).transpose([0, 3, 1, 2])
        imgs_1 = np.stack(imgs_1).transpose([0, 3, 1, 2])
        imgs_2 = np.stack(imgs_2).transpose([0, 3, 1, 2])
        imgs_3 = np.stack(imgs_3).transpose([0, 3, 1, 2])
        imgs = {}
        imgs["stage_0"] = imgs_0
        imgs["stage_1"] = imgs_1
        imgs["stage_2"] = imgs_2
        imgs["stage_3"] = imgs_3

        # proj_matrices: N*4*4
        proj_matrices_0 = np.stack(proj_matrices_0)
        proj_matrices_1 = np.stack(proj_matrices_1)
        proj_matrices_2 = np.stack(proj_matrices_2)
        proj_matrices_3 = np.stack(proj_matrices_3)
        proj = {}
        proj["stage_3"] = proj_matrices_3
        proj["stage_2"] = proj_matrices_2
        proj["stage_1"] = proj_matrices_1
        proj["stage_0"] = proj_matrices_0

        return {
            "idx": index,
            "imgs": imgs,  # N*3*H0*W0
            "proj_matrices": proj,  # N*4*4
            "depth_min": self.depth_metas[index, 0],  # scalar
            "depth_max": self.depth_metas[index, 1],  # scalar
            "filename": "{}/" + "{:0>8}".format(ref_key) + "{}",
        }
