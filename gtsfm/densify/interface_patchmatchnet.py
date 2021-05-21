"""Interface class from gtsfmData to patchmatchnetData

Authors: Ren Liu
"""
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from gtsfm.common.image import Image
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.densify.mvs_math import piecewise_gaussian

NUM_PATCHMATCHNET_STAGES = 4


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
        self.num_stages = NUM_PATCHMATCHNET_STAGES

        # Test data preparation
        self.keys = sorted(self.sfm_result.get_valid_camera_indices())
        self.num_images = len(self.keys)
        self.keys_map = {}
        for i in range(self.num_images):
            self.keys_map[self.keys[i]] = i
        self.images = images
        self.image_w = images[self.keys[0]].width
        self.image_h = images[self.keys[0]].height

        self.pairs, self.depth_ranges = self.configure()

    def configure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Configure pairs and depth_ranges for each view from sfm_result

        Returns:
            pairs: array of shape (num_images, num_views-1). Each row_id indicates the index of reference view
                in self.keys, with (num_views-1) values indicating the indices of source views in self.keys
            depth_ranges: array of shape (num_images, 2). Each row_id indicates the index of reference view
                in self.keys, with 2 values indicating [min_depth, max_depth]
        """
        num_images = self.num_images
        num_tracks = self.sfm_result.number_tracks()

        pair_scores = np.zeros((num_images, num_images))
        # initialize the pairwise scores between the same views as negative infinity
        np.fill_diagonal(pair_scores, -np.inf)

        depth_ranges = np.zeros((num_images, 2))
        depth_ranges[:, 0] = np.inf

        for i in range(num_tracks):
            track_i = self.sfm_result.get_track(i)
            num_measurements_i = track_i.number_measurements()
            measurements = [track_i.measurement(j) for j in range(num_measurements_i)]
            coord_world = track_i.point3()
            for cam_a in range(num_measurements_i):
                for cam_b in range(cam_a + 1, num_measurements_i):
                    cam_a_id = measurements[cam_a][0]
                    cam_b_id = measurements[cam_b][0]

                    # if both cameras are valid cameras
                    if cam_a_id in self.keys_map and cam_b_id in self.keys_map:

                        key_a_id = self.keys_map[cam_a_id]
                        key_b_id = self.keys_map[cam_b_id]

                        # calculate track_i's 3D coordinates in the camera pose
                        coord_cam_a = self.sfm_result.get_camera(cam_a_id).pose().transformTo(coord_world)
                        coord_cam_b = self.sfm_result.get_camera(cam_b_id).pose().transformTo(coord_world)

                        # calculate score for measurements of track_i in pair views (cam_a, cam_b)
                        score_a_b = piecewise_gaussian(p_a=coord_cam_a, p_b=coord_cam_b)

                        # sum up pair scores for each track_i
                        pair_scores[key_a_id, key_b_id] += score_a_b
                        pair_scores[key_b_id, key_a_id] += score_a_b

                        # update depth ranges
                        depth_ranges[(key_a_id, key_b_id), 0] = np.minimum(
                            depth_ranges[(key_a_id, key_b_id), 0], [coord_cam_a[-1], coord_cam_b[-1]]
                        )
                        depth_ranges[(key_a_id, key_b_id), 1] = np.maximum(
                            depth_ranges[(key_a_id, key_b_id), 1], [coord_cam_a[-1], coord_cam_b[-1]]
                        )

        # sort pair scores, for i-th row, choose the largest (num_views-1) scores, the corresponding views are selected
        #   as (num_views-1) source views for i-th reference view.
        pairs = np.argsort(pair_scores, axis=0)[:, -self.num_views + 1 :][:, ::-1]

        # convert float depth_ranges to integers
        depth_ranges[:, 0] = np.floor(depth_ranges[:, 0])
        depth_ranges[:, 1] = np.ceil(depth_ranges[:, 1])
        depth_ranges = depth_ranges.astype(np.int32)

        return pairs, depth_ranges

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

        imgs: List[np.ndarray] = [[] for _ in range(self.num_stages)]
        proj_mats: List[np.ndarray] = [[] for _ in range(self.num_stages)]

        for cam_key in cam_keys:
            img = self.images[cam_key].value_array
            np_img = np.array(img, dtype=np.float32) / 255.0
            np_img = cv2.resize(np_img, (self.image_w, self.image_h), interpolation=cv2.INTER_LINEAR)

            h, w, _ = np_img.shape

            intrinsics = self.sfm_result.get_camera(cam_key).calibration().K()
            extrinsics = self.sfm_result.get_camera(cam_key).pose().inverse().matrix()
            intrinsics[:2, :] /= 8.0

            for i in range(self.num_stages):
                imgs[i].append(cv2.resize(np_img, (w // (2 ** i), h // (2 ** i)), interpolation=cv2.INTER_LINEAR))
                proj_mat = extrinsics.copy()
                proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]
                intrinsics[:2, :] *= 2.0
                proj_mats[-1 - i].append(proj_mat)

        imgs_dict = {}
        proj_dict = {}
        for i in range(self.num_stages):
            imgs_dict[f"stage_{i}"] = np.stack(imgs[i]).transpose([0, 3, 1, 2])
            proj_dict[f"stage_{i}"] = np.stack(proj_mats[i])

        return {
            "idx": index,
            "imgs": imgs_dict,  # N*3*H0*W0
            "proj_matrices": proj_dict,  # N*4*4
            "depth_min": self.depth_ranges[index, 0],  # scalar
            "depth_max": self.depth_ranges[index, 1],  # scalar
            "filename": "{}/" + "{:0>8}".format(ref_key) + "{}",
        }
