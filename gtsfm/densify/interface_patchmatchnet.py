"""Interface class from GtsfmData to PatchmatchNetData

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
    """PatchmatchNetData class for PatchmatchNet. It contains the interface from GtsfmData.
    Wang's work in https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/datasets/dtu_yao_eval.py is referred to.
    """

    def __init__(self, images: Dict[int, Image], sfm_result: GtsfmData, num_views: int = 5) -> None:
        """Initialize method for PatchmatchNetData

        Args:
            images: input images (H, W, C) to GTSFM
            sfm_result: sfm results calculated by GTSFM
            num_views: number of views, containing 1 reference view and (num_views-1) source views
        """
        assert images is not None and len(images) > 1

        # cache sfm result
        self._sfm_result = sfm_result

        # PatchmatchNet meta
        self._num_views = num_views
        self._num_stages = NUM_PATCHMATCHNET_STAGES

        # Test data preparation
        self._keys = sorted(self._sfm_result.get_valid_camera_indices())
        self._num_images = len(self._keys)
        self._keys_map = {}
        for i in range(self._num_images):
            self._keys_map[self._keys[i]] = i
        self._images = images

        self._pairs, self._depth_ranges = self.configure()

    def configure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Configure pairs and depth_ranges for each view from sfm_result
        If there are N0 valid images and the patchmatchnet's number of views is num_views, the function does:
            1. Calculate the similarity scores between N0 images. Then for every image as the reference image, find
            (num_views - 1) most similar images as the source images;
            2. For every image as the reference image, calculate the depth range

        Returns:
            pairs: array of shape (num_images, num_views-1). Each row_id indicates the index of reference view
                in self.keys, with (num_views-1) values indicating the indices of source views in self.keys
            depth_ranges: array of shape (num_images, 2). Each row_id indicates the index of reference view
                in self.keys, with 2 values indicating [min_depth, max_depth]
        """
        num_images = self._num_images
        num_tracks = self._sfm_result.number_tracks()

        pair_scores = np.zeros((num_images, num_images))
        # initialize the pairwise scores between the same views as negative infinity
        np.fill_diagonal(pair_scores, -np.inf)

        depth_ranges = np.zeros((num_images, 2))
        depth_ranges[:, 0] = np.inf

        for j in range(num_tracks):
            track = self._sfm_result.get_track(j)
            num_measurements = track.number_measurements()
            measurements = [track.measurement(k) for k in range(num_measurements)]
            w_x = track.point3()
            for k1 in range(num_measurements):
                for k2 in range(k1 + 1, num_measurements):
                    i_a = measurements[k1][0]
                    i_b = measurements[k2][0]

                    key_a = -1
                    key_b = -1

                    # check if measurement j1 belongs to a valid camera a
                    if i_a in self._keys_map:
                        key_a = self._keys_map[i_a]
                        # calculate track_i's 3D coordinates in the camera pose
                        a_x = self._sfm_result.get_camera(i_a).pose().transformTo(w_x)
                        # update depth ranges
                        depth_ranges[key_a, 0] = min(depth_ranges[key_a, 0], a_x[-1])
                        depth_ranges[key_a, 1] = max(depth_ranges[key_a, 1], a_x[-1])

                    # check if measurement j2 belongs to a valid camera b
                    if i_b in self._keys_map:
                        key_b = self._keys_map[i_b]
                        # calculate track_i's 3D coordinates in the camera pose
                        b_x = self._sfm_result.get_camera(i_b).pose().transformTo(w_x)
                        # update depth ranges
                        depth_ranges[key_b, 0] = min(depth_ranges[key_b, 0], b_x[-1])
                        depth_ranges[key_b, 1] = max(depth_ranges[key_b, 1], b_x[-1])

                    # if both cameras are valid cameras
                    if key_a > 0 and key_b > 0:
                        # calculate score for measurements of track_i in pair views (cam_a, cam_b)
                        score_a_b = piecewise_gaussian(a_x=a_x, b_x=b_x)
                        # sum up pair scores for each track_i
                        pair_scores[key_a, key_b] += score_a_b
                        pair_scores[key_b, key_a] = pair_scores[key_a, key_b]

        # sort pair scores, for i-th row, choose the largest (num_views-1) scores, the corresponding views are selected
        #   as (num_views-1) source views for i-th reference view.
        pairs = np.argsort(pair_scores, axis=0)[:, -self._num_views + 1 :][:, ::-1]

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
        return self._num_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get test data to PatchmatchNet
        produce data containing _num_views images, the first images is the reference image

        Args:
            index: index of yield test data, the reference image ID of the test data is _keys[index]

        Returns:
            Dictionary containing:
                "idx" test image index: int
                "imgs" source and reference images: (num_views, image_channel, image_h, image_w)
                "proj_matrices" projection matrices: (num_views, 4, 4)
                "depth_min" minimum depth: int
                "depth_max" maximum depth: int
                "filename" output filename pattern: string
        """
        ref_key = self._keys[index]
        src_keys = [self._keys[src_index] for src_index in self._pairs[index]]

        cam_keys = [ref_key] + src_keys

        imgs: List[np.ndarray] = [[] for _ in range(self._num_stages)]
        proj_mats: List[np.ndarray] = [[] for _ in range(self._num_stages)]

        h, w, _ = self._images[self._keys[0]].value_array.shape
        for cam_key in cam_keys:
            img = self._images[cam_key].value_array
            np_img = np.array(img, dtype=np.float32) / 255.0
            np_img = cv2.resize(np_img, (w, h), interpolation=cv2.INTER_LINEAR)

            h, w, _ = np_img.shape

            intrinsics = self._sfm_result.get_camera(cam_key).calibration().K()
            extrinsics = self._sfm_result.get_camera(cam_key).pose().inverse().matrix()
            intrinsics[:2, :] /= 8.0

            for i in range(self._num_stages):
                imgs[i].append(cv2.resize(np_img, (w // (2 ** i), h // (2 ** i)), interpolation=cv2.INTER_LINEAR))
                proj_mat = extrinsics.copy()
                proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]
                intrinsics[:2, :] *= 2.0
                proj_mats[-1 - i].append(proj_mat)

        imgs_dict = {}
        proj_dict = {}
        for i in range(self._num_stages):
            imgs_dict[f"stage_{i}"] = np.stack(imgs[i]).transpose([0, 3, 1, 2])
            proj_dict[f"stage_{i}"] = np.stack(proj_mats[i])

        return {
            "idx": index,
            "imgs": imgs_dict,  # N*3*H0*W0
            "proj_matrices": proj_dict,  # N*4*4
            "depth_min": self._depth_ranges[index, 0],  # scalar
            "depth_max": self._depth_ranges[index, 1],  # scalar
            "filename": "{}/" + f"{ref_key:0>8}" + "{}",
        }
