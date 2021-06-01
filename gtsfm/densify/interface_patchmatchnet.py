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
        self._camera_centers = {}
        for i in range(self._num_images):
            self._keys_map[self._keys[i]] = i
            self._camera_centers[self._keys[i]] = self._sfm_result.get_camera(self._keys[i]).pose().translation()

        self._images = images

        self._pairs, self._depth_ranges = self.configure()

    def configure(self) -> Tuple[np.ndarray, np.ndarray]:
        """Configure pairs and depth_ranges for each view from sfm_result. The configure method is based on Wang's work
        https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/colmap_input.py (line 360-410)

        If there are N0 valid images and the patchmatchnet's number of views is num_views, the function does:
            1. Calculate the scores between N0 images. Then for every image as the reference image, find
            (num_views - 1) images with highest scores as the source images;
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

        # initialize empty lists to collect all possible depths for each view
        depths: List[List[float]] = [[] for _ in range(num_images)]
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
                        # calculate track_i's depth in the camera pose
                        a_z = self._sfm_result.get_camera(i_a).pose().transformTo(w_x)[-1]
                        # update depth ranges
                        depths[key_a].append(a_z)

                    # check if measurement j2 belongs to a valid camera b
                    if i_b in self._keys_map:
                        key_b = self._keys_map[i_b]
                        # calculate track_i's depth in the camera pose
                        b_z = self._sfm_result.get_camera(i_b).pose().transformTo(w_x)[-1]
                        # update depth ranges
                        depths[key_b].append(b_z)

                    # if both cameras are valid cameras
                    if key_a > 0 and key_b > 0:
                        # calculate score for track_i in the pair views (cam_a, cam_b)
                        score_a_b = piecewise_gaussian(
                            a_x=self._camera_centers[i_a] - w_x, b_x=self._camera_centers[i_b] - w_x
                        )
                        # sum up pair scores for each track_i
                        pair_scores[key_a, key_b] += score_a_b
                        pair_scores[key_b, key_a] = pair_scores[key_a, key_b]

        # sort pair scores, for i-th row, choose the largest (num_views-1) scores, the corresponding views are selected
        #   as (num_views-1) source views for i-th reference view.
        pairs = np.argsort(pair_scores, axis=0)[:, -self._num_views + 1 :][:, ::-1]

        # filter out depth outliers and calculate depth ranges
        for i in range(num_images):
            depths_sorted = sorted(depths[i])
            depth_ranges[i, 0] = depths_sorted[int(len(depths[i]) * 0.01)]
            depth_ranges[i, 1] = depths_sorted[int(len(depths[i]) * 0.99)]

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
            # In the multi-scale feature extraction, there are NUM_PATCHMATCHNET_STAGES stages.
            #   The scales are [2^0, 2^(-1), 2^(-2), ..., 2^(1-NUM_PATCHMATCHNET_STAGES)]
            #   Initially the intrinsics is scaled to fit the smallest image size
            intrinsics[:2, :] /= 2 ** (NUM_PATCHMATCHNET_STAGES - 1)

            for i in range(self._num_stages):
                imgs[i].append(cv2.resize(np_img, (w // (2 ** i), h // (2 ** i)), interpolation=cv2.INTER_LINEAR))
                proj_mat = extrinsics.copy()
                proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]
                # For the next stage, the image size is doubled, so the intrinsics should also doubled.
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
