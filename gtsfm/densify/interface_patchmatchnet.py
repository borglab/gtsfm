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


class PatchmatchNetData(Dataset):
    """PatchmatchNetData class for Patchmatch Net. It contains the interface from GtsfmData"""

    def __init__(self, images: Dict[int, Image], sfm_result: GtsfmData, num_views: int = 5) -> None:
        """Initialize method for PatchmatchnetData

        Args:
            images: input images to GTSFM
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

                        cam_a_pos_i = self.sfm_result.get_camera_poses(cam_a_id) @ position_3d
                        cam_b_pos_i = self.sfm_result.get_camera_poses(cam_b_id) @ position_3d

                        score_a_b = piecewise_gaussian(p_a=cam_a_pos_i, p_b=cam_b_pos_i)

                        pair_scores[key_a_id, key_b_id] += score_a_b
                        pair_scores[key_b_id, key_a_id] += score_a_b

                        depth_collection_views[key_a_id].append(cam_a_pos_i[-1])
                        depth_collection_views[key_b_id].append(cam_b_pos_i[-1])

        depth_metas[:, 0] = np.array([np.floor(np.min(depth_collection_views[i])) for i in range(num_images)])
        depth_metas[:, 1] = np.array([np.ceil(np.max(depth_collection_views[i])) for i in range(num_images)])
        pairs = np.argsort(pair_scores, axis=0)[:, 1:][:, ::-1]

        return pairs, depth_metas

    def __len__(self) -> int:
        """Get the number of images

        Returns:
            length of image dictionary's keys
        """

        return self.num_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get one test input to Patchmatch Net

        Args:
            index: index of yield item

        Returns:
            python dictionary stores test image index, source and reference images, projection matrices,
                minimum and maximum depth, and output filename pattern.
        """
        return super().__getitem__(index)
