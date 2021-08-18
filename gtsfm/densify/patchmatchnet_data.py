"""Interface class from GtsfmData to PatchmatchNetData

Authors: Ren Liu
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.densify.mvs_math import piecewise_gaussian

NUM_PATCHMATCHNET_STAGES = 4

MIN_DEPTH_PERCENTILE = 1
MAX_DEPTH_PERCENTILE = 99


class PatchmatchNetData(Dataset):
    """PatchmatchNetData class for PatchmatchNet. It contains the interface from GtsfmData.
    Ref: Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/datasets/dtu_yao_eval.py
    """

    def __init__(self, images: Dict[int, Image], sfm_result: GtsfmData, num_views: int = 5) -> None:
        """Cache image and camera pose metadata for PatchmatchNet inference.

        Args:
            images: input images (H, W, C) to GTSFM
            sfm_result: sfm results calculated by GTSFM
            num_views: number of views, containing 1 reference view and (num_views-1) source views
        """

        # Cache sfm result
        self._sfm_result = sfm_result

        # Test data preparation
        valid_camera_idxs = sorted(self._sfm_result.get_valid_camera_indices())
        self._patchmatchnet_idx_to_camera_idx = {i_pm: i for i_pm, i in enumerate(valid_camera_idxs)}
        #   the number of images with estimated poses，not the number of images provided to GTSFM
        self._num_images_cal = len(valid_camera_idxs)

        # Verify that there should be at least 2 valid images
        if self._num_images_cal <= 1:
            raise ValueError("At least 2 or more images with estimated poses must be provided")

        # PatchmatchNet meta
        self._num_stages = NUM_PATCHMATCHNET_STAGES
        #   the number of views must be no larger than the number of images
        self._num_views = min(num_views, self._num_images_cal)

        # Store locations of camera centers in the world frame
        self._camera_centers = {}
        for i in self._patchmatchnet_idx_to_camera_idx.values():
            self._camera_centers[i] = self._sfm_result.get_camera(i).pose().translation()

        # Create mapping from image indices (with some entries
        #   potentially missing) to [0,N-1] tensor indices for N inputs to PatchmatchNet
        self._camera_idx_to_patchmatchnet_idx = {i: i_pm for i_pm, i in self._patchmatchnet_idx_to_camera_idx.items()}

        self._images = images

        # Set image dimensions to those of zero’th image
        self._h, self._w = (
            self._images[self._patchmatchnet_idx_to_camera_idx[0]].height,
            self._images[self._patchmatchnet_idx_to_camera_idx[0]].width,
        )
        # Calculate the cropped size of each image so that the height and width of the image can be divided by 8 evenly,
        #   so that the image size can be matched after downsampling and then upsampling in PatchmatchNet
        self._cropped_h, self._cropped_w = (self._h - self._h % 8, self._w - self._w % 8)

        self._pairs, self._depth_ranges = self.select_src_views_depth_ranges()

    def select_src_views_depth_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Configure (ref_view, src_view) pairs and depth_ranges for each view from sfm_result.
        Ref: Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/colmap_input.py (line 360-410)

        If there are N0 valid images and the patchmatchnet's number of views is num_views, the function does:
            1. Calculate the scores between each pair of the N0 images. Then for every image as the reference image,
            find (num_views - 1) images with highest scores as the source images.
            2. The scores between each pair of the N0 images is the sum of piecewise Gaussian scores between vectors
            from a common track's coordinates to each camera center, which is mentioned in "View Selection" paragraphs
            in Yao's paper https://arxiv.org/abs/1804.02505.
            3. For every image as the reference image, calculate the depth range.

        Returns:
            pairs: array of shape (num_images, num_views-1). Each row_id indicates the index of reference view
                in self.keys, with (num_views-1) values indicating the indices of source views in self.keys
            depth_ranges: array of shape (num_images, 2). Each row_id indicates the index of reference view
                in self.keys, with 2 values indicating [min_depth, max_depth]
        """
        num_tracks = self._sfm_result.number_tracks()

        # Initialize the pairwise scores between the same views as negative infinity
        pair_scores = np.zeros((self._num_images_cal, self._num_images_cal))
        np.fill_diagonal(pair_scores, -np.inf)

        # Initialize empty lists to collect all possible depths for each view
        depths = defaultdict(list)

        for j in range(num_tracks):
            track = self._sfm_result.get_track(j)
            num_measurements = track.number_measurements()
            measurements = [track.measurement(k) for k in range(num_measurements)]
            w_x = track.point3()
            for k1 in range(num_measurements):
                i_a = measurements[k1][0]
                # Check if i_a is a valid image with estimated camera pose, then get its id for patchmatchnet
                i_pm_a = self._camera_idx_to_patchmatchnet_idx.get(i_a, -1)

                # Calculate track_i's depth in the camera pose
                z_a = self._sfm_result.get_camera(i_a).pose().transformTo(w_x)[-1]
                # Update image i's depth list only when i in a valid camera
                depths[i_pm_a].append(z_a)

                for k2 in range(k1 + 1, num_measurements):
                    i_b = measurements[k2][0]

                    # Check if i_b is a valid image with estimated camera pose, then get its id for patchmatchnet
                    i_pm_b = self._camera_idx_to_patchmatchnet_idx.get(i_b, -1)

                    # Calculate the pairwise gaussian score only if i_a and i_b are both images with estimated poses
                    if i_pm_a < 0 or i_pm_b < 0:
                        continue

                    # If both cameras are valid cameras,
                    #   calculate score for track_i in the pair views (cam_a, cam_b)
                    score_a_b = piecewise_gaussian(
                        xPa=self._camera_centers[i_a] - w_x, xPb=self._camera_centers[i_b] - w_x
                    )
                    # Sum up pair scores for each track_i
                    pair_scores[i_pm_b, i_pm_a] += score_a_b
                    pair_scores[i_pm_a, i_pm_b] += score_a_b

        # Sort pair scores, for i-th row, choose the largest (num_views-1) scores, the corresponding views are selected
        #   as (num_views-1) source views for i-th reference view.
        pairs = np.argsort(-pair_scores, axis=1)[:, : self._num_views - 1]

        # Filter out depth outliers and calculate depth ranges
        depth_ranges = np.zeros((self._num_images_cal, 2))
        for i in range(self._num_images_cal):
            depth_ranges[i, 0] = np.percentile(depths[i], MIN_DEPTH_PERCENTILE)
            depth_ranges[i, 1] = np.percentile(depths[i], MAX_DEPTH_PERCENTILE)

        return pairs, depth_ranges

    def __len__(self) -> int:
        """Get the length of the dataset
        In the PatchMatchNet workflow, each input image will be treated as a reference view once, and each input image
        corresponds to a estimated camera pose. So the length of the dataset is equal to the number of reference views.

        Returns:
            the length of the dataset
        """
        return self._num_images_cal

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get inference data to PatchmatchNet
        Produces data containing _num_views images, the first images is the reference image

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
        ref_key = self._patchmatchnet_idx_to_camera_idx[index]
        src_keys = [self._patchmatchnet_idx_to_camera_idx[src_index] for src_index in self._pairs[index]]

        cam_keys = [ref_key] + src_keys

        imgs: List[np.ndarray] = [[] for _ in range(self._num_stages)]
        proj_mats: List[np.ndarray] = [[] for _ in range(self._num_stages)]

        for cam_key in cam_keys:
            img = self._images[cam_key].value_array
            np_img = np.array(img, dtype=np.float32) / 255.0
            np_img = cv2.resize(np_img, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
            # Crop the image from the upper left corner, instead of from the center
            #   to align with the 2D coordinates in track's measurements.
            np_img = np_img[: self._cropped_h, : self._cropped_w, :]

            intrinsics = self._sfm_result.get_camera(cam_key).calibration().K()
            cTw = self._sfm_result.get_camera(cam_key).pose().inverse().matrix()
            # In the multi-scale feature extraction, there are NUM_PATCHMATCHNET_STAGES stages.
            #   The scales are [2^0, 2^(-1), 2^(-2), ..., 2^(1-NUM_PATCHMATCHNET_STAGES)]
            #   Initially the intrinsics is scaled to fit the smallest image size
            intrinsics[:2, :] /= 2 ** NUM_PATCHMATCHNET_STAGES

            for i in range(self._num_stages):
                imgs[i].append(
                    cv2.resize(
                        np_img,
                        (self._cropped_w // (2 ** i), self._cropped_h // (2 ** i)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
                proj_mat = cTw.copy()
                intrinsics[:2, :] *= 2.0
                proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]
                # For the next stage, the image size is doubled, so the intrinsics should also be doubled.
                proj_mats[-1 - i].append(proj_mat)

        imgs_dict = {}
        proj_dict = {}
        for i in range(self._num_stages):
            # Reshaping the images from (B, H, W, C) to (B, C, H, W)
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

    def get_packed_pairs(self) -> List[Dict[str, Any]]:
        """Pack view pair data in the form of (reference view index, source view indices) to fit with inference methods

        Returns:
            Packed pair data in {'ref_id': reference view index (int), 'src_ids': source view indices (List[int])},
                the length of source view indices is num_views
        """
        packed_pairs = []
        for idx in range(self._num_images_cal):
            ref_view = idx
            src_views = self._pairs[idx].tolist()
            packed_pairs.append({"ref_id": ref_view, "src_ids": src_views})
        return packed_pairs

    def get_camera_params(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera intrinsics and extrinsics parameters by image(or view) index

        Args:
            index: image(or view) index

        Returns:
            The camera parameter tuple of the input image(or view) index, (intrinsics (3, 3), extrinsics (4, 4))
        """
        cam_key = self._patchmatchnet_idx_to_camera_idx[index]
        intrinsics = self._sfm_result.get_camera(cam_key).calibration().K()
        cTw = self._sfm_result.get_camera(cam_key).pose().inverse().matrix()
        return (intrinsics, cTw)

    def get_image(self, index: int) -> np.ndarray:
        """Get preprocessed image by image(or view) index

        Args:
            index: image(or view) index

        Returns:
            Preprocessed image, of shape (cropped_h, cropped_w, 3)
        """
        cam_key = self._patchmatchnet_idx_to_camera_idx[index]
        img = self._images[cam_key].value_array
        np_img = np.array(img, dtype=np.float32) / 255.0
        np_img = cv2.resize(np_img, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        np_img = np_img[: self._cropped_h, : self._cropped_w, :]
        return np_img
