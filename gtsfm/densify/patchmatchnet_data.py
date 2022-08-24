"""Class that implements an interface from GtsfmData to PatchmatchNetData.

For terminology, we will estimate a depth map for each reference view, by warping features from the source views to 
frontoparallel planes of the reference view.

Authors: Ren Liu, John Lambert
"""
from collections import defaultdict
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from torch.utils.data import Dataset

import gtsfm.densify.mvs_utils as mvs_utils
import gtsfm.utils.logger as logger_utils
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image

logger = logger_utils.get_logger()

NUM_PATCHMATCHNET_STAGES = 4

MIN_DEPTH_PERCENTILE = 1
MAX_DEPTH_PERCENTILE = 99


class PatchmatchNetData(Dataset):
    """Converts the data format from GtsfmData to PatchmatchNet input, remapping camera indices.
    Ref: Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/datasets/dtu_yao_eval.py
    """

    def __init__(self, images: Dict[int, Image], sfm_result: GtsfmData, max_num_views: int = 5) -> None:
        """Cache images and GtsfmData for PatchmatchNet inference, including camera poses and tracks.

        Args:
            images: input images (H, W, C) to GTSFM
            sfm_result: sparse multiview reconstruction result
            max_num_views: defaults to 5, maximum number of views used to reconstruct one scene in PatchmatchNet,
                containing 1 reference view and (num_views-1) source views
        """

        # Cache sfm result
        self._sfm_result = sfm_result

        # Test data preparation
        valid_camera_idxs = sorted(self._sfm_result.get_valid_camera_indices())
        self._patchmatchnet_idx_to_camera_idx = {pm_i: i for pm_i, i in enumerate(valid_camera_idxs)}
        #   the number of images with estimated poses，not the number of images provided to GTSFM
        self._num_valid_cameras = len(valid_camera_idxs)

        # Verify that there should be at least 2 valid images
        if self._num_valid_cameras <= 1:
            raise ValueError("At least 2 or more images with estimated poses must be provided")

        # PatchmatchNet meta
        self._num_stages = NUM_PATCHMATCHNET_STAGES
        #   the number of views must be no larger than the number of images
        self._num_views = min(max_num_views, self._num_valid_cameras)

        # Create mapping from image indices (with some entries
        #   potentially missing) to [0,N-1] tensor indices for N inputs to PatchmatchNet
        self._camera_idx_to_patchmatchnet_idx = {i: pm_i for pm_i, i in self._patchmatchnet_idx_to_camera_idx.items()}

        self._images = images

        # Set image dimensions to those of zero’th image
        self._h, self._w = (
            self._images[self._patchmatchnet_idx_to_camera_idx[0]].height,
            self._images[self._patchmatchnet_idx_to_camera_idx[0]].width,
        )
        # Calculate the cropped size of each image so that the height and width of the image can be divided by 8 evenly,
        #   so that the image size can be matched after downsampling and then upsampling in PatchmatchNet
        self._cropped_h, self._cropped_w = (self._h - self._h % 8, self._w - self._w % 8)

        self._src_views_dict, self._depth_ranges = self.select_src_views_depth_ranges()

    def select_src_views_depth_ranges(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute image pair indices such that baseline angles are approximately 5 degree.
        Larger baseline angles are more acceptable than small baseline angles.

        Ref: Wang et al. https://github.com/FangjinhuaWang/PatchmatchNet/blob/main/colmap_input.py (line 360-410)

        If there are N0 valid images and the patchmatchnet's number of views is num_views, the function does:
            - Calculate the scores between each image pair among all the N0 images. Then for every image as
            the reference image, find (num_views - 1) images with the highest scores as the source images.
            The scores between each image pair is the summation of piecewise Gaussian scores of all common tracks
            in both images,which is mentioned in "View Selection" paragraphs in Yao's paper
            https://arxiv.org/abs/1804.02505.

            - For every image as the reference image, calculate the depth range.

        Returns:
            src_views_dict: 2d array of shape (num_images, num_views-1). Each row_id indicates the index of
                reference view in self.keys, with (num_views-1) values indicating the indices of source views
                in self.keys
            depth_ranges: 2d array of shape (num_images, 2). Each row_id indicates the index of reference view
                in self.keys, with 2 values indicating [min_depth, max_depth]
        """
        num_tracks = self._sfm_result.number_tracks()

        # Initialize the pairwise scores between the same views as negative infinity
        pair_scores = np.zeros((self._num_valid_cameras, self._num_valid_cameras))
        np.fill_diagonal(pair_scores, -np.inf)

        # Initialize empty lists to collect all possible depths for each view
        depths = defaultdict(list)

        for j in range(num_tracks):
            track = self._sfm_result.get_track(j)
            num_measurements = track.numberMeasurements()
            measurements = [track.measurement(k) for k in range(num_measurements)]
            wtj = track.point3()
            for k1 in range(num_measurements):
                i1, _ = measurements[k1]
                # Check if i1 is a valid image with estimated camera pose, then get its id for patchmatchnet
                if i1 not in self._camera_idx_to_patchmatchnet_idx:
                    # Calculate the depth only if i1 is an image with estimated pose
                    logger.info("Camera %d had no estimated pose, so skipping during MVS.", i1)
                    continue
                pm_i1 = self._camera_idx_to_patchmatchnet_idx[i1]

                # Calculate track j's depth in the camera i1 frame
                z1 = self._sfm_result.get_camera(i1).pose().transformTo(wtj)[-1]
                # Update image i1's depth list only when i1 in a valid camera
                depths[pm_i1].append(z1)

                for k2 in range(k1 + 1, num_measurements):
                    i2, _ = measurements[k2]

                    # Check if i2 is a valid image with estimated camera pose, then get its id for patchmatchnet
                    if i2 not in self._camera_idx_to_patchmatchnet_idx:
                        # Calculate the pairwise gaussian score only if i1 and i2 are both images with estimated poses
                        logger.info("Camera %d had no estimated pose, so skipping during MVS.", i2)
                        continue
                    pm_i2 = self._camera_idx_to_patchmatchnet_idx[i2]

                    # If both cameras are valid, calculate the score of track j in view pair (pm_i1, pm_i2)
                    #   1. calculate the baseline angle of track j
                    theta_i1_i2 = mvs_utils.calculate_triangulation_angle_in_degrees(
                        camera_1=self._sfm_result.get_camera(i1),
                        camera_2=self._sfm_result.get_camera(i2),
                        point_3d=wtj,
                    )
                    #   2. calculate the result of the Gaussian function as the score
                    score_i1_i2 = mvs_utils.piecewise_gaussian(theta=theta_i1_i2)
                    #   3. add the score of track j to the total score of view pair (pm_i1, pm_i2)
                    pair_scores[pm_i2, pm_i1] += score_i1_i2
                    pair_scores[pm_i1, pm_i2] += score_i1_i2

        # Sort pair scores, for i-th row, choose the largest (num_views-1) scores, the corresponding views are selected
        #   as (num_views-1) source views for i-th reference view.
        src_views_dict = np.argsort(-pair_scores, axis=1)[:, : self._num_views - 1]

        # Filter out depth outliers and calculate depth ranges
        depth_ranges = np.zeros((self._num_valid_cameras, 2))
        for i in range(self._num_valid_cameras):
            # image i must have at least 1 depth value, calculate the proper depth range
            depth_ranges[i, 0] = np.percentile(depths[i], MIN_DEPTH_PERCENTILE)
            depth_ranges[i, 1] = np.percentile(depths[i], MAX_DEPTH_PERCENTILE)

        return src_views_dict, depth_ranges

    def __len__(self) -> int:
        """Returns the number of reference views in the dataset. A depth map will be estimated for each.

        In the PatchmatchNet workflow, each input image will be treated as a reference view once, and each input
        image corresponds to a estimated camera pose.

        Returns:
            the length of the dataset.
        """
        return self._num_valid_cameras

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get inference data to PatchmatchNet
        Produces test data from one inference view.
        Returns meta data including ids and depth range, input images(_num_views images, the first images is the
        reference image) and projection matrices, as well as the output filename

        Args:
            index: index of yield test data, the reference image ID of the test data the patchmatchnet index (from 0 to
            number_of_views - 1), not the camera index in the sfm_result

        Returns:
            Dictionary containing:
                "idx" test image index: int
                "imgs" source and reference images: (num_views, image_channel, image_h, image_w)
                "proj_matrices" projection matrices: (num_views, 4, 4)
                "depth_min" minimum depth: int
                "depth_max" maximum depth: int
                "filename" output filename pattern: string
        """
        ref_idx = self._patchmatchnet_idx_to_camera_idx[index]
        src_idxs = [self._patchmatchnet_idx_to_camera_idx[src_index] for src_index in self._src_views_dict[index]]

        imgs: List[List[np.ndarray]] = [[] for _ in range(self._num_stages)]
        proj_mats: List[List[np.ndarray]] = [[] for _ in range(self._num_stages)]

        for i in [ref_idx] + src_idxs:
            img = self._images[i].value_array
            np_img = np.array(img, dtype=np.float32) / 255.0
            np_img = cv2.resize(np_img, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
            # Crop the image from the upper left corner, instead of from the center
            #   to align with the 2D coordinates in track's measurements.
            np_img = np_img[: self._cropped_h, : self._cropped_w, :]

            intrinsics = self._sfm_result.get_camera(i).calibration().K()
            cTw = self._sfm_result.get_camera(i).pose().inverse().matrix()
            # In the multi-scale feature extraction, there are NUM_PATCHMATCHNET_STAGES stages.
            #   Resize the image to scales in [2^0, 2^(-1), 2^(-2), ..., 2^(1-NUM_PATCHMATCHNET_STAGES)]
            #   Initially the intrinsics is scaled to fit the smallest image size
            intrinsics[:2, :] /= 2**NUM_PATCHMATCHNET_STAGES

            for s in range(self._num_stages):
                imgs[s].append(
                    cv2.resize(
                        np_img,
                        (self._cropped_w // (2**s), self._cropped_h // (2**s)),
                        interpolation=cv2.INTER_LINEAR,
                    )
                )
                proj_mat = cTw.copy()
                intrinsics[:2, :] *= 2.0
                proj_mat[:3, :4] = intrinsics @ proj_mat[:3, :4]
                # For the next stage, the image size is doubled, so the intrinsics should also be doubled.
                proj_mats[-1 - s].append(proj_mat)

        imgs_dict = {}
        proj_dict = {}
        for s in range(self._num_stages):
            # Reshaping the images from (B, H, W, C) to (B, C, H, W)
            imgs_dict[f"stage_{s}"] = np.stack(imgs[s]).transpose([0, 3, 1, 2])
            proj_dict[f"stage_{s}"] = np.stack(proj_mats[s])

        return {
            "idx": index,
            "imgs": imgs_dict,  # N*3*H0*W0
            "proj_matrices": proj_dict,  # N*4*4
            "depth_min": self._depth_ranges[index, 0],  # scalar
            "depth_max": self._depth_ranges[index, 1],  # scalar
            "filename": "{}/" + f"{ref_idx:0>8}" + "{}",
        }

    def get_packed_pairs(self) -> List[Dict[str, Any]]:
        """Pack view pair data in the form of (reference view index, source view indices) to fit with inference methods

        Returns:
            Packed pair data in {'ref_id': reference view index (int), 'src_ids': source view indices (List[int])},
                the length of source view indices is num_views
        """
        packed_pairs = []
        for pm_i in range(self._num_valid_cameras):
            ref_view = pm_i
            src_views = self._src_views_dict[pm_i].tolist()
            packed_pairs.append({"ref_id": ref_view, "src_ids": src_views})
        return packed_pairs

    def get_camera_params(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get camera intrinsics and extrinsics parameters by image(or view) index

        Args:
            index: image(or view) index, known as `pm_i` elsewhere

        Returns:
            The camera parameter tuple of the input image(or view) index, (intrinsics (3, 3), extrinsics (4, 4))
        """
        i = self._patchmatchnet_idx_to_camera_idx[index]
        intrinsics = self._sfm_result.get_camera(i).calibration().K()
        cTw = self._sfm_result.get_camera(i).pose().inverse().matrix()
        return (intrinsics, cTw)

    def get_image(self, index: int) -> np.ndarray:
        """Get preprocessed image by image(or view) index

        Args:
            index: image(or view) index, known as `pm_i` elsewher

        Returns:
            Preprocessed image, of shape (cropped_h, cropped_w, 3)
        """
        i = self._patchmatchnet_idx_to_camera_idx[index]
        img = self._images[i].value_array
        np_img = np.array(img, dtype=np.float32) / 255.0
        np_img = cv2.resize(np_img, (self._w, self._h), interpolation=cv2.INTER_LINEAR)
        np_img = np_img[: self._cropped_h, : self._cropped_w, :]
        return np_img
