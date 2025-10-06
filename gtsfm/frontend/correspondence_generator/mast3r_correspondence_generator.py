"""Correspondence generator using the MASt3R model.

https://github.com/naver/mast3r

Authors: Akshay Krishnan
"""

from typing import Any, Dict, List, Tuple
from pathlib import Path
import sys

HERE_PATH = Path(__file__).parent
MAST3R_REPO_PATH = HERE_PATH.parent.parent.parent / "thirdparty" / "mast3r"
if MAST3R_REPO_PATH.exists():
    # workaround for sibling import
    sys.path.insert(0, str(MAST3R_REPO_PATH))
else:
    raise ImportError(
        f"mast3r is not initialized, could not find: {MAST3R_REPO_PATH}.\n "
        "Did you forget to run 'git submodule update --init --recursive' ?"
    )

from mast3r.model import AsymmetricMASt3R
import mast3r.cloud_opt.sparse_ga as mast3r_ga

import numpy as np
import torch
import torchvision.transforms as tvf
from dask.distributed import Client, Future

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.utils import images as image_utils
from gtsfm.utils import logger as logger_utils
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import (
    CorrespondenceGeneratorBase,
)


logger = logger_utils.get_logger()
_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "thirdparty"
    / "mast3r"
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)
GRID_DIVISOR = 16
PATCH_MULTIPLIER = 8


class Mast3rCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Correspondence generator using the MASt3R model."""

    def __init__(self, deduplicate: bool = True, nms_merge_radius: float = 0.5, max_correspondences: int = 1000):
        """Initializes the Mast3rCorrespondenceGenerator.

        The Mast3r model currently generates keypoints in a fixed grid, so it does not use a keypoint aggregator.

        Args:
            deduplicate: Whether to deduplicate keypoints. Not currently used as Mast3r
                         generates a fixed set of keypoints per image pair.
            nms_merge_radius: Radius for Non-Maximum Suppression (NMS) merging. Not currently used.
            max_correspondences: The maximum number of correspondences to keep for each image pair.
        """
        super().__init__()
        self.max_correspondences = max_correspondences

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
        model = AsymmetricMASt3R.from_pretrained(_MODEL_PATH).eval()

        m = client.scatter(model, broadcast=False)

        pairwise_correspondence_futures = {
            (i1, i2): client.submit(
                Mast3rCorrespondenceGenerator.apply_mast3r,
                m,
                images[i1],
                images[i2],
            )
            for i1, i2 in image_pairs
        }
        pairwise_correspondences: Dict[Tuple[int, int], Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = (
            client.gather(pairwise_correspondence_futures)
        )
        logger.info(
            "Mast3r computed correspondences for %d edges, aggregating them...", len(pairwise_correspondence_futures)
        )

        keypoints_for_image = {}
        indices_for_image = {}
        putative_corr_idxs_dict = {}

        def update_keypoints(image_idx, keypoints, keypoints_idx):
            """Helper function to update and deduplicate keypoints for a given image.

            Args:
                image_idx: The index of the image.
                keypoints: The new keypoints to add.
                keypoints_idx: The original indices of the new keypoints in the feature grid.
            """
            existing_idx = indices_for_image.get(image_idx, np.array([], dtype=np.int64))
            indices_for_image[image_idx], unique_idx = np.unique(
                np.concatenate([existing_idx, keypoints_idx]), return_index=True
            )
            existing_keypoints = keypoints_for_image.get(image_idx, np.array([], dtype=np.float32).reshape(0, 2))
            keypoints_for_image[image_idx] = np.vstack([existing_keypoints, keypoints])[unique_idx]

        for (i1, i2), (keypoints_i1, keypoints_i2, idx1, idx2) in pairwise_correspondences.items():
            update_keypoints(i1, keypoints_i1, idx1)
            update_keypoints(i2, keypoints_i2, idx2)

        for (i1, i2), (keypoints_i1, keypoints_i2, idx1, idx2) in pairwise_correspondences.items():
            kp_idx1 = (indices_for_image[i1][None, :] == idx1[:, None]).argmax(axis=1)
            kp_idx2 = (indices_for_image[i2][None, :] == idx2[:, None]).argmax(axis=1)
            # TODO: use the matching score to sort/filter correspondences.
            putative_corr_idxs_dict[(i1, i2)] = np.stack([kp_idx1, kp_idx2], axis=-1).astype(np.int64)[
                : self.max_correspondences
            ]

        max_idx = max(keypoints_for_image.keys())
        keypoints_list: List[Keypoints] = [Keypoints(coordinates=np.array([]))] * (max_idx + 1)

        for i, kps in keypoints_for_image.items():
            keypoints_list[i] = Keypoints(coordinates=kps)

        logger.info("Correspondence aggregation complete!")

        return keypoints_list, putative_corr_idxs_dict

    @staticmethod
    def preprocess_image(image: Image, img_id: int, device) -> Tuple[Dict[str, Any], Tuple[int, int], float]:
        """Preprocesses image for MASt3R model: rescale, crop to a multiple of patch size, convert to tensor, normalize.

        Args:
            image: The input image to be preprocessed.
            img_id: An identifier for the image.
            device: The device (e.g., 'cpu' or 'cuda') to which the image tensor should be moved.

        Returns:
            A tuple containing:
            - A dictionary with the preprocessed image tensor, its true shape, id, and instance string.
            - A tuple (x, y) representing the start coordinates of the crop applied to the image.
            - The scaling factor applied to the image during resizing.
        """
        H1, _, _ = image.shape

        # resize to longest edge being `size`.
        img = image_utils.resize_to_max_size(image, long_edge_size=512).value_array
        H, W, _ = img.shape
        scale_factor = float(H) / H1

        cx, cy = W // 2, H // 2
        halfw, halfh = ((2 * cx) // GRID_DIVISOR) * PATCH_MULTIPLIER, ((2 * cy) // GRID_DIVISOR) * PATCH_MULTIPLIER
        crop_start = (cx - halfw, cy - halfh)
        img = img[crop_start[1] : cy + halfh, crop_start[0] : cx + halfw]
        ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        return (
            dict(
                img=ImgNorm(img)[None].to(device),
                true_shape=np.int32([img.shape[:2]]),
                idx=img_id,
                instance=str(img_id),
            ),
            crop_start,
            scale_factor,
        )

    @staticmethod
    def symmetric_inference(model, img1, img2, device):
        """Performs symmetric inference using the MASt3R model on a pair of images.

        By symmetry, we mean that the model computes 4 different point maps:
            - with img1 as the reference frame (res11, res21)
            - with img2 as the reference frame (res22, res12)

        Each result is a pixel-aligned point map in the respective coordinate frame, along with confidence.

        Args:
            model: The MASt3R model.
            img1: Preprocessed dictionary for the first image.
            img2: Preprocessed dictionary for the second image.
            device: The device (e.g., 'cpu' or 'cuda') to run the inference on.

        Returns:
            A tuple of four results (res11, res21, res22, res12) from the symmetric decoding process.
        """
        shape1 = torch.from_numpy(img1["true_shape"]).to(device, non_blocking=True)
        shape2 = torch.from_numpy(img2["true_shape"]).to(device, non_blocking=True)
        img1 = img1["img"].to(device, non_blocking=True)
        img2 = img2["img"].to(device, non_blocking=True)

        # compute encoder only once
        feat1, feat2, pos1, pos2 = model._encode_image_pairs(img1, img2, shape1, shape2)

        def decoder(feat1, feat2, pos1, pos2, shape1, shape2):
            dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)

            with torch.amp.autocast("cuda", enabled=False):
                res1 = model.downstream_head1([tok.float() for tok in dec1], shape1[0])
                res2 = model.downstream_head2([tok.float() for tok in dec2], shape2[0])
            return res1, res2

        # decoder 1-2
        res11, res21 = decoder(feat1, feat2, pos1, pos2, shape1, shape2)
        # decoder 2-1
        res22, res12 = decoder(feat2, feat1, pos2, pos1, shape2, shape1)

        return (res11, res21, res22, res12)

    @staticmethod
    def extract_correspondences(feats, qonfs, subsample=8, device=None, ptmap_key="pred_desc"):
        """Extracts correspondences from MASt3R features and confidence scores.

        Args:
            feats: A tuple of feature tensors (feat11, feat21, feat22, feat12) from symmetric inference.
            qonfs: A tuple of confidence score tensors (qonf11, qonf21, qonf22, qonf12) corresponding to the features.
            subsample: Subsampling rate for nearest neighbor search.
            device: The device (e.g., 'cpu' or 'cuda') for computation.
            ptmap_key: Key to determine the type of point map (e.g., "pred_desc" or "3d").

        Returns:
            A tuple containing:
            - idx1: Indices of correspondences in the first image's flattened feature space.
            - idx2: Indices of correspondences in the second image's flattened feature space.
            - xy1: 2D coordinates of correspondences in the first image.
            - xy2: 2D coordinates of correspondences in the second image.
            - scores: Confidence scores for each correspondence.
        """
        feat11, feat21, feat22, feat12 = feats
        qonf11, qonf21, qonf22, qonf12 = qonfs
        assert feat11.shape[:2] == feat12.shape[:2] == qonf11.shape == qonf12.shape
        assert feat21.shape[:2] == feat22.shape[:2] == qonf21.shape == qonf22.shape

        if "3d" in ptmap_key:
            opt = dict(device="cpu", workers=32)
        else:
            opt = dict(device=device, dist="dot", block_size=2**13)

        # matching the two pairs
        idx1 = []
        idx2 = []
        qonf1 = []
        qonf2 = []
        # TODO add non symmetric / pixel_tol options
        for A, B, QA, QB in [
            (feat11, feat21, qonf11.cpu(), qonf21.cpu()),
            (feat12, feat22, qonf12.cpu(), qonf22.cpu()),
        ]:
            nn1to2 = mast3r_ga.fast_reciprocal_NNs(A, B, subsample_or_initxy1=subsample, ret_xy=False, **opt)
            nn2to1 = mast3r_ga.fast_reciprocal_NNs(B, A, subsample_or_initxy1=subsample, ret_xy=False, **opt)

            idx1.append(np.r_[nn1to2[0], nn2to1[1]])
            idx2.append(np.r_[nn1to2[1], nn2to1[0]])
            qonf1.append(QA.ravel()[idx1[-1]])
            qonf2.append(QB.ravel()[idx2[-1]])

        # merge corres from opposite pairs
        H1, W1 = feat11.shape[:2]
        H2, W2 = feat22.shape[:2]
        cat = np.concatenate

        idx1, idx2, idx = mast3r_ga.merge_corres(cat(idx1), cat(idx2), (H1, W1), (H2, W2), ret_xy=False, ret_index=True)

        shape1 = (H1, W1)
        shape2 = (H2, W2)
        xy1 = np.unravel_index(idx1, shape1)
        xy2 = np.unravel_index(idx2, shape2)
        xy1 = xy1[0].base[:, ::-1]
        xy2 = xy2[0].base[:, ::-1]

        # corres = np.unique(np.c_[idx2, idx1].view(np.int64), return_index=ret_index)
        corres = (idx1, idx2, xy1.copy(), xy2.copy(), np.sqrt(cat(qonf1)[idx] * cat(qonf2)[idx]))

        return corres

    @staticmethod
    def apply_mast3r(model, image1: Image, image2: Image) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Applies the MASt3R model to a pair of images to generate correspondences.

        Args:
            model: The MASt3R model instance.
            image1: The first image.
            image2: The second image.

        Returns:
            A tuple containing:
            - matches_im1: Array of 2D coordinates of correspondences in the first image.
            - matches_im2: Array of 2D coordinates of correspondences in the second image.
            - idx1: Indices of correspondences in the first image's original feature grid.
            - idx2: Indices of correspondences in the second image's original feature grid.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image1_proc, crop1, scale1 = Mast3rCorrespondenceGenerator.preprocess_image(image1, 0, device)
        image2_proc, crop2, scale2 = Mast3rCorrespondenceGenerator.preprocess_image(image2, 1, device)

        subsample = 8

        with torch.no_grad():
            model = model.to(device)

            res = Mast3rCorrespondenceGenerator.symmetric_inference(model, image1_proc, image2_proc, device=device)
            X11, X21, X22, X12 = [r["pts3d"][0] for r in res]
            C11, C21, C22, C12 = [r["conf"][0] for r in res]
            descs = [r["desc"][0] for r in res]
            qonfs = [r["desc_conf"][0] for r in res]

            # We ignore the last result `scores` for now.
            idx1, idx2, matches_im1, matches_im2, _ = Mast3rCorrespondenceGenerator.extract_correspondences(
                descs, qonfs, device=device, subsample=subsample
            )
            # TODO: use the matching score to filter correspondences.
            # conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
            # matching_score = (float(conf_score), float(scores.sum()), len(scores))

            matches_im1 = ((matches_im1 + np.array(crop1)) + 0.5) / scale1
            matches_im2 = ((matches_im2 + np.array(crop2)) + 0.5) / scale2

        return (matches_im1, matches_im2, idx1, idx2)
