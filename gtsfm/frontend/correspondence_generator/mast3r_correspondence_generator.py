"""Correspondence generator using the MASt3R model.

https://github.com/naver/mast3r

Authors: Akshay Krishnan
"""

from typing import Any, Dict, List, Tuple
import torch
from dask.distributed import Client, Future
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
from mast3r.utils.misc import hash_md5
import mast3r.cloud_opt.sparse_ga as mast3r_ga

import numpy as np
import torchvision.transforms as tvf
import time

from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.utils import images as image_utils
from gtsfm.utils import logger as logger_utils
from gtsfm.frontend.correspondence_generator.correspondence_generator_base import (
    CorrespondenceGeneratorBase,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_base import KeypointAggregatorBase
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_dedup import (
    KeypointAggregatorDedup,
)
from gtsfm.frontend.correspondence_generator.keypoint_aggregator.keypoint_aggregator_unique import (
    KeypointAggregatorUnique,
)

ImgNorm = tvf.Compose([tvf.ToTensor(), tvf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
logger = logger_utils.get_logger()
MODEL_PATH = str(
    Path(__file__).resolve().parent.parent.parent.parent
    / "thirdparty"
    / "mast3r"
    / "checkpoints"
    / "MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth"
)


def resize_image(img: Image, long_edge_size: int) -> Image:
    """Resizes image such that longest edge is equal to long_edge_size."""
    max_size = max(img.height, img.width)
    ratio = float(long_edge_size) / max_size
    new_height = int(img.height * ratio)
    new_width = int(img.width * ratio)
    new_image = image_utils.resize_image(img, new_height, new_width)
    return new_image


def preprocess_image(image: Image, img_id: int, device) -> Tuple[Dict[str, Any], Tuple[int, int], float]:
    """Resizes image such that longest edge is equal to long_edge_size."""
    H1, _, _ = image.shape

    # resize to longest edge being `size`.
    img = resize_image(image, long_edge_size=512).value_array
    H, W, _ = img.shape
    scale_factor = float(H) / H1

    cx, cy = W // 2, H // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    crop_start = (cx - halfw, cy - halfh)
    img = img[crop_start[1] : cy + halfh, crop_start[0] : cx + halfw]

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


class Mast3rCorrespondenceGenerator(CorrespondenceGeneratorBase):
    """Base class for correspondence generators."""

    def __init__(self, deduplicate: bool = True, nms_merge_radius: float = 0.5, max_correspondences: int = 1000):
        super().__init__()
        self._aggregator: KeypointAggregatorBase = (
            KeypointAggregatorDedup(nms_merge_radius) if deduplicate else KeypointAggregatorUnique()
        )
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
        device = torch.device("cpu")
        model = AsymmetricMASt3R.from_pretrained(MODEL_PATH).eval()
        chkpt_tag = hash_md5(MODEL_PATH)

        m = client.scatter(model, broadcast=False)
        subsample = 8

        def apply_mast3r(model, image1: Image, image2: Image) -> Tuple[Keypoints, Keypoints]:

            image1_proc, crop1, scale1 = preprocess_image(image1, 0, device)
            image2_proc, crop2, scale2 = preprocess_image(image2, 1, device)

            with torch.no_grad():
                model = model.to(device)

                res = mast3r_ga.symmetric_inference(model, image1_proc, image2_proc, device=device)
                X11, X21, X22, X12 = [r["pts3d"][0] for r in res]
                C11, C21, C22, C12 = [r["conf"][0] for r in res]
                descs = [r["desc"][0] for r in res]
                qonfs = [r["desc_conf"][0] for r in res]

                matches_im1, matches_im2, scores = mast3r_ga.extract_correspondences(
                    descs, qonfs, device=device, subsample=subsample
                )
                conf_score = (C11.mean() * C12.mean() * C21.mean() * C22.mean()).sqrt().sqrt()
                # TODO: we dont currently use this
                matching_score = (float(conf_score), float(scores.sum()), len(scores))

                matches_im1 = ((matches_im1.cpu().numpy() + np.array(crop1)) + 0.5) / scale1
                matches_im2 = ((matches_im2.cpu().numpy() + np.array(crop2)) + 0.5) / scale2

            return (Keypoints(matches_im1), Keypoints(matches_im2))

        pairwise_correspondence_futures = {
            (i1, i2): client.submit(
                apply_mast3r,
                m,
                images[i1],
                images[i2],
            )
            for i1, i2 in image_pairs
        }
        pairwise_correspondences: Dict[Tuple[int, int], Tuple[Keypoints, Keypoints]] = client.gather(
            pairwise_correspondence_futures
        )

        keypoints_list, putative_corr_idxs_dict = self._aggregator.aggregate(keypoints_dict=pairwise_correspondences)
        return keypoints_list, putative_corr_idxs_dict
