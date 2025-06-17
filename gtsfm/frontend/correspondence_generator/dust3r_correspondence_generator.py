"""Correspondence generator using the DUSt3R model.

https://github.com/naver/dust3r

Authors: Akshay Krishnan
"""

from typing import Any, Dict, List, Tuple
import torch
from dask.distributed import Client, Future
from dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
import numpy as np
import torchvision.transforms as tvf
from pathlib import Path
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
    / "dust3r"
    / "DUSt3R_ViTLarge_BaseDecoder_512_dpt_params.pth"
)


def resize_image(img: Image, long_edge_size: int) -> Image:
    """Resizes image such that longest edge is equal to long_edge_size."""
    max_size = max(img.height, img.width)
    ratio = float(long_edge_size) / max_size
    new_height = int(img.height * ratio)
    new_width = int(img.width * ratio)
    new_image = image_utils.resize_image(img, new_height, new_width)
    return new_image


def preprocess_image(image: Image, img_id: int) -> Tuple[Dict[str, Any], Tuple[int, int], float]:
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
            img=ImgNorm(img)[None],
            true_shape=np.int32([img.shape[:2]]),
            idx=img_id,
            instance=str(img_id),
        ),
        crop_start,
        scale_factor,
    )


class Dust3rCorrespondenceGenerator(CorrespondenceGeneratorBase):
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
        model_name = "naver/DUSt3R_ViTLarge_BaseDecoder_512_dpt"
        model1 = AsymmetricCroCo3DStereo.from_pretrained(model_name).eval()
        m = client.scatter(model1, broadcast=False)

        def apply_dust3r(model, image1: Image, image2: Image) -> Tuple[Keypoints, Keypoints]:

            image1_proc, crop1, scale1 = preprocess_image(image1, 0)
            image2_proc, crop2, scale2 = preprocess_image(image2, 1)

            with torch.no_grad():
                model = model.to("cuda")
                output = inference(
                    [(image1_proc, image2_proc), (image2_proc, image1_proc)], model, "cuda", batch_size=1, verbose=False
                )

                scene = global_aligner(output, device="cuda", mode=GlobalAlignerMode.PairViewer, verbose=False)
                confidence_masks = scene.get_masks()
                confidences = scene.im_conf
                pts3d = scene.get_pts3d()

                conf1 = confidence_masks[0].cpu().numpy()
                points2d_1 = xy_grid(*image1_proc["true_shape"][0, ::-1])[conf1]
                points3d_1 = pts3d[0].detach().cpu().numpy()[conf1]
                conf_val1 = confidences[0].cpu().numpy()[conf1]

                conf2 = confidence_masks[1].cpu().numpy()
                points2d_2 = xy_grid(*image2_proc["true_shape"][0, ::-1])[conf2]
                points3d_2 = pts3d[1].detach().cpu().numpy()[conf2]
                conf_val2 = confidences[1].cpu().numpy()[conf2]

                reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(points3d_1, points3d_2)
                matches_im2 = points2d_2[reciprocal_in_P2]
                matches_im1 = points2d_1[nn2_in_P1][reciprocal_in_P2]

                match_conf_im2 = conf_val2[reciprocal_in_P2]
                match_conf_im1 = conf_val1[nn2_in_P1][reciprocal_in_P2]

                sort_idxs = np.argpartition(-match_conf_im2 * match_conf_im1, 10000)[:10000]
                matches_im1 = matches_im1[sort_idxs]
                matches_im2 = matches_im2[sort_idxs]

                matches_im1 = ((matches_im1 + np.array(crop1)) + 0.5) / scale1
                matches_im2 = ((matches_im2 + np.array(crop2)) + 0.5) / scale2

            return (Keypoints(matches_im1), Keypoints(matches_im2))

        pairwise_correspondence_futures = {
            (i1, i2): client.submit(
                apply_dust3r,
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
