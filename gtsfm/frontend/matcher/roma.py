"""RoMa image matcher.

The network was proposed in "RoMa: Revisiting Robust Losses for Dense Feature Matching".

References:
- https://arxiv.org/html/2305.15404v2

Authors: Travis Driver
"""
import sys
from typing import Optional, Tuple

import gtsfm.utils.viz as viz_utils
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch
import torch.nn.functional as F
from gtsfm.common.image import Image
from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.image_matcher_base import ImageMatcherBase

sys.path.append("thirdparty/RoMa")
from thirdparty.RoMa.roma import roma_indoor, roma_outdoor


class RoMa(ImageMatcherBase):
    """RoMa image matcher."""

    def __init__(
        self,
        use_outdoor_model: bool = True,
        use_cuda: bool = True,
        min_confidence: float = 0.4,
        max_keypoints: int = 5000,
    ) -> None:
        """Initialize the matcher.

        Args:
            use_outdoor_model (optional): use the outdoor pretrained model. Defaults to True.
            use_cuda (optional): use CUDA for inference on GPU. Defaults to True.
            min_confidence(optional): Minimum confidence required for matches. Defaults to 0.95.
            upsample_res: resolution of upsampled warp and certainty maps. Stored as (H, W).
        """
        super().__init__()
        self._min_confidence = min_confidence
        self._max_keypoints = max_keypoints

        # Initialize model.
        self._device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )
        if use_outdoor_model:
            self._matcher = roma_outdoor(self._device).eval()
        else:
            self._matcher = roma_indoor(self._device).eval()

    def match(
        self,
        image_i1: Image,
        image_i2: Image,
        keypoints_i1: Optional[Keypoints] = None,
        keypoints_i2: Optional[Keypoints] = None,
    ) -> Tuple[Keypoints, Keypoints]:
        """Identify feature matches across two images.

        Note: the matcher will run out of memory for large image sizes

        Args:
            image_i1: first input image of pair.
            image_i2: second input image of pair.

        Returns:
            Keypoints from image 1 (N keypoints will exist).
            Corresponding keypoints from image 2 (there will also be N keypoints). These represent feature matches.
        """
        # Compute dense warp and certainty maps.
        with torch.no_grad():
            im1 = PIL.Image.fromarray(image_i1.value_array).convert("RGB")
            im2 = PIL.Image.fromarray(image_i2.value_array).convert("RGB")
            warp, certainty = self._matcher.match(im1, im2, device=self._device)
        print("computed warp and certainty")

        # Sample keypoints and correspondences from warp.
        H1, W1 = image_i1.shape[:2]
        H2, W2 = image_i2.shape[:2]
        print(H1, W1, H2, W2)
        H, W = self._matcher.upsample_res
        if keypoints_i1 is not None and keypoints_i2 is not None:
            with torch.no_grad():
                (
                    norm_coords_i1,
                    norm_coords_i2,
                ) = self._matcher.to_normalized_coordinates(
                    (
                        torch.from_numpy(keypoints_i1.coordinates + 0.5).to(
                            self._device
                        ),
                        torch.from_numpy(keypoints_i2.coordinates + 0.5).to(
                            self._device
                        ),
                    ),
                    H1,
                    W1,
                    H2,
                    W2,
                )
                # rint(norm_coords_i1)
                # print(norm_coords_i1.shape)
                print("normalized coordinates")
                # mkpts1, mkpts2, norm_coords_i1_to_i2 = self.match_keypoints(
                #    norm_coords_i1, norm_coords_i2, warp, certainty
                # )
                print("matched keypoints")
                # coords_i1_i2 = to_pixel_coordinates(norm_coords_i1_to_i2)
                minds1, minds2, norm_coords_i1_to_i2 = self.match_keypoints(
                    norm_coords_i1,
                    norm_coords_i2,
                    warp[:, :W],
                    certainty,
                )
                print(norm_coords_i1_to_i2)
                coords_i1_to_i2 = to_pixel_coordinates(norm_coords_i1_to_i2, H2, W2)

                viz_utils.save_twoview_correspondences_viz(
                    image_i1,
                    image_i2,
                    keypoints_i1,
                    keypoints_i2,
                    np.hstack(
                        (
                            minds1.cpu().numpy()[..., None],
                            minds2.cpu().numpy()[..., None],
                        )
                    ),
                    # np.hstack(
                    #    (
                    #        np.arange(keypoints_i1.coordinates.shape[0])[..., None],
                    #        np.arange(keypoints_i1.coordinates.shape[0])[..., None],
                    #    )
                    # ),
                    two_view_report=None,
                    file_path="debug.jpg",
                )

                # plt.imshow(image_i1.value_array)
                # coords_i1_to_i2 = coords_i1_to_i2.cpu().numpy()
                # for i in range(keypoints_i1.coordinates.shape[0]):
                #    plt.plot(
                #        [keypoints_i1.coordinates[i, 0], coords_i1_to_i2[i, 0]],
                #        [keypoints_i1.coordinates[i, 1], coords_i1_to_i2[i, 1]],
                #    )
                # plt.savefig("debug.png", dpi=300)
                # plt.close()

                return np.hstack(
                    (minds1.cpu().numpy()[..., None], minds2.cpu().numpy()[..., None])
                )
        else:
            match, certs = self._matcher.sample(
                warp, certainty, num=self._max_keypoints
            )
            match = match[certs > self._min_confidence]
            mkpts1, mkpts2 = self._matcher.to_pixel_coordinates(match, H1, W1, H2, W2)

        # Convert to GTSfM keypoints and filter by mask.
        keypoints_i1 = Keypoints(coordinates=mkpts1.cpu().numpy())
        keypoints_i2 = Keypoints(coordinates=mkpts2.cpu().numpy())
        valid_ind = np.arange(len(keypoints_i1))
        if image_i1.mask is not None:
            _, valid_ind_i1 = keypoints_i1.filter_by_mask(image_i1.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i1)
        if image_i2.mask is not None:
            _, valid_ind_i2 = keypoints_i2.filter_by_mask(image_i2.mask)
            valid_ind = np.intersect1d(valid_ind, valid_ind_i2)

        return keypoints_i1.extract_indices(valid_ind), keypoints_i2.extract_indices(
            valid_ind
        )

    def match_keypoints(
        self,
        x_A: torch.Tensor,
        x_B: torch.Tensor,
        warp: torch.Tensor,
        certainty: torch.Tensor,
        dist_threshold: float = 1e-2,
    ):
        print("WARP SHAPE", warp.shape)
        print("WARP SHAPE", warp[..., -2:].permute(2, 0, 1)[None].shape)
        x_A_to_B = F.grid_sample(
            warp[..., -2:].permute(2, 0, 1)[None],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
            padding_mode="border",
        )[0, :, 0].mT
        print(x_A.shape)
        print(x_A_to_B.shape)
        cert_A_to_B = F.grid_sample(
            certainty[None, None, ...],
            x_A[None, None],
            align_corners=False,
            mode="bilinear",
            padding_mode="border",
        )[0, 0, 0]
        D = torch.cdist(x_A_to_B, x_B)
        inds_A, inds_B = torch.nonzero(
            (D == D.min(dim=-1, keepdim=True).values)
            * (D == D.min(dim=-2, keepdim=True).values)
            * (D.min(dim=-1, keepdim=True).values < 5e-3)
            * (D.min(dim=-2, keepdim=True).values < 5e-3)
            * (cert_A_to_B[:, None] > self._min_confidence),
            as_tuple=True,
        )
        print(D.shape)
        print(D.min(dim=-1, keepdim=True).values.shape)
        print(D.min(dim=-1, keepdim=True).values)
        print(inds_A.shape)
        print(inds_B.shape)
        # return x_A[inds_A], x_B[inds_B], x_A_to_B
        return inds_A, inds_B, x_A_to_B


def to_pixel_coordinates(norm_coords, H, W):
    coords = torch.stack(
        (W / 2 * (norm_coords[..., 0] + 1), H / 2 * (norm_coords[..., 1] + 1)), axis=-1
    )
    return coords
