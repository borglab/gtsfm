"""GPU/torch two-way (mutual nearest neighbor) matcher with Lowe ratio test.

Drop-in alternative to TwoWayMatcher: identical semantics (mutual NN + ratio test on L2
distances), implemented as one distance GEMM + topk per direction. On an idle datacenter GPU a
full 8192x8192 SIFT pair costs ~1-2 ms vs ~2-4 s for single-threaded OpenCV BFMatcher — the
matching stage drops from hours to minutes. Falls back to CPU torch (threads pinned to 1; dask
workers provide the parallelism) when CUDA is unavailable.
"""

from typing import Optional, Tuple

import numpy as np
import torch

from gtsfm.common.keypoints import Keypoints
from gtsfm.frontend.matcher.matcher_base import MatcherBase


class TorchTwoWayMatcher(MatcherBase):
    """Mutual-NN + ratio-test matcher on torch (CUDA when available)."""

    def __init__(self, ratio_test_threshold: Optional[float] = 0.8, device: Optional[str] = None) -> None:
        super().__init__()
        self._ratio_test_threshold = ratio_test_threshold
        self._device = device  # None => auto (cuda if available else cpu), resolved lazily per process

    def __repr__(self) -> str:
        return f"TorchTwoWayMatcher(ratio_test_threshold={self._ratio_test_threshold})"

    def _resolve_device(self) -> torch.device:
        if self._device is not None:
            return torch.device(self._device)
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    def match(
        self,
        keypoints_i1: Keypoints,
        keypoints_i2: Keypoints,
        descriptors_i1: np.ndarray,
        descriptors_i2: np.ndarray,
        im_shape_i1: Tuple[int, int],
        im_shape_i2: Tuple[int, int],
    ) -> np.ndarray:
        if descriptors_i1 is None or descriptors_i2 is None:
            return np.array([], dtype=np.uint32)
        if descriptors_i1.size == 0 or descriptors_i2.size == 0:
            return np.array([], dtype=np.uint32)

        valid_idx_i1 = np.nonzero(~(np.isnan(descriptors_i1).any(axis=1)))[0]
        valid_idx_i2 = np.nonzero(~(np.isnan(descriptors_i2).any(axis=1)))[0]
        if len(valid_idx_i1) < 2 or len(valid_idx_i2) < 2:
            return np.array([], dtype=np.uint32)

        device = self._resolve_device()
        if device.type == "cpu":
            torch.set_num_threads(1)  # dask workers are the parallelism (same rule as cv2/pycolmap)

        with torch.no_grad():
            d1 = torch.from_numpy(np.ascontiguousarray(descriptors_i1[valid_idx_i1], dtype=np.float32)).to(device)
            d2 = torch.from_numpy(np.ascontiguousarray(descriptors_i2[valid_idx_i2], dtype=np.float32)).to(device)
            # Squared L2 via the GEMM identity; monotone in L2 so NN/topk order is identical, and the
            # ratio test is applied on true (sqrt'd, clamped) distances to match OpenCV exactly.
            sq1 = (d1 * d1).sum(dim=1, keepdim=True)
            sq2 = (d2 * d2).sum(dim=1, keepdim=True)
            dist2 = sq1 + sq2.T - 2.0 * (d1 @ d2.T)
            dist2.clamp_(min=0.0)

            # direction 1->2
            two_12 = torch.topk(dist2, k=min(2, dist2.shape[1]), dim=1, largest=False)
            nn12 = two_12.indices[:, 0]
            # direction 2->1
            two_21 = torch.topk(dist2, k=min(2, dist2.shape[0]), dim=0, largest=False)
            nn21 = two_21.indices[0, :]

            mutual = nn21[nn12] == torch.arange(dist2.shape[0], device=device)

            if self._ratio_test_threshold is not None and dist2.shape[1] >= 2 and dist2.shape[0] >= 2:
                # OpenCV TwoWayMatcher ratio-tests EACH direction before the mutual intersection —
                # a match must pass Lowe's test both 1->2 and 2->1.
                dists_12 = torch.sqrt(two_12.values)
                ratio_ok_12 = dists_12[:, 0] <= self._ratio_test_threshold * dists_12[:, 1]
                dists_21 = torch.sqrt(two_21.values)  # (2, n2): first/second NN distance per d2 col
                ratio_ok_21 = dists_21[0, :] <= self._ratio_test_threshold * dists_21[1, :]
                keep = mutual & ratio_ok_12 & ratio_ok_21[nn12]
            else:
                keep = mutual
                dists_12 = torch.sqrt(two_12.values)

            rows = torch.nonzero(keep, as_tuple=False).squeeze(1)
            if rows.numel() == 0:
                return np.array([], dtype=np.uint32)
            # sort by match distance (ascending), mirroring the OpenCV path
            order = torch.argsort(dists_12[rows, 0])
            rows = rows[order]
            cols = nn12[rows]
            idx1 = valid_idx_i1[rows.cpu().numpy()]
            idx2 = valid_idx_i2[cols.cpu().numpy()]

        return np.column_stack([idx1, idx2]).astype(np.uint32)
