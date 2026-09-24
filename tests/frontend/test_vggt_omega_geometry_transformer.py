"""Unit tests for the VGGT-Omega image preprocessing (CPU-only; no weights required).

The omega loader is what keeps the depth lookup consistent with the model's output resolution, so its
contract — 16-patch-aligned sizes, a padded batch of mixed aspect ratios, and ``original_coords`` that map
loader pixels into the padded omega frame — is pinned here.
"""

import unittest

import numpy as np
import pytest
import torch

from gtsfm.common.image import Image

try:
    from gtsfm.frontend.vggt_omega_geometry_transformer import (
        VggtOmegaGeometryTransformer,
        load_image_batch_vggt_omega_loader,
    )
except ImportError:  # thirdparty/vggt-omega submodule not checked out
    pytest.skip("vggt-omega submodule not available", allow_module_level=True)

PATCH = 16
RESOLUTION = 512


class _ArrayLoader:
    """Minimal loader exposing ``get_image`` over in-memory RGB arrays."""

    def __init__(self, shapes: list[tuple[int, int]]) -> None:
        rng = np.random.default_rng(0)
        self._images = [Image(value_array=rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for h, w in shapes]

    def get_image(self, index: int) -> Image:
        return self._images[index]


class TestVggtOmegaLoader(unittest.TestCase):
    def test_balanced_mode_is_patch_aligned_and_padded(self) -> None:
        loader = _ArrayLoader([(480, 640), (640, 480), (300, 900)])  # landscape, portrait, wide
        batch, coords = load_image_batch_vggt_omega_loader(loader, [0, 1, 2], mode="balanced")

        self.assertEqual(batch.shape[0], 3)
        self.assertEqual(batch.shape[1], 3)
        self.assertEqual(batch.shape[2] % PATCH, 0)
        self.assertEqual(batch.shape[3] % PATCH, 0)
        self.assertEqual(coords.shape, (3, 6))
        self.assertEqual(coords.dtype, torch.float32)

    def test_max_size_mode_longest_side_is_resolution(self) -> None:
        loader = _ArrayLoader([(480, 640)])
        batch, _ = load_image_batch_vggt_omega_loader(loader, [0], mode="max_size")
        self.assertEqual(max(batch.shape[2], batch.shape[3]), RESOLUTION)

    def test_original_coords_map_loader_pixels_into_padded_frame(self) -> None:
        loader = _ArrayLoader([(480, 640), (640, 480)])
        batch, coords = load_image_batch_vggt_omega_loader(loader, [0, 1], mode="balanced")
        batch_h, batch_w = batch.shape[2], batch.shape[3]
        for i, (h, w) in enumerate([(480, 640), (640, 480)]):
            left, top, _, _, scaled_w, scaled_h = coords[i].tolist()
            # Mapping used by the frontend: u_omega = u_loader * scaled_w / loader_w - left.
            u = (w / 2) * scaled_w / w - left
            v = (h / 2) * scaled_h / h - top
            self.assertTrue(0 <= u < batch_w, f"image {i}: u={u} outside [0,{batch_w})")
            self.assertTrue(0 <= v < batch_h, f"image {i}: v={v} outside [0,{batch_h})")

    def test_invalid_mode_rejected_by_loader_but_normalized_by_transformer(self) -> None:
        loader = _ArrayLoader([(480, 640)])
        with self.assertRaises(ValueError):
            load_image_batch_vggt_omega_loader(loader, [0], mode="crop")
        # The transformer maps VGGT-style modes (crop/pad) onto omega's default.
        batch, coords = VggtOmegaGeometryTransformer().load_image_batch(loader, [0], mode="crop")
        self.assertEqual(batch.shape[0], 1)
        self.assertEqual(coords.shape, (1, 6))


if __name__ == "__main__":
    unittest.main()
