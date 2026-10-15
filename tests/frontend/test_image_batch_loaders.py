"""Golden regression tests for the VGGT and VGGT-Omega image-batch loaders.

The ``original_coords`` contract is the most fragile part of the geometry frontends: the depth lookup
indexes each model's output frame through it. These values were captured from the loaders' behavior
prior to factoring the shared batch-assembly out (``assemble_image_batch``), so any refactor of that
logic must reproduce them exactly. Row contract: ``[left, top, left + batch_w, top + batch_h,
scaled_w, scaled_h]`` with ``u_model = u_loader * scaled_w / loader_w - left``.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
import torch

from gtsfm.common.image import Image
from gtsfm.frontend.vggt_geometry_transformer import load_image_batch_vggt_loader

try:
    from gtsfm.frontend.vggt_omega_geometry_transformer import load_image_batch_vggt_omega_loader

    _HAS_OMEGA = True
except ImportError:  # thirdparty/vggt-omega submodule not checked out
    _HAS_OMEGA = False

# (height, width): two identical landscapes, one portrait, one extreme (4.5:1) aspect ratio.
_SHAPES = [(400, 600), (400, 600), (640, 480), (200, 900)]


class _ArrayLoader:
    """Minimal loader exposing ``get_image`` over constant in-memory RGB arrays."""

    def __init__(self):
        self._images = [Image(value_array=np.full((h, w, 3), 128, dtype=np.uint8)) for (h, w) in _SHAPES]

    def get_image(self, index: int) -> Image:
        return self._images[index]


def _check(testcase, batch, coords, expected_shape, expected_coords):
    testcase.assertEqual(tuple(batch.shape), expected_shape)
    np.testing.assert_allclose(coords.numpy(), np.asarray(expected_coords, dtype=np.float32), atol=1e-3)
    testcase.assertEqual(coords.dtype, torch.float32)


class TestVggtLoaderGolden(unittest.TestCase):
    def setUp(self):
        self.loader = _ArrayLoader()

    def test_crop_uniform_batch(self):
        batch, coords = load_image_batch_vggt_loader(self.loader, [0, 1], mode="crop")
        _check(self, batch, coords, (2, 3, 350, 518), [[0, 0, 518, 350, 518, 350]] * 2)

    def test_crop_mixed_batch(self):
        batch, coords = load_image_batch_vggt_loader(self.loader, [0, 2], mode="crop")
        _check(self, batch, coords, (2, 3, 518, 518), [[0, -84, 518, 434, 518, 350], [0, 84, 518, 602, 518, 686]])

    def test_crop_extreme_aspect(self):
        batch, coords = load_image_batch_vggt_loader(self.loader, [3], mode="crop")
        _check(self, batch, coords, (1, 3, 112, 518), [[0, 0, 518, 112, 518, 112]])

    def test_pad_uniform_batch(self):
        batch, coords = load_image_batch_vggt_loader(self.loader, [0, 1], mode="pad")
        _check(self, batch, coords, (2, 3, 518, 518), [[0, -84, 518, 434, 518, 350]] * 2)
        # Padded rows carry the 1.0 fill value; image rows carry the constant test image.
        self.assertTrue(torch.allclose(batch[0, :, 0, :], torch.ones(3, 518)))
        self.assertTrue(torch.allclose(batch[0, :, 259, :], torch.full((3, 518), 128 / 255.0), atol=1e-2))

    def test_pad_mixed_batch(self):
        batch, coords = load_image_batch_vggt_loader(self.loader, [0, 2], mode="pad")
        _check(self, batch, coords, (2, 3, 518, 518), [[0, -84, 518, 434, 518, 350], [-63, 0, 455, 518, 392, 518]])

    def test_pad_extreme_aspect(self):
        batch, coords = load_image_batch_vggt_loader(self.loader, [3], mode="pad")
        _check(self, batch, coords, (1, 3, 518, 518), [[0, -203, 518, 315, 518, 112]])


@unittest.skipUnless(_HAS_OMEGA, "vggt-omega submodule not available")
class TestVggtOmegaLoaderGolden(unittest.TestCase):
    def setUp(self):
        self.loader = _ArrayLoader()

    def test_balanced_uniform_batch(self):
        batch, coords = load_image_batch_vggt_omega_loader(self.loader, [0, 1], mode="balanced")
        _check(self, batch, coords, (2, 3, 416, 624), [[0, 0, 624, 416, 624, 416]] * 2)

    def test_balanced_mixed_batch(self):
        batch, coords = load_image_batch_vggt_omega_loader(self.loader, [0, 2], mode="balanced")
        _check(self, batch, coords, (2, 3, 592, 624), [[0, -88, 624, 504, 624, 416], [-88, 0, 536, 592, 448, 592]])

    def test_balanced_extreme_aspect_is_cropped(self):
        batch, coords = load_image_batch_vggt_omega_loader(self.loader, [3], mode="balanced")
        _check(self, batch, coords, (1, 3, 368, 720), [[450, 0, 1170, 368, 1620, 368]])

    def test_max_size_uniform_batch(self):
        batch, coords = load_image_batch_vggt_omega_loader(self.loader, [0, 1], mode="max_size")
        _check(self, batch, coords, (2, 3, 336, 512), [[0, 0, 512, 336, 512, 336]] * 2)

    def test_max_size_mixed_batch(self):
        batch, coords = load_image_batch_vggt_omega_loader(self.loader, [0, 2], mode="max_size")
        _check(self, batch, coords, (2, 3, 512, 512), [[0, -88, 512, 424, 512, 336], [-64, 0, 448, 512, 384, 512]])

    def test_max_size_extreme_aspect_is_cropped(self):
        batch, coords = load_image_batch_vggt_omega_loader(self.loader, [3], mode="max_size")
        _check(self, batch, coords, (1, 3, 256, 512), [[320, 0, 832, 256, 1152, 256]])


if __name__ == "__main__":
    unittest.main()
