"""Unit tests for VGGT glue.

Authors: Xinan Zhang and Frank Dellaert
"""

import math
import unittest
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.transforms import v2 as transforms  # type: ignore

import gtsfm.frontend.vggt as vggt
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.frontend.vggt import VggtConfiguration
from gtsfm.loader.olsson_loader import OlssonLoader
from gtsfm.utils import torch as torch_utils
from gtsfm.utils.tree import Tree  # PreOrderIter

LocalScene = tuple[Path, GtsfmData]
SceneTree = Tree[LocalScene]

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"
MAX_TRACKS_TO_DRAW = 200
MAX_POINTS_PER_FRAME = 200


def _vibrant_bgr_from_index(index: int) -> tuple[int, int, int]:
    """Generate a visually distinct BGR color using a hashed palette."""

    golden_ratio_hash = 0x9E3779B9
    hash_val = (index * golden_ratio_hash + 0xB5297A4D) & 0xFFFFFFFF

    def _component(shift: int) -> int:
        raw = (hash_val >> shift) & 0xFF
        return 64 + (raw * 191) // 255

    r = _component(16)
    g = _component(8)
    b = _component(0)
    return (b, g, r)


def _restore_images_to_original_scale(square_images: torch.Tensor, original_coords: torch.Tensor) -> torch.Tensor:
    """Crop padded square VGGT inputs back to their native aspect ratios."""

    if square_images.ndim != 4:
        raise ValueError(f"Expected square_images with 4 dims, got shape {tuple(square_images.shape)}")
    if original_coords.ndim != 2 or original_coords.shape[1] != 6:
        raise ValueError(f"original_coords must have shape (N,6); received {tuple(original_coords.shape)}")

    coords = original_coords.to(torch.float32)
    widths = coords[:, 4].round().clamp(min=1).to(torch.int64)
    heights = coords[:, 5].round().clamp(min=1).to(torch.int64)
    max_h = int(torch.max(heights).item())
    max_w = int(torch.max(widths).item())

    num_frames, num_channels, square_h, square_w = square_images.shape
    restored_frames: list[torch.Tensor] = []

    for idx in range(num_frames):
        x1, y1, x2, y2 = coords[idx, :4]

        x1i = int(torch.clamp(torch.floor(x1), 0, square_w - 1).item())
        y1i = int(torch.clamp(torch.floor(y1), 0, square_h - 1).item())
        x2i = int(torch.clamp(torch.ceil(x2), x1i + 1, square_w).item())
        y2i = int(torch.clamp(torch.ceil(y2), y1i + 1, square_h).item())

        crop = square_images[idx : idx + 1, :, y1i:y2i, x1i:x2i]
        if crop.numel() == 0:
            crop = square_images[idx : idx + 1]

        target_h = int(heights[idx].item())
        target_w = int(widths[idx].item())
        resized = F.interpolate(crop, size=(target_h, target_w), mode="bilinear", align_corners=False)

        canvas = torch.zeros((1, num_channels, max_h, max_w), dtype=square_images.dtype, device=square_images.device)
        canvas[:, :, :target_h, :target_w] = resized
        restored_frames.append(canvas)

    return torch.cat(restored_frames, dim=0).clamp(0.0, 1.0)


def run_vggt(
    image_batch: torch.Tensor,
    image_indices: list[int],
    original_coords,
    seed=42,
    conf_threshold_value=5.0,
    vggt_fixed_resolution=518,
    img_load_resolution=1024,
    max_query_pts=1000,
    query_frame_num=4,
    fine_tracking=True,
    vis_thresh=0.2,
    max_reproj_error=8.0,
    camera_type="SIMPLE_PINHOLE",
) -> GtsfmData:
    """Run VGGT on the given image keys and return GtsfmData."""

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Setting seed as: {seed}")

    device = torch_utils.default_device()
    dtype = vggt.default_dtype(device)
    print(f"Using device: {device.type}")
    print(f"Using dtype: {dtype}")

    model = vggt.load_model(device=device)
    print("Model loaded")

    image_batch = image_batch.to(device)
    print("image_batch: ", image_batch.shape)
    original_coords = original_coords.to(device)

    config = VggtConfiguration(
        vggt_fixed_resolution=vggt_fixed_resolution,
        img_load_resolution=img_load_resolution,
        max_query_pts=max_query_pts,
        query_frame_num=query_frame_num,
        fine_tracking=fine_tracking,
        track_vis_thresh=vis_thresh,
        max_reproj_error=max_reproj_error,
        confidence_threshold=conf_threshold_value,
    )

    result = vggt.run_reconstruction(
        image_batch,
        image_indices=image_indices,
        image_names=[f"image_{idx}" for idx in image_indices],
        original_coords=original_coords,
        config=config,
        model=model,
    )

    if result.points_3d.size == 0:
        print("VGGT produced no confident 3D structure.")

    return result.gtsfm_data


TEST_DATA = Path(__file__).parent.parent / "data"
PALACE = TEST_DATA / "palace-fine-arts-281"
DOOR = TEST_DATA / "set1_lund_door"


class TestVGGT(unittest.TestCase):

    def setUp(self) -> None:
        pass

    @unittest.skip("Skipping VGGT end-to-end test for now since it is slow and requires GPU.")
    def test_coordinate_round_trip(self) -> None:
        """Ensure the helper that maps VGGT grid coordinates back to original pixels is consistent.

        VGGT operates on square, padded inputs. The loader pads each image to a square, rescales it
        to ``img_load_resolution`` (1024 in our tests), and VGGT then down-samples that square to the
        inference resolution (518). ``_convert_measurement_to_original_resolution`` must undo both
        scaling steps and the padding offsets so that the merged reconstructions use the correct
        camera indices during alignment. This test uses an analytic example where we can derive the
        expected coordinates exactly and checks that the helper inverts the forward mapping.
        """

        width, height = 640, 480
        img_load_resolution = 1024
        inference_resolution = 518

        max_side = max(width, height)
        pad_left = (max_side - width) / 2.0
        pad_top = (max_side - height) / 2.0
        scale = img_load_resolution / max_side
        original_coord = np.array(
            [
                pad_left * scale,
                pad_top * scale,
                (pad_left + width) * scale,
                (pad_top + height) * scale,
                width,
                height,
            ],
            dtype=np.float32,
        )

        def forward_to_inference(u: float, v: float) -> tuple[float, float]:
            """Simulate the loader + VGGT downsampling path to produce inference-space coords."""

            u_padded = (u + pad_left) * scale
            v_padded = (v + pad_top) * scale
            shrink = inference_resolution / img_load_resolution
            return (u_padded * shrink, v_padded * shrink)

        # Corners and center stress the padding math.
        samples = [
            (0.0, 0.0),
            (width - 1.0, 0.0),
            (0.0, height - 1.0),
            (width - 1.0, height - 1.0),
            (width / 2.0, height / 2.0),
        ]

        for u_orig, v_orig in samples:
            uv_infer = forward_to_inference(u_orig, v_orig)
            u_back, v_back = vggt._convert_measurement_to_original_resolution(
                uv_infer,
                original_coord,
                inference_resolution,
                img_load_resolution,
            )
            self.assertAlmostEqual(u_back, u_orig, places=3)
            self.assertAlmostEqual(v_back, v_orig, places=3)

            uv_load = ((u_orig + pad_left) * scale, (v_orig + pad_top) * scale)
            u_back_load, v_back_load = vggt._convert_measurement_to_original_resolution(
                uv_load,
                original_coord,
                inference_resolution,
                img_load_resolution,
                measurement_in_load_resolution=True,
            )
            self.assertAlmostEqual(u_back_load, u_orig, places=3)
            self.assertAlmostEqual(v_back_load, v_orig, places=3)

    @unittest.skip("Skipping VGGT end-to-end test for now since it is slow and requires GPU.")
    def test_run_vggt_on_some_images(self):
        """Load four door images using Olsson loader and run vggt on them."""

        img_load_original_resolution = 760
        img_load_resolution = 1024
        loader = OlssonLoader(dataset_dir=str(DOOR), max_resolution=img_load_original_resolution)
        indices = [4, 11, 8, 2]

        # resize_transform = None
        resize_transform = transforms.Compose(
            [
                transforms.Lambda(lambda x: torch.from_numpy(x)),
                transforms.Lambda(lambda x: x.permute(2, 0, 1)),  # [H,W,C] → [C,H,W]
                transforms.Resize(size=(img_load_resolution, img_load_resolution), antialias=True),  # Expects [C,H,W]
            ]
        )
        # Transform 2: Convert to float32 and normalize to [0, 1]
        batch_transform = transforms.Lambda(lambda x: x.type(torch.float32) / 255.0)

        image_batch, original_coords = loader.load_image_batch_vggt(
            indices,
            img_load_resolution,
            resize_transform,
            batch_transform,
        )

        # image_batch, original_coords = loader.load_and_preprocess_images_square_vggt(indices, img_load_resolution)

        print("image_batch: ", image_batch.shape)

        with torch.no_grad():

            gtsfm_data = run_vggt(image_batch, indices, original_coords)

        self.assertIsNotNone(gtsfm_data)
        self.assertEqual(gtsfm_data.number_images(), len(indices))
        self.assertCountEqual(gtsfm_data.get_valid_camera_indices(), indices)

    @unittest.skip("Skipping because this test will be merged to the previous test.")
    def test_convert_measurement_to_original_resolution_door_extremes(self) -> None:
        """Ensure VGGT coordinate conversion preserves pixel centers for a real Door image."""

        img_load_resolution = 1024
        inference_resolution = vggt.DEFAULT_FIXED_RESOLUTION

        def _jpeg_size(path: Path) -> tuple[int, int]:
            with path.open("rb") as stream:
                if stream.read(2) != b"\xff\xd8":
                    raise ValueError("Not a JPEG file.")
                while True:
                    marker_start = stream.read(1)
                    if not marker_start:
                        raise ValueError("Reached EOF before finding SOF marker.")
                    if marker_start != b"\xff":
                        continue
                    marker_code = stream.read(1)
                    if not marker_code or marker_code == b"\x00":
                        continue
                    code = marker_code[0]
                    if code in {0xD8, 0xD9} or 0xD0 <= code <= 0xD7:
                        continue
                    length_bytes = stream.read(2)
                    if len(length_bytes) != 2:
                        raise ValueError("Unexpected end of marker payload.")
                    block_length = int.from_bytes(length_bytes, "big")
                    if code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                        payload = stream.read(block_length - 2)
                        if len(payload) != block_length - 2:
                            raise ValueError("Incomplete SOF payload.")
                        height = int.from_bytes(payload[1:3], "big")
                        width = int.from_bytes(payload[3:5], "big")
                        return width, height
                    stream.seek(block_length - 2, 1)

        image_path = DOOR / "images" / "DSC_0004.JPG"
        width, height = _jpeg_size(image_path)
        max_side = max(width, height)
        pad_width = max_side - width
        pad_height = max_side - height
        pad_left = pad_width // 2
        pad_top = pad_height // 2
        scale = img_load_resolution / max_side

        original_coord = np.array(
            [
                pad_left * scale,
                pad_top * scale,
                (pad_left + width) * scale,
                (pad_top + height) * scale,
                width,
                height,
            ],
            dtype=np.float32,
        )

        def forward_to_inference(u: float, v: float) -> tuple[float, float]:
            u_padded = (u + pad_left) * scale
            v_padded = (v + pad_top) * scale
            shrink = inference_resolution / img_load_resolution
            return (u_padded * shrink, v_padded * shrink)

        samples = [
            (0.5, 0.5),
            (width - 0.5, height - 0.5),
        ]

        for u_orig, v_orig in samples:
            uv_infer = forward_to_inference(u_orig, v_orig)
            u_back, v_back = vggt._convert_measurement_to_original_resolution(
                uv_infer,
                original_coord,
                inference_resolution,
                img_load_resolution,
            )
            self.assertAlmostEqual(u_back, u_orig, places=3)
            self.assertAlmostEqual(v_back, v_orig, places=3)

            uv_load = ((u_orig + pad_left) * scale, (v_orig + pad_top) * scale)
            u_back_load, v_back_load = vggt._convert_measurement_to_original_resolution(
                uv_load,
                original_coord,
                inference_resolution,
                img_load_resolution,
                measurement_in_load_resolution=True,
            )
            self.assertAlmostEqual(u_back_load, u_orig, places=3)
            self.assertAlmostEqual(v_back_load, v_orig, places=3)


if __name__ == "__main__":
    unittest.main()
