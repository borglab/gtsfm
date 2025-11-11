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


def _visualize_gtsfm_tracks_on_original_frames(
    square_images: torch.Tensor,
    original_coords: torch.Tensor,
    gtsfm_data: GtsfmData,
    image_indices: list[int],
    output_dir: Path,
) -> None:
    """Restore images to native scale and overlay 2D track measurements from ``GtsfmData``."""

    if gtsfm_data.number_tracks() == 0:
        return

    restored = _restore_images_to_original_scale(square_images, original_coords)

    num_frames = len(image_indices)
    index_lookup = {img_idx: local_idx for local_idx, img_idx in enumerate(image_indices)}
    per_frame_measurements: list[list[tuple[tuple[float, float], tuple[int, int, int]]]] = [
        [] for _ in range(num_frames)
    ]
    track_sequences: list[dict[str, Any]] = []

    # Tracks stored in ``gtsfm_data`` are already expressed in the original image coordinates by vggt.py.
    processed_tracks = 0
    for track_idx in range(gtsfm_data.number_tracks()):
        if processed_tracks >= MAX_TRACKS_TO_DRAW:
            break
        track = gtsfm_data.get_track(track_idx)
        if track is None:
            continue
        measurements: list[tuple[int, float, float]] = []
        for meas_idx in range(track.numberMeasurements()):
            img_idx, uv = track.measurement(meas_idx)
            local_idx = index_lookup.get(img_idx)
            if local_idx is None:
                continue
            if hasattr(uv, "x"):
                u = float(uv.x())
                v = float(uv.y())
            elif isinstance(uv, np.ndarray):
                u = float(uv[0])
                v = float(uv[1])
            else:
                # assume tuple-like
                u = float(uv[0])
                v = float(uv[1])
            measurements.append((local_idx, u, v))

        if len(measurements) == 0:
            continue

        color_bgr = _vibrant_bgr_from_index(track_idx)
        measurements = sorted(measurements, key=lambda m: m[0])
        point3 = track.point3() if hasattr(track, "point3") else None

        track_sequences.append(
            {
                "measurements": measurements,
                "color": color_bgr,
                "point3": point3,
            }
        )
        processed_tracks += 1

        for frame_idx, u, v in measurements:
            per_frame_measurements[frame_idx].append(((u, v), color_bgr))

    if not track_sequences:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    frames_per_row = min(4, max(1, num_frames))
    grid_rows = math.ceil(num_frames / frames_per_row)
    max_h = restored.shape[2]
    max_w = restored.shape[3]
    grid_canvas = np.zeros((grid_rows * max_h, frames_per_row * max_w, 3), dtype=np.uint8)

    restored_np = (restored.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

    for local_idx, frame in enumerate(restored_np):
        row = local_idx // frames_per_row
        col = local_idx % frames_per_row
        y0 = row * max_h
        x0 = col * max_w
        grid_canvas[y0 : y0 + frame.shape[0], x0 : x0 + frame.shape[1]] = frame

    grid_canvas = cv2.cvtColor(grid_canvas, cv2.COLOR_RGB2BGR)

    for track_info in track_sequences:
        measurements = track_info["measurements"]
        color_bgr = track_info["color"]
        last_grid_point = None
        for frame_idx, u, v in measurements:
            row = frame_idx // frames_per_row
            col = frame_idx % frames_per_row
            grid_pt = (int(round(col * max_w + u)), int(round(row * max_h + v)))
            cv2.circle(grid_canvas, grid_pt, radius=5, color=color_bgr, thickness=-1)
            if last_grid_point is not None:
                cv2.line(grid_canvas, last_grid_point, grid_pt, color=color_bgr, thickness=2, lineType=cv2.LINE_AA)
            last_grid_point = grid_pt if len(measurements) > 1 else None

    cv2.imwrite(str(output_dir / "tracks_grid.png"), grid_canvas)

    for local_idx, frame in enumerate(restored_np):
        canvas = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        for point_idx, ((u, v), color_bgr) in enumerate(per_frame_measurements[local_idx]):
            if point_idx >= MAX_POINTS_PER_FRAME:
                break
            pt = (int(round(u)), int(round(v)))
            cv2.circle(canvas, pt, radius=5, color=color_bgr, thickness=-1)
        cv2.imwrite(str(output_dir / f"frame_{local_idx:04d}.png"), canvas)

    for local_idx, frame in enumerate(restored_np):
        img_idx = image_indices[local_idx]
        camera = gtsfm_data.get_camera(img_idx)
        if camera is None:
            continue

        canvas = cv2.cvtColor(frame.copy(), cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        for track_info in track_sequences:
            point3 = track_info["point3"]
            if point3 is None:
                continue
            try:
                uv = camera.project(point3)
            except Exception:
                continue

            if hasattr(uv, "x"):
                u = float(uv.x())
                v = float(uv.y())
            elif isinstance(uv, np.ndarray):
                u = float(uv[0])
                v = float(uv[1])
            else:
                u = float(uv[0])
                v = float(uv[1])
            if not (0 <= u < width and 0 <= v < height):
                continue

            color_bgr = track_info["color"]
            pt = (int(round(u)), int(round(v)))
            cv2.drawMarker(
                canvas,
                pt,
                color=color_bgr,
                markerType=cv2.MARKER_CROSS,
                markerSize=10,
                thickness=2,
                line_type=cv2.LINE_AA,
            )

        cv2.imwrite(str(output_dir / f"reproj_{local_idx:04d}.png"), canvas)


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

    output_dir = DATA_ROOT_PATH / "vggt_test_output"
    output_subdir = "sparse_wo_ba"
    sparse_reconstruction_dir = output_dir / output_subdir
    print(f"Saving reconstruction to {sparse_reconstruction_dir}")
    sparse_reconstruction_dir.mkdir(parents=True, exist_ok=True)
    result.gtsfm_data.export_as_colmap_text(sparse_reconstruction_dir)

    square_images_cpu = image_batch.detach().cpu()
    original_coords_cpu = original_coords.detach().cpu()

    if result.gtsfm_data.number_tracks() > 0:
        track_output_dir = Path(__file__).resolve().parent / "track_visuals"
        _visualize_gtsfm_tracks_on_original_frames(
            square_images=square_images_cpu,
            original_coords=original_coords_cpu,
            gtsfm_data=result.gtsfm_data,
            image_indices=image_indices,
            output_dir=track_output_dir,
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

    # @unittest.skip("Skipping VGGT end-to-end test for now since it is slow and requires GPU.")
    def test_run_vggt_on_some_images(self):
        """Load four door images using Olsson loader and run vggt on them."""

        img_load_original_resolution = 760
        img_load_resolution = 1024
        loader = OlssonLoader(dataset_dir=str(PALACE), max_resolution=img_load_original_resolution)
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
