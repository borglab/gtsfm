"""Unit tests for the final_reconstruction export pieces.

Covers _colorize_scene_tracks_in_place (color sampled at the first in-bounds measurement, at loader
resolution, one image at a time) and the posed-only annex riding into an exported model.

Authors: Kathirvel Gounder
"""

import os
import tempfile
import unittest

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3, SfmTrack

from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.image import Image
from gtsfm.scene_optimizer import _colorize_scene_tracks_in_place

CALIB = Cal3Bundler(100.0, 0.0, 0.0, 32.0, 24.0)


class _StubLoader:
    """Serves a solid-color image per index; records which images were requested."""

    def __init__(self, colors):
        self.colors = colors
        self.requested = []

    def get_image(self, index):
        self.requested.append(index)
        arr = np.zeros((48, 64, 3), dtype=np.uint8)
        arr[:, :] = self.colors[index]
        return Image(value_array=arr)


class TestColorize(unittest.TestCase):
    def _scene(self):
        scene = GtsfmData(number_images=8)
        for i in (0, 1):
            scene.add_camera(i, PinholeCameraCal3Bundler(Pose3(Rot3(), np.array([i, 0.0, 0.0])), CALIB))
        # Track A: first measurement in cam 0 (in bounds) -> cam 0's color.
        ta = SfmTrack(Point3(0.0, 0.0, 5.0))
        ta.addMeasurement(0, np.array([10.0, 10.0]))
        ta.addMeasurement(1, np.array([20.0, 20.0]))
        assert scene.add_track(ta)
        # Track B: first measurement OUT of bounds -> left uncolored (no fallback to later
        # measurements; first-measurement-only is the intended, pinned behavior).
        tb = SfmTrack(Point3(1.0, 0.0, 5.0))
        tb.addMeasurement(1, np.array([500.0, 500.0]))
        tb.addMeasurement(0, np.array([5.0, 5.0]))
        assert scene.add_track(tb)
        return scene

    def test_colors_from_first_measurement_only(self) -> None:
        scene = self._scene()
        loader = _StubLoader({0: (200, 30, 40), 1: (10, 220, 50)})
        n = _colorize_scene_tracks_in_place(scene, loader)
        self.assertEqual(n, 1)
        ta = scene.get_track(0)
        self.assertEqual((int(ta.r), int(ta.g), int(ta.b)), (200, 30, 40))
        tb = scene.get_track(1)
        self.assertEqual((int(tb.r), int(tb.g), int(tb.b)), (0, 0, 0))
        # One load per referenced camera, no more.
        self.assertEqual(sorted(set(loader.requested)), [0, 1])

    def test_failed_image_load_is_nonfatal(self) -> None:
        scene = self._scene()

        class _Boom:
            def get_image(self, index):
                raise RuntimeError("disk gone")

        n = _colorize_scene_tracks_in_place(scene, _Boom())
        self.assertEqual(n, 0)

    def test_annex_cameras_ride_export(self) -> None:
        scene = self._scene()
        annex_cam = PinholeCameraCal3Bundler(Pose3(Rot3(), np.array([9.0, 0.0, 0.0])), CALIB)
        scene.add_camera(5, annex_cam)  # posed-only: no tracks reference camera 5
        with tempfile.TemporaryDirectory() as d:
            scene.export_as_colmap_text(d)
            ls = [l for l in open(os.path.join(d, "images.txt")) if not l.startswith("#")]
            pose_lines = [l for l in ls if len(l.split()) >= 10]
            ids = {int(l.split()[0]) for l in pose_lines}
            self.assertIn(5, ids)
            self.assertEqual(len(pose_lines), 3)


if __name__ == "__main__":
    unittest.main()
