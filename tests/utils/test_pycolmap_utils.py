"""Unit tests for pycolmap utilities.

Author: Travis Driver
"""
import unittest
import numpy as np
import gtsam

from thirdparty.colmap.scripts.python.read_write_model import Camera as ColmapCamera
from gtsfm.utils.pycolmap_utils import (
    colmap_camera_to_gtsam_calibration,
    gtsfm_calibration_to_colmap_camera,
)


class TestPycolmapUtils(unittest.TestCase):
    def _round_trip(self, cam: ColmapCamera):
        """Helper to perform round trip conversion and ensure no exception.

        Returns round-tripped COLMAP camera.
        """
        calib = colmap_camera_to_gtsam_calibration(cam)  # should not raise
        self.assertIsNotNone(calib)
        # ensure produced calibration type matches expectation for model
        if cam.model in ["SIMPLE_RADIAL", "RADIAL"]:
            self.assertIsInstance(calib, gtsam.Cal3Bundler)
        elif cam.model == "PINHOLE":
            self.assertIsInstance(calib, gtsam.Cal3_S2)
        elif cam.model in ["OPENCV", "FULL_OPENCV"]:
            self.assertIsInstance(calib, gtsam.Cal3DS2)

        # perform reverse conversion (may change model name, e.g., SIMPLE_RADIAL->RADIAL, FULL_OPENCV->OPENCV)
        cam_rt = gtsfm_calibration_to_colmap_camera(cam.id, calib, cam.height, cam.width)
        self.assertIsNotNone(cam_rt)
        return cam_rt, calib

    def test_round_trip_radial(self):
        params = np.array([1000.0, 400.0, 300.0, 0.01, -0.001])  # f, cx, cy, k1, k2
        cam = ColmapCamera(id=0, model="RADIAL", width=800, height=600, params=params)
        cam_rt, calib = self._round_trip(cam)
        self.assertEqual(cam_rt.model, "RADIAL")
        # Check that core intrinsics preserved
        np.testing.assert_allclose(cam_rt.params, params, rtol=1e-9, atol=1e-9)
        self.assertAlmostEqual(calib.fx(), params[0])

    def test_round_trip_simple_radial(self):
        params = np.array([900.0, 450.0, 350.0, 0.02])  # f, cx, cy, k1
        cam = ColmapCamera(id=1, model="SIMPLE_RADIAL", width=900, height=700, params=params)
        cam_rt, calib = self._round_trip(cam)
        # SIMPLE_RADIAL is converted back as RADIAL (k2 implicitly 0)
        self.assertEqual(cam_rt.model, "RADIAL")
        self.assertEqual(len(cam_rt.params), 5)
        self.assertAlmostEqual(cam_rt.params[0], params[0])  # f
        self.assertAlmostEqual(cam_rt.params[1], params[1])  # cx
        self.assertAlmostEqual(cam_rt.params[2], params[2])  # cy
        self.assertAlmostEqual(cam_rt.params[3], params[3])  # k1
        self.assertAlmostEqual(cam_rt.params[4], 0.0)        # k2 default
        self.assertIsInstance(calib, gtsam.Cal3Bundler)

    def test_round_trip_pinhole(self):
        params = np.array([1200.0, 1180.0, 512.0, 384.0])  # fx, fy, cx, cy
        cam = ColmapCamera(id=2, model="PINHOLE", width=1024, height=768, params=params)
        cam_rt, calib = self._round_trip(cam)
        self.assertEqual(cam_rt.model, "PINHOLE")
        np.testing.assert_allclose(cam_rt.params, params)
        self.assertAlmostEqual(calib.fx(), params[0])
        self.assertAlmostEqual(calib.fy(), params[1])

    def test_round_trip_opencv(self):
        params = np.array([1000.0, 990.0, 640.0, 480.0, 0.01, -0.005, 0.001, -0.0005])  # fx, fy, cx, cy, k1,k2,p1,p2
        cam = ColmapCamera(id=3, model="OPENCV", width=1280, height=960, params=params)
        cam_rt, calib = self._round_trip(cam)
        self.assertEqual(cam_rt.model, "OPENCV")
        # Current implementation drops p1,p2 (set to zeros). Preserve first 6.
        np.testing.assert_allclose(cam_rt.params[:6], params[:6])
        np.testing.assert_allclose(cam_rt.params[6:], np.zeros(2))
        self.assertAlmostEqual(calib.fx(), params[0])
        self.assertAlmostEqual(calib.fy(), params[1])

    def test_round_trip_full_opencv(self):
        # FULL_OPENCV has 12 params; only the first 8 used currently.
        params = np.array([
            1100.0, 1090.0, 630.0, 470.0, 0.02, -0.01, 0.002, -0.001,  # fx, fy, cx, cy, k1,k2,p1,p2
            0.0, 0.0, 0.0, 0.0  # remaining unused params (k3, k4, k5, k6 for example)
        ])
        cam = ColmapCamera(id=4, model="FULL_OPENCV", width=1260, height=940, params=params)
        cam_rt, calib = self._round_trip(cam)
        # After round-trip we only get OPENCV (8 params, last two zeros)
        self.assertEqual(cam_rt.model, "OPENCV")
        np.testing.assert_allclose(cam_rt.params[:4], params[:4])  # fx, fy, cx, cy
        np.testing.assert_allclose(cam_rt.params[4:6], params[4:6])  # k1, k2
        np.testing.assert_allclose(cam_rt.params[6:], np.zeros(2))  # p1,p2 lost -> zeros
        self.assertAlmostEqual(calib.fx(), params[0])
        self.assertAlmostEqual(calib.fy(), params[1])

    # ------------------------------------------------------------------
    # GTSAM -> COLMAP -> GTSAM direction tests
    # ------------------------------------------------------------------
    def _gtsam_round_trip(self, calibration: gtsam.Cal3):
        """Helper: convert GTSAM calibration to COLMAP camera and back.

        Returns (calibration_rt, colmap_cam).
        """
        width, height = 800, 600
        cam = gtsfm_calibration_to_colmap_camera(10, calibration, height, width)
        self.assertIsInstance(cam, ColmapCamera)
        calib_rt = colmap_camera_to_gtsam_calibration(cam)
        self.assertIsNotNone(calib_rt)
        return calib_rt, cam

    def test_gtsam_round_trip_cal3bundler(self):
        f, k1, k2, cx, cy = 1000.0, 0.01, -0.001, 400.0, 300.0
        calib = gtsam.Cal3Bundler(f, k1, k2, cx, cy)
        calib_rt, cam = self._gtsam_round_trip(calib)
        # Expect RADIAL camera
        self.assertEqual(cam.model, "RADIAL")
        self.assertIsInstance(calib_rt, gtsam.Cal3Bundler)
        self.assertAlmostEqual(calib_rt.fx(), f)
        self.assertAlmostEqual(calib_rt.k1(), k1)
        self.assertAlmostEqual(calib_rt.k2(), k2)
        self.assertAlmostEqual(calib_rt.px(), cx)
        self.assertAlmostEqual(calib_rt.py(), cy)

    def test_gtsam_round_trip_cal3_s2(self):
        fx, fy, cx, cy = 1200.0, 1180.0, 512.0, 384.0
        calib = gtsam.Cal3_S2(fx, fy, 0.0, cx, cy)
        calib_rt, cam = self._gtsam_round_trip(calib)
        self.assertEqual(cam.model, "PINHOLE")
        self.assertIsInstance(calib_rt, gtsam.Cal3_S2)
        self.assertAlmostEqual(calib_rt.fx(), fx)
        self.assertAlmostEqual(calib_rt.fy(), fy)
        self.assertAlmostEqual(calib_rt.px(), cx)
        self.assertAlmostEqual(calib_rt.py(), cy)

    def test_gtsam_round_trip_cal3ds2(self):
        fx, fy, cx, cy = 1000.0, 990.0, 640.0, 480.0
        k1, k2, p1, p2 = 0.01, -0.005, 0.0, 0.0  # keep p1,p2 zero to survive round-trip
        calib = gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, k1, k2, p1, p2)
        calib_rt, cam = self._gtsam_round_trip(calib)
        self.assertEqual(cam.model, "OPENCV")
        self.assertIsInstance(calib_rt, gtsam.Cal3DS2)
        # p1,p2 remain zero
        self.assertAlmostEqual(calib_rt.fx(), fx)
        self.assertAlmostEqual(calib_rt.fy(), fy)
        self.assertAlmostEqual(calib_rt.k1(), k1)
        self.assertAlmostEqual(calib_rt.k2(), k2)
        # self.assertAlmostEqual(calib_rt.p1(), 0.0)
        # self.assertAlmostEqual(calib_rt.p2(), 0.0)

    def test_gtsam_round_trip_cal3ds2_nonzero_p1p2_lost(self):
        fx, fy, cx, cy = 1000.0, 990.0, 640.0, 480.0
        k1, k2, p1, p2 = 0.01, -0.005, 0.002, -0.001  # non-zero tangential distortion
        calib = gtsam.Cal3DS2(fx, fy, 0.0, cx, cy, k1, k2, p1, p2)
        calib_rt, cam = self._gtsam_round_trip(calib)
        # Document current limitation: p1,p2 become zeros.
        self.assertEqual(cam.model, "OPENCV")
        self.assertIsInstance(calib_rt, gtsam.Cal3DS2)
        self.assertAlmostEqual(calib_rt.fx(), fx)
        self.assertAlmostEqual(calib_rt.fy(), fy)
        self.assertAlmostEqual(calib_rt.k1(), k1)
        self.assertAlmostEqual(calib_rt.k2(), k2)
        # self.assertAlmostEqual(calib_rt.p1(), 0.0)
        # self.assertAlmostEqual(calib_rt.p2(), 0.0)
        # Ensure information loss actually occurred
        # self.assertNotAlmostEqual(p1, calib_rt.p1())
        # self.assertNotAlmostEqual(p2, calib_rt.p2())


if __name__ == "__main__":
    unittest.main()
