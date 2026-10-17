"""Tests for the post-merge retriangulation stage (``_run_post_merge_retriangulation``).

The stage rebuilds sparse structure from the globally-verified 2D tracks against the merged cameras,
bundle-adjusts with the merge BA options, and — when the merge captured trackless good-pose cameras —
injects their poses so the geometric retriangulation can recover them.

Authors: Kathirvel Gounder
"""

import unittest

import numpy as np
from gtsam import Cal3Bundler, PinholeCameraCal3Bundler, Point3, Pose3, Rot3

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptions
from gtsfm.cluster_merging import MergingOptions
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.common.sfm_track import SfmMeasurement, SfmTrack2d
from gtsfm.scene_optimizer import _run_post_merge_retriangulation


def _make_setup(num_points: int = 24):
    """Four cameras on a baseline viewing a grid; camera 3 is 'trackless' (absent from the scene)."""
    cal = Cal3Bundler(100.0, 0.0, 0.0, 0.0, 0.0)
    cams = {i: PinholeCameraCal3Bundler(Pose3(Rot3(), Point3(0.5 * i, 0.0, 0.0)), cal) for i in range(4)}

    rng = np.random.default_rng(0)
    points = [Point3(x, y, 5.0 + z) for x, y, z in rng.uniform(-1.0, 1.0, size=(num_points, 3))]

    tracks_2d = [
        SfmTrack2d(measurements=[SfmMeasurement(i, np.asarray(cams[i].project(p))) for i in range(4)]) for p in points
    ]

    scene = GtsfmData(number_images=4)
    for i in range(3):  # camera 3 deliberately missing: the merge dropped it as trackless
        scene.add_camera(i, cams[i])
    for i in range(4):
        scene.set_image_info(i, name=f"img_{i}.jpg", shape=(200, 200))

    options = MergingOptions(
        recover_trackless_cameras_in_retriangulation=True,
        ba_options=BundleAdjustmentOptions(
            use_pose_prior_all_cameras=True,
            use_calibration_prior=True,
            calibration_prior_focal_sigma=10.0,
        ),
    )
    return scene, tracks_2d, cams, options


class TestPostMergeRetriangulation(unittest.TestCase):
    def test_rebuilds_structure_against_merged_cameras(self) -> None:
        scene, tracks_2d, cams, options = _make_setup()
        refined = _run_post_merge_retriangulation(scene, tracks_2d, options, trackless_cameras=None)
        self.assertGreater(refined.number_tracks(), 0)
        # Structure is rebuilt geometrically: a recovered point must sit near its generating depth band.
        pt = np.asarray(refined.get_track(0).point3())
        self.assertTrue(3.0 < pt[2] < 7.0, f"retriangulated point {pt} far from the generating z band")

    def test_trackless_camera_is_recovered(self) -> None:
        scene, tracks_2d, cams, options = _make_setup()
        self.assertIsNone(scene.get_camera(3))
        refined = _run_post_merge_retriangulation(scene, tracks_2d, options, trackless_cameras={3: cams[3]})
        self.assertIn(3, refined.get_valid_camera_indices())
        # The all-images metric denominator is preserved from the merged scene.
        self.assertEqual(refined.number_images(), 4)


if __name__ == "__main__":
    unittest.main()
