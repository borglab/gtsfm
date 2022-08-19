import unittest
from pathlib import Path

import numpy as np
from gtsam import Cal3Bundler, Pose3

import gtsfm.utils.io as io_utils
from gtsfm.loader.argoverse_dataset_loader import ArgoverseDatasetLoader

ARGOVERSE_DATA_ROOT_PATH = Path(__file__).resolve().parent.parent / "data" / "argoverse"


class TestArgoverseDatasetLoader(unittest.TestCase):
    """Tests a loader that loads Argoverse image and camera data."""

    def setUp(self) -> None:
        """ """
        self.loader = ArgoverseDatasetLoader(
            dataset_dir=ARGOVERSE_DATA_ROOT_PATH / "train1",
            log_id="273c1883-673a-36bf-b124-88311b1a80be",
            stride=1,
            max_num_imgs=2,
            max_lookahead_sec=50,
            camera_name="ring_front_center",
            max_resolution=1200,
        )
        assert len(self.loader)

    def test_len(self):
        """Test the number of entries in the loader."""
        self.assertEqual(2, len(self.loader))

    def test_get_image_valid_index(self):
        """Tests that get_image works for all valid indices."""

        for idx in range(len(self.loader)):
            self.assertIsNotNone(self.loader.get_image(idx))

    def test_get_image_invalid_index(self):
        """Test that get_image raises an exception on an invalid index."""

        # negative index
        with self.assertRaises(IndexError):
            self.loader.get_image(-1)
        # len() as index
        with self.assertRaises(IndexError):
            self.loader.get_image(3)
        # index > len()
        with self.assertRaises(IndexError):
            self.loader.get_image(15)

    def test_image_contents(self):
        """Test the actual image which is being fetched by the loader at an index.

        This test's primary purpose is to check if the ordering of filename is being respected by the loader
        """
        index_to_test = 1
        # corresponds to dataset split
        split = "train1"
        log_id = "273c1883-673a-36bf-b124-88311b1a80be"
        camera_name = "ring_front_center"
        fname = "ring_front_center_315975643412234000.jpg"

        file_path = ARGOVERSE_DATA_ROOT_PATH / split / log_id / camera_name / fname
        loader_image = self.loader.get_image(index_to_test)
        expected_image = io_utils.load_image(file_path)
        np.testing.assert_allclose(expected_image.value_array, loader_image.value_array)

    def test_get_camera_pose_exists(self):
        """Tests that the correct pose is fetched (present on disk)."""
        fetched_pose = self.loader.get_camera_pose(1)
        self.assertTrue(isinstance(fetched_pose, Pose3))

    def test_get_camera_invalid_index(self):
        """Tests that the camera pose is None, because it is missing on disk."""
        with self.assertRaises(IndexError):
            fetched_pose = self.loader.get_camera_pose(5)
            self.assertIsNone(fetched_pose)

    def test_get_camera_intrinsics_explicit(self):
        """Tests getter for intrinsics when explicit data.mat file with intrinsics are present on disk."""
        expected_fx = 1392.10693

        expected_px = 980.175985
        expected_py = 604.353418

        computed = self.loader.get_camera_intrinsics(0)
        expected = Cal3Bundler(fx=expected_fx, k1=0, k2=0, u0=expected_px, v0=expected_py)
        self.assertTrue(expected.equals(computed, 1e-3))
