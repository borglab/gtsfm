"""Unit tests for calculating overlap of frustums.

Author: Ren Liu
"""

import unittest

import numpy as np
from gtsam import Cal3_S2, PinholeCameraCal3_S2
from gtsam.examples import SFMdata

import gtsfm.utils.overlap_frustums as overlap_frustums_utils

CUBE_SIZE = 4
CUBE_RESOLUTION = 128
# set the dummy image size as 400x300
IMAGE_W = 400
IMAGE_H = 300

# set dummy camera intrinsics
CAMERA_INTRINSICS = Cal3_S2(
    fx=100.0,
    fy=100.0,
    s=0.0,
    u0=IMAGE_W // 2,
    v0=IMAGE_H // 2,
)
# set dummy camera poses as described in GTSAM example
CAMERA_POSES = SFMdata.createPoses(CAMERA_INTRINSICS)
# set dummy camera instances
CAMERAS = [PinholeCameraCal3_S2(CAMERA_POSES[i], CAMERA_INTRINSICS) for i in range(len(CAMERA_POSES))]

# set dummy sphere grid center
SPHERE_CENTER = np.array([-1.5, -1.5, -1.5])
# set dummy sphere grid radius
SPHERE_RADIUS = 0.2

UNIT_CUBE_CENTER = np.array([0.5, 0.5, 0.5])


class TestOverlapFrustums(unittest.TestCase):
    """Class containing all unit tests for overlap frustums utils."""

    def test_calculate_overlap_frustums(self) -> None:
        """Test whether the overlap frustum area is correct"""
        K = CAMERAS[0].calibration().K()
        iTw_list = [camera.pose().inverse().matrix() for camera in CAMERAS]

        overlap_grids = overlap_frustums_utils.calculate_overlap_frustums(
            CUBE_SIZE, CUBE_RESOLUTION, IMAGE_W, IMAGE_H, K, iTw_list
        )

        overlap_center = overlap_grids.mean(axis=0)

        # the overlapping area should be located in the center of the xy plane
        np.testing.assert_almost_equal(overlap_center[0], 0)
        np.testing.assert_almost_equal(overlap_center[1], 0)

    def test_transform_to_unit_cube(self) -> None:
        """Test whether the transformation to fit with the unit cube is correct"""
        # 1. generate a sphere grid centered at SPHERE_CENTER with radius SPHERE_RADIUS as dummy overlap grid
        cube_grid = overlap_frustums_utils.gen_cube_voxels(-2, -1, 64)
        sphere_grid_id = np.square(cube_grid - SPHERE_CENTER).sum(axis=1) <= SPHERE_RADIUS**2
        sphere_grid = cube_grid[sphere_grid_id]

        # 2. perform transformation
        _, scale, offset = overlap_frustums_utils.transform_to_unit_cube(sphere_grid)

        np.testing.assert_almost_equal(scale, 1.0 / (2 * SPHERE_RADIUS), decimal=1)
        np.testing.assert_almost_equal(offset, UNIT_CUBE_CENTER - SPHERE_CENTER * scale, decimal=1)


if __name__ == "__main__":
    unittest.main()
