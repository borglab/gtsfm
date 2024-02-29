"""Unit tests for rotation utils.

Authors: Ayush Baid
"""

import unittest

import gtsfm.utils.geometry_comparisons as geometry_comparisons
import gtsfm.utils.rotation as rotation_util
import tests.data.sample_poses as sample_poses

ROTATION_ANGLE_ERROR_THRESHOLD_DEG = 2


class TestRotationUtil(unittest.TestCase):
    def test_mst_initialization(self):
        """Test for 4 poses in a circle, with a pose connected all others."""
        i2Ri1_dict, wRi_expected = sample_poses.convert_data_for_rotation_averaging(
            sample_poses.CIRCLE_ALL_EDGES_GLOBAL_POSES, sample_poses.CIRCLE_ALL_EDGES_RELATIVE_POSES
        )

        wRi_computed = rotation_util.initialize_global_rotations_using_mst(
            len(wRi_expected),
            i2Ri1_dict,
            edge_weights={(i1, i2): (i1 + i2) * 100 for i1, i2 in i2Ri1_dict.keys()},
        )
        self.assertTrue(
            geometry_comparisons.compare_rotations(wRi_computed, wRi_expected, ROTATION_ANGLE_ERROR_THRESHOLD_DEG)
        )


if __name__ == "__main__":
    unittest.main()
