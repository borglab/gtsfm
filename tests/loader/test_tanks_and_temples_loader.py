"""Unit tests for Tanks & Temple dataset loader.

Author: John Lambert
"""

import unittest
from pathlib import Path

_TEST_DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "tanks_and_temples_barn"


class TanksAndTemplesLoaderTest(unittest.TestCase):

    def setUp(self) -> None:

        img_dir = _TEST_DATA_ROOT / 'Barn'
        log_fpath = _TEST_DATA_ROOT / 'Barn_COLMAP_SfM.log'
        ply_fpath = _TEST_DATA_ROOT / 'Barn.ply'
        #ply_fpath = scene_data_dir / 'Barn_COLMAP.ply'
        ply_alignment_fpath = _TEST_DATA_ROOT / 'Barn_trans.txt'
        bounding_polyhedron_json_fpath = _TEST_DATA_ROOT / 'Barn.json'

        loader = TanksAndTemplesLoader(
        img_dir=img_dir,
        poses_fpath=log_fpath,
        ply_fpath=ply_fpath,
        ply_alignment_fpath=ply_alignment_fpath,
        bounding_polyhedron_json_fpath=bounding_polyhedron_json_fpath,
        )

    def test_get_camera_intrinsics_full_res(self) -> None:

        intrinsics = loader.get_camera_intrinsics_full_res(index=0)

    def test_get_camera_pose(self) -> None:

        wTi = loader.get_camera_pose(index=0)
        """
        Unit Test
        (Pdb) p wTi_gt[0]
        array([[-0.43321999, -0.05555365, -0.89957447,  3.24710662],
        [ 0.05678138,  0.99443357, -0.08875668,  0.14032715],
        [ 0.89949781, -0.08953024, -0.42765409,  0.55723886],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
        """
