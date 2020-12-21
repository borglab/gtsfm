"""Unit tests for the scene-optimizer class.

Authors: Ayush Baid
"""
import unittest
from pathlib import Path
from types import SimpleNamespace

import dask
import numpy as np
from gtsam import EssentialMatrix, Rot3, Unit3

from averaging.rotation.shonan import ShonanRotationAveraging
from averaging.translation.averaging_1dsfm import TranslationAveraging1DSFM
from common.sfm_result import SfmResult
from data_association.data_assoc import TriangulationParam
from frontend.detector_descriptor.sift import SIFTDetectorDescriptor
from frontend.matcher.twoway_matcher import TwoWayMatcher
from frontend.verifier.degensac import Degensac
from loader.folder_loader import FolderLoader
from scene_optimizer import SceneOptimizer

DATA_ROOT_PATH = Path(__file__).resolve().parent / "data"


class TestSceneOptimizer(unittest.TestCase):
    """Unit test for SceneOptimizer, which runs SfM for a scene."""

    def setUp(self) -> None:
        self.loader = FolderLoader(
            str(DATA_ROOT_PATH / "set1_lund_door"), image_extension="JPG"
        )
        assert len(self.loader)
        config = SimpleNamespace(
            **{
                "reproj_error_thresh": 5,
                "min_track_len": 2,
                "triangulation_mode": TriangulationParam.RANSAC_SAMPLE_BIASED_BASELINE,
                "num_ransac_hypotheses": 20,
            }
        )
        self.obj = SceneOptimizer(
            detector_descriptor=SIFTDetectorDescriptor(),
            matcher=TwoWayMatcher(),
            verifier=Degensac(),
            rot_avg_module=ShonanRotationAveraging(),
            trans_avg_module=TranslationAveraging1DSFM(),
            config=config,
            debug_mode=True,
        )

    def test_find_largest_connected_component(self):
        """Tests the function to prune the scene graph to its largest connected
        component."""

        # create a graph with two connected components of length 4 and 3.
        input_essential_matrices = {
            (0, 1): generate_random_essential_matrix(),
            (1, 5): None,
            (3, 1): generate_random_essential_matrix(),
            (3, 2): generate_random_essential_matrix(),
            (2, 7): None,
            (4, 6): generate_random_essential_matrix(),
            (6, 7): generate_random_essential_matrix(),
        }

        # generate Rot3 and Unit3 inputs
        input_relative_rotations = dict()
        input_relative_unit_translations = dict()
        for (i1, i2), i2Ei1 in input_essential_matrices.items():
            if i2Ei1 is None:
                input_relative_rotations[(i1, i2)] = None
                input_relative_unit_translations[(i1, i2)] = None
            else:
                input_relative_rotations[(i1, i2)] = i2Ei1.rotation()
                input_relative_unit_translations[(i1, i2)] = i2Ei1.direction()

        expected_edges = [(0, 1), (3, 2), (3, 1)]

        (
            computed_relative_rotations,
            computed_relative_unit_translations,
        ) = self.obj.multiview_optimizer.select_largest_connected_component(
            input_relative_rotations, input_relative_unit_translations
        )

        # check the edges in the pruned graph
        self.assertCountEqual(
            list(computed_relative_rotations.keys()), expected_edges
        )
        self.assertCountEqual(
            list(computed_relative_unit_translations.keys()), expected_edges
        )

        # check the actual Rot3 and Unit3 values
        for (i1, i2) in expected_edges:
            self.assertTrue(
                computed_relative_rotations[(i1, i2)].equals(
                    input_relative_rotations[(i1, i2)], 1e-2
                )
            )
            self.assertTrue(
                computed_relative_unit_translations[(i1, i2)].equals(
                    input_relative_unit_translations[(i1, i2)], 1e-2
                )
            )

    def test_create_computation_graph(self):

        use_intrinsics_in_verification = False

        # generate the dask computation graph
        sfm_result_graph, viz_graph = self.obj.create_computation_graph(
            len(self.loader),
            self.loader.get_valid_pairs(),
            self.loader.create_computation_graph_for_images(),
            self.loader.create_computation_graph_for_intrinsics(),
            use_intrinsics_in_verification=use_intrinsics_in_verification,
        )

        with dask.config.set(scheduler="single-threaded"):
            sfm_result = dask.compute(sfm_result_graph, viz_graph)[0]

        self.assertIsInstance(sfm_result, SfmResult)


def generate_random_essential_matrix() -> EssentialMatrix:
    rotation_angles = np.random.uniform(low=0.0, high=2 * np.pi, size=(3,))
    R = Rot3.RzRyRx(rotation_angles[0], rotation_angles[1], rotation_angles[2])
    t = np.random.uniform(low=-1.0, high=1.0, size=(3,))

    return EssentialMatrix(R, Unit3(t))


if __name__ == "__main__":
    unittest.main()
