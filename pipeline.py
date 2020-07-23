"""
End-to-end pipeline for GTSFM.

Authors: Ayush Baid
"""

import gtsam

from averaging.rotation.rotation_averaging_base import RotationAveragingBase
from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase
from frontend.frontend_base import FrontEndBase
from loader.loader_base import LoaderBase


class Pipeline:
    """End-to-end pipeline for GTSFM.."""

    def __init__(self, frontend: FrontEndBase,
                 rotation_averaging: RotationAveragingBase,
                 translation_averaging: TranslationAveragingBase):
        """
        Initialize all the modules of GTSFM.

        Args:
            frontend (FrontEndBase): [description]
            rotation_averaging (RotationAveragingBase): [description]
            translation_averaging (TranslationAveragingBase): [description]
        """
        self.frontend = frontend
        self.rotation_averaging = rotation_averaging
        self.translation_averaging = translation_averaging

    def run(self, loader: LoaderBase):
        """Run the pipeline on a dataset.

        Args:
            loader (LoaderBase): loader for the dataset
        """
        frontend_results = self.frontend.run(loader)

        camera_intrinsics = [
            loader.get_intrinsic_matrix(x) for x in range(len(loader))
        ]
        relative_poses = frontend_results.get_relative_poses(camera_intrinsics)

        # TODO: name these as camj_R_cami, or camj_R_cami
        relative_rotations = [x.rotation() for x in relative_poses]
        relative_translations = [x.rotation() for x in relative_poses]

        # TODO: name these as world_R_camera, or camera_R_world
        global_rotations = self.rotation_averaging.run(relative_rotations)

        global_poses = self.translation_averaging.run(
            relative_translations, global_rotations)
