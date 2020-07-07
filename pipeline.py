"""
End-to-end pipeline for GTSFM    
"""

import gtsam

from averaging.rotation.rotation_averaging_base import RotationAvergagingBase
from averaging.translation.translation_averaging_base import \
    TranslationAveragingBase
from frontend.frontend_base import FrontEndBase
from loader.loader_base import LoaderBase


class Pipeline:
    def __init__(self, frontend: FrontEndBase, rotation_averaging: RotationAvergagingBase, translation_averaging: TranslationAveragingBase):
        self.frontend = frontend
        self.rotation_averaging = rotation_averaging
        self.translation_averaging = translation_averaging

    def run(self, loader: LoaderBase):
        frontend_results = self.frontend.run(loader)

        global_rotations = self.rotation_averaging.run(
            frontend_results.get_relative_rotations())

        global_translations = self.translation_averaging.run(
            frontend_results.get_relative_translations(), global_rotations)

        global_poses = [gtsam.Pose3(R, t) for (R, t) in zip(
            global_rotations, global_translations)]
