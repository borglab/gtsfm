"""[summary]
"""

from typing import NamedTuple, List

from gtsam import PinholeCameraCal3Bundler, Point3


class SfmResult(NamedTuple):
    # use either this or Pose3, Intrinsics to allow for different camera types.
    cameras: List[PinholeCameraCal3Bundler]
    points3d: List[Point3]  # 3D points

    def get_camera_rotations(self):
        return [x.pose().rotation() for x in self.cameras]

    def get_camera_translations(self):
        return [x.pose().translations() for x in self.cameras]

    def get_camera_intrinsics(self):
        return [x.pose().calibration() for x in self.cameras]
