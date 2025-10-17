"""Container for data associated with a single view in the scene.

Authors: Frank Dellaert, Ayush Baid
"""

from dataclasses import dataclass
from typing import Optional

from dask.delayed import Delayed
from gtsam import Pose3  # type: ignore

import gtsfm.common.types as gtsfm_types
from gtsfm.common.pose_prior import PosePrior
from gtsfm.common.types import CALIBRATION_TYPE


@dataclass(frozen=True)
class OneViewData:
    """Aggregates per-view data items keyed by image index."""

    image_delayed: Delayed
    image_fname: str
    intrinsics: CALIBRATION_TYPE
    absolute_pose_prior: Optional[PosePrior]
    camera_gt: Optional[gtsfm_types.CAMERA_TYPE]
    pose_gt: Optional[Pose3]
