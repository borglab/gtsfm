"""Interface from OpenMVG output to GTSFM data types.

Authors: John Lambert
"""

from pathlib import Path
from typing import Dict

import gtsfm.utils.io as io_utils
import numpy as np
from gtsam import Rot3, Pose3


def load_openmvg_reconstructions_from_json(json_fpath: str) -> Dict[str, Pose3]:
    """Read OpenMVG-specific format ("sfm_data.json") to GTSFM generic types.

    Note: OpenSfM only returns the largest connected component for incremental
    (See Pierre Moulon's comment here: https://github.com/openMVG/openMVG/issues/1938)

    Args:
        json_fpath: path to reconstruction/sfm_data.json file

    Returns:
        pose_dict: mapping from filename to camera global pose.
    """
    if not Path(json_fpath).exists():
        raise ValueError(f"Path does not exist: {json_fpath}")

    data = io_utils.read_json_file(json_fpath)
    assert data["sfm_data_version"] == "0.3"

    intrinsics = data["intrinsics"] # noqa
    #print("OpenMVG Estimated Instrinsics: ", intrinsics)
    view_metadata = data["views"] # noqa
    #print("OpenMVG Estimated View Metadata: ", view_metadata)
    extrinsics = data["extrinsics"]

    key_to_fname_dict = {}
    for view in data["views"]:
        openmvg_key = view["key"]
        filename = view["value"]["ptr_wrapper"]["data"]["filename"]
        key_to_fname_dict[openmvg_key] = filename

    pose_dict = {}

    for ext_info in extrinsics:
        openmvg_key = ext_info["key"]
        R = np.array(ext_info["value"]["rotation"])
        # See https://github.com/openMVG/openMVG/issues/671
        # and http://openmvg.readthedocs.io/en/latest/openMVG/cameras/cameras/#pinhole-camera-model
        t = -R @ np.array(ext_info["value"]["center"])

        cTw = Pose3(Rot3(R), t)
        wTc = cTw.inverse()

        filename = key_to_fname_dict[openmvg_key]
        pose_dict[filename] = wTc

    # TODO: load camera properties
    # TODO: load points w/ RGB colors
    return pose_dict
