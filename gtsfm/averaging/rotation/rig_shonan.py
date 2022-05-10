"""Shonan Rotation Averaging for the rig multi-sensory devices.

Right now, this class is specific to the Hilti 2022 challenge.
TODO(Ayush): generalize.

This class builds upon the ShonanAveraging class, by adding priors from the top facing camera to 4 other cameras at the
same timestamp. The reason for doing this is cam2 does not have overlap with any other camera and is not matched with\
other cameras, leading to a disconnected component.

Authors: Ayush Baid
"""
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from gtsam import Rot3

import gtsfm.utils.logger as logger_utils
from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging
from gtsfm.common.pose_prior import PosePrior


logger = logger_utils.get_logger()

TEST_DATA_ROOT: Path = Path(__file__).resolve().parent.parent.parent.parent / "tests" / "data"
HILTI_TEST_DATA_PATH: Path = TEST_DATA_ROOT / "hilti_exp4_small"


class RigShonanRotationAveraging(ShonanRotationAveraging):
    """Performs Shonan rotation averaging."""

    def run(
        self,
        num_images: int,
        i2Ri1_dict: Dict[Tuple[int, int], Optional[Rot3]],
        i2Ti1_priors: Dict[Tuple[int, int], Optional[PosePrior]],
    ) -> List[Optional[Rot3]]:
        """Run the rotation averaging on a connected graph with arbitrary keys, where each key is a image/pose index.

        Note: run() functions as a wrapper that re-orders keys to prepare a graph w/ N keys ordered [0,...,N-1].
        All input nodes must belong to a single connected component, in order to obtain an absolute pose for each
        camera in a single, global coordinate frame.

        Args:
            num_images: number of images. Since we have one pose per image, it is also the number of poses.
            i2Ri1_dict: relative rotations for each image pair-edge as dictionary (i1, i2): i2Ri1.
            i2Ti1_priors: priors on relative poses.

        Returns:
            Global rotations for each camera pose, i.e. wRi, as a list. The number of entries in the list is
                `num_images`. The list may contain `None` where the global rotation could not be computed (either
                underconstrained system or ill-constrained system), or where the camera pose had no valid observation
                in the input to run().
        """
        if len(i2Ri1_dict) == 0:
            logger.warning("Shonan cannot proceed: No cycle-consistent triplets found after filtering.")
            wRi_list = [None] * num_images
            return wRi_list

        # Hack: Add the edges from cam2 to the same rig to i2Ri1_dict
        for i in range(2, num_images, 5):
            i2Ri1_dict[(i - 2, i)] = i2Ti1_priors[(i - 2, i)].value.rotation()
            i2Ri1_dict[(i - 1, i)] = i2Ti1_priors[(i - 1, i)].value.rotation()
            i2Ri1_dict[(i, i + 1)] = i2Ti1_priors[(i, i + 1)].value.rotation()
            i2Ri1_dict[(i, i + 2)] = i2Ti1_priors[(i, i + 2)].value.rotation()

        nodes_with_edges = set()
        for (i1, i2) in i2Ri1_dict.keys():
            nodes_with_edges.add(i1)
            nodes_with_edges.add(i2)

        nodes_with_edges = sorted(list(nodes_with_edges))

        # given original index, this map gives back a new temporary index, starting at 0
        reordered_idx_map = {}
        for (new_idx, i) in enumerate(nodes_with_edges):
            reordered_idx_map[i] = new_idx

        # now, map the original indices to reordered indices
        i2Ri1_dict_reordered = {}
        for (i1, i2), i2Ri1 in i2Ri1_dict.items():
            i1_ = reordered_idx_map[i1]
            i2_ = reordered_idx_map[i2]
            i2Ri1_dict_reordered[(i1_, i2_)] = i2Ri1

        wRi_list_subset = self._run_with_consecutive_ordering(
            num_connected_nodes=len(nodes_with_edges), i2Ri1_dict=i2Ri1_dict_reordered
        )

        wRi_list = [None] * num_images
        for remapped_i, original_i in enumerate(nodes_with_edges):
            wRi_list[original_i] = wRi_list_subset[remapped_i]

        return wRi_list
