"""Retriever for camera rigs that come with a constraints file.

Author: Frank Dellaert
"""

from typing import List, Tuple

from gtsfm.loader.hilti_loader import HiltiLoader

import gtsfm.utils.logger as logger_utils
from gtsfm.common.constraint import Constraint
from gtsfm.loader.loader_base import LoaderBase
from gtsfm.retriever.retriever_base import RetrieverBase

logger = logger_utils.get_logger()

INTRA_RIG_VALID_PAIRS = {(0, 1), (0, 3), (1, 4)}

CAM2_FRAME_LOOKAHEAD = 2  # number of frames to look ahead for cam2


class RigRetriever(RetrieverBase):
    """Retriever for camera rigs inspired by the Hilti challenge."""

    def __init__(self, subsample: int, threshold: int = 100):
        """Create RigRetriever

        Args:
            threshold (int, optional): amount of "proxy" correspondences that will trigger an image-pair. Default 100.
        """
        self._subsample = subsample
        self._threshold = threshold

    def __accept_constraint(self, constraint: Constraint) -> bool:
        if not self._subsample:
            return True

        return constraint.a % self._subsample == 0 and constraint.b % self._subsample == 0

    def run(self, loader: LoaderBase) -> List[Tuple[int, int]]:
        """Compute potential image pairs for *visual matching*.

        Args:
            loader: image loader.

        Return:
            pair_indices: (i1,i2) image pairs.
        """
        # Get between-rig constraints from HiltiLoader.
        assert isinstance(loader, HiltiLoader)
        constraints: List[Constraint] = loader.get_all_constraints()

        # Get pairs from those constraints.
        unique_pairs = set(
            sum([c.predicted_pairs(self._threshold) for c in constraints if self.__accept_constraint(c)], [])
        )

        num_cam2_pairs = list(filter(lambda edge: edge[0] % 5 == 2 or edge[1] % 5 == 2, unique_pairs))

        logger.info(f"Received {len(constraints)} constraints from loader")
        logger.info(f"Found {len(unique_pairs)} pairs in the constraints file")
        logger.info(f"Found {len(num_cam2_pairs)} pairs with cam2 in the constraints file")

        # Translate all rig level constraints to CAM2-CAM2 constraints
        for constraint in constraints:
            if not self.__accept_constraint(constraint):
                continue
            a = constraint.a
            b = constraint.b

            unique_pairs.add((loader.image_from_rig_and_camera(a, 2), loader.image_from_rig_and_camera(b, 2)))

        # Add all intra-rig pairs even if no LIDAR signal.
        for rig_index in range(0, loader.num_rig_poses, self._subsample if self._subsample else 1):
            for c1, c2 in INTRA_RIG_VALID_PAIRS:
                unique_pairs.add(
                    (loader.image_from_rig_and_camera(rig_index, c1), loader.image_from_rig_and_camera(rig_index, c2))
                )

        pairs = list(unique_pairs)
        pairs.sort()

        logger.info(f"RigRetriever finally created {len(pairs)} pairs.")
        return pairs
