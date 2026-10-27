"""Bounded rotation averaging for controlled benchmark experiments.

This retains Shonan's rank-three objective and chordal initialization, but does
not attempt a global-optimality certificate. It is not a production setting.
"""

from typing import List, Optional

import gtsam
from gtsam import Rot3

from gtsfm.averaging.rotation.shonan import ShonanRotationAveraging


class UncertifiedRotationAveraging(ShonanRotationAveraging):
    """Solve Shonan's rank-three problem without optimality certification."""

    def _run_with_consecutive_ordering(
        self, num_connected_nodes: int, measurements: gtsam.BinaryMeasurementsRot3
    ) -> List[Optional[Rot3]]:
        """Estimate rotations for a connected, consecutively numbered graph.

        Args:
            num_connected_nodes: Number of cameras in the connected graph.
            measurements: Relative rotation measurements.

        Returns:
            Estimated rotations in camera-index order.
        """
        parameters = gtsam.ShonanAveragingParameters3(gtsam.LevenbergMarquardtParams.CeresDefaults())
        parameters.setUseHuber(False)
        parameters.setCertifyOptimality(False)
        parameters.setGaugesWeight(0.0)
        parameters.setKarcherWeight(0.0)
        parameters.setAnchorWeight(1.0)
        parameters.setAnchor(measurements[0].key1(), Rot3())
        solver = gtsam.ShonanAveraging3(measurements, parameters)
        result, _ = solver.run(self.chordal_initialize(measurements), 3, 3)
        return [result.atRot3(i) if result.exists(i) else None for i in range(num_connected_nodes)]
