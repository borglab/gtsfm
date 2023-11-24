"""General multi-view bundle adjustment."""

from gtsfm.bundle.bundle_adjustment import BundleAdjustmentOptimizer
from gtsfm.common.gtsfm_data import GtsfmData
from gtsfm.ui.gtsfm_process import GTSFMProcess, UiMetadata


class GlobalBundleAdjustment(GTSFMProcess, BundleAdjustmentOptimizer):
    """Adds UI metadata for global bundle adjustment to BundleAdjustmentOptimizer."""

    def get_ui_metadata() -> UiMetadata:
        """Returns data needed to display node and edge info for this process in the process graph."""

        return UiMetadata(
            display_name="Global Bundle Adjustment",
            input_products=(
                "Absolute Pose Priors",
                "Relative Pose Priors",
                "Camera Intrinsics",
                "Global Translations",
                "Global Rotations",
                "3D Tracks",
            ),
            output_products=("Optimized Camera Poses", "Optimized 3D Tracks"),
            parent_plate="Sparse Reconstruction",
        )

    def is_two_view_ba(self, initial_data: GtsfmData) -> bool:
        """Determines whether two-view bundle adjustment is being executed."""
        return False
